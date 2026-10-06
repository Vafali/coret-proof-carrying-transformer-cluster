#!/usr/bin/env python3
"""Independent O(gD) containment checker for native precise-dot radii.

The checker consumes immutable IEEE binary32 operand blocks.  It does not
import Torch, NumPy, producer code, or the pair-enumerating reference checker.
CUDA is used only to perform three deterministic positive FP64 sums.  Their
roundoff is enclosed with 256-bit directed MPFR before the factorized bound is
formed.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

import gmpy2


SCHEMA = "CORET_FACTORIZED_PRECISE_DOT_CHECKER_V1"
PRECISION = 256


CUDA_SOURCE = r'''
__device__ __forceinline__ long loff(int h, int g, int r, int d,
                                     int ng, int rows, int inner) {
  return (((long)h * (ng + 1) + g) * rows + r) * inner + d;
}
__device__ __forceinline__ long roff(int h, int g, int c, int d,
                                     int ng, int columns, int inner) {
  return (((long)h * (ng + 1) + g) * columns + c) * inner + d;
}

extern "C" __global__ void factor_sums_f32(
    const float* left, const float* right,
    double* asum, double* bsum, double* dsum,
    int heads, int ga, int gb, int rows, int columns, int inner) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long acount = (long)heads * rows * inner;
  long bcount = (long)heads * columns * inner;
  long dcount = (long)heads * rows * columns * inner;
  if (tid < acount) {
    int d = tid % inner;
    int q = (tid / inner) % rows;
    int h = tid / ((long)rows * inner);
    double value = 0.0;
    for (int g = 0; g < ga; ++g) {
      value = __dadd_rn(value, fabs((double)left[
          loff(h, g + 1, q, d, ga, rows, inner)]));
    }
    asum[tid] = value;
  }
  if (tid < bcount) {
    int d = tid % inner;
    int c = (tid / inner) % columns;
    int h = tid / ((long)columns * inner);
    double value = 0.0;
    for (int g = 0; g < gb; ++g) {
      value = __dadd_rn(value, fabs((double)right[
          roff(h, g + 1, c, d, gb, columns, inner)]));
    }
    bsum[tid] = value;
  }
  if (tid < dcount) {
    int d = tid % inner;
    int c = (tid / inner) % columns;
    int q = (tid / ((long)inner * columns)) % rows;
    int h = tid / ((long)rows * columns * inner);
    int gmin = ga < gb ? ga : gb;
    double value = 0.0;
    for (int g = 0; g < gmin; ++g) {
      double av = (double)left[loff(h, g + 1, q, d,
                                         ga, rows, inner)];
      double bv = (double)right[roff(h, g + 1, c, d,
                                          gb, columns, inner)];
      value = __dadd_rn(value, fabs(__dmul_rn(av, bv)));
    }
    dsum[tid] = value;
  }
}
'''
CUDA_SOURCE_SHA256 = "4103762e82560a17d401075770b1beab7d5c9be0c5640eac141b3dbe1a7e8282"


def _context(rounding):
    return gmpy2.local_context(
        gmpy2.context(), precision=PRECISION, round=rounding)


def _mp(value):
    with _context(gmpy2.RoundToNearest):
        return gmpy2.mpfr(value)


def _gamma(count: int):
    if count < 0:
        raise AssertionError("negative operation count")
    with _context(gmpy2.RoundUp):
        u = gmpy2.mpfr(2) ** -53
        product = count * u
        if product >= 1:
            raise AssertionError("FP64 error-model precondition failed")
        return product / (1 - product)


def directed_factorized_coordinate(asums, bsums, dsums, ga: int, gb: int):
    """Return a directed upper bound from deterministic FP64 summaries."""
    if not (len(asums) == len(bsums) == len(dsums)):
        raise AssertionError("factorized summary width mismatch")
    gmin = min(ga, gb)
    if any(not math.isfinite(float(item)) or float(item) < 0
           for item in (*asums, *bsums, *dsums)):
        raise AssertionError("nonfinite or negative factorized summary")
    ga_error, gb_error, gd_error = (_gamma(ga), _gamma(gb), _gamma(gmin))
    with _context(gmpy2.RoundUp):
        result = gmpy2.mpfr(0)
    for ahat, bhat, dhat in zip(asums, bsums, dsums):
        with _context(gmpy2.RoundUp):
            a_upper = _mp(float(ahat)) / (1 - ga_error)
            b_upper = _mp(float(bhat)) / (1 - gb_error)
        # fl(sum positive exact products) <= (1+gamma) * exact.  This
        # quotient is a lower bound and therefore must be rounded down.
        with _context(gmpy2.RoundDown):
            d_lower = _mp(float(dhat)) / (1 + gd_error)
        with _context(gmpy2.RoundUp):
            term = a_upper * b_upper - d_lower / 2
            if term < 0:
                # The exact term is nonnegative.  A negative value would mean
                # the summary/error model is internally inconsistent.
                raise AssertionError("negative factorized coordinate term")
            result += term
    return result


class _NVRTC:
    def __init__(self):
        environment_candidates = sorted(
            (Path(sys.prefix) / "lib").glob(
                "python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*"))
        candidates = [
            os.environ.get("CORET_NVRTC_LIBRARY"),
            *(str(path) for path in environment_candidates),
            "/home/vafali_ubuntu/anaconda3/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
            ctypes.util.find_library("nvrtc"),
        ]
        for candidate in candidates:
            if candidate:
                try:
                    self.lib = ctypes.CDLL(candidate)
                    self.path = candidate
                    break
                except OSError:
                    pass
        else:
            raise RuntimeError("NVRTC library is unavailable")
        self.lib.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.nvrtcGetProgramLogSize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.nvrtcGetPTXSize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    @staticmethod
    def _check(code, label):
        if code:
            raise RuntimeError(f"NVRTC {label} failed with code {code}")

    def compile(self, source: str, architecture: str):
        program = ctypes.c_void_p()
        encoded = source.encode()
        self._check(self.lib.nvrtcCreateProgram(
            ctypes.byref(program), encoded, b"factorized_checker.cu",
            0, None, None), "create")
        options = [f"--gpu-architecture=compute_{architecture}".encode(),
                   b"--std=c++11", b"--fmad=false", b"--ftz=false"]
        array_type = ctypes.c_char_p * len(options)
        status = self.lib.nvrtcCompileProgram(
            program, len(options), array_type(*options))
        log_size = ctypes.c_size_t()
        self.lib.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value or 1)
        self.lib.nvrtcGetProgramLog(program, log)
        if status:
            raise RuntimeError(
                f"NVRTC compile failed with code {status}:\n{log.value.decode()}")
        ptx_size = ctypes.c_size_t()
        self._check(self.lib.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)),
                    "get PTX size")
        ptx = ctypes.create_string_buffer(ptx_size.value)
        self._check(self.lib.nvrtcGetPTX(program, ptx), "get PTX")
        self.lib.nvrtcDestroyProgram(ctypes.byref(program))
        return ptx.raw, log.value.decode()


class _CUDA:
    def __init__(self, device_index=0):
        candidates = [os.environ.get("CORET_CUDA_DRIVER"),
                      "/usr/lib/wsl/lib/libcuda.so.1",
                      ctypes.util.find_library("cuda")]
        for candidate in candidates:
            if candidate:
                try:
                    self.lib = ctypes.CDLL(candidate)
                    self.path = candidate
                    break
                except OSError:
                    pass
        else:
            raise RuntimeError("CUDA driver library is unavailable")
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self.lib.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        self.lib.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.lib.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.lib.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self._check(self.lib.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), device_index),
                    "cuDeviceGet")
        self.context = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(
            ctypes.byref(self.context), 0, self.device), "cuCtxCreate")
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(
            ctypes.byref(major), 75, self.device), "compute capability major")
        self._check(self.lib.cuDeviceGetAttribute(
            ctypes.byref(minor), 76, self.device), "compute capability minor")
        self.architecture = f"{major.value}{minor.value}"
        self.allocations = []

    @staticmethod
    def _check(code, label):
        if code:
            raise RuntimeError(f"CUDA {label} failed with code {code}")

    def alloc(self, size):
        pointer = ctypes.c_uint64()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(pointer), size),
                    "cuMemAlloc")
        self.allocations.append(pointer.value)
        return pointer.value

    def upload(self, raw):
        pointer = self.alloc(len(raw))
        buffer = ctypes.create_string_buffer(raw)
        self._check(self.lib.cuMemcpyHtoD_v2(
            ctypes.c_uint64(pointer), buffer, len(raw)), "cuMemcpyHtoD")
        return pointer

    def download(self, pointer, size):
        buffer = ctypes.create_string_buffer(size)
        self._check(self.lib.cuMemcpyDtoH_v2(
            buffer, ctypes.c_uint64(pointer), size), "cuMemcpyDtoH")
        return buffer.raw

    def module(self, ptx):
        module = ctypes.c_void_p()
        data = ctypes.create_string_buffer(ptx)
        self._check(self.lib.cuModuleLoadData(
            ctypes.byref(module), data), "cuModuleLoadData")
        return module, data

    def function(self, module, name):
        function = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(
            ctypes.byref(function), module, name.encode()),
            f"cuModuleGetFunction({name})")
        return function

    def launch(self, function, grid, block, arguments):
        pointers = (ctypes.c_void_p * len(arguments))(
            *(ctypes.cast(ctypes.byref(item), ctypes.c_void_p)
              for item in arguments))
        self._check(self.lib.cuLaunchKernel(
            function, grid, 1, 1, block, 1, 1, 0, None, pointers, None),
            "cuLaunchKernel")

    def synchronize(self):
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self):
        for pointer in reversed(self.allocations):
            self.lib.cuMemFree_v2(ctypes.c_uint64(pointer))
        self.allocations.clear()
        if self.context:
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = None


def _pointer(value):
    return ctypes.c_uint64(value)


def _integer(value):
    return ctypes.c_int(value)


def _unpack_doubles(raw):
    return tuple(item[0] for item in struct.iter_unpack("<d", raw))


def check_factorized_f32(*, left_raw: bytes, right_raw: bytes,
                         left_shape: tuple[int, int, int, int],
                         right_shape: tuple[int, int, int, int],
                         stored_radii_raw: bytes | None = None,
                         device_index: int = 0) -> dict[str, Any]:
    """Compute the directed factorized radius bound for every coordinate."""
    heads, ga1, rows, inner = left_shape
    rheads, gb1, columns, rinner = right_shape
    ga, gb = ga1 - 1, gb1 - 1
    if heads <= 0 or min(ga, gb, rows, columns, inner) < 0:
        raise AssertionError("invalid factorized checker dimensions")
    if (rheads, rinner) != (heads, inner):
        raise AssertionError("factorized checker shape mismatch")
    expected = (math.prod(left_shape) * 4, math.prod(right_shape) * 4)
    if (len(left_raw), len(right_raw)) != expected:
        raise AssertionError("factorized checker byte count mismatch")
    outputs = heads * rows * columns
    if stored_radii_raw is not None and len(stored_radii_raw) != outputs * 4:
        raise AssertionError("stored radius byte count mismatch")
    source_hash = hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()
    if source_hash != CUDA_SOURCE_SHA256:
        raise AssertionError("factorized CUDA source identity mismatch")
    started = time.perf_counter()
    cuda = _CUDA(device_index)
    try:
        compiler = _NVRTC()
        compile_started = time.perf_counter()
        ptx, compile_log = compiler.compile(CUDA_SOURCE, cuda.architecture)
        compile_seconds = time.perf_counter() - compile_started
        module, module_buffer = cuda.module(ptx)
        function = cuda.function(module, "factor_sums_f32")
        left_p, right_p = cuda.upload(left_raw), cuda.upload(right_raw)
        acount = heads * rows * inner
        bcount = heads * columns * inner
        dcount = outputs * inner
        asum_p = cuda.alloc(acount * 8)
        bsum_p = cuda.alloc(bcount * 8)
        dsum_p = cuda.alloc(dcount * 8)
        maximum = max(acount, bcount, dcount)
        arguments = [
            _pointer(left_p), _pointer(right_p), _pointer(asum_p),
            _pointer(bsum_p), _pointer(dsum_p),
            *(_integer(item) for item in
              (heads, ga, gb, rows, columns, inner))]
        kernel_started = time.perf_counter()
        cuda.launch(function, (maximum + 255) // 256, 256, arguments)
        cuda.synchronize()
        kernel_seconds = time.perf_counter() - kernel_started
        asums = _unpack_doubles(cuda.download(asum_p, acount * 8))
        bsums = _unpack_doubles(cuda.download(bsum_p, bcount * 8))
        dsums = _unpack_doubles(cuda.download(dsum_p, dcount * 8))
    finally:
        cuda.close()
    stored = (None if stored_radii_raw is None else
              tuple(item[0] for item in
                    struct.iter_unpack("<f", stored_radii_raw)))
    bounds, etas = [], []
    maximum_bound, maximum_eta = _mp(0), _mp(0)
    for h in range(heads):
        for q in range(rows):
            abase = (h * rows + q) * inner
            avec = asums[abase:abase + inner]
            for c in range(columns):
                bbase = (h * columns + c) * inner
                dbase = ((h * rows + q) * columns + c) * inner
                bound = directed_factorized_coordinate(
                    avec, bsums[bbase:bbase + inner],
                    dsums[dbase:dbase + inner], ga, gb)
                bounds.append(str(bound))
                maximum_bound = max(maximum_bound, bound)
                if stored is not None:
                    value = _mp(stored[len(bounds) - 1])
                    eta = max(_mp(0), bound - value)
                    etas.append(str(eta))
                    maximum_eta = max(maximum_eta, eta)
    return {
        "schema": SCHEMA,
        "coordinates_checked": outputs,
        "factorized_upper_bounds": bounds,
        "required_extra_eta": etas if stored is not None else None,
        "maximum_factorized_upper": str(maximum_bound),
        "maximum_required_extra_eta": (
            str(maximum_eta) if stored is not None else None),
        "compile_seconds": compile_seconds,
        "kernel_seconds": kernel_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_device_bytes": (len(left_raw) + len(right_raw)
                              + (acount + bcount + dcount) * 8),
        "architecture": cuda.architecture,
        "compiler_flags": ["--std=c++11", "--fmad=false", "--ftz=false"],
        "source_sha256": source_hash,
        "cuda_driver": cuda.path,
        "nvrtc_library": compiler.path,
        "compile_log": compile_log,
        "scientific_queries": 0,
        "bound_calls": 0,
    }
