#!/usr/bin/env python3
"""Independent rigorous-FP64 backend for frozen float32 precise dot.

This module deliberately imports neither Torch nor producer code.  CUDA source
is compiled by NVRTC and invoked through the CUDA driver API.  The native
kernel returns deterministic FP64 summaries; this module converts them into a
directed MPFR upper bound using the documented rounding model.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.util
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

import gmpy2


SCHEMA = "CORET_RIGOROUS_PRECISE_DOT_BACKEND_V1"
BLOCK_THREADS = 256
STAT_FIELDS = 13
PRECISION = 256
UNIT_ROUNDOFF_HEX = "0x1.0000000000000p-53"


CUDA_SOURCE = r'''
#define THREADS 256
#define FIELDS 13

__device__ __forceinline__ long loff(int h, int g, int r, int d,
                                     int ng, int rows, int inner) {
  return (((long)h * (ng + 1) + g) * rows + r) * inner + d;
}

__device__ __forceinline__ long roff(int h, int g, int c, int d,
                                     int ng, int columns, int inner) {
  return (((long)h * (ng + 1) + g) * columns + c) * inner + d;
}

__device__ __forceinline__ long ooff(int h, int g, int r, int c,
                                     int ng, int rows, int columns) {
  return (((long)h * ng + g) * rows + r) * columns + c;
}

extern "C" __global__ void build_active_f32(
    const float* left, const float* right, int* indices, int* counts,
    int heads, int ga, int gb, int rows, int columns, int inner) {
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  int outputs = heads * rows * columns;
  if (out >= outputs) return;
  int c = out % columns;
  int q = (out / columns) % rows;
  int h = out / (rows * columns);
  int gmax = ga > gb ? ga : gb;
  int count = 0;
  for (int g = 0; g < gmax; ++g) {
    bool active = false;
    if (g < ga) {
      for (int d = 0; d < inner; ++d) {
        if (left[loff(h, g + 1, q, d, ga, rows, inner)] != 0.0f) {
          active = true; break;
        }
      }
    }
    if (!active && g < gb) {
      for (int d = 0; d < inner; ++d) {
        if (right[roff(h, g + 1, c, d, gb, columns, inner)] != 0.0f) {
          active = true; break;
        }
      }
    }
    if (active) indices[(long)out * gmax + count++] = g;
  }
  counts[out] = count;
}

extern "C" __global__ void precise_stats_f32(
    const float* left, const float* right, const float* output,
    const int* indices, const int* counts, double* stats,
    int heads, int ga, int gb, int rows, int columns, int inner) {
  int out = blockIdx.x;
  int outputs = heads * rows * columns;
  if (out >= outputs) return;
  int tid = threadIdx.x;
  int c = out % columns;
  int q = (out / columns) % rows;
  int h = out / (rows * columns);
  int gmax = ga > gb ? ga : gb;
  int gmin = ga < gb ? ga : gb;
  int fresh = heads * rows * columns;
  int output_ng = 1 + gmax + fresh;

  __shared__ double sh_ret_h[THREADS];
  __shared__ double sh_ret_s[THREADS];
  __shared__ double sh_ret_m[THREADS];
  __shared__ double sh_rad_h[THREADS];
  __shared__ double sh_rad_s[THREADS];
  __shared__ unsigned long long sh_rt[THREADS];
  __shared__ unsigned long long sh_rp[THREADS];
  __shared__ unsigned long long sh_qt[THREADS];
  __shared__ unsigned long long sh_qp[THREADS];

  double ret_h = 0.0, ret_s = 0.0, ret_m = 0.0;
  unsigned long long ret_terms = 0, ret_products = 0;
  for (int g = tid; g < gmax; g += blockDim.x) {
    double value = 0.0, absolute_products = 0.0;
    if (g < gb) {
      for (int d = 0; d < inner; ++d) {
        double av = (double)left[loff(h, 0, q, d, ga, rows, inner)];
        double bv = (double)right[roff(h, g + 1, c, d, gb, columns, inner)];
        value = __fma_rn(av, bv, value);
        absolute_products = __dadd_rn(absolute_products,
                                      fabs(__dmul_rn(av, bv)));
        ++ret_products;
      }
    }
    if (g < ga) {
      for (int d = 0; d < inner; ++d) {
        double av = (double)left[loff(h, g + 1, q, d, ga, rows, inner)];
        double bv = (double)right[roff(h, 0, c, d, gb, columns, inner)];
        value = __fma_rn(av, bv, value);
        absolute_products = __dadd_rn(absolute_products,
                                      fabs(__dmul_rn(av, bv)));
        ++ret_products;
      }
    }
    double stored = (double)output[ooff(
        h, g + 1, q, c, output_ng, rows, columns)];
    ret_h = __dadd_rn(ret_h, fabs(__dsub_rn(value, stored)));
    ret_s = __dadd_rn(ret_s, absolute_products);
    ret_m = __dadd_rn(ret_m,
                      __dadd_rn(fabs(value), fabs(stored)));
    ++ret_terms;
  }

  double rad_h = 0.0, rad_s = 0.0;
  unsigned long long rad_terms = 0, rad_products = 0;
  int active = counts[out];
  const int* active_indices = indices + (long)out * gmax;
  for (int ii = tid; ii < active; ii += blockDim.x) {
    int i = active_indices[ii];
    if (i < gmin) {
      double value = 0.0, absolute_products = 0.0;
      bool possible = false;
      for (int d = 0; d < inner; ++d) {
        double av = (double)left[loff(h, i + 1, q, d, ga, rows, inner)];
        double bv = (double)right[roff(h, i + 1, c, d, gb, columns, inner)];
        if (av != 0.0 && bv != 0.0) possible = true;
        value = __fma_rn(av, bv, value);
        absolute_products = __dadd_rn(absolute_products,
                                      fabs(__dmul_rn(av, bv)));
      }
      if (possible) {
        rad_h = __dadd_rn(rad_h, __dmul_rn(0.5, fabs(value)));
        rad_s = __dadd_rn(rad_s, __dmul_rn(0.5, absolute_products));
        ++rad_terms; rad_products += inner;
      }
    }
    for (int jj = ii + 1; jj < active; ++jj) {
      int j = active_indices[jj];
      double value = 0.0, absolute_products = 0.0;
      bool possible = false;
      if (i < ga && j < gb) {
        for (int d = 0; d < inner; ++d) {
          double av = (double)left[loff(h, i + 1, q, d, ga, rows, inner)];
          double bv = (double)right[roff(h, j + 1, c, d, gb, columns, inner)];
          if (av != 0.0 && bv != 0.0) possible = true;
          value = __fma_rn(av, bv, value);
          absolute_products = __dadd_rn(absolute_products,
                                        fabs(__dmul_rn(av, bv)));
          rad_products++;
        }
      }
      if (j < ga && i < gb) {
        for (int d = 0; d < inner; ++d) {
          double av = (double)left[loff(h, j + 1, q, d, ga, rows, inner)];
          double bv = (double)right[roff(h, i + 1, c, d, gb, columns, inner)];
          if (av != 0.0 && bv != 0.0) possible = true;
          value = __fma_rn(av, bv, value);
          absolute_products = __dadd_rn(absolute_products,
                                        fabs(__dmul_rn(av, bv)));
          rad_products++;
        }
      }
      if (possible) {
        rad_h = __dadd_rn(rad_h, fabs(value));
        rad_s = __dadd_rn(rad_s, absolute_products);
        ++rad_terms;
      }
    }
  }

  sh_ret_h[tid] = ret_h; sh_ret_s[tid] = ret_s; sh_ret_m[tid] = ret_m;
  sh_rad_h[tid] = rad_h; sh_rad_s[tid] = rad_s;
  sh_rt[tid] = ret_terms; sh_rp[tid] = ret_products;
  sh_qt[tid] = rad_terms; sh_qp[tid] = rad_products;
  __syncthreads();
  for (int stride = THREADS / 2; stride; stride >>= 1) {
    if (tid < stride) {
      sh_ret_h[tid] = __dadd_rn(sh_ret_h[tid], sh_ret_h[tid + stride]);
      sh_ret_s[tid] = __dadd_rn(sh_ret_s[tid], sh_ret_s[tid + stride]);
      sh_ret_m[tid] = __dadd_rn(sh_ret_m[tid], sh_ret_m[tid + stride]);
      sh_rad_h[tid] = __dadd_rn(sh_rad_h[tid], sh_rad_h[tid + stride]);
      sh_rad_s[tid] = __dadd_rn(sh_rad_s[tid], sh_rad_s[tid + stride]);
      sh_rt[tid] += sh_rt[tid + stride]; sh_rp[tid] += sh_rp[tid + stride];
      sh_qt[tid] += sh_qt[tid + stride]; sh_qp[tid] += sh_qp[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    double center = 0.0, center_s = 0.0;
    unsigned long long center_products = 0;
    for (int d = 0; d < inner; ++d) {
      double av = (double)left[loff(h, 0, q, d, ga, rows, inner)];
      double bv = (double)right[roff(h, 0, c, d, gb, columns, inner)];
      center = __fma_rn(av, bv, center);
      center_s = __dadd_rn(center_s, fabs(__dmul_rn(av, bv)));
      ++center_products;
    }
    for (int g = 0; g < gmin; ++g) {
      for (int d = 0; d < inner; ++d) {
        double av = __dmul_rn(0.5, (double)left[
            loff(h, g + 1, q, d, ga, rows, inner)]);
        double bv = (double)right[roff(h, g + 1, c, d, gb, columns, inner)];
        center = __fma_rn(av, bv, center);
        center_s = __dadd_rn(center_s, fabs(__dmul_rn(av, bv)));
        ++center_products;
      }
    }
    long base = (long)out * FIELDS;
    stats[base + 0] = center;
    stats[base + 1] = center_s;
    stats[base + 2] = sh_ret_h[0];
    stats[base + 3] = sh_ret_s[0];
    stats[base + 4] = sh_ret_m[0];
    stats[base + 5] = sh_rad_h[0];
    stats[base + 6] = sh_rad_s[0];
    stats[base + 7] = (double)center_products;
    stats[base + 8] = (double)sh_rt[0];
    stats[base + 9] = (double)sh_rp[0];
    stats[base + 10] = (double)sh_qt[0];
    stats[base + 11] = (double)sh_qp[0];
    stats[base + 12] = (double)active;
  }
}
'''
CUDA_SOURCE_SHA256 = (
    "131e466c835de9a20b7848849c1dac1ada887ecba8e020487def4d4e7c36546b")


def _context(rounding):
    return gmpy2.local_context(
        gmpy2.context(), precision=PRECISION, round=rounding)


def _mp(value):
    with _context(gmpy2.RoundToNearest):
        return gmpy2.mpfr(value)


def _up(function):
    with _context(gmpy2.RoundUp):
        return +function()


def _gamma(count: int):
    if count < 0:
        raise AssertionError("negative operation count")
    with _context(gmpy2.RoundUp):
        u = gmpy2.mpfr(2) ** -53
        product = count * u
        if product >= 1:
            raise AssertionError("FP64 error model precondition failed")
        return product / (1 - product)


def _float32_fraction(value: float) -> Fraction:
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = -1 if word >> 31 else 1
    exponent = (word >> 23) & 255
    fraction = word & 0x7fffff
    if exponent == 255:
        raise AssertionError("nonfinite binary32 input")
    if exponent == 0:
        mantissa, power = fraction, -149
    else:
        mantissa, power = (1 << 23) | fraction, exponent - 127 - 23
    numerator = sign * mantissa
    return (Fraction(numerator << power, 1) if power >= 0
            else Fraction(numerator, 1 << -power))


def reference_cross_f32(first_a, first_b, second_a, second_b) -> Fraction:
    """Exact-dyadic bounded oracle for one symmetric cross term."""
    if not (len(first_a) == len(first_b) == len(second_a) == len(second_b)):
        raise AssertionError("cross vectors have unequal widths")
    result = Fraction(0)
    for left, right in ((first_a, second_b), (second_a, first_b)):
        for av, bv in zip(left, right):
            result += _float32_fraction(av) * _float32_fraction(bv)
    return result


def rigorous_cross_f32(first_a, first_b, second_a, second_b):
    """Host analogue of the kernel schedule, for bounded differential tests."""
    terms = []
    for left, right in ((first_a, second_b), (second_a, first_b)):
        for av, bv in zip(left, right):
            # Both operands are first rounded to binary32 by the witness.
            af = struct.unpack("<f", struct.pack("<f", av))[0]
            bf = struct.unpack("<f", struct.pack("<f", bv))[0]
            terms.append(float(af) * float(bf))
    approximation = 0.0
    absolute_sum = 0.0
    for product in terms:
        approximation += product
        absolute_sum += abs(product)
    with _context(gmpy2.RoundUp):
        product_upper = _mp(absolute_sum) / (1 - _gamma(len(terms)))
        error = _gamma(len(terms)) * product_upper
        lower = _mp(approximation) - error
        upper = _mp(approximation) + error
    return {
        "approximation_hex": approximation.hex(),
        "absolute_product_sum_hex": absolute_sum.hex(),
        "lower": str(lower), "upper": str(upper),
        "term_count": len(terms),
    }


@dataclass(frozen=True)
class CoordinateResult:
    required_upper: str
    numerical_sidecar_radius: str
    stored_radius_hex: str
    accepted: bool
    center_upper: str
    retained_upper: str
    radius_upper: str
    active_generators: int
    retained_terms: int
    radius_terms: int
    radius_products: int


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
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p)]
        self.lib.nvrtcGetProgramLogSize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.nvrtcGetPTXSize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvrtcCreateProgram.restype = ctypes.c_int
        self.lib.nvrtcCompileProgram.restype = ctypes.c_int

    def _check(self, code, label):
        if code:
            raise RuntimeError(f"NVRTC {label} failed with code {code}")

    def compile(self, source: str, architecture: str) -> tuple[bytes, str]:
        program = ctypes.c_void_p()
        encoded = source.encode("utf-8")
        self._check(self.lib.nvrtcCreateProgram(
            ctypes.byref(program), encoded, b"checker.cu", 0, None, None),
            "create")
        options = [
            f"--gpu-architecture=compute_{architecture}".encode(),
            b"--std=c++11", b"--fmad=false", b"--ftz=false",
        ]
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
    def __init__(self, device_index: int = 0):
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
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self._check(self.lib.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), device_index),
                    "cuDeviceGet")
        self.context = ctypes.c_void_p()
        create = getattr(self.lib, "cuCtxCreate_v2")
        self._check(create(ctypes.byref(self.context), 0, self.device),
                    "cuCtxCreate")
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(
            ctypes.byref(major), 75, self.device), "compute capability major")
        self._check(self.lib.cuDeviceGetAttribute(
            ctypes.byref(minor), 76, self.device), "compute capability minor")
        self.architecture = f"{major.value}{minor.value}"
        self.allocations: list[int] = []

    @staticmethod
    def _check(code, label):
        if code:
            raise RuntimeError(f"CUDA {label} failed with code {code}")

    def alloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(pointer), size),
                    "cuMemAlloc")
        self.allocations.append(pointer.value)
        return pointer.value

    def upload(self, raw: bytes) -> int:
        pointer = self.alloc(len(raw))
        buffer = ctypes.create_string_buffer(raw)
        self._check(self.lib.cuMemcpyHtoD_v2(
            ctypes.c_uint64(pointer), buffer, len(raw)), "cuMemcpyHtoD")
        return pointer

    def download(self, pointer: int, size: int) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        self._check(self.lib.cuMemcpyDtoH_v2(
            buffer, ctypes.c_uint64(pointer), size), "cuMemcpyDtoH")
        return buffer.raw

    def module(self, ptx: bytes):
        module = ctypes.c_void_p()
        data = ctypes.create_string_buffer(ptx)
        self._check(self.lib.cuModuleLoadData(
            ctypes.byref(module), data), "cuModuleLoadData")
        return module, data

    def function(self, module, name: str):
        function = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(
            ctypes.byref(function), module, name.encode()),
            f"cuModuleGetFunction({name})")
        return function

    def launch(self, function, grid: int, block: int, arguments: list[Any]):
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


def _pointer(value: int):
    return ctypes.c_uint64(value)


def _integer(value: int):
    return ctypes.c_int(value)


def _stats_upper(values: tuple[float, ...], stored_center: float,
                 stored_radius: float, *, inner: int | None = None,
                 ga: int | None = None, gb: int | None = None
                 ) -> CoordinateResult:
    (center, center_s, ret_h, ret_s, ret_m, rad_h, rad_s,
     center_products_f, ret_terms_f, ret_products_f,
     rad_terms_f, rad_products_f, active_f) = values
    counts_f = (center_products_f, ret_terms_f, ret_products_f,
                rad_terms_f, rad_products_f, active_f)
    if any(not math.isfinite(item) or item < 0 or item != int(item)
           for item in counts_f):
        raise AssertionError("backend count metadata is invalid")
    center_products, ret_terms, ret_products, rad_terms, rad_products, active = (
        int(item) for item in counts_f)
    if any(item is not None for item in (inner, ga, gb)):
        if inner is None or ga is None or gb is None:
            raise AssertionError("incomplete backend dimension metadata")
        gmin, gmax = min(ga, gb), max(ga, gb)
        if center_products != inner * (1 + gmin):
            raise AssertionError("backend center operation count mismatch")
        if ret_terms != gmax or ret_products != inner * (ga + gb):
            raise AssertionError("backend retained operation count mismatch")
        if active > gmax or rad_terms > active * (active + 1) // 2:
            raise AssertionError("backend radius inventory count mismatch")
        maximum_products = inner * active * active
        if rad_products > maximum_products:
            raise AssertionError(
                "backend radius product count mismatch: "
                f"inner={inner} terms={rad_terms} products={rad_products} "
                f"active={active}")
    if not all(math.isfinite(item) and item >= 0 for item in
               (center_s, ret_h, ret_s, ret_m, rad_h, rad_s, stored_radius)):
        raise AssertionError("backend numerical summary is invalid")
    with _context(gmpy2.RoundUp):
        u = gmpy2.mpfr(2) ** -53
        gc = _gamma(center_products)
        center_prod = _mp(center_s) / (1 - gc)
        center_upper = abs(_mp(center) - _mp(stored_center)) + gc * center_prod

        grt = _gamma(ret_terms)
        grp = _gamma(ret_products)
        retained_difference = _mp(ret_h) / (1 - grt)
        retained_magnitude = _mp(ret_m) / (1 - _gamma(2 * ret_terms))
        retained_products = _mp(ret_s) / (1 - grp)
        retained_upper = (retained_difference + u * retained_magnitude
                          + _gamma(64) * retained_products)

        gqt = _gamma(rad_terms)
        gqp = _gamma(rad_products)
        radius_products = _mp(rad_s) / (1 - gqp)
        radius_upper = (_mp(rad_h) / (1 - gqt)
                        + _gamma(64) * radius_products)
        # Native precise-dot fresh rows represent only the quadratic radius.
        # Center/retained machine-rounding discrepancies are separate
        # checker-only numerical sources; charging them to a nonexistent
        # native fresh row would change native symbol semantics.
        required = radius_upper
        sidecar = center_upper + retained_upper
    return CoordinateResult(
        required_upper=str(required),
        numerical_sidecar_radius=str(sidecar),
        stored_radius_hex=float(stored_radius).hex(),
        accepted=_mp(stored_radius) >= required,
        center_upper=str(center_upper), retained_upper=str(retained_upper),
        radius_upper=str(radius_upper), active_generators=active,
        retained_terms=ret_terms, radius_terms=rad_terms,
        radius_products=rad_products)


def check_f32(*, left_raw: bytes, right_raw: bytes, output_raw: bytes,
              left_shape: tuple[int, int, int, int],
              right_shape: tuple[int, int, int, int],
              output_shape: tuple[int, int, int, int],
              device_index: int = 0) -> dict[str, Any]:
    """Recompute and check a frozen float32 QK/A.V precise-dot transition."""
    heads, ga1, rows, inner = left_shape
    rheads, gb1, columns, rinner = right_shape
    ga, gb = ga1 - 1, gb1 - 1
    gmax = max(ga, gb)
    fresh = heads * rows * columns
    expected_output = (heads, 1 + gmax + fresh, rows, columns)
    if (rheads, rinner) != (heads, inner) or output_shape != expected_output:
        raise AssertionError("precise-dot backend shape mismatch")
    expected_bytes = (math.prod(left_shape) * 4,
                      math.prod(right_shape) * 4,
                      math.prod(output_shape) * 4)
    if tuple(map(len, (left_raw, right_raw, output_raw))) != expected_bytes:
        raise AssertionError("precise-dot backend byte count mismatch")
    source_hash = hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()
    if source_hash != CUDA_SOURCE_SHA256:
        raise AssertionError("checker CUDA arithmetic source identity mismatch")
    started = time.perf_counter()
    cuda = _CUDA(device_index)
    peak_bytes = 0
    try:
        compiler = _NVRTC()
        compile_started = time.perf_counter()
        ptx, compile_log = compiler.compile(CUDA_SOURCE, cuda.architecture)
        compile_seconds = time.perf_counter() - compile_started
        module, module_buffer = cuda.module(ptx)
        active_fn = cuda.function(module, "build_active_f32")
        stats_fn = cuda.function(module, "precise_stats_f32")
        left_p, right_p, output_p = (cuda.upload(raw) for raw in
                                     (left_raw, right_raw, output_raw))
        outputs = fresh
        indices_bytes = outputs * gmax * 4
        counts_bytes = outputs * 4
        stats_bytes = outputs * STAT_FIELDS * 8
        indices_p = cuda.alloc(indices_bytes)
        counts_p = cuda.alloc(counts_bytes)
        stats_p = cuda.alloc(stats_bytes)
        peak_bytes = sum(expected_bytes) + indices_bytes + counts_bytes + stats_bytes
        integer_args = [_integer(item) for item in
                        (heads, ga, gb, rows, columns, inner)]
        cuda.launch(active_fn, (outputs + 127) // 128, 128,
                    [_pointer(left_p), _pointer(right_p), _pointer(indices_p),
                     _pointer(counts_p), *integer_args])
        cuda.synchronize()
        compute_started = time.perf_counter()
        cuda.launch(stats_fn, outputs, BLOCK_THREADS,
                    [_pointer(left_p), _pointer(right_p), _pointer(output_p),
                     _pointer(indices_p), _pointer(counts_p), _pointer(stats_p),
                     *integer_args])
        cuda.synchronize()
        compute_seconds = time.perf_counter() - compute_started
        raw_stats = cuda.download(stats_p, stats_bytes)
    finally:
        cuda.close()
    decoded = struct.iter_unpack("<" + "d" * STAT_FIELDS, raw_stats)
    results = []
    maximum_required = _mp(0)
    maximum_sidecar = _mp(0)
    minimum_slack = None
    minimum_slack_index = None
    evaluated_pairs = 0
    evaluated_products = 0
    for out, values in enumerate(decoded):
        h = out // (rows * columns)
        within = out % (rows * columns)
        q, c = divmod(within, columns)
        center_offset = (((h * output_shape[1]) * rows + q) * columns + c) * 4
        fresh_index = 1 + gmax + out
        fresh_offset = (((h * output_shape[1] + fresh_index) * rows + q)
                        * columns + c) * 4
        stored_center = struct.unpack_from("<f", output_raw, center_offset)[0]
        stored_radius = struct.unpack_from("<f", output_raw, fresh_offset)[0]
        result = _stats_upper(
            values, stored_center, stored_radius, inner=inner, ga=ga, gb=gb)
        results.append(result)
        required = _mp(result.required_upper)
        slack = _mp(stored_radius) - required
        maximum_required = max(maximum_required, required)
        maximum_sidecar = max(
            maximum_sidecar, _mp(result.numerical_sidecar_radius))
        if minimum_slack is None or slack < minimum_slack:
            minimum_slack = slack
            minimum_slack_index = out
        evaluated_pairs += result.radius_terms
        evaluated_products += result.radius_products
    accepted = all(item.accepted for item in results)
    return {
        "schema": SCHEMA,
        "accepted": accepted,
        "coordinates_checked": outputs,
        "evaluated_radius_terms": evaluated_pairs,
        "evaluated_scalar_products": evaluated_products,
        "maximum_required_upper": str(maximum_required),
        "maximum_numerical_sidecar_radius": str(maximum_sidecar),
        "minimum_soundness_slack": str(minimum_slack),
        "minimum_soundness_slack_index": minimum_slack_index,
        "compile_seconds": compile_seconds,
        "kernel_seconds": compute_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_device_bytes": peak_bytes,
        "architecture": cuda.architecture,
        "block_threads": BLOCK_THREADS,
        "compiler_flags": ["--std=c++11", "--fmad=false", "--ftz=false"],
        "explicit_fma": "__fma_rn",
        "explicit_add": "__dadd_rn/__dsub_rn",
        "tf32": False,
        "tensor_cores": False,
        "fast_math": False,
        "unit_roundoff_hex": UNIT_ROUNDOFF_HEX,
        "source_sha256": source_hash,
        "nvrtc_library": compiler.path,
        "cuda_driver": cuda.path,
        "compile_log": compile_log,
        "coordinate_results": [item.__dict__ for item in results],
    }
