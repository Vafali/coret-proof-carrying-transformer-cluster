from __future__ import annotations

from fractions import Fraction
import ast
import math
import random
import struct
import sys
from pathlib import Path

import gmpy2
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_hab"))

import coret_rigorous_precise_dot_backend_v1 as backend


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def pack(values):
    return struct.pack("<" + "f" * len(values), *values)


def next_f32(value, positive=True):
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    if value < 0:
        word += -1 if positive else 1
    else:
        word += 1 if positive else -1
    return struct.unpack("<f", struct.pack("<I", word))[0]


def exact_inside(report, exact):
    lower = gmpy2.mpfr(report["lower"])
    upper = gmpy2.mpfr(report["upper"])
    exact_mp = gmpy2.mpfr(exact.numerator) / exact.denominator
    return lower <= exact_mp <= upper


def test_1000_exact_dyadic_differential_cases():
    rng = random.Random(20260928)
    adversarial = [
        [f32(math.ldexp(1.0, -149))] * 32,
        [f32(math.ldexp(1.0, -126))] * 32,
        [f32((-1.0) ** i * math.ldexp(1.0, 60 - (i % 5)))
         for i in range(32)],
        [f32((-1.0) ** i * math.ldexp(1.0, -60 + (i % 5)))
         for i in range(32)],
    ]
    checked = 0
    for case in range(1000):
        vectors = []
        for vector in range(4):
            if case < len(adversarial):
                values = list(adversarial[(case + vector) % len(adversarial)])
            else:
                values = []
                for coordinate in range(32):
                    exponent = rng.randint(-125, 100)
                    sign = -1.0 if rng.randrange(2) else 1.0
                    mantissa = rng.uniform(0.5, 1.0)
                    value = f32(sign * math.ldexp(mantissa, exponent))
                    if (case + vector + coordinate) % 37 == 0:
                        value = 0.0
                    values.append(value)
            vectors.append(values)
        exact = backend.reference_cross_f32(*vectors)
        rigorous = backend.rigorous_cross_f32(*vectors)
        assert exact_inside(rigorous, exact)
        checked += 1
    assert checked == 1000


def _tiny_inputs():
    left = [1., 2., 3., 4., 1., 0., 0., 0., 0., 2., 0., 0.]
    right = [2., 1., 0., -1., 0., 3., 0., 0., 5., 0., 0., 0.]
    output = [0., 8., 7., 13.]
    return left, right, output


def test_cuda_kernel_accepts_and_one_ulp_radius_narrowing_rejects(monkeypatch):
    monkeypatch.setenv("CORET_CUDA_DRIVER", "/usr/lib/wsl/lib/libcuda.so.1")
    left, right, output = _tiny_inputs()
    common = dict(
        left_raw=pack(left), right_raw=pack(right),
        left_shape=(1, 3, 1, 4), right_shape=(1, 3, 1, 4),
        output_shape=(1, 4, 1, 1))
    generous = backend.check_f32(output_raw=pack(output), **common)
    assert generous["accepted"] is True
    required = float(generous["coordinate_results"][0]["required_upper"])
    tight = next_f32(f32(required), positive=True)
    output[-1] = tight
    accepted = backend.check_f32(output_raw=pack(output), **common)
    assert accepted["accepted"] is True
    output[-1] = next_f32(tight, positive=False)
    rejected = backend.check_f32(output_raw=pack(output), **common)
    assert rejected["accepted"] is False


def test_pruning_is_reconstructed_from_exact_bits(monkeypatch):
    monkeypatch.setenv("CORET_CUDA_DRIVER", "/usr/lib/wsl/lib/libcuda.so.1")
    left, right, output = _tiny_inputs()
    common = dict(
        right_raw=pack(right), output_raw=pack(output),
        left_shape=(1, 3, 1, 4), right_shape=(1, 3, 1, 4),
        output_shape=(1, 4, 1, 1))
    original = backend.check_f32(left_raw=pack(left), **common)
    # A previously zero coefficient becomes the smallest positive binary32.
    # No producer support/skipped-work metadata is supplied to this backend.
    left[5] = math.ldexp(1.0, -149)
    mutated = backend.check_f32(left_raw=pack(left), **common)
    assert original["source_sha256"] == mutated["source_sha256"]
    assert mutated["evaluated_scalar_products"] >= original["evaluated_scalar_products"]
    assert mutated["coordinate_results"][0]["active_generators"] == 2


def test_error_metadata_mutations_fail_closed():
    values = (0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
              32.0, 2.0, 64.0, 3.0, 96.0, 3.0)
    valid = backend._stats_upper(values, 0.0, 100.0)
    assert valid.accepted
    narrowed_s = list(values)
    narrowed_s[6] = math.nextafter(narrowed_s[6], -math.inf)
    narrowed = backend._stats_upper(tuple(narrowed_s), 0.0, 100.0)
    assert backend._mp(narrowed.required_upper) < backend._mp(valid.required_upper)
    for index in (7, 8, 9, 10, 11):
        broken = list(values)
        broken[index] = -1.0
        with pytest.raises(AssertionError, match="count metadata"):
            backend._stats_upper(tuple(broken), 0.0, 100.0)


def test_kernel_inventory_metadata_is_independently_constrained():
    # inner=4, ga=gb=2 fixes the center and retained operation counts.
    valid = (0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
             12.0, 2.0, 16.0, 3.0, 12.0, 2.0)
    assert backend._stats_upper(
        valid, 0.0, 100.0, inner=4, ga=2, gb=2).accepted
    mutations = ((7, 13.0, "center"), (8, 3.0, "retained"),
                 (9, 17.0, "retained"), (10, 4.0, "radius"),
                 (11, 17.0, "radius"), (12, 3.0, "radius"))
    for index, replacement, message in mutations:
        mutated = list(valid)
        mutated[index] = replacement
        with pytest.raises(AssertionError, match=message):
            backend._stats_upper(
                tuple(mutated), 0.0, 100.0, inner=4, ga=2, gb=2)


def test_backend_import_boundary_excludes_producer_torch_and_numpy():
    tree = ast.parse(Path(backend.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not ({"torch", "numpy"} & imported)
    assert not any(name.startswith("coret_") for name in imported)


def test_unknown_or_nonfinite_inputs_fail_closed():
    with pytest.raises(AssertionError, match="nonfinite"):
        backend.reference_cross_f32([math.inf], [1.0], [0.0], [0.0])


def test_kernel_arithmetic_source_mutation_rejects_before_execution(monkeypatch):
    left, right, output = _tiny_inputs()
    monkeypatch.setattr(backend, "CUDA_SOURCE", backend.CUDA_SOURCE + "\n// mutation")
    with pytest.raises(AssertionError, match="source identity"):
        backend.check_f32(
            left_raw=pack(left), right_raw=pack(right), output_raw=pack(output),
            left_shape=(1, 3, 1, 4), right_shape=(1, 3, 1, 4),
            output_shape=(1, 4, 1, 1))
