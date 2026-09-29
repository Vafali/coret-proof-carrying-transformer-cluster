#!/usr/bin/env python3
"""Immutable numerical witnesses for the native-semantics checker.

This module is producer-side serialization only.  It does not decide whether
an enclosure is sound.  Tensor values are stored as their exact native bits in
content-addressed blocks so the independent checker need not trust decimal
printing, producer reductions, or producer-reported envelope summaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


SCHEMA = "CORET_NATIVE_NUMERICAL_WITNESS_V1"
BLOCK_SCHEMA = "CORET_EXACT_TENSOR_BLOCK_V1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def support_record(proof) -> dict[str, Any]:
    value = {
        "num_tokens": int(proof.num_tokens),
        "masks": [int(item) for item in proof.masks],
        "ids": list(proof.ids),
        "reasons": list(proof.reasons),
    }
    value["canonical_sha256"] = _digest(value)
    return value


def ranges_record(z) -> dict[str, Any]:
    low, high = z.error_term_range_low, z.error_term_range_high
    if low is None:
        if high is not None:
            raise RuntimeError("incomplete native range metadata")
        value = {"kind": "implicit_minus1_plus1",
                 "count": int(z.num_error_terms)}
    else:
        if high is None or low.numel() != high.numel():
            raise RuntimeError("invalid native range metadata")
        low_values = [float(item).hex() for item in low.detach().cpu().reshape(-1)]
        high_values = [float(item).hex() for item in high.detach().cpu().reshape(-1)]
        value = {"kind": "explicit", "count": int(low.numel()),
                 "low_hex": low_values, "high_hex": high_values}
    value["canonical_sha256"] = _digest(value)
    return value


class ContentAddressedWitnessStore:
    """Write exact tensor bits once, with bounded transient host storage.

    A call copies one tensor to contiguous CPU storage, writes it immediately,
    and returns only compact metadata.  No tensor is retained by the store.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.blocks = self.root / "blocks"
        self.blocks.mkdir(parents=True, exist_ok=True)

    def tensor(self, value: torch.Tensor, label: str) -> dict[str, Any]:
        cpu = value.detach().to(device="cpu").contiguous()
        if cpu.dtype == torch.float32:
            dtype, itemsize = "float32", 4
        elif cpu.dtype == torch.float64:
            dtype, itemsize = "float64", 8
        else:
            raise RuntimeError(f"unsupported witness tensor dtype: {cpu.dtype}")
        raw = cpu.numpy().tobytes(order="C")
        expected = cpu.numel() * itemsize
        if len(raw) != expected:
            raise RuntimeError("tensor witness byte count differs")
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path("blocks") / f"{digest}.{dtype}.le.bin"
        destination = self.root / relative
        if destination.exists():
            if (destination.stat().st_size != len(raw)
                    or hashlib.sha256(destination.read_bytes()).hexdigest() != digest):
                raise RuntimeError("existing content-addressed block differs")
        else:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=self.blocks)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return {
            "schema": BLOCK_SCHEMA, "label": str(label),
            "dtype": dtype, "byte_order": "little",
            "shape": [int(item) for item in cpu.shape],
            "element_count": int(cpu.numel()), "byte_count": len(raw),
            "sha256": digest, "relative_path": relative.as_posix(),
        }

    def precise_dot(self, *, mode: str, left, right, output,
                    left_support, right_support, output_support,
                    call_index: int) -> dict[str, Any]:
        if mode not in {"QK", "A.V"}:
            raise ValueError("precise-dot witness mode must be QK or A.V")
        output_block = self.tensor(output.zonotope_w, f"{mode}.output")
        value = {
            "schema": SCHEMA,
            "kind": "precise_dot_structural",
            "family": mode,
            "call_index": int(call_index),
            "equation": (
                "center=a0.b0+0.5*sum_i(ai.bi);"
                "retained_i=a0.bi+ai.b0;"
                "fresh=0.5*sum_i|ai.bi|+sum_i<j|ai.bj+aj.bi|"),
            "left": self.tensor(left.zonotope_w, f"{mode}.left"),
            "right": self.tensor(right.zonotope_w, f"{mode}.right"),
            "output": output_block,
            # The certificate-level commitment is intentionally distinct
            # from the block's self-hash: mutation tests must not be able to
            # silently substitute a different (even still-sound) machine
            # result for the frozen producer output.
            "frozen_output_sha256": output_block["sha256"],
            "left_generator_count": int(left.num_error_terms),
            "right_generator_count": int(right.num_error_terms),
            "output_generator_count": int(output.num_error_terms),
            "left_special_prefix_count": int(
                left.num_input_error_terms_special_norm),
            "right_special_prefix_count": int(
                right.num_input_error_terms_special_norm),
            "output_special_prefix_count": int(
                output.num_input_error_terms_special_norm),
            "left_support": support_record(left_support),
            "right_support": support_record(right_support),
            "output_support": support_record(output_support),
            "left_ranges": ranges_record(left),
            "right_ranges": ranges_record(right),
            "output_ranges": ranges_record(output),
        }
        value["canonical_sha256"] = _digest(value)
        return value

    def native_affine_sqrt(self, *, source, output,
                           fresh_flat_indices) -> dict[str, Any]:
        """Serialize exact bits for the pinned affine sqrt transition.

        ``fresh_flat_indices`` is evidence to be checked, not a trusted
        producer conclusion.  The independent checker recomputes membership
        and row-major order from the immutable input coefficients/ranges.
        """
        output_block = self.tensor(output.zonotope_w, "sqrt.output")
        value = {
            "schema": SCHEMA,
            "kind": "native_affine_sqrt",
            "equation": (
                "l,u=concretize(x);"
                "t=((u-l)/(2*(sqrt(u)-sqrt(l))))^2;"
                "lambda=(sqrt(u)-sqrt(l))/(u-l);"
                "native midpoint/fresh affine relaxation"),
            "input": self.tensor(source.zonotope_w, "sqrt.input"),
            "output": output_block,
            "frozen_output_sha256": output_block["sha256"],
            "input_generator_count": int(source.num_error_terms),
            "output_generator_count": int(output.num_error_terms),
            "input_special_prefix_count": int(
                source.num_input_error_terms_special_norm),
            "fresh_flat_indices": [int(item) for item in fresh_flat_indices],
            "input_ranges": ranges_record(source),
            "output_ranges": ranges_record(output),
        }
        value["canonical_sha256"] = _digest(value)
        return value

    def native_affine_sqrt_shadow(
            self, *, source, output, fresh_flat_indices,
            coordinate_numerical_radii) -> dict[str, Any]:
        """Serialize native sqrt unchanged plus a separate FP sidecar."""
        value = self.native_affine_sqrt(
            source=source, output=output,
            fresh_flat_indices=fresh_flat_indices)
        value.pop("canonical_sha256")
        value["kind"] = "native_affine_sqrt_shadow"
        value["coordinate_numerical_radius_hex"] = [
            float(item).hex() for item in coordinate_numerical_radii]
        if any(not math.isfinite(float.fromhex(item))
               or float.fromhex(item) < 0
               for item in value["coordinate_numerical_radius_hex"]):
            raise ValueError("invalid native affine sqrt numerical sidecar")
        value["canonical_sha256"] = _digest(value)
        return value


def scalar_interval_witness(family: str, lower: float, upper: float,
                            output_lower: float, output_upper: float) -> dict[str, Any]:
    """Small fixture witness for independently directed unary endpoints."""
    if family not in {"sqrt", "reciprocal", "exp", "tanh"}:
        raise ValueError("unsupported scalar numerical witness family")
    value = {
        "schema": SCHEMA, "kind": "scalar_interval", "family": family,
        "input_lower_hex": float(lower).hex(),
        "input_upper_hex": float(upper).hex(),
        "output_lower_hex": float(output_lower).hex(),
        "output_upper_hex": float(output_upper).hex(),
    }
    value["canonical_sha256"] = _digest(value)
    return value


def numerical_shadow_scalar_chain_witness(
        *, input_value: float, stored_sqrt: float, sqrt_error_radius: float,
        stored_reciprocal: float, reciprocal_error_radius: float,
        scale: float, shift: float, stored_output: float,
        output_error_radius: float) -> dict[str, Any]:
    """Serialize a bounded feasibility witness for a numerical sidecar.

    The three error radii are representation evidence, not native DeepT
    generators.  In particular, all native fresh-symbol counts remain zero.
    The independent checker recomputes the composed MPFR image and does not
    trust these radii or the producer's elementary-function arithmetic.
    """
    fields = {
        "input_hex": float(input_value).hex(),
        "stored_sqrt_hex": float(stored_sqrt).hex(),
        "sqrt_error_radius_hex": float(sqrt_error_radius).hex(),
        "stored_reciprocal_hex": float(stored_reciprocal).hex(),
        "reciprocal_error_radius_hex": float(reciprocal_error_radius).hex(),
        "scale_hex": float(scale).hex(),
        "shift_hex": float(shift).hex(),
        "stored_output_hex": float(stored_output).hex(),
        "output_error_radius_hex": float(output_error_radius).hex(),
    }
    if not all(math.isfinite(float.fromhex(item)) for item in fields.values()):
        raise ValueError("nonfinite numerical shadow scalar witness")
    if any(float.fromhex(fields[name]) < 0 for name in (
            "sqrt_error_radius_hex", "reciprocal_error_radius_hex",
            "output_error_radius_hex")):
        raise ValueError("negative numerical shadow radius")
    value = {
        "schema": SCHEMA,
        "kind": "numerical_shadow_scalar_chain",
        "equation": "sqrt(x);reciprocal;scale*y+shift",
        "native_fresh_counts": [0, 0, 0],
        **fields,
    }
    value["canonical_sha256"] = _digest(value)
    return value


def ghost_precise_square_witness(*, center: float, coefficients,
                                 stored_center: float,
                                 stored_retained, stored_fresh: float):
    """Serialize a scalar precise-square fixture with persistent ghost IDs.

    ``coefficients`` and ``stored_retained`` are ordered ``(ghost_id, value)``
    pairs.  Ghost identities are checker-only correlation evidence and are
    never installed as native DeepT rows.
    """
    coefficient_rows = [
        {"ghost_id": str(identifier), "value_hex": float(value).hex(),
         "range_low_hex": (-1.0).hex(), "range_high_hex": (1.0).hex()}
        for identifier, value in coefficients]
    retained_rows = [
        {"ghost_id": str(identifier), "value_hex": float(value).hex()}
        for identifier, value in stored_retained]
    value = {
        "schema": SCHEMA,
        "kind": "ghost_precise_square",
        "equation": (
            "center=c*c+0.5*sum_i(ai*ai);"
            "retained_i=2*c*ai;"
            "fresh=0.5*sum_i|ai*ai|+sum_i<j|2*ai*aj|"),
        "center_hex": float(center).hex(),
        "coefficients": coefficient_rows,
        "stored_center_hex": float(stored_center).hex(),
        "stored_retained": retained_rows,
        "stored_fresh_hex": float(stored_fresh).hex(),
    }
    value["canonical_sha256"] = _digest(value)
    return value
