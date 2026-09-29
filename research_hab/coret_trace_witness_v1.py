#!/usr/bin/env python3
"""Graph-level content-addressed witnesses for a bounded native-formula graph.

This is producer-side serialization.  It deliberately makes no soundness
decision.  The companion checker imports neither this module nor Torch/NumPy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "CORET_TRACE_WITNESS_V1"
BLOB_SCHEMA = "CORET_TRACE_BLOB_V1"
PINNED_REVISION = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def seal(value):
    body = dict(value)
    body.pop("canonical_sha256", None)
    value["canonical_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    return value


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def fadd(a, b): return f32(f32(a) + f32(b))
def fsub(a, b): return f32(f32(a) - f32(b))
def fmul(a, b): return f32(f32(a) * f32(b))
def fdiv(a, b): return f32(f32(a) / f32(b))


@dataclass(frozen=True)
class State:
    weights: tuple[tuple[float, ...], ...]
    generator_ids: tuple[str, ...]
    ranges: tuple[tuple[float, float], ...]
    numerical_radius: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if len(self.weights) != 1 + len(self.generator_ids):
            raise ValueError("state generator count differs")
        if len(self.ranges) != len(self.generator_ids):
            raise ValueError("state range count differs")
        width = len(self.weights[0])
        if any(len(row) != width for row in self.weights):
            raise ValueError("ragged state")
        if len(self.numerical_radius) != len(self.weights):
            raise ValueError("numerical radius row count differs")
        if any(len(row) != width for row in self.numerical_radius):
            raise ValueError("ragged numerical radius")


class BlobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)

    def _write(self, raw, dtype, shape, label):
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path("blobs") / f"{digest}.{dtype}.le.bin"
        target = self.root / relative
        if target.exists():
            if target.read_bytes() != raw:
                raise RuntimeError("content-address collision")
        else:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=self.blobs)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(raw)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
        return seal({
            "schema": BLOB_SCHEMA, "label": str(label), "dtype": dtype,
            "byte_order": "little", "shape": list(shape),
            "element_count": math.prod(shape), "byte_count": len(raw),
            "sha256": digest, "relative_path": relative.as_posix(),
        })

    def f32_matrix(self, rows, label):
        rows = tuple(tuple(f32(item) for item in row) for row in rows)
        width = len(rows[0])
        if any(len(row) != width for row in rows): raise ValueError("ragged")
        raw = b"".join(struct.pack("<f", item) for row in rows for item in row)
        return self._write(raw, "float32", (len(rows), width), label)


def _zeros(rows, cols):
    return [[f32(0)] * cols for _ in range(rows)]


def concretize(state):
    low, high = list(state.weights[0]), list(state.weights[0])
    for row, (lo, hi) in zip(state.weights[1:], state.ranges):
        for j, value in enumerate(row):
            a, b = fmul(value, lo), fmul(value, hi)
            low[j] = fadd(low[j], min(a, b))
            high[j] = fadd(high[j], max(a, b))
    return low, high


def affine(state, matrix, bias, prefix="affine"):
    output = []
    for row_index, row in enumerate(state.weights):
        values = []
        for out, weights in enumerate(matrix):
            acc = f32(bias[out] if row_index == 0 else 0)
            for value, coefficient in zip(row, weights):
                acc = fadd(acc, fmul(value, coefficient))
            values.append(acc)
        output.append(tuple(values))
    return State(tuple(output), state.generator_ids, state.ranges,
                 tuple(tuple(f32(.125) for _ in output[0]) for _ in output))


def _append_coordinate(rows, values, active):
    width = len(rows[0]); result = [list(row) for row in rows]
    for coordinate in active:
        row = [f32(0)] * width
        row[coordinate] = f32(values[coordinate])
        result.append(row)
    return result


def layer_norm(state, gamma, beta, epsilon=1e-12, label="ln"):
    d = len(state.weights[0]); g = len(state.generator_ids)
    centered = []
    for row in state.weights:
        total = f32(0)
        for value in row: total = fadd(total, value)
        mean = fdiv(total, d)
        centered.append([fsub(value, mean) for value in row])

    variance = _zeros(2 + g, d)
    acc = f32(0)
    for value in centered[0]: acc = fadd(acc, fmul(value, value))
    variance[0] = [fadd(fdiv(acc, d), epsilon)] * d
    for generator in range(g):
        acc = f32(0)
        for c, a in zip(centered[0], centered[1 + generator]):
            acc = fadd(acc, fmul(c, a))
        value = fdiv(fmul(2.0, acc), d)
        variance[1 + generator] = [value] * d
    width = f32(0)
    for coordinate in range(d):
        support = f32(0)
        for generator in range(g):
            support = fadd(support, abs(centered[1 + generator][coordinate]))
        width = fadd(width, fmul(support, support))
    width = fdiv(width, d)
    variance[-1] = [width] * d
    ids = list(state.generator_ids) + [f"{label}.variance.000000"]
    ranges = list(state.ranges) + [(-1.0, 1.0)]

    def unary_sqrt(rows, ids, ranges):
        temp = State(tuple(map(tuple, rows)), tuple(ids), tuple(ranges),
                     tuple(tuple(0.0 for _ in range(d)) for _ in rows))
        lower, upper = concretize(temp); active = [i for i in range(d) if lower[i] != upper[i]]
        output = _zeros(len(rows), d); fresh = [f32(0)] * d
        for i, (lo, hi) in enumerate(zip(lower, upper)):
            if lo <= 0: raise AssertionError("sqrt domain")
            if lo == hi:
                output[0][i] = f32(math.sqrt(lo)); continue
            sl, su = f32(math.sqrt(lo)), f32(math.sqrt(hi))
            t = fmul(fdiv(fsub(hi, lo), fmul(2.0, fsub(su, sl))),
                     fdiv(fsub(hi, lo), fmul(2.0, fsub(su, sl))))
            lam = fdiv(fsub(su, sl), fsub(hi, lo))
            x = fsub(sl, fmul(lam, lo))
            ft = f32(math.sqrt(t))
            const = fmul(.5, fadd(fsub(ft, fmul(lam, t)), x))
            fresh[i] = fmul(.5, fadd(fsub(fmul(lam, t), ft), x))
            output[0][i] = fadd(fmul(lam, rows[0][i]), const)
            for r in range(1, len(rows)): output[r][i] = fmul(rows[r][i], lam)
        output = _append_coordinate(output, fresh, active)
        ids = ids + [f"{label}.sqrt.{i:06d}" for i in active]
        ranges = ranges + [(-1.0, 1.0)] * len(active)
        return output, ids, ranges, active

    sqrt_rows, ids, ranges, sqrt_active = unary_sqrt(variance, ids, ranges)

    def unary_recip(rows, ids, ranges):
        temp = State(tuple(map(tuple, rows)), tuple(ids), tuple(ranges),
                     tuple(tuple(0.0 for _ in range(d)) for _ in rows))
        lower, upper = concretize(temp); active = [i for i in range(d) if lower[i] != upper[i]]
        output = _zeros(len(rows), d); fresh = [f32(0)] * d
        for i, (lo, hi) in enumerate(zip(lower, upper)):
            if lo <= 0: raise AssertionError("reciprocal domain")
            if lo == hi:
                output[0][i] = fdiv(1.0, lo); continue
            lam = fdiv(-1.0, fmul(hi, hi))
            bottom = fsub(fdiv(1.0, hi), fmul(lam, hi))
            top = fsub(fdiv(1.0, lo), fmul(lam, lo))
            const = fmul(.5, fadd(top, bottom)); fresh[i] = fmul(.5, fsub(top, bottom))
            output[0][i] = fadd(fmul(lam, rows[0][i]), const)
            for r in range(1, len(rows)): output[r][i] = fmul(rows[r][i], lam)
        output = _append_coordinate(output, fresh, active)
        ids = ids + [f"{label}.reciprocal.{i:06d}" for i in active]
        ranges = ranges + [(-1.0, 1.0)] * len(active)
        return output, ids, ranges, active

    reciprocal, ids, ranges, reciprocal_active = unary_recip(sqrt_rows, ids, ranges)
    centered += [[f32(0)] * d for _ in range(len(reciprocal) - len(centered))]
    product = _zeros(len(reciprocal), d)
    for i in range(d):
        product[0][i] = fmul(centered[0][i], reciprocal[0][i])
        for r in range(1, len(reciprocal)):
            product[r][i] = fadd(fmul(centered[0][i], reciprocal[r][i]),
                                 fmul(reciprocal[0][i], centered[r][i]))
    product_fresh = []
    for i in range(d):
        left = f32(0); right = f32(0)
        for r in range(1, len(reciprocal)):
            left = fadd(left, abs(centered[r][i])); right = fadd(right, abs(reciprocal[r][i]))
        product_fresh.append(fmul(left, right))
    product = _append_coordinate(product, product_fresh, list(range(d)))
    ids += [f"{label}.product.{i:06d}" for i in range(d)]
    ranges += [(-1.0, 1.0)] * d
    for r in range(len(product)):
        for i in range(d):
            product[r][i] = fmul(product[r][i], gamma[i])
            if r == 0: product[r][i] = fadd(product[r][i], beta[i])
    # This bounded fixture deliberately uses a generous checker-only sidecar.
    # It is not a native generator and never changes the producer tensor.
    state_out = State(tuple(map(tuple, product)), tuple(ids), tuple(ranges),
                      tuple(tuple(f32(.125) for _ in range(d)) for _ in product))
    tau = {"mode": "standard", "epsilon_hex": float(epsilon).hex(),
           "variance_fresh_ids": [f"{label}.variance.000000"],
           "sqrt_active_flat_indices": sqrt_active,
           "reciprocal_active_flat_indices": reciprocal_active,
           "product_active_flat_indices": list(range(d)),
           "fresh_generator_ids": list(state_out.generator_ids[g:]),
           "branch": "positive_variance_standard_layernorm"}
    return state_out, tau


def relu(state, label="relu"):
    lower, upper = concretize(state); active = []
    output = _zeros(len(state.weights), len(lower)); fresh = [f32(0)] * len(lower)
    cases = []
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo >= 0:
            cases.append("active")
            for r in range(len(state.weights)): output[r][i] = state.weights[r][i]
        elif hi <= 0:
            cases.append("inactive")
        else:
            cases.append("crossing"); active.append(i)
            lam = fdiv(hi, fadd(fsub(hi, lo), 1e-12))
            delta = max(fmul(-lam, lo), fmul(fsub(1.0, lam), hi))
            output[0][i] = fadd(fmul(lam, state.weights[0][i]), fmul(.5, delta))
            for r in range(1, len(state.weights)): output[r][i] = fmul(lam, state.weights[r][i])
            fresh[i] = fmul(.5, delta)
    output = _append_coordinate(output, fresh, active)
    ids = state.generator_ids + tuple(f"{label}.{i:06d}" for i in active)
    ranges = state.ranges + tuple((-1.0, 1.0) for _ in active)
    result = State(tuple(map(tuple, output)), ids, ranges,
                   tuple(tuple(f32(.125) for _ in output[0]) for _ in output))
    return result, {"coordinate_cases": cases, "active_flat_indices": active,
                    "fresh_generator_ids": list(ids[len(state.generator_ids):])}


def state_record(store, state, state_id):
    generator_map = [{"native_row": i + 1, "ghost_id": identifier,
                      "relation": "identity"}
                     for i, identifier in enumerate(state.generator_ids)]
    value = {
        "state_id": state_id,
        "producer_tensor_content_ids": {
            "weights": store.f32_matrix(state.weights, f"{state_id}.weights"),
            "numerical_radius": store.f32_matrix(
                state.numerical_radius, f"{state_id}.numerical_radius")},
        "centers": {"row": 0}, "native_generator_coefficients": {"rows_start": 1},
        "generator_ids": list(state.generator_ids),
        "explicit_ranges": [[float(lo).hex(), float(hi).hex()] for lo, hi in state.ranges],
        "dtype": "float32", "shape": [len(state.weights), len(state.weights[0])],
        "numerical_sidecar_linkage": {"kind": "coefficient_symmetric_radius",
                                      "content": "numerical_radius"},
        "ghost_state_linkage": {"ordered_native_to_ghost": generator_map},
    }
    return seal(value)


def build_minimal_trace(root, *, instrument=True):
    store = BlobStore(root)
    rho = f32(0.125)
    source = State(((.75, -.25), (rho, 0.0), (0.0, rho)),
                   ("source.embedding.000", "source.embedding.001"),
                   ((-1.0, 1.0), (-1.0, 1.0)),
                   ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))
    a1 = affine(source, ((1.0, .25), (-.5, 1.0)), (.125, -.25), "affine0")
    ln, ln_tau = layer_norm(a1, (1.0, .75), (.1, -.2), label="layernorm0")
    activation, relu_tau = relu(ln, "relu0")
    margin = affine(activation, ((1.25, -.75),), (.2,), "classifier")
    states = [source, a1, ln, activation, margin]
    state_ids = ["s0_source", "s1_affine", "s2_layernorm", "s3_relu", "s4_margin"]
    if not instrument:
        return states, None
    records = [state_record(store, state, identifier)
               for state, identifier in zip(states, state_ids)]
    transitions = [
        seal({"transition_id": "t0", "operator_family": "affine",
              "input_state_ids": [state_ids[0]], "output_state_ids": [state_ids[1]],
              "tau_k": {"branch": "dense_affine", "generator_order": list(source.generator_ids)},
              "operator_witness": {"matrix_hex": [[float(x).hex() for x in row] for row in ((1.0,.25),(-.5,1.0))],
                                   "bias_hex": [float(x).hex() for x in (.125,-.25)]}}),
        seal({"transition_id": "t1", "operator_family": "LayerNorm",
              "input_state_ids": [state_ids[1]], "output_state_ids": [state_ids[2]],
              "tau_k": ln_tau,
              "operator_witness": {"gamma_hex": [float(x).hex() for x in (1.0,.75)],
                                   "beta_hex": [float(x).hex() for x in (.1,-.2)]}}),
        seal({"transition_id": "t2", "operator_family": "ReLU",
              "input_state_ids": [state_ids[2]], "output_state_ids": [state_ids[3]],
              "tau_k": relu_tau, "operator_witness": {}}),
        seal({"transition_id": "t3", "operator_family": "classifier_affine",
              "input_state_ids": [state_ids[3]], "output_state_ids": [state_ids[4]],
              "tau_k": {"branch": "target_label_minus_other", "generator_order": list(activation.generator_ids)},
              "operator_witness": {"matrix_hex": [[float(x).hex() for x in (1.25,-.75)]],
                                   "bias_hex": [float(.2).hex()]}}),
    ]
    # The producer supplies a candidate only.  The independent checker derives
    # its own lower bound from the entire trace.  This value is the greatest
    # binary64 fixture candidate below that independently derived bound.
    candidate_lower = 1.4227110949613724
    final = seal({"state_id": state_ids[-1], "target": "label0-label1",
                  "claimed_lower_hex": float(candidate_lower).hex(),
                  "certified": True})
    source_domain = seal({
        "benchmark": "p100_single_token_embedding_linf", "p_cli": 100,
        "interpreted_domain": "Linf", "epsilon_hex": float(rho).hex(),
        "source_symbol_ids": list(source.generator_ids),
        "source_ranges": [[(-1.0).hex(), (1.0).hex()]] * 2,
        "input_source_mask": [0, 1], "perturbed_token": 0,
        "model_identity": "bounded_trace_fixture_model_v1",
        "property_identity": "bounded_trace_fixture_margin_v1"})
    graph = {
        "schema": SCHEMA,
        "run_manifest": seal({"pinned_deept_revision": PINNED_REVISION,
                              "purpose": "bounded_minimal_trace_chain",
                              "scientific_query": False}),
        "source_domain": source_domain,
        "graph_nodes": state_ids,
        "content_store": {"schema": BLOB_SCHEMA, "root": "."},
        "state_records": records,
        "transition_records": transitions,
        "final_property_record": final,
    }
    seal(graph)
    Path(root, "trace.json").write_bytes(canonical_bytes(graph) + b"\n")
    return states, graph
