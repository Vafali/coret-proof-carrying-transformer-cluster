#!/usr/bin/env python3
"""Frozen identities and read-only helpers for full depth-6 A40 production."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import depth_sanity_common as depth_common


REPO = Path(__file__).resolve().parents[1]
FROZEN_ROOT = REPO / "frozen/depth6_full"
FULL_MANIFEST_PATH = (
    REPO / "frozen/depth_manifests/deept_table7_stdln6_full_manifest_v1.json")
PRODUCTION_MANIFEST_PATH = FROZEN_ROOT / "depth6_full_production_manifest_v1.json"
PLAN_PATH = FROZEN_ROOT / "depth6_full_a40_plan_v1.json"
CHUNK_ROOT = FROZEN_ROOT / "chunks"
RESULT_RELATIVE = Path("research_hab/results/coret_depth6_full_a40_v1")
EXPECTED_SOURCE_COMMIT = "876f325e4e5813be5393e87777736caf03a8059a"
EXPECTED_FULL_MANIFEST_SHA = (
    "b9401cc378ddcdaa7b68429d3f84a779b448b16960ade1add0a3a325d46d9eb2")
EXPECTED_CHECKPOINT_SHA = (
    "6e75fe827f7259db65a19a1fda152eb644cb44ead6d2d5a2e92b0950f368a236")
EXPECTED_PROPERTY_COUNT = 137
INITIAL_RHO = 1.0 / 1600.0
WORKER_COUNT = 2
CHUNKS_PER_WORKER = 4
FRESH_ROOT_BASENAME = "coret-a40-historical-depth6-full-v1"

# Accepted measurements only: the sanity analysis projected 45.1 hours on two
# A40s before packed metadata; its bounded packed-metadata audit projected a
# 1.08x--1.12x whole-property improvement.  These are planning estimates, not
# newly measured production timings.
PRE_PACKED_TWO_A40_WALL_HOURS = 45.1
PACKED_SPEEDUP_LOWER = 1.08
PACKED_SPEEDUP_UPPER = 1.12
ESTIMATED_TWO_A40_WALL_LOWER_HOURS = (
    PRE_PACKED_TWO_A40_WALL_HOURS / PACKED_SPEEDUP_UPPER)
ESTIMATED_TWO_A40_WALL_UPPER_HOURS = (
    PRE_PACKED_TWO_A40_WALL_HOURS / PACKED_SPEEDUP_LOWER)
ESTIMATED_TWO_A40_WALL_CENTRAL_HOURS = (
    (ESTIMATED_TWO_A40_WALL_LOWER_HOURS
     + ESTIMATED_TWO_A40_WALL_UPPER_HOURS) / 2.0)

SCIENTIFIC_SOURCE_HASHES = {
    "research_hab/coret_optimized_historical_127_v1.py":
        "b62494ba6230773a227f1599684faac5832442189ca555339cdb893d8179fd70",
    "research_hab/coret_structural_support_precise_dot_v1.py":
        "effd8af7b0a15a5417f9143a333402b013c70886f6332e8d11b56e4c35502e2e",
    "research_hab/coret_bounded_native_depth_graph_v1.py":
        "ff72b8fe24f9ecf42c352af25a9ab0b4426455fe44f93362f80b1062c98f1a95",
    "research_hab/coret_native_semantics_checker_v1.py":
        "516e67a9574b63229f03cb563960a502d85a22c41dbc840bd9253c9c265729f8",
}


def canonical(payload: dict | list) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, key: str = "canonical_manifest_sha256") -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.get(key)
    body = dict(payload)
    body.pop(key, None)
    if claimed != canonical(body):
        raise RuntimeError(f"canonical identity mismatch: {path}")
    return payload


def full_manifest() -> dict:
    value = verified(FULL_MANIFEST_PATH)
    if (value["canonical_manifest_sha256"] != EXPECTED_FULL_MANIFEST_SHA
            or int(value["property_count"]) != EXPECTED_PROPERTY_COUNT
            or int(value["depth"]) != 6
            or value["model"]["checkpoint"]["sha256"]
            != EXPECTED_CHECKPOINT_SHA):
        raise RuntimeError("frozen depth-6 scientific manifest differs")
    ids = [row["property_id"] for row in value["properties"]]
    if len(ids) != len(set(ids)) or len(ids) != EXPECTED_PROPERTY_COUNT:
        raise RuntimeError("depth-6 property universe is not exact and unique")
    return value


def _reference_radii(full: dict) -> dict[tuple[int, int], float]:
    rows = csv.DictReader(io.StringIO(depth_common.git_blob(
        full["reference_csv"]["git_path"]).decode("utf-8")))
    values = {(int(row["sentence"]), int(row["position"])): float(row["eps"])
              for row in rows}
    if len(values) != int(full["reference_csv"]["row_count"]):
        raise RuntimeError("cached DeepT reference coverage differs")
    return values


def expected_production_manifest() -> dict:
    full = full_manifest()
    references = _reference_radii(full)
    examples = {int(row["accepted_draw_ordinal"]): row
                for row in full["accepted_examples"]}
    properties = []
    for ordinal, item in enumerate(full["properties"]):
        draw = int(item["accepted_draw_ordinal"])
        key = (draw, int(item["token_position"]))
        if key not in references:
            raise RuntimeError("property absent from cached DeepT references")
        example = examples[draw]
        properties.append({
            **item,
            "benchmark_ordinal": ordinal,
            "sentence_ordinal": draw,
            "sequence_length": int(item["tokenized_length"]),
            "source_dimension": 128,
            "token_ids": list(example["token_ids"]),
            "cached_DeepT_reference": {
                "certified_lower_endpoint_binary64": references[key],
                "certified_lower_endpoint_binary64_hex": references[key].hex(),
                "reference_csv_sha256": full["reference_csv"]["sha256"],
                "sentence": draw,
                "position": int(item["token_position"]),
            },
            "frozen_proof_reference": None,
        })
    return {
        "schema": "CORET_DEPTH6_FULL_A40_PRODUCTION_MANIFEST_V1",
        "status": "FROZEN_BEFORE_FULL_DEPTH6_PRODUCTION",
        "scientific_source_commit": EXPECTED_SOURCE_COMMIT,
        "scientific_source_hashes": SCIENTIFIC_SOURCE_HASHES,
        "depth": 6,
        "property_count": EXPECTED_PROPERTY_COUNT,
        "properties": properties,
        "frozen_scientific_manifest": {
            "canonical_sha256": full["canonical_manifest_sha256"],
            "file_sha256": sha(FULL_MANIFEST_PATH),
            "property_count": EXPECTED_PROPERTY_COUNT,
        },
        "model": full["model"],
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "pinned_DeepT_revision": full["pinned_DeepT_revision"],
        "dataset": full["dataset"],
        "vocabulary_sha256": full["model"]["vocab"]["sha256"],
        "threat_model": {
            "p": 100,
            "perturbed_words": 1,
            "source_dimension": 128,
            "single_token_embedding_Linf": True,
            "perturbation_boundary": "before_embedding_LayerNorm",
        },
        "search": {
            "initial_rho": INITIAL_RHO,
            "initial_rho_binary64_hex": INITIAL_RHO.hex(),
            "factor_two_bracketing": True,
            "maximum_factor_two_refinements": 12,
            "midpoint_iterations": 10,
            "native_typed_fail_closed_diagnostics": [
                "PINNED_DEEPT_POSITIVITY_DOMAIN_FAILURE",
                "PINNED_DEEPT_RECIPROCAL_NAN_DOMAIN_FAILURE",
                "PINNED_DEEPT_ZONOTOPE_NAN_DOMAIN_FAILURE",
                "PINNED_DEEPT_EXP_MARK_DOMAIN_FAILURE",
            ],
            "unexpected_exception_policy": "fatal",
        },
        "execution": {
            "proof_carrying_native_semantics": True,
            "independent_checker_required": True,
            "provenance_structural_support": True,
            "support_and_reduction_recomputed_each_radius": True,
            "packed_integer_metadata": True,
            "requested_AV_generator_tile": 112,
            "grouped_temporary_cap_bytes": 128 * 1024 * 1024,
            "pre_B2_QK_allocator_lifetime_fence": True,
            "generic_fallback_allowed": False,
            "fused_exploratory_backend_allowed": False,
            "workers": WORKER_COUNT,
            "maximum_concurrent_jobs": WORKER_COUNT,
        },
        "output_root_basename": FRESH_ROOT_BASENAME,
        "preparation_scientific_queries": 0,
        "preparation_bound_entrypoint_calls": 0,
        "training_runs": 0,
    }


def production_manifest() -> dict:
    stored = verified(PRODUCTION_MANIFEST_PATH)
    body = dict(stored)
    body.pop("canonical_manifest_sha256")
    if body != expected_production_manifest():
        raise RuntimeError("stored full depth-6 production manifest differs")
    return stored


def plan() -> dict:
    value = verified(PLAN_PATH)
    if (value["production_manifest_sha256"]
            != production_manifest()["canonical_manifest_sha256"]):
        raise RuntimeError("full depth-6 plan parent differs")
    return value


def chunk_path(worker_id: int, chunk_index: int) -> Path:
    return CHUNK_ROOT / f"worker_{worker_id}_chunk_{chunk_index}.json"


def chunk(worker_id: int, chunk_index: int) -> dict:
    value = verified(chunk_path(worker_id, chunk_index))
    parent = plan()
    if (value["parent_plan_sha256"] != parent["canonical_manifest_sha256"]
            or int(value["worker_id"]) != worker_id
            or int(value["chunk_index"]) != chunk_index):
        raise RuntimeError("full depth-6 chunk identity differs")
    expected = parent["workers"][worker_id]["chunks"][chunk_index]
    if value["properties"] != expected["properties"]:
        raise RuntimeError("full depth-6 chunk inventory differs")
    return value


def assigned_ids(worker_id: int) -> set[str]:
    return {row["property_id"] for item in plan()["workers"][worker_id]["chunks"]
            for row in item["properties"]}


def worker_dir(fresh_root: Path, worker_id: int) -> Path:
    root = fresh_root.resolve()
    result = (root / f"worker_{worker_id}").resolve()
    if not result.is_relative_to(root):
        raise RuntimeError("worker directory escapes fresh root")
    return result


def worker_result_root(directory: Path) -> Path:
    result = (directory.resolve() / RESULT_RELATIVE).resolve()
    if not result.is_relative_to(directory.resolve()):
        raise RuntimeError("result directory escapes worker root")
    return result
