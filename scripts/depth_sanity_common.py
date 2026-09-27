#!/usr/bin/env python3
"""Frozen identities and read-only helpers for A40 depth sanity runs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEEPT = REPO / "research_hab/public_benchmarks/DeepT"
DEEPT_REVISION = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
MANIFEST_ROOT = REPO / "frozen/depth_manifests"
PRODUCTION_ROOT = REPO / "frozen/depth_sanity"
RESULT_RELATIVE = Path("research_hab/results/coret_depth_sanity_a40_v1")
INITIAL_RHO = 1.0 / 1600.0
DEPTHS = {
    6: {
        "full": "deept_table7_stdln6_full_manifest_v1.json",
        "sanity": "deept_table7_stdln6_sanity10_manifest_v1.json",
        "full_sha": "b9401cc378ddcdaa7b68429d3f84a779b448b16960ade1add0a3a325d46d9eb2",
        "sanity_sha": "e87e900ebd76e95031c34cf09799019767f691d9bf8646b963cba0295b43c586",
        "checkpoint_sha": "6e75fe827f7259db65a19a1fda152eb644cb44ead6d2d5a2e92b0950f368a236",
        "full_count": 137,
    },
    12: {
        "full": "deept_table7_stdln12_full_manifest_v1.json",
        "sanity": "deept_table7_stdln12_sanity10_manifest_v1.json",
        "full_sha": "752e466fcb6a1a7e93e974e8db495f7d038b5aaca0a7831a219fc2c3ce0a819f",
        "sanity_sha": "f7df56ff334f6ec1f29125cfcb1fcf1a49bff55a4f679377b51511ea8d2f24b0",
        "checkpoint_sha": "0686f41444b94d7f81aad874ef522a882f0a77b888e537d80d056275c6221c25",
        "full_count": 117,
    },
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
    expected = payload.get(key)
    body = dict(payload)
    body.pop(key, None)
    if expected != canonical(body):
        raise RuntimeError(f"canonical identity mismatch: {path}")
    return payload


def git_blob(path: str) -> bytes:
    return subprocess.check_output([
        "git", "-C", str(DEEPT), "show", f"{DEEPT_REVISION}:{path}"])


def manifests(depth: int) -> tuple[dict, dict]:
    spec = DEPTHS[depth]
    full = verified(MANIFEST_ROOT / spec["full"])
    sanity = verified(MANIFEST_ROOT / spec["sanity"])
    if full["canonical_manifest_sha256"] != spec["full_sha"]:
        raise RuntimeError("frozen full manifest identity differs")
    if sanity["canonical_manifest_sha256"] != spec["sanity_sha"]:
        raise RuntimeError("frozen sanity manifest identity differs")
    if int(full["depth"]) != depth or int(sanity["depth"]) != depth:
        raise RuntimeError("cross-depth manifest binding")
    if sanity["parent_full_manifest_canonical_sha256"] != spec["full_sha"]:
        raise RuntimeError("sanity/full parent identity differs")
    if len(full["properties"]) != spec["full_count"]:
        raise RuntimeError("frozen full property count differs")
    if len(sanity["properties"]) != 10:
        raise RuntimeError("frozen sanity property count differs")
    return full, sanity


def _reference_radii(full: dict) -> dict[tuple[int, int], float]:
    rows = csv.DictReader(io.StringIO(
        git_blob(full["reference_csv"]["git_path"]).decode("utf-8")))
    values = {(int(row["sentence"]), int(row["position"])): float(row["eps"])
              for row in rows}
    if len(values) != int(full["reference_csv"]["row_count"]):
        raise RuntimeError("DeepT reference CSV coverage differs")
    return values


def expected_production_manifest(depth: int) -> dict:
    full, sanity = manifests(depth)
    references = _reference_radii(full)
    examples = {int(row["accepted_draw_ordinal"]): row
                for row in full["accepted_examples"]}
    properties = []
    for ordinal, item in enumerate(sanity["properties"]):
        draw = int(item["accepted_draw_ordinal"])
        example = examples[draw]
        key = (draw, int(item["token_position"]))
        if key not in references:
            raise RuntimeError("sanity property absent from DeepT reference CSV")
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
            "worker_id": ordinal % 2,
        })
    model = full["model"]
    return {
        "schema": "CORET_DEPTH_SANITY_A40_PRODUCTION_MANIFEST_V1",
        "status": "FROZEN_BEFORE_DEPTH_SANITY_EXECUTION",
        "depth": depth,
        "property_count": 10,
        "properties": properties,
        "frozen_full_manifest": {
            "canonical_sha256": full["canonical_manifest_sha256"],
            "file_sha256": sha(MANIFEST_ROOT / DEPTHS[depth]["full"]),
            "property_count": len(full["properties"]),
        },
        "frozen_sanity_manifest": {
            "canonical_sha256": sanity["canonical_manifest_sha256"],
            "file_sha256": sha(MANIFEST_ROOT / DEPTHS[depth]["sanity"]),
            "selection_rule": sanity["selection_rule"],
        },
        "model": model,
        "checkpoint_sha256": model["checkpoint"]["sha256"],
        "pinned_DeepT_revision": full["pinned_DeepT_revision"],
        "dataset": full["dataset"],
        "vocabulary_sha256": model["vocab"]["sha256"],
        "threat_model": {
            "p": 100, "perturbed_words": 1,
            "source_dimension": 128, "single_token_embedding_Linf": True,
        },
        "search": {
            "initial_rho": INITIAL_RHO,
            "initial_rho_binary64_hex": INITIAL_RHO.hex(),
            "factor_two_bracketing": True,
            "maximum_factor_two_refinements": 12,
            "midpoint_iterations": 10,
            "typed_native_domain_failure_prefixes": [
                "sqrt: Bounds must be positive",
                "reciprocal: Bounds must be positive",
                "Reciprocal: there are NaNs in the new COEFFS, pre-condition not met",
            ],
            "unexpected_exception_policy": "fatal",
        },
        "execution": {
            "proof_carrying_native_semantics": True,
            "provenance_structural_support": True,
            "support_and_reduction_recomputed_each_radius": True,
            "requested_AV_generator_tile": 112,
            "grouped_temporary_cap_bytes": 128 * 1024 * 1024,
            "pre_B2_QK_allocator_lifetime_fence": True,
            "generic_fallback_allowed": False,
            "fused_exploratory_backend_allowed": False,
            "workers": 2,
            "assignment": "benchmark_ordinal_mod_2",
        },
        "optimized_smoke_reuse": {"count": 0, "records": []},
        "preparation_scientific_queries": 0,
        "preparation_bound_entrypoint_calls": 0,
        "training_runs": 0,
    }


def production_manifest_path(depth: int) -> Path:
    return PRODUCTION_ROOT / f"depth{depth}_sanity_a40_manifest_v1.json"


def production_manifest(depth: int) -> dict:
    stored = verified(production_manifest_path(depth))
    body = dict(stored)
    body.pop("canonical_manifest_sha256")
    if body != expected_production_manifest(depth):
        raise RuntimeError("stored depth production manifest differs")
    return stored


def assigned_properties(depth: int, worker_id: int) -> list[dict]:
    if worker_id not in (0, 1):
        raise RuntimeError("worker ID must be 0 or 1")
    return [row for row in production_manifest(depth)["properties"]
            if int(row["worker_id"]) == worker_id]


def worker_dir(fresh_root: Path, depth: int, worker_id: int) -> Path:
    result = (fresh_root.resolve() / f"depth_{depth}" / f"worker_{worker_id}").resolve()
    if not result.is_relative_to(fresh_root.resolve()):
        raise RuntimeError("worker directory escapes fresh root")
    return result

def worker_result_root(directory: Path) -> Path:
    result = (directory.resolve() / RESULT_RELATIVE).resolve()
    if not result.is_relative_to(directory.resolve()):
        raise RuntimeError("result directory escapes worker root")
    return result
