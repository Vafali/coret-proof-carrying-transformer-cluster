#!/usr/bin/env python3
"""Run one immutable, resumable full depth-6 A40 production chunk."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import depth6_full_common as common
import depth_sanity_common as depth_common


REPO = Path(__file__).resolve().parents[1]
RH = REPO / "research_hab"
ISOLATION_VARIABLES = (
    "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR",
    "PYTHONPYCACHEPREFIX",
)


def _verified_record(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.get("record_sha256")
    body = dict(payload)
    body.pop("record_sha256", None)
    if claimed != common.canonical(body):
        raise RuntimeError(f"record identity mismatch: {path}")
    return payload


def _initialize(worker_dir: Path, worker_id: int) -> Path:
    for relative in ("runtime", "tmp", "cache", "cuda_cache",
                     "torch_extensions", "pycache", "chunk_records"):
        (worker_dir / relative).mkdir(parents=True, exist_ok=True)
    output = common.worker_result_root(worker_dir)
    (output / "properties").mkdir(parents=True, exist_ok=True)
    marker = worker_dir / "full_depth6_worker_identity_v1.json"
    expected = {
        "schema": "CORET_DEPTH6_FULL_A40_WORKER_IDENTITY_V1",
        "worker_id": worker_id,
        "production_manifest_sha256": common.production_manifest()[
            "canonical_manifest_sha256"],
        "plan_sha256": common.plan()["canonical_manifest_sha256"],
        "fresh_root_basename": common.FRESH_ROOT_BASENAME,
        "imported_results": 0,
    }
    expected["record_sha256"] = common.canonical(expected)
    if marker.exists():
        if _verified_record(marker) != expected:
            raise RuntimeError("worker identity marker differs")
    else:
        marker.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
    return output


def _verify_isolation(worker_dir: Path) -> None:
    if Path.cwd().resolve() != (worker_dir / "runtime").resolve():
        raise RuntimeError("worker must execute from isolated runtime directory")
    for name in ISOLATION_VARIABLES:
        value = os.environ.get(name)
        if not value or not Path(value).resolve().is_relative_to(
                worker_dir.resolve()):
            raise RuntimeError(f"worker isolation variable differs: {name}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("exactly one A40 must be visible")


def configure(worker_dir: Path):
    sys.path.insert(0, str(RH)) if str(RH) not in sys.path else None
    import coret_bounded_native_depth_graph_v1 as graph
    import coret_deept_depth_adapter_v1 as depth_adapter
    import coret_optimized_historical_127_v1 as runner

    manifest = common.production_manifest()
    output = common.worker_result_root(worker_dir)
    graph.configure_depth(6)
    runner.WORKTREE = REPO
    runner.OUT = output
    runner.MANIFEST = common.PRODUCTION_MANIFEST_PATH
    runner.PREFLIGHT = output / "depth6_full_preflight_v1.json"
    runner.SUMMARY = output / "depth6_full_summary_v1.json"
    runner.graph = graph
    runner.validate_manifest = common.production_manifest
    runner._reuse_record = lambda manifest_arg, property_id: None
    runner.adapter.load_native_model = (
        lambda modules, device, dtype: depth_adapter.load_native_model(
            modules, device, 6, dtype))
    runner.adapter.build_deept_args = (
        lambda modules, device: depth_adapter.build_deept_args(
            modules, device, 6))
    examples = {int(row["sentence_ordinal"]): {"token_ids": row["token_ids"]}
                for row in manifest["properties"]}
    runner._example = lambda prop: examples[int(prop["sentence_ordinal"])]
    return runner, manifest, output


def _validate_complete(output: Path, property_id: str) -> dict:
    manifest = common.production_manifest()
    expected = {row["property_id"] for row in manifest["properties"]}
    if property_id not in expected:
        raise RuntimeError("completed property is outside frozen universe")
    directory = output / "properties" / property_id
    result = _verified_record(directory / "result_v1.json")
    if (result.get("terminal_status") != "COMPLETE"
            or result.get("canonical_manifest_sha256")
            != manifest["canonical_manifest_sha256"]
            or result.get("property_id") != property_id
            or result.get("complete_certificate") is not True
            or result.get("independent_checker_accepted") is not True
            or result.get("checker_failure_count") != 0
            or result.get("soundness_or_proof_failure_count") != 0
            or result.get("generic_fallback_count") != 0
            or result.get("all_support_claims_validated") is not True
            or result.get("all_provenance_consistent") is not True
            or result.get("outward_fresh_radius_envelopes") is not True
            or result.get("deterministic_support_reduction_lineage") is not True):
        raise RuntimeError(f"nonaccepted completed property: {property_id}")
    queries = sorted((directory / "queries").glob("query_*_result_v1.json"))
    rows = [_verified_record(path) for path in queries]
    if [row["query_ordinal"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("query ordinal chain differs")
    events_path = directory / "query_events_v1.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()
              if line]
    for event in events:
        body = dict(event)
        claimed = body.pop("event_sha256")
        if common.canonical(body) != claimed:
            raise RuntimeError("query journal identity differs")
    if {row["record_sha256"] for row in rows} != {
            event["query_result_record_sha256"] for event in events}:
        raise RuntimeError("journal/query inventory differs")
    return result


def _invoke_query(worker_dir: Path, property_id: str,
                  rho: float, ordinal: int) -> None:
    subprocess.run([
        sys.executable, str(Path(__file__).resolve()), "query",
        "--worker-dir", str(worker_dir), "--property-id", property_id,
        "--rho-hex", float(rho).hex(), "--query-ordinal", str(ordinal),
    ], check=True, cwd=worker_dir / "runtime", env={
        **os.environ,
        "PYTHONPATH": f"{REPO / 'scripts'}:{RH}",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })


def run_property(worker_dir: Path, property_id: str) -> None:
    runner, _, _ = configure(worker_dir)
    runner._invoke_query = lambda pid, rho, ordinal: _invoke_query(
        worker_dir, pid, rho, ordinal)
    runner.run_property(property_id, True)


def preflight(fresh_root: Path) -> dict:
    manifest = common.production_manifest()
    plan = common.plan()
    properties = manifest["properties"]
    ids = [row["property_id"] for row in properties]
    planned = [row["property_id"] for worker in plan["workers"]
               for item in worker["chunks"] for row in item["properties"]]
    if (len(ids) != 137 or len(set(ids)) != 137
            or len(planned) != 137 or len(set(planned)) != 137
            or set(ids) != set(planned)):
        raise RuntimeError("full depth-6 property coverage differs")
    if fresh_root.name != common.FRESH_ROOT_BASENAME:
        raise RuntimeError("fresh output-root identity differs")
    for path, expected in common.SCIENTIFIC_SOURCE_HASHES.items():
        if common.sha(REPO / path) != expected:
            raise RuntimeError(f"scientific source identity differs: {path}")
    checkpoint = depth_common.git_blob(
        manifest["model"]["checkpoint"]["git_path"])
    if common.hashlib.sha256(checkpoint).hexdigest() \
            != common.EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("depth-6 checkpoint identity differs")
    if (manifest["execution"]["generic_fallback_allowed"] is not False
            or manifest["execution"]["fused_exploratory_backend_allowed"]
            is not False):
        raise RuntimeError("production fallback/backend policy differs")
    return {
        "status": "PASS_NO_SOLVE",
        "scientific_source_commit": common.EXPECTED_SOURCE_COMMIT,
        "scientific_manifest_sha256": common.EXPECTED_FULL_MANIFEST_SHA,
        "production_manifest_sha256": manifest["canonical_manifest_sha256"],
        "plan_sha256": plan["canonical_manifest_sha256"],
        "checkpoint_sha256": common.EXPECTED_CHECKPOINT_SHA,
        "property_count": len(ids),
        "unique_property_count": len(set(ids)),
        "planned_property_count": len(planned),
        "worker_count": 2,
        "chunk_count": 8,
        "fresh_root": str(fresh_root),
        "generic_fallback_allowed": False,
        "scientific_queries": 0,
        "bound_entrypoint_calls": 0,
        "training_runs": 0,
    }


def run_chunk(worker_id: int, chunk_index: int, worker_dir: Path) -> None:
    output = _initialize(worker_dir, worker_id)
    _verify_isolation(worker_dir)
    item = common.chunk(worker_id, chunk_index)
    allowed = common.assigned_ids(worker_id)
    existing_dirs = {path.parent.name for path in
                     (output / "properties").glob("*/result_v1.json")}
    if not existing_dirs.issubset(allowed):
        raise RuntimeError("worker contains result owned by another worker")
    completed = []
    for prop in item["properties"]:
        property_id = prop["property_id"]
        result = output / "properties" / property_id / "result_v1.json"
        if result.exists():
            _validate_complete(output, property_id)
        else:
            run_property(worker_dir, property_id)
            _validate_complete(output, property_id)
        completed.append(property_id)
    record = {
        "schema": "CORET_DEPTH6_FULL_A40_CHUNK_COMPLETION_V1",
        "plan_sha256": common.plan()["canonical_manifest_sha256"],
        "chunk_manifest_sha256": item["canonical_manifest_sha256"],
        "worker_id": worker_id,
        "chunk_index": chunk_index,
        "completed_properties": completed,
        "completed_property_count": len(completed),
    }
    record["record_sha256"] = common.canonical(record)
    target = worker_dir / "chunk_records" / f"chunk_{chunk_index}_complete_v1.json"
    if target.exists():
        if _verified_record(target) != record:
            raise RuntimeError("existing chunk completion record differs")
    else:
        target.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "DEPTH6_FULL_CHUNK_COMPLETE",
                      "worker_id": worker_id, "chunk_index": chunk_index,
                      "property_count": len(completed)}, sort_keys=True))


def query(worker_dir: Path, property_id: str,
          rho_hex: str, ordinal: int) -> None:
    runner, _, _ = configure(worker_dir)
    runner.generate_query(property_id, float.fromhex(rho_hex), ordinal, True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "chunk", "query"))
    parser.add_argument("--fresh-root")
    parser.add_argument("--worker-id", type=int, choices=(0, 1))
    parser.add_argument("--chunk-index", type=int,
                        choices=range(common.CHUNKS_PER_WORKER))
    parser.add_argument("--worker-dir")
    parser.add_argument("--property-id")
    parser.add_argument("--rho-hex")
    parser.add_argument("--query-ordinal", type=int)
    args = parser.parse_args()
    if args.mode == "preflight":
        if args.fresh_root is None:
            parser.error("preflight requires --fresh-root")
        print(json.dumps(preflight(Path(args.fresh_root).resolve()), sort_keys=True))
    elif args.mode == "chunk":
        if None in (args.worker_id, args.chunk_index, args.worker_dir):
            parser.error("chunk arguments incomplete")
        run_chunk(args.worker_id, args.chunk_index,
                  Path(args.worker_dir).resolve())
    else:
        if None in (args.worker_dir, args.property_id, args.rho_hex,
                    args.query_ordinal):
            parser.error("query arguments incomplete")
        query(Path(args.worker_dir).resolve(), args.property_id,
              args.rho_hex, args.query_ordinal)


if __name__ == "__main__":
    main()
