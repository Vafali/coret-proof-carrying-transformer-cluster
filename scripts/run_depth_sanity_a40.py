#!/usr/bin/env python3
"""Isolated, resumable two-worker A40 depth sanity production."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import depth_sanity_common as common


REPO = Path(__file__).resolve().parents[1]
RH = REPO / "research_hab"
ISOLATION_VARIABLES = (
    "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR",
    "PYTHONPYCACHEPREFIX",
)


def _verified_record(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("record_sha256")
    body = dict(payload)
    body.pop("record_sha256", None)
    if expected != common.canonical(body):
        raise RuntimeError(f"result record identity mismatch: {path}")
    return payload


def configure(depth: int, directory: Path):
    sys.path.insert(0, str(RH)) if str(RH) not in sys.path else None
    import coret_bounded_native_depth_graph_v1 as graph
    import coret_deept_depth_adapter_v1 as depth_adapter
    import coret_optimized_historical_127_v1 as runner

    manifest = common.production_manifest(depth)
    output = common.worker_result_root(directory)
    graph.configure_depth(depth)
    runner.WORKTREE = REPO
    runner.OUT = output
    runner.MANIFEST = common.production_manifest_path(depth)
    runner.PREFLIGHT = output / "depth_sanity_preflight_v1.json"
    runner.SUMMARY = output / "depth_sanity_summary_v1.json"
    runner.graph = graph
    runner.validate_manifest = lambda: common.production_manifest(depth)
    runner._reuse_record = lambda manifest_arg, property_id: None
    runner.adapter.load_native_model = (
        lambda modules, device, dtype: depth_adapter.load_native_model(
            modules, device, depth, dtype))
    runner.adapter.build_deept_args = (
        lambda modules, device: depth_adapter.build_deept_args(
            modules, device, depth))
    examples = {int(row["sentence_ordinal"]): {"token_ids": row["token_ids"]}
                for row in manifest["properties"]}
    runner._example = lambda prop: examples[int(prop["sentence_ordinal"])]
    return runner, manifest, output


def _initialize(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for relative in ("runtime", "tmp", "cache", "cuda_cache",
                     "torch_extensions", "pycache", "chunk_records"):
        (directory / relative).mkdir(parents=True, exist_ok=True)


def _verify_isolation(directory: Path) -> None:
    if Path.cwd().resolve() != (directory / "runtime").resolve():
        raise RuntimeError("worker must execute from isolated runtime directory")
    for name in ISOLATION_VARIABLES:
        value = os.environ.get(name)
        if not value or not Path(value).resolve().is_relative_to(directory.resolve()):
            raise RuntimeError(f"worker isolation variable differs: {name}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("exactly one A40 must be visible")


def _validate_complete(output: Path, manifest: dict, property_id: str) -> dict:
    result = output / "properties" / property_id / "result_v1.json"
    row = _verified_record(result)
    if (row.get("terminal_status") != "COMPLETE"
            or row.get("canonical_manifest_sha256") != manifest["canonical_manifest_sha256"]
            or row.get("property_id") != property_id
            or row.get("complete_certificate") is not True
            or row.get("independent_checker_accepted") is not True
            or row.get("generic_fallback_count") != 0
            or row.get("all_support_claims_validated") is not True
            or row.get("all_provenance_consistent") is not True
            or row.get("outward_fresh_radius_envelopes") is not True):
        raise RuntimeError(f"nonaccepted completed depth property: {property_id}")
    return row


def preflight(depth: int) -> dict:
    manifest = common.production_manifest(depth)
    properties = manifest["properties"]
    if len(properties) != 10 or {row["worker_id"] for row in properties} != {0, 1}:
        raise RuntimeError("depth sanity assignment differs")
    if [len(common.assigned_properties(depth, worker)) for worker in (0, 1)] != [5, 5]:
        raise RuntimeError("depth sanity worker split differs")
    checkpoint_path = manifest["model"]["checkpoint"]["git_path"]
    checkpoint = common.git_blob(checkpoint_path)
    if common.hashlib.sha256(checkpoint).hexdigest() != manifest["checkpoint_sha256"]:
        raise RuntimeError("depth checkpoint identity differs")
    config_blob = common.git_blob(manifest["model"]["config"]["git_path"])
    if common.hashlib.sha256(config_blob).hexdigest() != manifest["model"][
            "config"]["sha256"]:
        raise RuntimeError("depth configuration identity differs")
    config = json.loads(config_blob)
    if config != manifest["model"]["configuration"] \
            or int(config["num_hidden_layers"]) != depth:
        raise RuntimeError("depth configuration semantics differ")
    if any("fused" in path.name.lower() for path in (
            RH / "coret_deept_depth_adapter_v1.py",
            RH / "coret_bounded_native_depth_graph_v1.py")):
        raise RuntimeError("exploratory fused backend is reachable by filename")
    return {
        "status": "PASS_NO_SOLVE",
        "depth": depth,
        "property_count": 10,
        "worker_property_counts": [5, 5],
        "manifest_sha256": manifest["canonical_manifest_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_sha256": manifest["model"]["config"]["sha256"],
        "generic_fallback_allowed": False,
        "fused_exploratory_backend_reachable": False,
        "scientific_queries": 0,
        "bound_entrypoint_calls": 0,
        "training_runs": 0,
    }


def _invoke_query(depth: int, directory: Path, property_id: str,
                  rho: float, ordinal: int) -> None:
    subprocess.run([
        sys.executable, str(Path(__file__).resolve()), "query",
        "--depth", str(depth), "--worker-dir", str(directory),
        "--property-id", property_id, "--rho-hex", float(rho).hex(),
        "--query-ordinal", str(ordinal),
    ], check=True, cwd=directory / "runtime", env={
        **os.environ, "PYTHONPATH": f"{REPO / 'scripts'}:{RH}",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })


def run_worker(depth: int, worker_id: int, directory: Path) -> None:
    _initialize(directory)
    _verify_isolation(directory)
    runner, manifest, output = configure(depth, directory)
    expected_directory = common.worker_dir(directory.parents[1], depth, worker_id)
    if directory.resolve() != expected_directory:
        raise RuntimeError("worker/depth directory identity differs")
    runner._invoke_query = lambda pid, rho, ordinal: _invoke_query(
        depth, directory, pid, rho, ordinal)
    completed = []
    for prop in common.assigned_properties(depth, worker_id):
        property_id = prop["property_id"]
        result = output / "properties" / property_id / "result_v1.json"
        if result.exists():
            _validate_complete(output, manifest, property_id)
        else:
            runner.run_property(property_id, True)
            _validate_complete(output, manifest, property_id)
        completed.append(property_id)
    payload = {
        "schema": "CORET_DEPTH_SANITY_A40_WORKER_COMPLETE_V1",
        "depth": depth, "worker_id": worker_id,
        "manifest_sha256": manifest["canonical_manifest_sha256"],
        "completed_properties": completed,
        "completed_property_count": len(completed),
    }
    payload["record_sha256"] = common.canonical(payload)
    target = directory / "chunk_records/worker_complete_v1.json"
    if target.exists():
        if _verified_record(target) != payload:
            raise RuntimeError("existing worker completion record differs")
    else:
        target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "DEPTH_SANITY_WORKER_COMPLETE",
                      "depth": depth, "worker_id": worker_id,
                      "completed": len(completed)}, sort_keys=True))


def query(depth: int, directory: Path, property_id: str,
          rho_hex: str, ordinal: int) -> None:
    runner, _, _ = configure(depth, directory)
    runner.generate_query(property_id, float.fromhex(rho_hex), ordinal, True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "worker", "query"))
    parser.add_argument("--depth", type=int, choices=(6, 12), required=True)
    parser.add_argument("--worker-id", type=int, choices=(0, 1))
    parser.add_argument("--worker-dir")
    parser.add_argument("--property-id")
    parser.add_argument("--rho-hex")
    parser.add_argument("--query-ordinal", type=int)
    args = parser.parse_args()
    if args.mode == "preflight":
        print(json.dumps(preflight(args.depth), sort_keys=True))
    elif args.mode == "worker":
        if args.worker_id is None or args.worker_dir is None:
            parser.error("worker requires worker ID and directory")
        run_worker(args.depth, args.worker_id, Path(args.worker_dir).resolve())
    else:
        if None in (args.worker_dir, args.property_id, args.rho_hex,
                    args.query_ordinal):
            parser.error("query arguments incomplete")
        query(args.depth, Path(args.worker_dir).resolve(), args.property_id,
              args.rho_hex, args.query_ordinal)


if __name__ == "__main__":
    main()
