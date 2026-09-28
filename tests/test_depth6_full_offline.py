from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import analyze_depth6_full as analyzer
import build_depth6_full_plan as builder
import depth6_full_common as common
import run_depth6_full_chunk as runner


def _record(path: Path, payload: dict) -> None:
    value = dict(payload)
    value["record_sha256"] = common.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_frozen_scientific_manifest_and_checkpoint_identity():
    full = common.full_manifest()
    production = common.production_manifest()
    assert full["canonical_manifest_sha256"] == common.EXPECTED_FULL_MANIFEST_SHA
    assert production["frozen_scientific_manifest"]["canonical_sha256"] == \
        common.EXPECTED_FULL_MANIFEST_SHA
    assert production["checkpoint_sha256"] == common.EXPECTED_CHECKPOINT_SHA
    assert production["scientific_source_commit"] == common.EXPECTED_SOURCE_COMMIT
    assert len(production["properties"]) == 137


def test_exact_complete_disjoint_137_property_plan():
    production = common.production_manifest()
    plan = common.plan()
    expected = [row["property_id"] for row in production["properties"]]
    actual = [row["property_id"] for worker in plan["workers"]
              for item in worker["chunks"] for row in item["properties"]]
    assert len(expected) == len(set(expected)) == 137
    assert len(actual) == len(set(actual)) == 137
    assert set(actual) == set(expected)
    assert [worker["property_count"] for worker in plan["workers"]] == [69, 68]
    assert sum(len(worker["chunks"]) for worker in plan["workers"]) == 8


def test_plan_rebuild_is_deterministic_and_under_twelve_hours():
    first = builder.build_plan(common.production_manifest())
    second = builder.build_plan(common.production_manifest())
    assert first == second == common.plan()
    for worker in first["workers"]:
        assert worker["physical_gpu"] == f"gpu{worker['worker_id']}"
        for item in worker["chunks"]:
            assert item["predicted_hours_upper"] < 12.0


def test_chunk_files_match_parent_and_inventory():
    for worker_id in (0, 1):
        for chunk_index in range(4):
            item = common.chunk(worker_id, chunk_index)
            assert item["worker_id"] == worker_id
            assert item["chunk_index"] == chunk_index
            assert item["parent_plan_sha256"] == common.plan()[
                "canonical_manifest_sha256"]


def test_production_policy_is_fail_closed_and_prefused():
    execution = common.production_manifest()["execution"]
    assert execution["generic_fallback_allowed"] is False
    assert execution["fused_exploratory_backend_allowed"] is False
    assert execution["packed_integer_metadata"] is True
    assert execution["requested_AV_generator_tile"] == 112
    assert execution["grouped_temporary_cap_bytes"] == 128 << 20
    tree = ast.parse((common.REPO / "scripts/run_depth6_full_chunk.py").read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("fused" in name.lower() for name in imports)


def test_zero_solve_preflight():
    with tempfile.TemporaryDirectory() as temporary:
        fresh = Path(temporary) / common.FRESH_ROOT_BASENAME
        result = runner.preflight(fresh)
    assert result["status"] == "PASS_NO_SOLVE"
    assert result["property_count"] == 137
    assert result["planned_property_count"] == 137
    assert result["scientific_queries"] == 0
    assert result["bound_entrypoint_calls"] == 0
    assert result["training_runs"] == 0


def test_configure_is_idempotent_and_worker_isolated():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / common.FRESH_ROOT_BASENAME
        worker = common.worker_dir(root, 0)
        first = runner.configure(worker)
        second = runner.configure(worker)
        assert first[2] == second[2] == common.worker_result_root(worker)
        assert first[2].is_relative_to(worker)
        assert common.worker_dir(root, 0) != common.worker_dir(root, 1)


def test_existing_completed_result_is_skipped_in_chunk_loop():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / common.FRESH_ROOT_BASENAME
        worker = common.worker_dir(root, 0)
        output = runner._initialize(worker, 0)
        item = common.chunk(0, 0)
        for prop in item["properties"]:
            path = output / "properties" / prop["property_id"] / "result_v1.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        old_cwd = Path.cwd()
        old_env = {name: os.environ.get(name) for name in runner.ISOLATION_VARIABLES}
        old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.chdir(worker / "runtime")
            for name, child in zip(runner.ISOLATION_VARIABLES,
                                   ("tmp", "cache", "cuda_cache",
                                    "torch_extensions", "pycache")):
                os.environ[name] = str(worker / child)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            with mock.patch.object(runner, "_validate_complete",
                                   return_value={}), \
                    mock.patch.object(runner, "run_property") as execute:
                runner.run_chunk(0, 0, worker)
                execute.assert_not_called()
        finally:
            os.chdir(old_cwd)
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            if old_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_visible


def test_analyzer_reports_exact_full_synthetic_inventory_read_only():
    manifest = common.production_manifest()
    owner = {row["property_id"]: worker["worker_id"]
             for worker in common.plan()["workers"] for item in worker["chunks"]
             for row in item["properties"]}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / common.FRESH_ROOT_BASENAME
        for prop in manifest["properties"]:
            deep = float(prop["cached_DeepT_reference"][
                "certified_lower_endpoint_binary64"])
            payload = {
                "terminal_status": "COMPLETE",
                "canonical_manifest_sha256": manifest[
                    "canonical_manifest_sha256"],
                "property_id": prop["property_id"],
                "benchmark_ordinal": prop["benchmark_ordinal"],
                "sentence_ordinal": prop["sentence_ordinal"],
                "certified_radius": deep,
                "cached_DeepT_certified_radius": deep,
                "radius_ratio_to_cached_DeepT": 1.0,
                "total_wall_time_seconds": 1.0,
                "checker_failure_count": 0,
                "all_provenance_consistent": True,
                "all_support_claims_validated": True,
                "generic_fallback_count": 0,
                "peak_CPU_RSS_bytes": 1,
                "peak_GPU_allocated_bytes": 1,
                "peak_GPU_reserved_bytes": 1,
            }
            worker = common.worker_dir(root, owner[prop["property_id"]])
            path = (common.worker_result_root(worker) / "properties"
                    / prop["property_id"] / "result_v1.json")
            _record(path, payload)
        result = analyzer.analyze(root)
    assert result["classification"] == "DEPTH6_FULL_137_COMPLETE"
    assert result["completed_properties"] == 137
    assert result["missing_property_ids"] == []
    assert result["duplicate_property_ids"] == []
    assert result["ratio_summary"]["median"] == 1.0
    assert result["scientific_queries_executed_by_analysis"] == 0


def test_slurm_scripts_bind_one_distinct_gpu_and_twelve_hours():
    left = (common.REPO / "slurm/depth6_full_worker0.sbatch").read_text()
    right = (common.REPO / "slurm/depth6_full_worker1.sbatch").read_text()
    for text in (left, right):
        assert "#SBATCH --partition=gpu-a40" in text
        assert "#SBATCH --nodelist=afrodita" in text
        assert "#SBATCH --time=12:00:00" in text
        assert "flock -n" in text
    assert "gpu:gpu0:1" in left and "CUDA_VISIBLE_DEVICES=0" in left
    assert "gpu:gpu1:1" in right and "CUDA_VISIBLE_DEVICES=1" in right
