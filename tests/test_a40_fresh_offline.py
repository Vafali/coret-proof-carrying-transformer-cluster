from __future__ import annotations

import json
from pathlib import Path

import a40_fresh_common as fresh
import cluster_common as common
from build_a40_fresh_plan import build


def artifact() -> Path:
    return common.artifact_root()


def test_plan_is_deterministic_complete_and_disjoint():
    plan = fresh.load_plan()
    manifest = common.load_production_manifest(artifact())
    assert plan == build(manifest)
    rows = [row for worker in plan["workers"]
            for chunk in worker["chunks"] for row in chunk["properties"]]
    ids = [row["property_id"] for row in rows]
    assert len(ids) == len(set(ids)) == 127
    assert set(ids) == {row["property_id"] for row in manifest["properties"]}
    assert [worker["property_count"] for worker in plan["workers"]] == [63, 64]
    assert plan["imported_A4000_completed_properties"] == 0
    assert plan["imported_A4000_query_records"] == 0


def test_chunks_are_frozen_and_conservatively_below_limit():
    plan = fresh.load_plan()
    for worker in plan["workers"]:
        assert len(worker["chunks"]) == 4
        for chunk in worker["chunks"]:
            frozen = fresh.load_chunk(worker["worker_id"],
                                      chunk["chunk_index"])
            assert frozen["properties"] == chunk["properties"]
            hours = chunk["predicted_seconds"] / 3600
            assert 6.0 <= hours < 8.0 < 12.0


def test_historical_duplicate_is_preserved():
    manifest = common.load_production_manifest(artifact())
    ids = {row["property_id"] for worker in fresh.load_plan()["workers"]
           for chunk in worker["chunks"] for row in chunk["properties"]}
    duplicate_ids = {row["property_id"]
                     for row in manifest["historical_duplicate_entries"]}
    assert len(duplicate_ids) == 2
    assert duplicate_ids.issubset(ids)


def test_fresh_workers_import_no_a4000_state_and_are_isolated(tmp_path):
    workers = []
    for worker_id in range(2):
        worker = tmp_path / f"worker_{worker_id}"
        fresh.initialize_fresh_worker(artifact(), worker, worker_id)
        fresh.verify_no_imported_results(worker)
        marker = common.verified_json(worker / "fresh_worker_seed_v1.json")
        assert marker["imported_A4000_completed_properties"] == 0
        assert marker["imported_A4000_query_records"] == 0
        workers.append(worker)
    for name in ("runtime", "tmpdir", "cache", "cuda_cache",
                 "torch_extensions", "pycache", "chunk_records"):
        assert (workers[0] / name).resolve() != (workers[1] / name).resolve()


def test_slurm_is_a40_only_and_worker_specific():
    for worker_id in range(2):
        text = (common.REPO / f"slurm/a40_worker{worker_id}_chunk.sbatch").read_text()
        assert "#SBATCH --partition=gpu-a40" in text
        assert "#SBATCH --nodelist=afrodita" in text
        assert f"#SBATCH --gres=gpu:gpu{worker_id}:1" in text
        assert "#SBATCH --time=11:30:00" in text
        assert f'worker_{worker_id}' in text
        assert "TMPDIR" in text and "XDG_CACHE_HOME" in text
        assert "portobelo" not in text.lower()
        assert "l40" not in text.lower()


def test_production_path_remains_prefused():
    assert common.sha256(
        common.RESEARCH / "coret_structural_support_precise_dot_v1.py") == (
        "86ba56410c713283251264c618d7652f1f741531bda4bfdc0dcdf38df7a2a0c6")
    scientific = list(common.RESEARCH.glob("*.py"))
    assert not any("fused_av" in path.read_text(errors="ignore")
                   for path in scientific)


def test_fresh_runner_disables_prior_smoke_reuse_only_in_orchestration():
    text = (common.REPO / "scripts/run_a40_fresh_chunk.py").read_text()
    assert "runner._reuse_record = lambda manifest, pid: None" in text
    scientific = (common.RESEARCH
                  / "coret_optimized_historical_127_v1.py").read_text()
    assert "def _reuse_record" in scientific
