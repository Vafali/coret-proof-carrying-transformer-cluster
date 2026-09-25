from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

import a40_fresh_common as fresh
import cluster_common as common
import portable_runner
import run_a40_fresh_chunk
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


def test_portable_configure_is_idempotent_and_workers_are_isolated(tmp_path):
    import coret_optimized_historical_127_v1 as runner

    mutable = ("OUT", "SUMMARY", "HISTORICAL_MANIFEST", "DEEPT_POSITIONS",
               "DEEPT_RESULT", "validate_manifest")
    original = {name: getattr(runner, name) for name in mutable}
    try:
        worker0 = tmp_path / "worker_0"
        worker1 = tmp_path / "worker_1"
        roots = []
        for _ in range(4):
            configured, _, root = portable_runner.configure(worker0, artifact())
            assert configured is runner
            roots.append(root)
            assert root.is_relative_to(worker0.resolve())
            assert runner.OUT == root
        assert roots == [worker0.resolve() / common.BASELINE_REL] * 4

        _, _, other = portable_runner.configure(worker1, artifact())
        assert other == worker1.resolve() / common.BASELINE_REL
        assert other.is_relative_to(worker1.resolve())
        assert other != roots[0]
    finally:
        for name, value in original.items():
            setattr(runner, name, value)


def test_synthetic_two_property_chunk_reconfigures_without_bound_call(
        tmp_path, monkeypatch):
    worktree = tmp_path / "repository"
    fake = SimpleNamespace(
        WORKTREE=worktree,
        OUT=worktree / common.BASELINE_REL,
    )
    manifest = {"identity": "offline-only"}
    patched = []

    def fake_patch(runner, artifact, result_root, loaded_manifest):
        assert loaded_manifest is manifest
        runner.OUT = result_root
        patched.append((artifact, result_root))

    monkeypatch.setitem(sys.modules,
                        "coret_optimized_historical_127_v1", fake)
    monkeypatch.setattr(portable_runner, "load_production_manifest",
                        lambda artifact: manifest)
    monkeypatch.setattr(portable_runner, "patch_runner", fake_patch)

    worker = tmp_path / "worker_0"
    artifact_root = tmp_path / "artifact"
    resolved = []
    for property_id in ("fake_property_0", "fake_property_1"):
        configured, loaded, result_root = portable_runner.configure(
            worker, artifact_root)
        assert property_id.startswith("fake_property_")
        assert configured is fake and loaded is manifest
        assert result_root.is_relative_to(worker.resolve())
        resolved.append(result_root)

    assert resolved == [worker.resolve() / common.BASELINE_REL] * 2
    assert len(patched) == 2
    assert not hasattr(fake, "generate_query")
    assert not hasattr(fake, "run_property")


@pytest.mark.parametrize(("worker_id", "property_id"), [
    (0, "deept_table7_stdln3_s000_line504_tok05"),
    (1, "deept_table7_stdln3_s000_line504_tok10"),
])
def test_chunk_zero_resume_validates_and_skips_completed_property(
        tmp_path, monkeypatch, worker_id, property_id):
    worker = tmp_path / f"worker_{worker_id}"
    for name in ("runtime", "tmpdir", "cache", "cuda_cache",
                 "torch_extensions", "pycache", "chunk_records"):
        (worker / name).mkdir(parents=True)
    root = fresh.result_root(worker)
    result = root / "properties" / property_id / "result_v1.json"
    result.parent.mkdir(parents=True)
    result.write_text("offline resume sentinel\n")

    monkeypatch.chdir(worker / "runtime")
    for name, directory in zip(run_a40_fresh_chunk.ISOLATION_VARIABLES,
            ("tmpdir", "cache", "cuda_cache", "torch_extensions", "pycache")):
        monkeypatch.setenv(name, str(worker / directory))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(worker_id))
    monkeypatch.setattr(sys, "argv", ["run_a40_fresh_chunk.py",
        "--worker-id", str(worker_id), "--chunk-index", "0",
        "--worker-dir", str(worker), "--artifact-root", str(tmp_path / "artifact")])
    monkeypatch.setattr(run_a40_fresh_chunk, "artifact_root",
                        lambda value: tmp_path / "artifact")
    monkeypatch.setattr(run_a40_fresh_chunk, "initialize_fresh_worker",
                        lambda artifact, directory, identity: root)
    monkeypatch.setattr(run_a40_fresh_chunk, "load_plan", lambda: {
        "canonical_manifest_sha256": "offline_plan",
    })
    monkeypatch.setattr(run_a40_fresh_chunk, "assigned_properties",
                        lambda plan, identity: {property_id})
    monkeypatch.setattr(run_a40_fresh_chunk, "load_chunk", lambda identity, index: {
        "canonical_manifest_sha256": "offline_chunk",
        "properties": [{"property_id": property_id}],
    })
    monkeypatch.setattr(run_a40_fresh_chunk, "result_root",
                        lambda directory: root)
    validated = []
    monkeypatch.setattr(run_a40_fresh_chunk, "validate_property",
                        lambda directory, identity: validated.append(
                            (directory, identity)))

    def forbidden_run(*args, **kwargs):
        raise AssertionError("completed property was recomputed")

    monkeypatch.setattr(run_a40_fresh_chunk, "run_property", forbidden_run)
    run_a40_fresh_chunk.main()
    assert validated == [(root, property_id)]
