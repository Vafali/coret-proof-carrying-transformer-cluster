#!/usr/bin/env python3
"""Invoke the frozen runner with portable artifact/output locations."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cluster_common import artifact_root, initialize_worker, load_production_manifest, patch_runner


def configure(worker_dir: Path, artifact: Path):
    import coret_optimized_historical_127_v1 as runner
    manifest = load_production_manifest(artifact)
    result_root = worker_dir / runner.OUT.relative_to(runner.WORKTREE)
    patch_runner(runner, artifact, result_root, manifest)
    return runner, manifest, result_root


def query(args) -> None:
    artifact = artifact_root(args.artifact_root)
    runner, _, _ = configure(Path(args.worker_dir).resolve(), artifact)
    runner.generate_query(args.property_id, float.fromhex(args.rho_hex),
                          args.query_ordinal, True)


def property_run(args) -> None:
    artifact = artifact_root(args.artifact_root)
    worker_dir = Path(args.worker_dir).resolve()
    initialize_worker(artifact, worker_dir)
    runner, _, _ = configure(worker_dir, artifact)

    def invoke(property_id: str, rho: float, ordinal: int) -> None:
        command = [sys.executable, str(Path(__file__).resolve()), "query",
                   "--worker-dir", str(worker_dir),
                   "--artifact-root", str(artifact),
                   "--property-id", property_id,
                   "--rho-hex", float(rho).hex(),
                   "--query-ordinal", str(ordinal)]
        subprocess.run(command, check=True, env={**os.environ,
            "PYTHONPATH": str(runner.WORKTREE / "research_hab")})

    runner._invoke_query = invoke
    runner.run_property(args.property_id, True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("query", "property"):
        item = sub.add_parser(name)
        item.add_argument("--worker-dir", required=True)
        item.add_argument("--artifact-root")
        item.add_argument("--property-id", required=True)
        if name == "query":
            item.add_argument("--rho-hex", required=True)
            item.add_argument("--query-ordinal", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "query": query(args)
    else: property_run(args)


if __name__ == "__main__":
    main()
