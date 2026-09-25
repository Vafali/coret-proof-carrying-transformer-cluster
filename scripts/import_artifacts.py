#!/usr/bin/env python3
"""Verify and import an unpacked immutable artifact bundle."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cluster_common import verify_artifact_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    verify_artifact_manifest(source)
    destination = Path(args.destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    shutil.copytree(source, destination)
    verify_artifact_manifest(destination)
    print(f"ARTIFACT_IMPORT_PASS {destination}")


if __name__ == "__main__": main()
