#!/usr/bin/env python3
"""Run one preregistered frozen query in an isolated calibration directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cluster_common import artifact_root
from portable_runner import configure


CASES = {
    "short": {"property_id": "deept_table7_stdln3_s000_line504_tok03",
              "rho": 0.0009375, "query_ordinal": 2,
              "reference": "query_02_0x1deb851eb851eb8pm11_result_v1.json"},
    "long": {"property_id": "deept_table7_stdln3_s001_line1794_tok05",
             "rho": 0.0009375, "query_ordinal": 2,
             "reference": "query_02_0x1deb851eb851eb8pm11_result_v1.json"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("calibration requires exactly one visible physical GPU")
    case = CASES[args.case]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact = artifact_root(args.artifact_root)
    runner, _, _ = configure(output, artifact)
    runner.generate_query(case["property_id"], case["rho"],
                          case["query_ordinal"], True)
    print(json.dumps({"status": "CALIBRATION_QUERY_COMPLETE", "case": args.case,
                      "property_id": case["property_id"], "rho": case["rho"],
                      "output": str(output)}, sort_keys=True))


if __name__ == "__main__": main()
