#!/usr/bin/env python3
"""Metadata-only preregistration. Never imports or executes the verifier."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import cluster_common as common


REPO = Path(__file__).resolve().parents[1]
SCHEMA = "CORET_PREREGISTERED_TRANSFORMER_BENCHMARK24_V1"
ALGORITHM = "historical_radius_tertiles_balanced_sentences_spaced_tokens_v1"
RADIUS_SOURCE = "cached_DeepT_reference.certified_lower_endpoint_binary64"
STRATA = ("historical_radius_lower_tertile", "historical_radius_middle_tertile",
          "historical_radius_upper_tertile")
# Selection reads ONLY this whitelist and the historical radius. It does not
# inspect previous/current CoReT results, runtimes, margins, certificates or logs.
FIELDS = ("property_id", "benchmark_ordinal", "sentence_id", "sentence_ordinal",
          "source_test_line", "canonical_binary_test_index", "raw_sentence_sha256",
          "sequence_length", "source_dimension", "token_position", "token",
          "token_id", "clean_label", "nominal_prediction")
EXECUTION_FILES = ("scripts/run_sound_fp64_3l_campaign.py",
                   "scripts/run_sound_fp64_finish_3l_v1.py",
                   "scripts/cluster_common.py", "scripts/a40_fresh_common.py")


def radius(row):
    reference = row.get("cached_DeepT_reference", {})
    if "certified_lower_endpoint_binary64" not in reference:
        raise RuntimeError("authoritative historical radius is missing")
    value = float(reference["certified_lower_endpoint_binary64"])
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("historical radius is not finite/nonnegative")
    return value


def population_projection(properties):
    if len(properties) != 127:
        raise RuntimeError("source population must contain exactly 127 properties")
    rows = [{**{key: row[key] for key in FIELDS}, "tested_radius": radius(row),
             "tested_radius_hex": radius(row).hex()} for row in properties]
    rows.sort(key=lambda row: row["benchmark_ordinal"])
    if (len({row["property_id"] for row in rows}) != 127 or
            [row["benchmark_ordinal"] for row in rows] != list(range(127))):
        raise RuntimeError("source population IDs/order differ")
    return rows


def _completion_possible(pool, selected, groups):
    chosen = {row["property_id"] for row in selected}
    used = Counter(row["raw_sentence_sha256"] for row in selected)
    quota = Counter(row["radius_stratum"] for row in selected)
    need = tuple(8 - quota[s] for s in STRATA)
    capacities = []
    for group in groups:
        avail = Counter(row["radius_stratum"] for row in pool
                        if row["raw_sentence_sha256"] == group and row["property_id"] not in chosen)
        capacities.append((max(0, 2 - used[group]), 3 - used[group], tuple(avail[s] for s in STRATA)))

    @lru_cache(None)
    def visit(i, a, b, c):
        if min(a, b, c) < 0:
            return False
        if i == len(capacities):
            return a == b == c == 0
        if not sum(v[0] for v in capacities[i:]) <= a + b + c <= sum(v[1] for v in capacities[i:]):
            return False
        minimum, maximum, available = capacities[i]
        for x in range(min(maximum, available[0], a) + 1):
            for y in range(min(maximum - x, available[1], b) + 1):
                for z in range(min(maximum - x - y, available[2], c) + 1):
                    if minimum <= x + y + z <= maximum and visit(i + 1, a - x, b - y, c - z):
                        return True
        return False
    return visit(0, *need)


def select(properties):
    rows = population_projection(properties)
    ranked = sorted(rows, key=lambda row: (row["tested_radius"], row["benchmark_ordinal"]))
    annotated = {}
    for i, row in enumerate(ranked):
        annotated[row["property_id"]] = {**row, "radius_stratum": STRATA[min(2, i * 3 // 127)],
                                         "historical_radius_rank": i}
    # Duplicate historical draws remain in the authenticated population digest,
    # but the same raw sentence/token pair appears at most once in this subset.
    unique = {}
    for row in rows:
        unique.setdefault((row["raw_sentence_sha256"], row["token_position"]), annotated[row["property_id"]])
    pool = list(unique.values())
    groups = sorted({row["raw_sentence_sha256"] for row in pool})
    if len(groups) != 9 or not _completion_possible(pool, [], groups):
        raise RuntimeError("fixed 24-property stratification is impossible; do not replace properties")
    selected = []
    for step in range(24):
        stratum = STRATA[step % 3]
        chosen = {row["property_id"] for row in selected}
        group_counts = Counter(row["raw_sentence_sha256"] for row in selected)
        group_strata = Counter((row["raw_sentence_sha256"], row["radius_stratum"]) for row in selected)
        token_counts = Counter(row["token_position"] for row in selected)

        def priority(row):
            group = row["raw_sentence_sha256"]
            positions = [item["token_position"] for item in selected if item["raw_sentence_sha256"] == group]
            separation = min((Fraction(abs(row["token_position"] - p), row["sequence_length"] - 1)
                              for p in positions), default=Fraction(0))
            tie = hashlib.sha256((SCHEMA + ":" + row["property_id"]).encode()).hexdigest()
            return (group_counts[group], group_strata[group, stratum], -separation,
                    token_counts[row["token_position"]], tie, row["benchmark_ordinal"])

        candidates = sorted((row for row in pool if row["radius_stratum"] == stratum and
                             row["property_id"] not in chosen and group_counts[row["raw_sentence_sha256"]] < 3),
                            key=priority)
        for row in candidates:
            if _completion_possible(pool, selected + [row], groups):
                selected.append(row)
                break
        else:
            raise RuntimeError("deterministic selection failed; no replacement policy")
    for i, row in enumerate(selected):
        row["selection_rationale"] = {
            "selection_ordinal": i, "algorithm": ALGORITHM,
            "radius_round": i // 3, "radius_stratum": row["radius_stratum"],
            "sentence_quota": "2 or 3 per unique raw sentence; 9 sentences total",
            "token_diversity": "prefer farthest normalized position within sentence, then least repeated position globally",
            "completion_guard": "exact discrete remaining-quota feasibility, metadata only",
            "tie_break": "SHA256(schema + ':' + property_id), then historical ordinal",
        }
    return selected


def capabilities():
    return {
        "final_proof_status": {"available": True, "source": "campaign terminal_status and final_sound_lower_margin",
                               "independently_certified_requires": "complete independent checker PASS"},
        "independent_certificate_check": {"available": False, "status": "NOT_AVAILABLE",
            "reason": "current complete sound-FP64 path emits integrity/support checks and optional MPFR spots, not an independent complete-graph numerical verdict"},
        "certificate_integrity": {"available": True, "source": "campaign._verified_result artifact/report hashes",
                                  "not_independent_numerical_soundness": True},
        "failure_stage_operator": {"available": True, "source": "failure_stage/reason; domain_failure_diagnostic when emitted",
                                   "limitation": "non-domain Block2 exceptions may identify only block2_to_margin"},
        "runtime": {"available": True, "source": "runtime_seconds; entire property"},
        "peak_memory": {"available": True, "source": "peak_gpu_allocated/reserved_bytes and peak_cpu_rss_bytes",
                        "scope": "GPU peak generally Block2-to-margin only; CPU RSS is process lifetime; unavailable values stay null"},
        "generator_count": {"available": True, "source": "final_generator_count if completed/domain-failed"},
        "reduction_counts": {"available": True, "source": "certificate_report.reductions",
                             "scope": "Block2-to-head only; earlier per-stage reports are discarded by existing campaign"},
        "numerical_diagnostics": {"available": True, "source": "numerical_widening, max_numerical_native_ratio, report.stages/MPFR spots if emitted",
                                  "full_containment_checker_status": "NOT_AVAILABLE"},
        "cpu_execution": {"available": False, "reason": "finish3l.execute explicitly requires CUDA"},
    }


def backend_pins():
    files = sorted(set(EXECUTION_FILES) | {str(path.relative_to(REPO)) for path in
                    (REPO / "research_hab").rglob("*") if path.suffix in (".py", ".cpp", ".cu", ".h")})
    return {name: common.sha256(REPO / name) for name in files}


def build_manifest(production_path, scientific_path, positions_path):
    production = common.verified_json(production_path, "canonical_manifest_sha256")
    scientific = common.verified_json(scientific_path, "canonical_manifest_sha256")
    if (production["canonical_manifest_sha256"] != common.PRODUCTION_MANIFEST_SHA or
            scientific["canonical_manifest_sha256"] != common.SCIENTIFIC_MANIFEST_SHA or
            common.sha256(positions_path) != common.DEEPT_CACHE_SHA):
        raise RuntimeError("frozen population/artifact identities differ")
    if common.sha256(scientific_path) != production["historical_scientific_manifest"]["file_sha256"]:
        raise RuntimeError("scientific manifest file SHA differs")
    cached = [json.loads(line) for line in positions_path.read_text().splitlines() if line.strip()]
    by_id = {row["property_id"]: row for row in cached}
    if len(cached) != 127 or len(by_id) != 127:
        raise RuntimeError("cached population is not exactly 127")
    original_historical = {row["property_id"]: row for row in scientific["properties"]}
    # The scientific schema encodes benchmark order by array position, whereas
    # the production schema explicitly stores benchmark_ordinal.
    historical = {row["property_id"]: {**row, "benchmark_ordinal": ordinal}
                  for ordinal, row in enumerate(scientific["properties"])}
    if set(historical) != set(by_id) or set(historical) != {row["property_id"] for row in production["properties"]}:
        raise RuntimeError("population IDs differ between frozen sources")
    for row in production["properties"]:
        if by_id[row["property_id"]] != row["cached_DeepT_reference"]:
            raise RuntimeError("historical candidate record differs")
        if any(row[key] != historical[row["property_id"]][key] for key in FIELDS):
            raise RuntimeError("historical property identity differs")
    selected = select(production["properties"])
    source_digest = common.canonical(population_projection(production["properties"]))
    identities = {
        "scientific_manifest": {"canonical_sha256": common.SCIENTIFIC_MANIFEST_SHA,
                                "sha256": common.sha256(scientific_path),
                                "relative_path": "payload/" + str(common.HISTORICAL_REL / scientific_path.name)},
        "production_manifest": {"canonical_sha256": common.PRODUCTION_MANIFEST_SHA,
                                "sha256": common.sha256(production_path),
                                "relative_path": "payload/" + str(common.BASELINE_REL / production_path.name)},
        "DeepT_positions": {"sha256": common.DEEPT_CACHE_SHA,
                            "relative_path": "payload/" + str(common.HISTORICAL_REL / positions_path.name)},
        "model": production["model"], "dataset_sha256": scientific["dataset_sha256"],
        "dataset_member": scientific["dataset_member"], "pinned_DeepT_revision": production["pinned_DeepT_revision"],
        "threat_model": production["threat_model"],
    }
    examples = {row["sentence_ordinal"]: row for row in scientific["examples"]}
    for row in selected:
        example = examples[row["sentence_ordinal"]]
        tokens = example["token_ids"]
        if (len(tokens) != row["sequence_length"] or row["clean_label"] != row["nominal_prediction"] or
                tokens[row["token_position"]] != row["token_id"] or
                any(example[key] != row[key] for key in ("sentence_id", "source_test_line", "raw_sentence_sha256", "clean_label", "nominal_prediction"))):
            raise RuntimeError("authoritative token/source identity differs")
        row.update(token_ids=tokens, token_ids_sha256=common.canonical(tokens),
                   radius_source=RADIUS_SOURCE, source_population_sha256=source_digest,
                   historical_property_row_sha256=common.canonical(original_historical[row["property_id"]]),
                   frozen_artifact_identifiers=identities,
                   sequence_length_stratum=("short_le12" if row["sequence_length"] <= 12 else
                                            "medium_13to20" if row["sequence_length"] <= 20 else "long_gt20"),
                   measurement_inventory=capabilities())
    payload = {
        "schema": SCHEMA, "property_count": 24, "source_property_count": 127,
        "source_population_sha256": source_digest, "selection_algorithm": ALGORITHM,
        "selection_input_whitelist": list(FIELDS) + [RADIUS_SOURCE],
        "selection_order": "low/middle/high historical radius rank tertile, 8 rounds",
        "no_outcome_dependent_selection": True, "membership_replacement_allowed": False,
        "radius_policy": "unchanged historical DeepT candidate, one evaluation, no shrink/search",
        "source_artifacts": identities, "frozen_execution_source_hashes": backend_pins(),
        "measurement_inventory": capabilities(), "properties": selected,
        "preparation_scientific_queries": 0, "preparation_bound_calls": 0,
    }
    payload["manifest_sha256"] = common.canonical(payload)
    return payload


def load_manifest(path):
    manifest = common.verified_json(path, "manifest_sha256")
    if manifest["schema"] != SCHEMA or manifest["property_count"] != 24 or len(manifest["properties"]) != 24:
        raise RuntimeError("preregistered benchmark identity/count differs")
    if len({row["property_id"] for row in manifest["properties"]}) != 24:
        raise RuntimeError("duplicate preregistered ID")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-manifest", required=True, type=Path)
    parser.add_argument("--scientific-manifest", required=True, type=Path)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite frozen membership")
    manifest = build_manifest(args.production_manifest, args.scientific_manifest, args.positions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"], "file_sha256": common.sha256(output),
                      "source_population_sha256": manifest["source_population_sha256"],
                      "property_count": 24, "scientific_queries": 0, "bound_calls": 0,
                      "ids": [row["property_id"] for row in manifest["properties"]]}, indent=2))


if __name__ == "__main__":
    main()
