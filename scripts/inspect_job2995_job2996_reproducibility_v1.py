#!/usr/bin/env python3
"""Read-only job-2995/job-2996 semantic reproducibility audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import inspect_block2_layernorm_capture_linkage_v1 as linkage


SCHEMA = "CORET_JOB2995_JOB2996_REPRODUCIBILITY_V1"
JOB2995_FEATURE_COMMIT = "8409f2c451a0b60aa2cdea196771c9adb92e116d"
JOB2996_FEATURE_COMMIT = "f1374ed5264b26dd7ad496bf34f8ea4d4a22c024"
TARGET_STATE = "post_attention_ln_pre_reduction"


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["record_sha256"] = _canonical(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _state_record(name: str, state: dict) -> dict:
    record = linkage._legacy_state_record(state)
    record["state_key"] = name
    return record


def _ordered_frontier(path: Path, label: str) -> dict:
    loaded = linkage._load_frontier(path, label)
    states = loaded["state"]  # Force authentication of the target state.
    del states
    manifest = loaded["manifest"]
    try:
        payload = torch.load(
            loaded["artifact_path"], map_location="cpu", weights_only=False,
            mmap=True)
    except TypeError:
        payload = torch.load(
            loaded["artifact_path"], map_location="cpu", weights_only=False)
    states = payload["states"]
    rows = manifest["variants"]
    if [row["state_key"] for row in rows] != list(states):
        raise RuntimeError(f"{label}: state ordering differs")
    for row in rows:
        name = row["state_key"]
        linkage._validate_state(states[name], f"{label}/{name}")
        actual = _state_record(name, states[name])
        if actual != row:
            fields = sorted(set(actual) | set(row))
            first = next(key for key in fields
                         if actual.get(key) != row.get(key))
            raise RuntimeError(
                f"{label}: state manifest differs at {name}/{first}")
    return {**loaded, "states": states,
            "state_names": [row["state_key"] for row in rows]}


def _metadata_comparison(left: dict, right: dict) -> dict:
    lp, rp = left["proof"], right["proof"]
    fields = {
        "source_ids": (lp["ids"], rp["ids"]),
        "masks": (lp["masks"], rp["masks"]),
        "provenance": (lp["reasons"], rp["reasons"]),
        "ranges_low": (left["range_low"], right["range_low"]),
        "ranges_high": (left["range_high"], right["range_high"]),
    }
    result = {}
    for name, (a, b) in fields.items():
        if isinstance(a, torch.Tensor):
            row = linkage._tensor_comparison(a, b)
            result[name] = {
                "exact_equal": row["exact_equal"],
                "maximum_absolute_difference": row[
                    "maximum_absolute_difference"],
            }
        else:
            result[name] = {
                "exact_equal": a == b,
                "expected_sha256": _canonical(a),
                "reproduced_sha256": _canonical(b),
                "first_difference": linkage._first_difference(a, b),
            }
    result["generator_counts_equal"] = (
        len(lp["ids"]) == len(rp["ids"]))
    return result


def compare_all_states(expected_path: Path, reproduced_path: Path) -> dict:
    expected = _ordered_frontier(expected_path, "job2995")
    reproduced = _ordered_frontier(reproduced_path, "job2996")
    if expected["state_names"] != reproduced["state_names"]:
        raise RuntimeError("frontier common state inventory/order differs")
    rows = []
    first = None
    for name in expected["state_names"]:
        left, right = expected["states"][name], reproduced["states"][name]
        center = linkage._tensor_comparison(
            left["weights"][0], right["weights"][0])
        generators = linkage._tensor_comparison(
            left["weights"][1:], right["weights"][1:])
        metadata = _metadata_comparison(left, right)
        equal = (center["exact_equal"] and generators["exact_equal"]
                 and all(row.get("exact_equal", row) is True
                         for key, row in metadata.items()
                         if key != "generator_counts_equal")
                 and metadata["generator_counts_equal"] is True)
        if not equal and first is None:
            first = name
        rows.append({
            "state_name": name,
            "semantic_equal": equal,
            "center": center,
            "generators": generators,
            "metadata": metadata,
            "expected_generator_count": len(left["proof"]["ids"]),
            "reproduced_generator_count": len(right["proof"]["ids"]),
        })
    return {
        "state_count": len(rows),
        "state_order": expected["state_names"],
        "first_common_state_that_differs": first,
        "states": rows,
    }


def _verified_trace(path: Path, label: str) -> dict:
    value = linkage._verified_json(path.expanduser().resolve())
    if (value.get("property_id") != linkage.PROPERTY_ID
            or value.get("tested_radius") != linkage.TESTED_RADIUS
            or value.get("tested_radius_hex") != linkage.TESTED_RADIUS_HEX
            or not isinstance(value.get("invocations"), list)):
        raise RuntimeError(f"{label}: invocation trace identity differs")
    return value


def _flatten(value, prefix="") -> dict:
    result = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            result.update(_flatten(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        result[prefix] = value
    return result


def _compact_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"canonical_sha256": _canonical(value), "type": type(value).__name__}


def compare_traces(expected_path: Path, reproduced_path: Path) -> dict:
    expected = _verified_trace(expected_path, "job2995")
    reproduced = _verified_trace(reproduced_path, "job2996")
    left = [row for row in expected["invocations"]
            if row.get("stage") == linkage.TARGET_STAGE]
    right = [row for row in reproduced["invocations"]
             if row.get("stage") == linkage.TARGET_STAGE]
    if len(left) != 1 or len(right) != 1:
        raise RuntimeError("target LayerNorm trace inventory differs")
    left, right = left[0], right[0]
    lf, rf = _flatten(left), _flatten(right)
    paths = sorted(set(lf) | set(rf))
    differences = [{
        "field": path,
        "expected": _compact_value(lf.get(path)),
        "reproduced": _compact_value(rf.get(path)),
    } for path in paths if lf.get(path) != rf.get(path)]
    input_left = left.get("state_identity")
    input_right = right.get("state_identity")
    interesting_words = (
        "state_identity", "variance", "second_moment", "centered",
        "generator", "range", "reserve", "widen", "lower", "upper",
        "reduction", "source_count",
    )
    available = sorted(path for path in paths
                       if any(word in path for word in interesting_words))
    return {
        "trace_schema_expected": expected.get("schema"),
        "trace_schema_reproduced": reproduced.get("schema"),
        "invocation_count_expected": len(expected["invocations"]),
        "invocation_count_reproduced": len(reproduced["invocations"]),
        "target_ordinal_expected": left.get("ordinal"),
        "target_ordinal_reproduced": right.get("ordinal"),
        "target_layernorm_index_expected": left.get("layernorm_index"),
        "target_layernorm_index_reproduced": right.get("layernorm_index"),
        "target_input_state_identity_available_in_both": (
            isinstance(input_left, dict) and isinstance(input_right, dict)),
        "target_input_state_identity_equal": input_left == input_right,
        "target_input_state_identity_expected": input_left,
        "target_input_state_identity_reproduced": input_right,
        "available_input_or_numerical_fields": available,
        "field_difference_count": len(differences),
        "field_differences": differences,
    }


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
        timeout=10)
    return completed.stdout.strip()


def code_provenance(repo: Path) -> dict:
    commits_present = {}
    for name, commit in (("job2995_feature_commit", JOB2995_FEATURE_COMMIT),
                         ("job2996_feature_commit", JOB2996_FEATURE_COMMIT)):
        try:
            _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
            commits_present[name] = True
        except (subprocess.SubprocessError, OSError):
            commits_present[name] = False
    changed = _git(
        repo, "diff", "--name-only",
        f"{JOB2995_FEATURE_COMMIT}..{JOB2996_FEATURE_COMMIT}").splitlines()
    classifications = []
    executed_runner = "scripts/run_sound_fp64_3l_psd_layernorm_experiment_v1.py"
    for path in changed:
        if path == executed_runner:
            category = "SEMANTICS_AFFECTING"
            reason = (
                "adds an executed pre-LayerNorm wrapper with two state hashes "
                "and a synchronous GPU-to-CPU snapshot before the original "
                "variance routine")
        elif path.startswith("scripts/analyze_") or path.startswith("tests/"):
            category = "DIAGNOSTIC_ONLY"
            reason = "not imported by the scientific execution path"
        elif path == "scripts/decide_block2_output_zero_variance_v1.py":
            category = "DIAGNOSTIC_ONLY"
            reason = "standalone CPU decision utility"
        else:
            category = "UNKNOWN"
            reason = "not statically classified"
        classifications.append({"path": path, "classification": category,
                                "reason": reason})
    protected_paths = (
        "scripts/run_sound_fp64_finish_3l_v1.py",
        "scripts/run_sound_fp64_3l_campaign.py",
        "scripts/run_sound_fp64_3l_psd_state_capture_v1.py",
        "research_hab/coret_psd_layernorm_experiment_v1.py",
        "research_hab/coret_sound_fp64_block0_feasibility_v1.py",
        "research_hab/coret_structural_support_precise_dot_v1.py",
    )
    unchanged = [path for path in protected_paths if path not in changed]
    return {
        "feature_introduction_commits_not_job_provenance": {
            "job2995_frontier_capture_introduced": JOB2995_FEATURE_COMMIT,
            "job2996_input_capture_introduced": JOB2996_FEATURE_COMMIT,
        },
        "actual_job_commit_identity": "UNKNOWN_UNLESS_RECORDED_IN_LOG",
        "commit_objects_present": commits_present,
        "changed_files": classifications,
        "relevant_unchanged_files": unchanged,
        "only_changed_executed_scientific_runner": (
            [row["path"] for row in classifications
             if row["classification"] == "SEMANTICS_AFFECTING"]
            == [executed_runner]),
    }


RUNTIME_PATTERNS = {
    "git_commit": re.compile(r"(?:git_commit|commit|HEAD)[=: ]+([0-9a-f]{40})",
                             re.IGNORECASE),
    "CUDA_VISIBLE_DEVICES": re.compile(r"CUDA_VISIBLE_DEVICES[=: ]+([^\s]+)"),
    "CUBLAS_WORKSPACE_CONFIG": re.compile(r"CUBLAS_WORKSPACE_CONFIG[=: ]+([^\s]+)"),
    "torch_version": re.compile(r"(?:torch|pytorch)(?:_version)?[=: ]+([^\s]+)",
                                re.IGNORECASE),
    "cuda_runtime_version": re.compile(r"CUDA(?: runtime)?(?:_version)?[=: ]+([^\s]+)",
                                       re.IGNORECASE),
    "cudnn_version": re.compile(r"cuDNN(?:_version)?[=: ]+([^\s]+)",
                                 re.IGNORECASE),
    "gpu_model": re.compile(r"GPU(?: model)?[=: ]+(.+)$", re.IGNORECASE | re.MULTILINE),
}


def _extract_logs(paths: list[Path]) -> dict:
    values = {key: set() for key in RUNTIME_PATTERNS}
    authenticated = []
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        authenticated.append({"path": str(path),
                              "sha256": linkage._sha256(path)})
        for key, pattern in RUNTIME_PATTERNS.items():
            values[key].update(match.strip() for match in pattern.findall(text))
    return {"files": authenticated,
            "values": {key: sorted(items) for key, items in values.items()}}


def runtime_configuration(job2995_logs: list[Path],
                          job2996_logs: list[Path]) -> dict:
    left, right = _extract_logs(job2995_logs), _extract_logs(job2996_logs)
    rows = {}
    for key in RUNTIME_PATTERNS:
        a, b = left["values"][key], right["values"][key]
        status = "UNKNOWN" if not a or not b else (
            "SAME" if a == b else "DIFFERENT")
        rows[key] = {"status": status, "job2995": a, "job2996": b}
    static = {
        "property_radius": "SAME",
        "semantic_manifest_identity": "SAME",
        "dtype": "SAME",
        "frontier_stage": "SAME",
        "source_set_model": "SAME",
        "token_ids": "UNKNOWN",
        "campaign_root": "UNKNOWN",
        "artifact_root": "UNKNOWN",
        "prior_experiment_report": "UNKNOWN",
        "checkpoint": "UNKNOWN",
        "generator_cap": "UNKNOWN",
        "reduction_policy": "UNKNOWN",
        "numerical_reserve_policy": "UNKNOWN",
        "layernorm_epsilon": "UNKNOWN",
        "deterministic_algorithms": "UNKNOWN",
        "tf32_settings": "UNKNOWN",
        "cublas_version": "UNKNOWN",
    }
    return {"log_evidence": {"job2995": left, "job2996": right},
            "extracted": rows, "other_fields": static}


def capture_hook_audit(repo: Path) -> dict:
    runner = repo / "scripts/run_sound_fp64_3l_psd_layernorm_experiment_v1.py"
    return {
        "file": str(runner),
        "pre_original_operations": [
            "experiment.state_identity(state, proof)",
            "_frontier_snapshot(state, proof)",
        ],
        "in_place_tensor_operations": False,
        "argument_substitution": False,
        "rng_use": False,
        "dtype_or_device_mutation": False,
        "global_state_change": (
            "temporary monkey-patch of finish3l._layernorm_variance_state"),
        "synchronization_or_order_change": True,
        "synchronization_evidence": (
            "state_identity and _frontier_snapshot both call detach().cpu() "
            "before invoking the original routine; CUDA-to-CPU copies wait "
            "for prior stream work"),
        "gpu_allocator_change_before_original": False,
        "cpu_allocations_before_original": True,
        "calls_original_once_with_same_objects": True,
        "post_original_operation": (
            "a second state_identity hash in finally verifies no mutation"),
        "scientific_conclusion": (
            "The hook is value-passive but not schedule-passive. This is a "
            "possible perturbation mechanism, not proof that it caused the "
            "different LayerNorm output."),
    }


def classify(*, traces: dict, runtime: dict, provenance: dict) -> dict:
    runtime_differences = [key for key, row in runtime["extracted"].items()
                           if row["status"] == "DIFFERENT"]
    commit_rows = runtime["extracted"]["git_commit"]
    if commit_rows["status"] == "DIFFERENT":
        return {"root_cause": "ROOT_CAUSE_CODE_VERSION_DRIFT",
                "evidence": "recorded job commits differ",
                "differing_fields": runtime_differences}
    if runtime_differences:
        return {"root_cause": "ROOT_CAUSE_RUNTIME_CONFIGURATION_DRIFT",
                "evidence": "recorded runtime configuration differs",
                "differing_fields": runtime_differences}
    unknown_runtime = [key for key, row in runtime["extracted"].items()
                       if row["status"] == "UNKNOWN"]
    unknown_runtime += [key for key, value in runtime["other_fields"].items()
                        if value == "UNKNOWN"]
    if (not unknown_runtime
            and commit_rows["status"] == "SAME"
            and traces["target_input_state_identity_equal"] is True):
        return {"root_cause": "ROOT_CAUSE_NUMERICAL_NONDETERMINISM",
                "evidence": (
                    "identical recorded code/runtime and identical target "
                    "input precede different output coefficients"),
                "differing_fields": []}
    return {
        "root_cause": "ROOT_CAUSE_UNRESOLVED",
        "evidence": (
            "semantic output mismatch is established, but actual job commit "
            "and/or numerically relevant runtime provenance is absent; the "
            "capture hook adds synchronization but no causal mechanism is "
            "proved"),
        "unknown_fields": sorted(set(unknown_runtime)),
        "feature_commit_diff_is_not_job_identity": True,
        "only_candidate_executed_code_delta": (
            provenance["only_changed_executed_scientific_runner"]),
    }


def build_report(args) -> dict:
    all_states = compare_all_states(
        args.job2995_frontier_manifest, args.job2996_frontier_manifest)
    traces = compare_traces(args.job2995_trace, args.job2996_trace)
    provenance = code_provenance(args.repo.resolve())
    runtime = runtime_configuration(args.job2995_log, args.job2996_log)
    hook = capture_hook_audit(args.repo.resolve())
    classification = classify(
        traces=traces, runtime=runtime, provenance=provenance)
    return {
        "schema": SCHEMA,
        "property_id": linkage.PROPERTY_ID,
        "tested_radius": linkage.TESTED_RADIUS,
        "tested_radius_hex": linkage.TESTED_RADIUS_HEX,
        "frontier_state_comparison": all_states,
        "psd_invocation_trace_comparison": traces,
        "code_provenance": provenance,
        "runtime_configuration": runtime,
        "capture_hook_passivity_audit": hook,
        "classification": classification,
        "gpu_rerun_recommendation": {
            "necessary_now": False,
            "reason": (
                "No rerun isolates a single cause until missing commit/runtime "
                "provenance is recovered or a schedule-passive capture is "
                "preregistered as a specific hook-perturbation test."),
        },
        "scientific_queries": 0,
        "cuda_executions": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job2995-frontier-manifest", required=True, type=Path)
    parser.add_argument("--job2996-frontier-manifest", required=True, type=Path)
    parser.add_argument("--job2995-trace", required=True, type=Path)
    parser.add_argument("--job2996-trace", required=True, type=Path)
    parser.add_argument("--job2995-log", action="append", default=[], type=Path)
    parser.add_argument("--job2996-log", action="append", default=[], type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args)
    _write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
