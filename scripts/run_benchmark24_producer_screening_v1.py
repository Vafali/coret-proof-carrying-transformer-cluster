#!/usr/bin/env python3
"""Two-worker screening ONLY. No checker, radius search or verifier changes.

All scientific execution delegates to the unchanged benchmark24 one runner.
This module itself imports only metadata/stdlib helpers. flock requires a
shared filesystem supporting POSIX locks (both workers run on afrodita).
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import cluster_common as C
import run_transformer_benchmark24_v1 as B


QUEUE_PATH = C.REPO / "frozen/benchmark24/producer_screening_queue_v1.json"
PRIOR_IDS = (
    "deept_table7_stdln3_s001_line1794_tok05",
    "deept_table7_stdln3_s004_line216_tok09",
    "deept_table7_stdln3_s002_line1468_tok06",
)
QUEUE_SCHEMA = "CORET_BENCHMARK24_PRODUCER_SCREENING_QUEUE_V1"
CAMPAIGN_SCHEMA = "CORET_BENCHMARK24_PRODUCER_SCREENING_CAMPAIGN_V1"
SUCCESS_THRESHOLD, NON_SUCCESS_THRESHOLD = 8, 17


def queue_manifest(manifest):
    by_id = {row["property_id"]: row for row in manifest["properties"]}
    if len(by_id) != 24 or not set(PRIOR_IDS) <= set(by_id):
        raise RuntimeError("frozen population/prior IDs differ")
    remaining = sorted((row for row in by_id.values() if row["property_id"] not in PRIOR_IDS),
                       key=lambda row: (row["sequence_length"], row["property_id"]))
    queue = {"schema": QUEUE_SCHEMA, "benchmark_manifest_sha256": manifest["manifest_sha256"],
             "source_population_sha256": manifest["source_population_sha256"],
             "property_count": 24, "remaining_count": 21, "worker_ids": [0, 1],
             "prior_property_ids": list(PRIOR_IDS), "ordering_fields": ["sequence_length", "property_id"],
             "success_threshold": SUCCESS_THRESHOLD, "non_success_threshold": NON_SUCCESS_THRESHOLD,
             "success_predicate": "producer_proof_status == CERTIFIED_MARGIN AND final_sound_lower_margin > 0 AND certificate_integrity_status == PASS",
             "independent_checker_authority": "NONE; positive producer records remain NOT_AVAILABLE/INCONCLUSIVE",
             "properties": [{key: row[key] for key in ("property_id", "sequence_length", "tested_radius", "tested_radius_hex")}
                            for row in remaining]}
    queue["queue_sha256"] = C.canonical(queue)
    return queue


def load_queue(manifest, path=QUEUE_PATH):
    queue = C.verified_json(Path(path), "queue_sha256")
    if queue != queue_manifest(manifest):
        raise RuntimeError("queue identity/order/gate differs from frozen metadata")
    return queue


def _payload(value):
    return {key: item for key, item in value.items() if key != "record_sha256"}


def _atomic(path, value, *, replace=False):
    """fsync + atomic publication; immutable records use no-clobber link."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = {**value, "record_sha256": C.canonical(value)}
    data = (json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".screening-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise RuntimeError(f"immutable record differs: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sealed


def _copy_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if C.sha256(source) != C.sha256(destination):
            raise RuntimeError("prior immutable copy differs")
    else:
        fd, temporary = tempfile.mkstemp(prefix=".prior-copy-", dir=destination.parent)
        try:
            with source.open("rb") as incoming, os.fdopen(fd, "wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush(); os.fsync(outgoing.fileno())
            if C.sha256(source) != C.sha256(Path(temporary)):
                raise RuntimeError("prior source changed during import")
            os.link(temporary, destination)
        finally:
            os.unlink(temporary)
    if C.sha256(source) != C.sha256(destination):
        raise RuntimeError("prior immutable copy verification failed")


@contextmanager
def _lock(path, *, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as error:
            raise RuntimeError("worker already running; no oversubscription") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _safe_root(root):
    root = Path(root).resolve()
    if root == Path('/') or root == Path.home() or root == C.REPO or root.is_relative_to(C.REPO / "frozen"):
        raise RuntimeError("campaign root must be an isolated result directory")
    return root


def success(record):
    margin = record.get("final_sound_lower_margin")
    return (record.get("producer_proof_status") == "CERTIFIED_MARGIN" and B._finite(margin)
            and margin > 0 and record.get("certificate_integrity_status") == "PASS")


def gate(records):
    if len(records) > 24 or len({r["property_id"] for r in records}) != len(records):
        raise RuntimeError("duplicate/oversized screening population")
    successes = sum(success(row) for row in records)
    non_successes = len(records) - successes
    decision = ("CHECKER_BUILD_JUSTIFIED" if successes >= SUCCESS_THRESHOLD else
                "CHECKER_BUILD_NOT_JUSTIFIED" if non_successes >= NON_SUCCESS_THRESHOLD else "CONTINUE")
    if len(records) == 24 and decision == "CONTINUE":
        raise RuntimeError("gate arithmetic invariant failed")
    return {"decision": decision, "successes": successes, "non_successes": non_successes,
            "evaluated": len(records), "not_evaluated": 24 - len(records),
            "success_threshold": SUCCESS_THRESHOLD, "non_success_threshold": NON_SUCCESS_THRESHOLD,
            "producer_status_histogram": dict(Counter(r.get("producer_proof_status") for r in records)),
            "independent_certifications": 0}


def _artifact_details(manifest, normalized, directory):
    """Hash/integrity checking only: never deserialize tensors or claim soundness."""
    fields = {name: None for name in ("certificate_path", "certificate_sha256", "certificate_report_path",
                                    "certificate_report_sha256", "producer_result_path", "producer_result_sha256")}
    raw_path = directory / "result.json"
    raw = C.verified_json(raw_path) if raw_path.is_file() else None
    if raw is not None:
        row = next(r for r in manifest["properties"] if r["property_id"] == normalized["property_id"])
        if (raw.get("schema") != "CORET_SOUND_FP64_3L_PROPERTY_RESULT_V1" or
                raw.get("property_id") != normalized["property_id"] or
                raw.get("historical_candidate_radius_hex") != normalized["tested_radius_hex"] or
                raw.get("candidate_source") != normalized["radius_source"] or
                raw.get("binary_search_performed") is not False or
                raw.get("clean_label") != row["clean_label"]):
            raise RuntimeError("producer artifact property/radius identity differs")
        fields.update(producer_result_path=str(raw_path.resolve()), producer_result_sha256=C.sha256(raw_path))
    if normalized.get("certificate_integrity_status") == "PASS":
        if raw is None or raw.get("terminal_status") != "COMPLETE":
            raise RuntimeError("PASS requires authenticated completed producer artifacts")
        for filename, path_key, hash_key in (("certificate.pt", "certificate_path", "certificate_sha256"),
                ("certificate_report.json", "certificate_report_path", "certificate_report_sha256")):
            path = directory / filename
            if not path.is_file() or C.sha256(path) != raw.get(hash_key):
                raise RuntimeError(f"certificate/report hash mismatch: {filename}")
            fields[path_key], fields[hash_key] = str(path.resolve()), raw[hash_key]
        report = json.loads((directory / "certificate_report.json").read_text())
        row = next(r for r in manifest["properties"] if r["property_id"] == normalized["property_id"])
        if (raw.get("final_sound_lower_margin") != normalized.get("final_sound_lower_margin") or
                raw.get("certified_at_historical_radius") is not (normalized["final_sound_lower_margin"] > 0) or
                raw.get("scientific_evaluation_complete") is not True or
                raw.get("generic_fallback_count") != 0 or
                report.get("output_artifact_schema") != "CORET_SOUND_FP64_3L_FINAL_V1" or
                report.get("output_artifact_sha256") != fields["certificate_sha256"] or
                report.get("final_sound_margin") != normalized.get("final_sound_lower_margin") or
                report.get("fixture_rho_hex") != row["tested_radius_hex"] or
                report.get("fixture_token_ids") != row["token_ids"]):
            raise RuntimeError("certificate report/source identity differs")
    return fields


def _augment(manifest, source_path, producer_dir, *, prior=False, worker_id=None):
    original = C.verified_json(source_path)
    B.validate_result(manifest, original)
    if original["final_proof_status"] == "CERTIFIED":
        raise RuntimeError("screening cannot promote independently unchecked results")
    if original.get("producer_proof_status") == "CERTIFIED_MARGIN" and (
            original.get("independently_checked_certificate_status") != "NOT_AVAILABLE" or not success(original)):
        raise RuntimeError("positive screening result/checker status differs")
    row = next(r for r in manifest["properties"] if r["property_id"] == original["property_id"])
    result = {**_payload(original), "sequence_length": row["sequence_length"],
              "screening_prior_result": prior, "screening_worker_id": worker_id,
              "normalized_source_path": str(source_path.resolve()), "normalized_source_sha256": C.sha256(source_path),
              **_artifact_details(manifest, original, producer_dir)}
    for name in ("failure_stage", "failure_reason", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes",
                 "peak_cpu_rss_bytes", "final_generator_count", "numerical_widening", "max_numerical_native_ratio",
                 "reduction_count", "reduction_inflation_max"):
        result.setdefault(name, None)
    return result


def _validate_screening(manifest, record):
    B.validate_result(manifest, record)
    row = next(r for r in manifest["properties"] if r["property_id"] == record["property_id"])
    if record.get("sequence_length") != row["sequence_length"] or record["final_proof_status"] == "CERTIFIED":
        raise RuntimeError("screening identity/certification differs")
    if success(record) and record.get("independently_checked_certificate_status") != "NOT_AVAILABLE":
        raise RuntimeError("screening independent-checker promotion forbidden")
    source = Path(record["normalized_source_path"])
    if not source.is_file() or C.sha256(source) != record["normalized_source_sha256"]:
        raise RuntimeError("immutable normalized source missing/corrupt")
    original = C.verified_json(source)
    expected = _augment(manifest, source,
                        Path(record["producer_result_path"]).parent if record.get("producer_result_path") else source.parent,
                        prior=record["screening_prior_result"], worker_id=record["screening_worker_id"])
    if _payload(record) != expected:
        raise RuntimeError("screening completion differs from source")


def _campaign(root, queue):
    record = C.verified_json(root / "campaign.json")
    if (record.get("schema") != CAMPAIGN_SCHEMA or record.get("queue_sha256") != queue["queue_sha256"] or
            set(record.get("prior_source_sha256", {})) != set(PRIOR_IDS) or record.get("worker_ids") != [0, 1]):
        raise RuntimeError("campaign/queue identity differs")
    return record


def _claim_record(path, queue):
    record = C.verified_json(path)
    if (record.get("queue_sha256") != queue["queue_sha256"] or record.get("worker_id") not in (0, 1) or
            path.stem != record.get("property_id") or record["property_id"] not in {r["property_id"] for r in queue["properties"]}):
        raise RuntimeError("claim identity differs")
    return record


def _records(root, manifest):
    records = [C.verified_json(path) for path in sorted((root / "results").glob("*.json"))]
    for record in records:
        _validate_screening(manifest, record)
    return records


def _publish(root, manifest, queue, property_id, worker_id):
    path = root / "results" / f"{property_id}.json"
    if path.exists():
        record = C.verified_json(path)
        _validate_screening(manifest, record)
        return record
    worker = root / "workers" / f"worker_{worker_id}"
    value = _augment(manifest, worker / "records" / f"{property_id}.json",
                     worker / "producer/properties" / property_id, worker_id=worker_id)
    return _atomic(path, value)


def _status_locked(root, manifest, queue):
    _campaign(root, queue)
    # Recover already atomic completions even if their owning worker died just
    # before publishing global status. Never rerun them or omit gate evidence.
    for path in sorted((root / "claims").glob("*.json")):
        claim = _claim_record(path, queue)
        normalized = root / "workers" / f"worker_{claim['worker_id']}" / "records" / f"{claim['property_id']}.json"
        if normalized.exists():
            _publish(root, manifest, queue, claim["property_id"], claim["worker_id"])
    records = _records(root, manifest)
    if not set(PRIOR_IDS) <= {row["property_id"] for row in records}:
        raise RuntimeError("immutable prior completion missing; never reset baseline counts")
    state = {"schema": "CORET_BENCHMARK24_SCREENING_STATUS_V1", "queue_sha256": queue["queue_sha256"], **gate(records)}
    if state["decision"] != "CONTINUE":
        _decision = root / "terminal_decision.json"
        if not _decision.exists():
            _atomic(_decision, state)
        elif C.verified_json(_decision)["decision"] != state["decision"]:
            raise RuntimeError("terminal decision changed")
    _atomic(root / "status.json", state, replace=True)
    return state


def prepare(manifest, queue, root, prior_roots):
    root = _safe_root(root)
    sources = {}
    for property_id in PRIOR_IDS:
        candidates = {Path(base).resolve() / "records" / f"{property_id}.json" for base in prior_roots}
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) != 1:
            raise RuntimeError(f"require exactly one immutable prior source for {property_id}; found {len(candidates)}")
        source = candidates[0]
        if root.is_relative_to(source.parent.parent) or source.is_relative_to(root):
            raise RuntimeError("new campaign root must not overlap prior results")
        producer = source.parent.parent / "producer/properties" / property_id
        value = _augment(manifest, source, producer, prior=True)
        sources[property_id] = (source, producer, value)
    if (sum(success(value) for _, _, value in sources.values()) != 1 or
            sources[PRIOR_IDS[2]][2]["final_sound_lower_margin"] != 5.292863350147482 or
            any(sources[pid][2].get("producer_proof_status") != "UNCERTIFIED" or
                sources[pid][2].get("inconclusive_reason") != "SOUND_DOMAIN_FAILURE" for pid in PRIOR_IDS[:2])):
        raise RuntimeError("three prior outcomes do not match the accepted 1-success/2-domain-failure baseline")
    with _lock(root / ".campaign.lock"):
        if (root / "campaign.json").exists():
            previous = _campaign(root, queue)
            if previous["prior_source_sha256"] != {pid: C.sha256(item[0]) for pid, item in sources.items()}:
                raise RuntimeError("prior source identity changed on prepare/resume")
            return _status_locked(root, manifest, queue)
        for property_id, (source, producer, _) in sources.items():
            copied = root / "priors/records" / source.name
            _copy_once(source, copied)
            destination = root / "priors/producer" / property_id
            for filename in ("result.json", "certificate.pt", "certificate_report.json"):
                path = producer / filename
                if path.is_file():
                    _copy_once(path, destination / filename)
            _atomic(root / "results" / source.name, _augment(manifest, copied, destination, prior=True))
        _atomic(root / "campaign.json", {"schema": CAMPAIGN_SCHEMA,
                "queue_sha256": queue["queue_sha256"], "benchmark_manifest_sha256": manifest["manifest_sha256"],
                "prior_source_sha256": {pid: C.sha256(item[0]) for pid, item in sources.items()},
                "worker_ids": [0, 1], "no_radius_search": True, "no_independent_checker": True})
        return _status_locked(root, manifest, queue)


def claim_next(root, manifest, queue, worker_id):
    if worker_id not in (0, 1):
        raise RuntimeError("exactly worker IDs 0/1 are permitted")
    with _lock(root / ".campaign.lock"):
        state = _status_locked(root, manifest, queue)
        if state["decision"] != "CONTINUE":
            return None, state
        # Only the owning worker resumes an interrupted claim. Its lifetime
        # lease ensures the former owner cannot still be running concurrently.
        for row in queue["properties"]:
            path = root / "claims" / f"{row['property_id']}.json"
            if path.exists() and not (root / "results" / path.name).exists():
                old = _claim_record(path, queue)
                if old["worker_id"] == worker_id:
                    return old, state
        for row in queue["properties"]:
            path = root / "claims" / f"{row['property_id']}.json"
            if not path.exists() and not (root / "results" / path.name).exists():
                return _atomic(path, {"property_id": row["property_id"], "worker_id": worker_id,
                                     "queue_sha256": queue["queue_sha256"]}), state
        return None, state


def complete(root, manifest, queue, claim):
    with _lock(root / ".campaign.lock"):
        stored = _claim_record(root / "claims" / f"{claim['property_id']}.json", queue)
        if stored != claim:
            raise RuntimeError("completion claim differs")
        _publish(root, manifest, queue, claim["property_id"], claim["worker_id"])
        return _status_locked(root, manifest, queue)


def worker(root, manifest, queue, artifact_root, worker_id, *, runner=None, preflight=True):
    root = _safe_root(root)
    if worker_id not in (0, 1):
        raise RuntimeError("worker ID must be 0 or 1")
    with _lock(root / "workers" / f"worker_{worker_id}" / ".worker.lock", nonblocking=True):
        with _lock(root / ".campaign.lock"):
            existing = _status_locked(root, manifest, queue)
        if existing["decision"] != "CONTINUE":
            return existing
        if preflight:
            errors = B.execution_source_errors(manifest) + B.artifact_errors(manifest, artifact_root)
            if errors:
                raise RuntimeError(f"preflight failed BEFORE any claim: {errors}")
        while True:
            claim, state = claim_next(root, manifest, queue, worker_id)
            print(json.dumps({"stage": "screening_gate", "worker_id": worker_id, **state}), flush=True)
            if claim is None:
                return state
            print(json.dumps({"stage": "screening_claim", **claim}), flush=True)
            (runner or B.run_one)(manifest, claim["property_id"], artifact_root,
                                 root / "workers" / f"worker_{worker_id}", "cuda:0")
            state = complete(root, manifest, queue, claim)
            print(json.dumps({"stage": "screening_completion", "property_id": claim["property_id"], **state}), flush=True)
            if state["decision"] != "CONTINUE":
                return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write-queue", "prepare", "status", "worker"))
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--prior-root", type=Path, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path, default=C.REPO / "runtime_inputs")
    parser.add_argument("--worker-id", type=int, choices=(0, 1))
    args = parser.parse_args()
    manifest = B.read_protocol()
    if args.mode == "write-queue":
        queue = queue_manifest(manifest)
        if args.queue.exists():
            raise RuntimeError("refusing to overwrite deterministic queue")
        args.queue.parent.mkdir(parents=True, exist_ok=True)
        args.queue.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"queue_sha256": queue["queue_sha256"], "remaining_count": 21}), flush=True)
        return
    if args.campaign_root is None:
        parser.error("--campaign-root required")
    root = _safe_root(args.campaign_root)
    if root.is_relative_to(args.artifact_root.resolve()):
        parser.error("campaign root cannot overlap frozen artifact inputs")
    queue = load_queue(manifest, args.queue)
    if args.mode == "prepare":
        if not args.prior_root:
            parser.error("prepare requires --prior-root (repeat for separate prior roots)")
        report = prepare(manifest, queue, root, args.prior_root)
    elif args.mode == "status":
        with _lock(root / ".campaign.lock"):
            report = _status_locked(root, manifest, queue)
    else:
        if args.worker_id is None:
            parser.error("worker requires --worker-id")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.worker_id):
            parser.error("launcher must explicitly bind physical gpu0/gpu1, one visible GPU per process")
        report = worker(root, manifest, queue, args.artifact_root, args.worker_id)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
