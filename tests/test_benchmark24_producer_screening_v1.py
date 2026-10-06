"""Synthetic filesystem/orchestration tests; never import a scientific backend."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cluster_common as C
import run_transformer_benchmark24_v1 as B
import run_benchmark24_producer_screening_v1 as S


@pytest.fixture
def manifest():
    return B.read_protocol()


def fake_property(manifest, property_id, workspace, *, positive=True, prior=False):
    row = next(r for r in manifest["properties"] if r["property_id"] == property_id)
    margin = 5.292863350147482 if prior and positive else 2.0 if positive else None
    raw = {"schema": "CORET_SOUND_FP64_3L_PROPERTY_RESULT_V1", "property_id": property_id,
           "historical_candidate_radius_hex": row["tested_radius_hex"], "candidate_source": row["radius_source"],
           "clean_label": row["clean_label"], "binary_search_performed": False,
           "terminal_status": "COMPLETE" if positive else "UNCERTIFIED_DOMAIN_FAILURE",
           "scientific_evaluation_complete": True, "certified_at_historical_radius": positive,
           "final_sound_lower_margin": margin, "runtime_seconds": 936.5 if positive else 318.7,
           "final_generator_count": 14000, "generic_fallback_count": 0,
           "failure_stage": None if positive else "block2_to_margin",
           "failure_reason": None if positive else "SOUND_FP64_LAYERNORM_VARIANCE_DOMAIN_FAILURE",
           "peak_gpu_allocated_bytes": 100, "peak_gpu_reserved_bytes": 200, "peak_cpu_rss_bytes": 300,
           "numerical_widening": .001 if positive else None, "max_numerical_native_ratio": .01 if positive else None}
    directory = workspace / "producer/properties" / property_id
    directory.mkdir(parents=True, exist_ok=True)
    report = None
    if positive:
        certificate = directory / "certificate.pt"
        # Deliberately not a scientific tensor artifact: only hash plumbing is
        # exercised. This test never claims independent numerical acceptance.
        certificate.write_bytes(b"synthetic certificate bits: " + property_id.encode())
        report = {"output_artifact_schema": "CORET_SOUND_FP64_3L_FINAL_V1",
                  "output_artifact_sha256": C.sha256(certificate), "final_sound_margin": margin,
                  "fixture_rho_hex": row["tested_radius_hex"], "fixture_token_ids": row["token_ids"],
                  "reductions": [{"support_inflation": 1e-13}], "representative_mpfr_checks_performed": False}
        report_path = directory / "certificate_report.json"
        report_path.write_text(json.dumps(report))
        raw.update(certificate_sha256=C.sha256(certificate), certificate_report_sha256=C.sha256(report_path))
    B.atomic_record(directory / "result.json", raw)
    normalized = B.normalize(manifest, row, raw, report)
    normalized["device_requested"] = "cuda:0"  # String only; no GPU access.
    return B.atomic_record(workspace / "records" / f"{property_id}.json", normalized)


@pytest.fixture
def prepared(tmp_path, manifest):
    roots = [tmp_path / name for name in ("smoke2", "mid-pilot", "upper-pilot")]
    originals = {}
    for index, (pid, root) in enumerate(zip(S.PRIOR_IDS, roots)):
        fake_property(manifest, pid, root, positive=index == 2, prior=True)
        originals.update({path: path.read_bytes() for path in root.rglob('*') if path.is_file()})
    campaign = tmp_path / "new-screening"
    queue = S.queue_manifest(manifest)
    initial = S.prepare(manifest, queue, campaign, roots)
    assert (initial['successes'], initial['non_successes'], initial['evaluated']) == (1, 2, 3)
    assert all(path.read_bytes() == data for path, data in originals.items())
    return campaign, queue, roots


def status(root, manifest, queue):
    with S._lock(root / '.campaign.lock'):
        return S._status_locked(root, manifest, queue)


def test_queue_is_21_metadata_ordered_with_exact_exclusions(manifest):
    queue = S.queue_manifest(manifest)
    assert len(queue['properties']) == 21
    assert not set(S.PRIOR_IDS) & {r['property_id'] for r in queue['properties']}
    assert queue['properties'] == sorted(queue['properties'], key=lambda r: (r['sequence_length'], r['property_id']))
    mutated = deepcopy(manifest)
    mutated['properties'].reverse()
    for row in mutated['properties']:
        row.update(producer_proof_status='CERTIFIED_MARGIN', runtime_seconds=-123, final_sound_lower_margin=123.)
    assert S.queue_manifest(mutated) == queue
    assert S.load_queue(manifest) == queue


def test_existing_three_counted_copied_unchanged_and_prepare_idempotent(prepared, manifest):
    root, queue, priors = prepared
    before = {p: p.read_bytes() for p in (root / 'results').glob('*.json')}
    result = S.prepare(manifest, queue, root, priors)
    assert result['successes'] == 1 and result['non_successes'] == 2
    assert all(p.read_bytes() == data for p, data in before.items())
    records = S._records(root, manifest)
    positive = next(r for r in records if S.success(r))
    assert positive['independently_checked_certificate_status'] == 'NOT_AVAILABLE'
    assert positive['final_proof_status'] == 'INCONCLUSIVE'
    assert positive['certificate_path'].startswith(str(root / 'priors'))


def test_worker_stops_at_eight_successes_and_never_claims_again(prepared, manifest):
    root, queue, _ = prepared
    calls = []
    def fake(m, pid, artifact_root, workspace, device):
        calls.append(pid)
        return fake_property(m, pid, workspace)
    result = S.worker(root, manifest, queue, root / 'inputs', 0, runner=fake, preflight=False)
    assert len(calls) == 7 and result['decision'] == 'CHECKER_BUILD_JUSTIFIED'
    assert result['successes'] == 8
    assert S.claim_next(root, manifest, queue, 1)[0] is None
    resumed = S.worker(root, manifest, queue, root / 'inputs', 0, runner=lambda *_: pytest.fail('must not rerun'), preflight=False)
    assert resumed['decision'] == result['decision']


def test_worker_stops_at_seventeen_non_successes(prepared, manifest):
    root, queue, _ = prepared
    calls = []
    def fake(m, pid, artifact_root, workspace, device):
        calls.append(pid)
        return fake_property(m, pid, workspace, positive=False)
    result = S.worker(root, manifest, queue, root / 'inputs', 1, runner=fake, preflight=False)
    assert len(calls) == 15 and result['non_successes'] == 17
    assert result['decision'] == 'CHECKER_BUILD_NOT_JUSTIFIED'
    assert S.claim_next(root, manifest, queue, 0)[0] is None


def test_distinct_claims_and_owner_only_unfinished_resume(prepared, manifest):
    root, queue, _ = prepared
    first, _ = S.claim_next(root, manifest, queue, 0)
    second, _ = S.claim_next(root, manifest, queue, 1)
    assert [first['property_id'], second['property_id']] == [r['property_id'] for r in queue['properties'][:2]]
    resumed, _ = S.claim_next(root, manifest, queue, 0)
    assert resumed == first
    assert len(list((root / 'claims').glob('*.json'))) == 2


def test_atomic_persisted_result_recovered_before_claim_without_reexecution(prepared, manifest):
    root, queue, _ = prepared
    claim, _ = S.claim_next(root, manifest, queue, 0)
    fake_property(manifest, claim['property_id'], root / 'workers/worker_0')
    # Crash after the one-runner atomic normalized record, before completion.
    next_claim, current = S.claim_next(root, manifest, queue, 1)
    assert current['evaluated'] == 4 and current['successes'] == 2
    assert next_claim['property_id'] != claim['property_id']
    assert S.complete(root, manifest, queue, claim)['evaluated'] == 4


def test_global_gate_recovers_final_success_before_another_claim(prepared, manifest):
    root, queue, _ = prepared
    for _ in range(6):
        claim, _ = S.claim_next(root, manifest, queue, 0)
        fake_property(manifest, claim['property_id'], root / 'workers/worker_0')
        S.complete(root, manifest, queue, claim)
    last, _ = S.claim_next(root, manifest, queue, 0)
    fake_property(manifest, last['property_id'], root / 'workers/worker_0')
    no_claim, current = S.claim_next(root, manifest, queue, 1)
    assert no_claim is None and current['successes'] == 8


def test_two_concurrent_workers_have_no_duplicate_property_execution(prepared, manifest):
    root, queue, _ = prepared
    calls = []
    def fake(m, pid, artifact_root, workspace, device):
        calls.append(pid)
        return fake_property(m, pid, workspace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(S.worker, root, manifest, queue, root / 'inputs', wid,
                                   runner=fake, preflight=False) for wid in (0, 1)]
        for future in futures:
            future.result(timeout=10)
    result = status(root, manifest, queue)
    assert result['decision'] == 'CHECKER_BUILD_JUSTIFIED'
    assert len(calls) == len(set(calls))
    assert 7 <= len(calls) <= 8  # At most one already-in-flight completion.
    assert not set(calls) & set(S.PRIOR_IDS)


def test_worker_lifetime_lease_rejects_duplicate_process(prepared, manifest):
    root, queue, _ = prepared
    with S._lock(root / 'workers/worker_0/.worker.lock'):
        with pytest.raises(RuntimeError, match='already running'):
            S.worker(root, manifest, queue, root / 'inputs', 0, preflight=False)


def test_missing_or_corrupt_prior_artifact_rejected_without_reset(tmp_path, manifest):
    roots = [tmp_path / str(i) for i in range(3)]
    for i, pid in enumerate(S.PRIOR_IDS):
        fake_property(manifest, pid, roots[i], positive=i == 2, prior=True)
    certificate = roots[2] / 'producer/properties' / S.PRIOR_IDS[2] / 'certificate.pt'
    certificate.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        S.prepare(manifest, S.queue_manifest(manifest), tmp_path / 'new', roots)


def test_conflicting_prior_sources_reject(prepared, manifest, tmp_path):
    _, queue, roots = prepared
    extra = tmp_path / 'extra'
    fake_property(manifest, S.PRIOR_IDS[0], extra, positive=False, prior=True)
    with pytest.raises(RuntimeError, match='exactly one'):
        S.prepare(manifest, queue, tmp_path / 'other-campaign', roots + [extra])


def test_success_predicate_requires_all_three_conditions_and_no_independent_promotion():
    valid = {'producer_proof_status': 'CERTIFIED_MARGIN', 'final_sound_lower_margin': 1., 'certificate_integrity_status': 'PASS'}
    assert S.success(valid)
    for changed in ({'producer_proof_status': 'UNCERTIFIED'}, {'final_sound_lower_margin': 0.},
                    {'final_sound_lower_margin': None}, {'certificate_integrity_status': 'FAIL'}):
        assert not S.success({**valid, **changed})
    rows = [{'property_id': str(i), **valid} for i in range(8)]
    assert S.gate(rows)['decision'] == 'CHECKER_BUILD_JUSTIFIED'
    bad = [{'property_id': str(i), **valid, 'certificate_integrity_status': 'FAIL'} for i in range(17)]
    assert S.gate(bad)['decision'] == 'CHECKER_BUILD_NOT_JUSTIFIED'


def test_preflight_fails_before_claiming_any_property(prepared, manifest, monkeypatch):
    root, queue, _ = prepared
    monkeypatch.setattr(B, 'artifact_errors', lambda *_: [{'kind': 'MISSING'}])
    with pytest.raises(RuntimeError, match='BEFORE any claim'):
        S.worker(root, manifest, queue, root / 'inputs', 0)
    assert not list((root / 'claims').glob('*.json'))


def test_no_prior_loss_or_corruption_is_silently_counted(prepared, manifest):
    root, queue, _ = prepared
    path = root / 'results' / f'{S.PRIOR_IDS[0]}.json'
    path.unlink()
    with pytest.raises(RuntimeError, match='prior completion missing'):
        status(root, manifest, queue)
