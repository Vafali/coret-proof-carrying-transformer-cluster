# Benchmark24 producer-screening handoff

This is producer screening, not independent certification. It delegates to the
unchanged `run_transformer_benchmark24_v1.run_one` path. No radius search or
independent checker is added. Positive producer results retain independent
checker status `NOT_AVAILABLE` and final proof status `INCONCLUSIVE`.

## Frozen queue and gate

The existing frozen 24-property manifest is unchanged. The separate queue
excludes exactly the three accepted prior property IDs and sorts the remaining
21 by `(sequence_length, property_id)`. It does not inspect screening outcomes.
Queue canonical SHA256:
`bc23bb8d2ad7dbe074326123d2e948ed0a420ee2fb4e22f564e2ab1a547982f4`.

Success requires all three: `producer_proof_status == CERTIFIED_MARGIN`, finite
`final_sound_lower_margin > 0`, and `certificate_integrity_status == PASS`.
Every other completed screening record is a non-success for this precommitted
decision gate; errors remain explicitly distinguished from scientific domain
failures in the preserved result fields and status histogram.

- Initial counts: one success, two non-successes.
- Eight successes: `CHECKER_BUILD_JUSTIFIED`.
- Seventeen non-successes: `CHECKER_BUILD_NOT_JUSTIFIED`.

The gate is checked under a shared POSIX lock before every claim and after each
completion. At most one already-running property can finish after the first
terminal decision. No new property is claimed after that decision. An in-flight
evaluation is not forcibly interrupted or discarded.

## Manual ISIS setup

Run after these files have been transferred to the existing repository. The
three original roots are separate, read-only inputs, not one shared directory.
The importer requires exactly one matching record per prior ID, validates its
frozen identity and producer artifact hashes, and makes byte-identical copies.
It does not deserialize scientific tensors or claim numerical verification.
The actual ISIS prior files are not available in the local coding environment;
their authentication happens during this manual setup.

```bash
cd "$HOME/projects/coret-proof-carrying-transformer-cluster"
mkdir -p slurm_logs
"$HOME/.conda/envs/coret-cluster-a268be7/bin/python" scripts/run_benchmark24_producer_screening_v1.py prepare \
  --campaign-root "$HOME/coret-benchmark24-producer-screening-v1" \
  --prior-root "$HOME/coret-benchmark24-v1-smoke2" \
  --prior-root "$HOME/coret-benchmark24-v1-mid-pilot" \
  --prior-root "$HOME/coret-benchmark24-v1-upper-pilot"
```

Expected setup status: `CONTINUE`, `successes=1`, `non_successes=2`, `evaluated=3`.
Stop if authentication fails; do not replace a prior property or reset counts.

## Exactly two manual submissions

From the repository directory above:

```bash
sbatch slurm/benchmark24_screening_worker0.sbatch
sbatch slurm/benchmark24_screening_worker1.sbatch
```

Both jobs target `afrodita` / `gpu-a40`. Worker 0 requests `gpu:gpu0:1` and sets
`CUDA_VISIBLE_DEVICES=0`; worker 1 requests `gpu:gpu1:1` and sets
`CUDA_VISIBLE_DEVICES=1`. Each starts one Python process with device `cuda:0`
inside its single-visible-GPU namespace. The existing cluster environment is
used; no dependencies are installed. Each worker has isolated tmp/cache/CUDA
cache/extensions/pycache directories. Lifetime worker locks reject duplicate
active worker IDs. Do not submit additional workers or oversubscribe the GPUs.

The launchers set a six-hour Slurm wall limit. If a job is interrupted, resubmit
only that worker's same command after the old job has ended. It resumes its own
unfinished claim and skips authenticated atomic completions. It cannot steal a
claim from another live worker. Missing/corrupt prior or completed artifacts are
fatal integrity errors, not permission to silently rerun them.

## Outputs and status

Root: `$HOME/coret-benchmark24-producer-screening-v1`.

- `campaign.json`: queue identity and original prior-record file hashes.
- `priors/records/<property_id>.json`: byte-identical prior record copies.
- `priors/producer/<property_id>/`: copied producer artifacts when present.
- `claims/<property_id>.json`: immutable atomic claim/owner audit.
- `results/<property_id>.json`: sealed normalized screening results, including
  preserved diagnostics, authoritative source paths/hashes and artifact paths/hashes.
- `workers/worker_N/records/` and `workers/worker_N/producer/properties/`: existing
  one-property runner outputs. A completed worker record is recovered into the
  global gate after a crash without recomputation.
- `status.json`: current counts, decision and producer-status histogram.
- `terminal_decision.json`: immutable first terminal gate decision.
- `slurm_logs/benchmark24_screening_wN_<job_id>.out`: flushed claims/gates/completions.

Unavailable diagnostics are explicitly null, not fabricated. Integrity PASS
means artifact/hash consistency only, never independent numerical soundness.

```bash
"$HOME/.conda/envs/coret-cluster-a268be7/bin/python" scripts/run_benchmark24_producer_screening_v1.py status \
  --campaign-root "$HOME/coret-benchmark24-producer-screening-v1"
```

## Runtime planning

Observed runtimes: 318.7, 457.4, 936.5 seconds. If every remaining evaluation
takes no longer than the observed maximum, 21 evaluations require at most
19,666.5 GPU-seconds, or approximately 2h52m on two dynamically assigned workers
(11 x 936.5 seconds), excluding startup/I/O. This is a conditional planning
estimate, not a guaranteed runtime bound. Early stopping usually reduces work;
the six-hour job limit and atomic resume protect against overruns.

No verifier code, frozen benchmark membership, historical radii, solver settings,
or global causal-oracle implementation is changed by this handoff.
