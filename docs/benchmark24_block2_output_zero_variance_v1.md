# s004/tok04: passive block2_output capture and exact zero decision

Only `deept_table7_stdln3_s004_line216_tok04` is executable by this capture
wrapper. It uses the unchanged benchmark24 historical radius
`0x1.4dc28f5c28f5bp-10` (`0.0012731933593749997`) and all 12 frozen token IDs.
No benchmark population, producer formula, LayerNorm, reduction, widening,
solver setting, or global causal-oracle machinery changes.

The wrapper calls the existing benchmark one-property runner and supplies its
existing `experimental_layernorm_failure_capture` callback. The callback holds
references only. CPU snapshotting starts after the property returns. No
post-attention experimental LayerNorm override is installed. Only the complete
post-reduction input of `block2_output` is captured; pre-reduction states are
not copied. A new isolated root is required and existing records are not edited.

The captured state includes weights `[14001,12,128]`, all 14,000 generator
ranges, ordered IDs, masks, reasons, and token count. Its manifest binds the
artifact and component hashes, failure result and diagnostic to the frozen
benchmark/property/radius/token/source/model identities. Token 4 MUST be the
diagnostic's minimum token, in the native zero-based tensor convention. This is
not a conversion from the property's perturbed-token suffix. The weights alone
occupy 172,044,288 bytes (approximately 164 MiB).

## Manual single-property capture

Run in a single allocated GPU shell with the existing environment (not on the
ISIS CPU login node). This is one deliberate rerun, not a campaign launcher.

```bash
cd "$HOME/projects/coret-proof-carrying-transformer-cluster"
CUDA_VISIBLE_DEVICES=0 "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u \
  scripts/capture_benchmark24_block2_output_zero_variance_v1.py \
  --property-id deept_table7_stdln3_s004_line216_tok04 \
  --artifact-root "$HOME/projects/coret-proof-carrying-transformer-cluster/runtime_inputs" \
  --output-root "$HOME/coret-s004-tok04-zero-variance-v1" \
  --device cuda:0
```

Outputs: `capture_manifest.json`, `pre_block2_output_layernorm_state.pt`, and
the unchanged runner records under `execution/`. A different failure stage,
property identity, radius, input population, token minimum, or generator count
fails authentication. The small prior screening records remain untouched.

## FIRST CPU oracle: token 4 ONLY

```bash
cd "$HOME/projects/coret-proof-carrying-transformer-cluster"
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u \
  scripts/decide_block2_output_zero_variance_v1.py \
  --capture-manifest "$HOME/coret-s004-tok04-zero-variance-v1/capture_manifest.json" \
  --expected-property-id deept_table7_stdln3_s004_line216_tok04 \
  --variant complete_post_reduction --token-index 4 \
  --fast-first --exact-only-if-near-zero --exact-solve-timeout-seconds 180 \
  --output "$HOME/coret-s004-tok04-zero-variance-v1/token4_decision.json"
```

Each equation uses exact differences of original binary64 coefficients, not
differences of a rounded mean. For coordinate j<127, h_j=c_j-c_127 and
A_ji=G_ij-G_i,127; solve `h+A*xi=0` with every authenticated range.
Numerical LP, least-squares, QR selection and dual searches are proposals only.

For this capture schema, `--exact-solve-timeout-seconds` also imposes a hard
wall-clock deadline on the entire per-token attempt AFTER capture authentication:
proposal solvers, dual search, exact construction, reconstruction, persistence
and full replay. A supervised CPU child is terminated at the deadline (with
bounded termination cleanup); the parent writes `INCONCLUSIVE` with reason
`EXACT_EXCLUSION_TIMEOUT`, elapsed time and the last reported stage. This does
not change any solver settings or mathematical acceptance condition. The
original legacy oracle's selected-integer-solve timer remains unchanged.

- `EXACT_ZERO_VARIANCE_FEASIBLE`: a rational witness replays all 127 original
  equations and all 14,000 ranges exactly. The JSON witness is reread/rechecked.
  This proves the captured incoming abstract state admits a constant vector at
  token 4. Merely tightening its variance lower bound cannot exclude that vector.
- `EXACT_ZERO_VARIANCE_EXCLUDED`: a rational Farkas certificate is replayed
  against the original equations and every bound. This result concerns token 4
  only, not the whole LayerNorm. No repair is implemented automatically.
- `INCONCLUSIVE`: no exact certificate verified. Tiny residual, numerical
  infeasibility, search failure and timeout never authorize a claim.

The compact Farkas witness serializes signed equality multipliers y. The
checker independently derives a=y*A, nonnegative lower/upper bound multipliers
`max(a,0)` / `max(-a,0)`, verifies cancellation exactly, and requires
`-y*h-sum(min(a_i*low_i,a_i*high_i)) < 0` exactly. It never checks the rounded
solver matrix. This is an exact linear exclusion certificate, not SDP/SOS or
the perspective/ReLU phase oracle. No giant rational matrix is constructed.

## SECOND CPU command: ONLY after token-4 exact exclusion

```bash
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u \
  scripts/decide_block2_output_zero_variance_v1.py \
  --capture-manifest "$HOME/coret-s004-tok04-zero-variance-v1/capture_manifest.json" \
  --expected-property-id deept_table7_stdln3_s004_line216_tok04 \
  --variant complete_post_reduction --token-index all \
  --token4-exclusion-report "$HOME/coret-s004-tok04-zero-variance-v1/token4_decision.json" \
  --fast-first --exact-only-if-near-zero --exact-solve-timeout-seconds 180 \
  --output "$HOME/coret-s004-tok04-zero-variance-v1/all_tokens_decision.json"
```

This command is rejected without a matching token-4 report AND independently
replayed exact exclusion certificate. It reuses that certificate and checks
remaining tokens in ascending index order. One exact feasible token stops the
existential search. Whole-LayerNorm exclusion requires exact exclusions for
ALL 12 tokens. Any unresolved token prevents a whole-LayerNorm exclusion claim.

The result and compact certificates bind the input problem SHA and ordered IDs;
verification also binds the raw artifact, ranges, masks/provenance and result
hashes. This establishes properties of the captured affine box, not whether
its witness is realizable by the original network input. It is not a complete
independent end-to-end certificate for the producer's preceding graph.

No real property, captured production artifact or CUDA operation is evaluated
in local validation. Tests use two-generator CPU fixtures and mock callbacks;
production authentication has no CLI escape from the exact 14,000-row count.
The original legacy capture/oracle interface remains available unchanged.
