# s003/token 8: one passive capture, one exact CPU oracle

Target: `deept_table7_stdln3_s003_line2031_tok08`, `block2_output`, native
zero-based token **8 only**, frozen radius `0.00130126953125`
(`0x1.551eb851eb852p-10`). The frozen benchmark has 17 tokens.

The capture uses the existing `SOUND_FP64_SEPARATING_LAYERNORM_V1` runner and
its authenticated source revision. It only installs the existing failure
callback; it does not replace any numerical operator. The callback retains
references and CPU persistence occurs after execution returns. It requires
the unresolved separator token to be 8. The original generic variance-minimum
index is preserved as separate diagnostic information: it need not identify
the unresolved token after attempted repairs.

Authentication binds the unchanged benchmark/model/data/radius/token identity,
current approved producer source hashes, raw artifact and component hashes,
failure record and diagnostic. The captured state contains weights
`[14001,17,128]`, all 14,000 ordered IDs, support masks, provenance reasons, and
all exact binary64 generator ranges. It is the existing callback's complete
post-reduction affine state, not a synthetic or regenerated state.

## Manual capture on ISIS: one A40 property rerun

Preconditions: repository files transferred, existing authenticated
`runtime_inputs`, `~/slurm_logs` exists, and the output root does not exist.
Do not launch another benchmark property or campaign.

```bash
sbatch --job-name=coret-s003-tok08-zero-capture \
  --partition=gpu-a40 --nodelist=afrodita --gres=gpu:gpu0:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
  --output="$HOME/slurm_logs/coret-s003-tok08-zero-capture-%j.out" \
  --error="$HOME/slurm_logs/coret-s003-tok08-zero-capture-%j.err" \
  --wrap='env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u "$HOME/projects/coret-proof-carrying-transformer-cluster/scripts/capture_benchmark24_block2_output_zero_variance_v1.py" --property-id deept_table7_stdln3_s003_line2031_tok08 --artifact-root "$HOME/projects/coret-proof-carrying-transformer-cluster/runtime_inputs" --output-root "$HOME/coret-s003-tok08-zero-variance-v1" --device cuda:0'
```

Output: `capture_manifest.json`, `pre_block2_output_layernorm_state.pt`, and
unchanged runner records under `execution/`. Prior screening records are not
read as new numerical states and are not overwritten. If the requested failure
callback/token is not reached, capture authentication fails rather than
substituting another stage/property/token.

## Manual CPU oracle: token 8 only

```bash
timeout --signal=INT --kill-after=10s 240s \
  env CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u \
  "$HOME/projects/coret-proof-carrying-transformer-cluster/scripts/decide_block2_output_zero_variance_v1.py" \
  --capture-manifest "$HOME/coret-s003-tok08-zero-variance-v1/capture_manifest.json" \
  --expected-property-id deept_table7_stdln3_s003_line2031_tok08 \
  --variant complete_post_reduction --token-index 8 \
  --fast-first --exact-only-if-near-zero --exact-solve-timeout-seconds 180 \
  --output "$HOME/coret-s003-tok08-zero-variance-v1/token8_decision.json"
```

The existing CPU-child watchdog bounds the entire token attempt after capture
authentication by 180 seconds, not only the selected-system solve. It emits
`INCONCLUSIVE / EXACT_EXCLUSION_TIMEOUT` with stage/elapsed diagnostics on
timeout. The external timeout is a separate emergency guard.

All 127 equations use exact coordinate differences of original binary64
coefficients, never a rounded mean. Every generator and original box bound is
included. The numerical solvers have no proof authority:

- `EXACT_ZERO_VARIANCE_FEASIBLE` requires persisted rational witness replay
  against every original equation and bound.
- `EXACT_ZERO_VARIANCE_EXCLUDED` requires rational Farkas replay against the
  complete original equation/box system, and concerns **token 8 only**.
- Otherwise the result is `INCONCLUSIVE`; numerical infeasibility, objective
  zero, search failure or timeout does not imply either exact decision.

No all-token command is enabled for this property. The legacy s004 capture
schema/commands remain compatible; no separator, exact solver, reduction,
LayerNorm, benchmark or certificate mathematics changes are made here.
