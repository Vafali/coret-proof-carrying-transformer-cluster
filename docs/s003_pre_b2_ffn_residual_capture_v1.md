# s003/token8: causal pre/post b2_ffn_residual comparison

Only target: `deept_table7_stdln3_s003_line2031_tok08`, native token **8**,
frozen radius `0.00130126953125` (`0x1.551eb851eb852p-10`).

The existing `run_sound_fp64_finish_3l_v1.py` failure callback already exposes
`pre_output_reduction`: the exact state after residual addition and its usual
FP64 reserve injection, immediately before `_maybe_reduce(...,
"b2_ffn_residual", ...)`. No producer/reduction hook or formula is changed.

The new capture callback retains references only. CPU snapshots/hashes happen
after the unchanged property execution returns. Only the pre-state is persisted,
with its full weights, **all** actual pre-reduction generators (not capped or
truncated to 14k), ranges, IDs, masks, provenance and token count. The native
reduction record binds input/output counts and membership. Authentication
requires a genuine reduction, the existing source/model/benchmark/radius/token
identity, and a rerun post-state bitwise/component-hash identical to the existing
authenticated post-state. A mismatch aborts comparison, not a scientific result.

The post-state is checked transiently and is NOT dumped again. No earlier
operator state is captured. Existing input artifacts are never overwritten.

## ONE manual A40 capture on ISIS

Requires the updated files, existing `runtime_inputs`, `~/slurm_logs`, the
existing post-capture bundle, and a NEW output root.

```bash
sbatch --job-name=coret-s003-pre-ffn-residual \
  --partition=gpu-a40 --nodelist=afrodita --gres=gpu:gpu0:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
  --output="$HOME/slurm_logs/coret-s003-pre-ffn-residual-%j.out" \
  --error="$HOME/slurm_logs/coret-s003-pre-ffn-residual-%j.err" \
  --wrap='env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u "$HOME/projects/coret-proof-carrying-transformer-cluster/scripts/capture_s003_b2_ffn_residual_pre_reduction_v1.py" --post-capture-manifest "$HOME/coret-s003-tok08-zero-variance-v1/capture_manifest.json" --artifact-root "$HOME/projects/coret-proof-carrying-transformer-cluster/runtime_inputs" --output-root "$HOME/coret-s003-tok08-pre-b2-ffn-residual-v1" --device cuda:0'
```

Outputs:

- `~/coret-s003-tok08-pre-b2-ffn-residual-v1/pre_b2_ffn_residual_state.pt`
- `~/coret-s003-tok08-pre-b2-ffn-residual-v1/capture_manifest.json`
- Existing runner records under that new root's `execution/`.

## ONE token8 CPU separator certificate command

```bash
timeout --signal=INT --kill-after=10s 180s \
  env CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u \
  "$HOME/projects/coret-proof-carrying-transformer-cluster/scripts/certify_s003_pre_reduction_separating_variance_v1.py" \
  --capture-manifest "$HOME/coret-s003-tok08-pre-b2-ffn-residual-v1/capture_manifest.json" \
  --proposal-timeout-seconds 60 \
  --output "$HOME/coret-s003-tok08-pre-b2-ffn-residual-v1/token8_pre_separator.json"
```

This reuses the SAME bounded max-separation LP from
`certify_block2_output_separating_variance_v1.py`. It is an untrusted proposal
only. The unchanged, independent standard-library arithmetic in
`separating_variance_checker_v1.py` constructs and replays a persisted rational
certificate: exact `v=D^T y`, every binary64 coefficient and original range,
strict support separation, norm, and `delta^2/(128*||v||^2)`. The report records
both exact rational and downward-safe binary64 lower bounds. No primal
zero-variance reconstruction, least-squares, Bareiss, Farkas or new solver is
invoked. There is no all-token command.

On exact positive replay, `final_status` is
`CERTIFIED_SEPARATING_VARIANCE_LOWER_BOUND`. If the linked authenticated post
capture also records the failed token8 separator attempt, `causal_decision`
is `REDUCTION_PRECISION_LOSS_CAUSAL`, with the certified pre bound. Otherwise
it is `REDUCTION_NOT_YET_CAUSALLY_ISOLATED` and no other method is launched.
This is the requested pre-bound/post-search-failure gate, **not** an exact
post-state zero-variance witness or a complete end-to-end proof certificate.

The capture/authentication code checks boundary continuity but does not add a
new reduction theorem: production's established native reduction checker
remains unchanged. The independent separator checker proves the bound of the
captured pre-state using all its generators/ranges.
