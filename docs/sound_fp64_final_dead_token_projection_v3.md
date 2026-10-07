# Final encoder dead-token projection (producer revision V3)

`SOUND_FP64_FINAL_DEAD_TOKEN_PROJECTION_V3` changes verifier precision only.
The frozen benchmark, historical radii, LayerNorm/native/separator/epsilon-floor
rules, 14,000-generator cap and reduction formulas remain unchanged. The old
producer revision records remain immutable; V3 has its own source inventory.

## Placement and semantics

After `block2_output` LayerNorm and its unchanged `_inject` reserve, before
`_maybe_reduce(..., "b2_output_layernorm", ...)`, the full hidden state is
projected to token 0. The frozen final head performs token-0 selection, a
feature-affine pooler, elementwise tanh, a feature-affine classifier difference,
and margin concretization. There is no later attention, token sum/mean, or
other neural cross-token operation. The recenter/reduction representation
boundaries now also operate on the projected one-token state.

Projection copies the center and retained coefficients bit-for-bit. A row is
deleted only if **every selected coefficient is exactly zero**, including signed
zero; masks alone never authorize deletion. Surviving rows retain their relative
order, IDs, exact ranges and provenance/numerical classification. Their masks
are restricted to bit 0 and the output token universe is one. This is an exact
affine-set projection, not a new abstraction or a new reduction policy.

Existing `_maybe_reduce` then ranks/reduces this one-token state unchanged.
The coordinate-box replacement count is consequently `1 * hidden_dimension`,
as required by that existing algorithm, rather than full-sequence coordinates.

## Local independent obligation and persistence

`final_token_projection_checker_v1` imports only hashlib/json/math/struct.
It streams the complete authenticated predecessor coefficients, validates
finite values/ranges/support, independently selects token-0 bytes and identifies
exact-zero rows, and compares output bytes and the ordered transition map.
The external caller supplies authenticated predecessor/output identities,
execution identity and the frozen final-head region specification. A complete
graph checker must authenticate that region in its chain; this local obligation
does not certify earlier Transformer operations. Independent complete-graph
certificate status remains `NOT_AVAILABLE`.

`final_token_projection_witness.pt` contains full immutable input/output binary64
blocks and canonical identities, the original token count, retained token set,
ordered pre/post IDs, exact-zero discarded IDs, execution/source/model hashes,
the frozen no-cross-token rule, counts before/after projection/after reduction,
and the existing subsequent reduction telemetry. Its path/SHA and summary are
bound in the final certificate report and property result. The already-emitted
LayerNorm separator/epsilon-floor witnesses refer to their original full states
and are not rewritten by projection. Witness size is one pre-projection state
plus one single-token projected state and metadata; no intermediate graph dump.

## ONE manual ISIS rerun (not executed locally)

Transfer these changes and the V3 source revision. Use the same authenticated
`runtime_inputs` and a NEW result root. Only s003/token8 is evaluated, at its
unchanged frozen historical radius `0.00130126953125`; no radius search.

```bash
sbatch --job-name=coret-s003-tok08-final-projection-v3 \
  --partition=gpu-a40 --nodelist=afrodita --gres=gpu:gpu0:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
  --output="$HOME/slurm_logs/coret-s003-tok08-final-projection-v3-%j.out" \
  --error="$HOME/slurm_logs/coret-s003-tok08-final-projection-v3-%j.err" \
  --wrap='env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u "$HOME/projects/coret-proof-carrying-transformer-cluster/scripts/run_sound_fp64_epsilon_floor_property_v1.py" --property-id deept_table7_stdln3_s003_line2031_tok08 --artifact-root "$HOME/projects/coret-proof-carrying-transformer-cluster/runtime_inputs" --result-root "$HOME/coret-s003-tok08-final-projection-v3" --device cuda:0'
```
