# Semantic epsilon-floor LayerNorm revision

Producer revision: `SOUND_FP64_SEMANTIC_EPSILON_FLOOR_V1`. The separate
`frozen/layernorm_epsilon_floor_source_revision_v1.json` authenticates its
sources. The frozen benchmark manifest and historical candidate radii are
unchanged. The previous separator revision's pins intentionally reject this
new code; they are not rewritten.

## Trigger and semantics

1. Every generic variance lower is positive: identical native dispatch and
   original numerical reserve, without new payload construction.
2. Otherwise the existing exact-replayed separator preparation succeeds:
   identical separator execution, topology, numerical reserve and witness.
3. Otherwise the semantic epsilon-floor path replaces each unresolved token
   (generic lower <= 0) by fresh coordinate interval-box symbols. Tokens with
   positive generic bounds still undergo native tokenwise LayerNorm with its
   original majorant and numerical reserve. This third path intentionally
   drops failed-token affine correlations; no relaxed coefficient is clamped.

The incoming complete affine state, including every numerical generator and
explicit source range, is independently authenticated. For each failed token
the checker computes exact centered affine coefficients `P(c+G xi)` and their
box support. With exact centered intervals `[l_j,u_j]`,

```
0 <= V <= sum_j max(abs(l_j), abs(u_j))^2 / d.
```

This upper bound is reconstructed from the incoming proof state, not trusted
from a machine variance zonotope. The regularized domain uses exact binary64
epsilon `0x1.19799812dea11p-40` (bits `3d719799812dea11`), with
`h_low = nextafter(epsilon,-inf) = 9.999999999999998e-13`, and an outward upper
bound for `V_upper+epsilon`. Exact rational-square tests check directed sqrt
endpoints; exact rational division checks reciprocal endpoints from that
authenticated sqrt range. The producer reuses the PSD directed-sqrt primitive.

Exact interval products enclose `(P x)_j / sqrt(V+epsilon)`, then frozen gamma
and beta. A machine midpoint and an outward radius contain this image;
midpoint rounding is also covered by an explicit reserve subsequently embedded
through the existing `_inject` path. There is no native sqrt or reciprocal on
the failed token's unconstrained negative variance assignments. No derivative
majorant of that invalid affine domain is used. The resulting state is one
ordinary relational zonotope, with the established downstream/reduction rules.

## Proof payload and independence

Tagged `CORET_SEMANTIC_EPSILON_FLOOR_LAYERNORM_V1` witnesses contain immutable
binary64 input/output blocks, canonical source/output identities, source-domain
identity, frozen gamma/beta hashes and blocks, per-token semantic endpoints,
exact upper-bound/support provenance, box containment/rounding evidence,
ordered fresh IDs/masks/reasons/ranges and numerical reserves. The existing
stage witness transport persists them; mixed archives use
`CORET_SOUND_FP64_LAYERNORM_DOMAIN_ARCHIVE_V2`. No prior artifact is retrofitted.

`semantic_epsilon_floor_checker_v1.replay_persisted` takes **externally
authenticated** predecessor, parameter, source-domain and output identities.
It verifies all canonical components, source support, every generator/range,
PSD rule applicability, every interval operation, output coefficients,
allocation order/provenance and reserves. It imports only the standard
library and the existing stdlib exact-support checker, not Torch, NumPy,
producer kernels, majorant helpers or the production support validator.

This verifies the epsilon-floor local transition. Successful-token native
transitions retain their previous obligations. It is not a new complete graph
checker: `independently_checked_certificate_status` remains `NOT_AVAILABLE`.

This interval-box fallback may be substantially coarser than the native or
separator path. Passing its domain obligation does not promise a positive
final margin. Witness storage uses one full input and output binary64 block
per fallback LayerNorm (`O(g*t*d)` each), plus exact bounds and proof metadata.

## ONE manual ISIS property validation (not executed locally)

Preconditions: these source files and the new revision manifest transferred;
existing authenticated `runtime_inputs`; `~/slurm_logs` exists; result root is
new. Only the frozen s003/token8 property is run, at its unchanged historical
radius `0.00130126953125`. No search or benchmark campaign is launched.

```bash
sbatch --job-name=coret-s003-tok08-epsilon-floor \
  --partition=gpu-a40 --nodelist=afrodita --gres=gpu:gpu0:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
  --output="$HOME/slurm_logs/coret-s003-tok08-epsilon-floor-%j.out" \
  --error="$HOME/slurm_logs/coret-s003-tok08-epsilon-floor-%j.err" \
  --wrap='env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u "$HOME/projects/coret-proof-carrying-transformer-cluster/scripts/run_sound_fp64_epsilon_floor_property_v1.py" --property-id deept_table7_stdln3_s003_line2031_tok08 --artifact-root "$HOME/projects/coret-proof-carrying-transformer-cluster/runtime_inputs" --result-root "$HOME/coret-s003-tok08-epsilon-floor-v1" --device cuda:0'
```
