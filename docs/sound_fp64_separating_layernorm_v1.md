# Conditional separating-variance LayerNorm repair

The real sound-FP64 3L property path checks every LayerNorm token. If all native
cheap variance lower bounds are strictly positive, it invokes the original
native dispatch, rounding reserves and provenance path unchanged. No separator
proposal or payload is produced on that fast path.

Only tokens with `generic_variance_low[token].min() <= 0` enter the fallback.
At most eight coordinate differences and four fixed sparse two-difference
directions are tried. Their ordering is numerical and untrusted. Each candidate
is accepted only by exact rational support replay over every input generator
and its actual range. If necessary, the existing normalized LP proposal uses
the remaining per-token 60-second proposal budget with a CPU child watchdog.
There is no least-squares, Bareiss, SDP, branching, alternative solver portfolio
or hard-coded property/certificate.

For `v=D^T y`, exact replay proves `sum(v)=0`, computes the exact support
`[L,U]`, requires strict zero exclusion, and proves
`Var(x) >= delta^2/(d*||v||^2)`. The production bound is rounded down to
binary64 and must still be positive. ALL failing tokens must be repaired;
otherwise the LayerNorm remains fail-closed. No uncertified LP conclusion
can authorize execution.

Repaired calls reuse the established native PSD-domain sequence: the variance
affine coefficients are unchanged; the additional semantic lower constraint
is consumed by sqrt and its directed result range by reciprocal. Only failed
tokens receive new range selections. Successful tokens retain native
reciprocal inputs. Fresh membership/order is traced from actual native
predicates, support is validated, and the existing FP64 majorant/reserve
formula is evaluated using the checked positive domain. Reduction policy,
generator cap, epsilon, model, radius and downstream operators are unchanged.

The archived payload contains compact CPU binary64 operand blocks only for
repaired tokens (all generator rows), original ranges/ordered IDs/provenance,
source identity/domain, generic and semantic bounds, rational separator
certificates, proposal diagnostics and native range/membership evidence.
Exact directed sqrt range relations are checked on replay. A token operand
block at 14k generators and width 128 is about 13.67 MiB before metadata.
This avoids dumping all tokens of every state. The campaign retains
`properties/ID/layernorm_separator_witnesses.pt` with a SHA in the atomic
result before deleting temporary stage artifacts. Block-2 witness data also
lives in the final certificate artifact.

This checks the separating-variance obligation, not the complete FP64 graph.
Complete independent certificate status remains `NOT_AVAILABLE`.

The historical benchmark manifest retains its original population/radii AND
source hashes. The new manual entry point labels the producer revision
`SOUND_FP64_SEPARATING_LAYERNORM_V1`, checks its separate source-revision manifest
and rejects unexpected changes outside the four authorized integration files.
It requires a NEW result root, never overwrites prior benchmark records and
never runs binary search.

One manual validation job on ISIS (do not launch the benchmark campaign):

```bash
sbatch --job-name=coret-s004-tok04-separating-ln \
  --partition=gpu-a40 --nodelist=afrodita --gres=gpu:gpu0:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
  --output="$HOME/slurm_logs/coret-s004-tok04-separating-ln-%j.out" \
  --error="$HOME/slurm_logs/coret-s004-tok04-separating-ln-%j.err" \
  --wrap='env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u "$HOME/projects/coret-proof-carrying-transformer-cluster/scripts/run_sound_fp64_separator_property_v1.py" --property-id deept_table7_stdln3_s004_line216_tok04 --artifact-root "$HOME/projects/coret-proof-carrying-transformer-cluster/runtime_inputs" --result-root "$HOME/coret-s004-tok04-separating-ln-v1" --device cuda:0'
```

Precondition: `~/slurm_logs` exists and the result root does not. The frozen
historical radius is used exactly; the token-4 diagnostic certificate is not
hard-coded or reused as authorization for a different live state.
