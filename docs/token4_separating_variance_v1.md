# Token-4 separating variance certificate

This standalone CPU diagnostic targets only
`deept_table7_stdln3_s004_line216_tok04`, `block2_output`, native token index 4.
It uses the existing capture; there is no property execution, state mutation,
least squares, Bareiss, phase enumeration, SDP, or production integration.
The captured artifact SHA is pinned to
`5203d9dac6dee0d0a4a1f3df4ee7ec6eb3c2237cf14753c8ac11fc0b01249e33`.
The existing capture verifier authenticates the frozen identity and full state.

For `D_j=e_j-e_127`, the proposal LP minimizes
`-z^T y + sum_i s_i` under `+/-H_i y <= s_i`, `+/-y_j <= a_j`,
`sum a <= 1`, where `z=D(c+G midpoint)` and `H_i=(Dg_i)*halfwidth_i`.
Every generator, including FP64 numerical rows, enters the proposal.
All floating model construction and solving is confined to a supervised CPU
child with a maximum 60-second computation deadline and bounded kill cleanup.
Solver status and objective have **no proof authority**.

The proposed binary64 `y` is stored as exact rational values. The separate
standard-library checker constructs `v=[y,-sum y]`, verifies zero sum and
positive exact norm, and recomputes the exact support interval using all 14,000
original binary64 coefficient rows and authenticated asymmetric ranges.
It uses integer power-of-two alignment for dot products and exact rational
endpoint arithmetic, not rounded mean subtraction or a dense rational system.
If the interval strictly excludes zero, it proves
`Var(x) >= delta^2/(128*||v||^2)` by Cauchy--Schwarz.
The certificate binds the full capture identity, token coefficient/range bits,
ordered IDs, direction, interval, delta, norm and bound. Persisted claims are
recomputed; their hashes alone never establish soundness.

Terminal outcomes are `CERTIFIED_SEPARATING_VARIANCE_LOWER_BOUND` or
`NO_CERTIFIED_SEPARATOR`. The latter does not assert zero-variance feasibility
or imply that no separator exists. It is the final outcome of this one proposal.
Authentication or implementation errors fail loudly. A positive result is only
a candidate LayerNorm repair; production behavior is unchanged.

On ISIS, run exactly one CPU diagnostic:

```bash
timeout --signal=INT --kill-after=10s 240s \
  env CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$HOME/.conda/envs/coret-cluster-a268be7/bin/python" -u \
  "$HOME/projects/coret-proof-carrying-transformer-cluster/scripts/certify_block2_output_separating_variance_v1.py" \
  --capture-manifest "$HOME/coret-s004-tok04-zero-variance-v1/capture_manifest.json" \
  --proposal-timeout-seconds 60 \
  --output "$HOME/coret-s004-tok04-zero-variance-v1/token4_separating_variance.json"
```

If successful, the sibling `token4_separating_variance.certificate.json` holds
the exact rational certificate. `--verify-certificate PATH` replays a saved
certificate without calling a proposal solver. There is no all-token mode and
no alternative solver portfolio.
