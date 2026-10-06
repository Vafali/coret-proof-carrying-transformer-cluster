# Partial Shor numerical go/no-go probe (no proof authority)

Local SDP backend inventory found no suitable solver. No dependency was installed,
no SDP was solved, and no production instance was constructed locally.
The implemented Python layer only constructs the model and numerical diagnostics.
The solve below is a MANUAL experiment in a separately approved CPU environment
with CVXPY and SCS available. It cannot certify exclusion or an exact witness.

## Mathematical model

Let B=W1 Gamma and alpha=W1 beta+b1. Use q=[c,t,u] and symmetric Q, with
M=[[1,q^T],[q,Q]] PSD (dimension 258 at d=128). Keep all 14,000 xi variables
ordinary/unlifted. Substitute g=Bc+alpha*t exactly in the existing canonical
linear rows, preserving the shared-source equations, all 127 cancellation rows,
source bounds, certified scale bounds, and existing sound ReLU triangles/stable
phases. Explicitly retain u>=0 and u>=g.

The new moment equalities are

- d Q[t,t] - sum_j Q[c_j,c_j] = d epsilon;
- Q[u_i,u_i] - sum_j B[i,j] Q[u_i,c_j] - alpha[i] Q[u_i,t] = 0.

Off-diagonal terms are indexed entries of Q, NOT doubled trace coefficients.
Only rank one is relaxed; there is no lifting of xi, SOS/RLT hierarchy, phase
branching, numerical Farkas authority, or change to the existing oracle.
No additional relationship between Q and xi moments is imposed.

## One manual production numerical probe

Use the exact same authenticated downstream report that was supplied to v10.
Its actual ISIS filename is not available in the local repository, so the
command requires CORET_DOWNSTREAM_REPORT explicitly instead of guessing it.
Set that variable to the existing report path before running this command.
Run from the cluster repository with its scientific artifacts accessible and
CVXPY/SCS already available in the chosen Python environment. No automatic
installation is performed.

```bash
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 3600s python - <<'PY'
import importlib.util, os, sys, time
from pathlib import Path
import numpy as np
import cvxpy as cp

path = Path("scripts/build_block2_partial_shor_sdp_v1.py").resolve()
spec = importlib.util.spec_from_file_location("manual_partial_shor", path)
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)
if "SCS" not in cp.installed_solvers():
    raise RuntimeError("SCS PSD backend is required; nothing was installed")
downstream = Path(os.environ["CORET_DOWNSTREAM_REPORT"]).expanduser().resolve()
output = Path.home() / "coret-block2-partial-shor-probe-v1"
output.mkdir(exist_ok=False)
started = time.perf_counter()
model = S.prepare_production(
    Path.home() / "coret-block2-post-attention-ln-input-capture-v2", downstream)
record = model.record()
model_sha = S.oracle._sha_json(record)
record["model_sha256"] = model_sha
S.oracle._atomic_json(output / "exact_model.json", record)
problem, variables = S.as_cvxpy_problem(model)
solve_started = time.perf_counter()
failure = None
try:
    problem.solve(solver="SCS", use_indirect=True, verbose=True,
                  eps_abs=1e-5, eps_rel=1e-5, max_iters=5000,
                  time_limit_secs=1800)
except cp.error.SolverError as error:
    failure = str(error)
report = S.numerical_telemetry(
    model, variables["xi"].value, variables["q"].value, variables["Q"].value,
    solver="SCS", status=problem.status, solver_stats=problem.solver_stats)
report.update(model_sha256=model_sha, solver_inventory=S.solver_inventory(),
              solve_wall_seconds=time.perf_counter()-solve_started,
              total_wall_seconds=time.perf_counter()-started,
              solver_failure=failure, scientific_queries=0, bound_calls=0,
              scientific_exclusion_authorized=False,
              exact_witness_authorized=False)
if report.get("primal_present") and not report.get("malformed_primal"):
    np.savez_compressed(output / "numerical_primal.npz",
                        xi=variables["xi"].value, q=variables["q"].value,
                        Q=variables["Q"].value)
    report["numerical_primal_sha256"] = S.oracle.cluster_common.sha256(
        output / "numerical_primal.npz")
S.oracle._atomic_json(output / "report.json", report)
print("NUMERICAL_ONLY_REPORT", output / "report.json", flush=True)
print("REPORT_SHA256", S.oracle.cluster_common.sha256(output / "report.json"), flush=True)
PY
```

The SCS 1,800-second limit covers solver work, not CVXPY canonicalization or
artifact authentication/export. The outer 3,600-second timeout bounds the whole
manual command; hitting it gives no scientific result. The output directory is
new and isolated, and a repeated invocation refuses to overwrite it.

## Size and interpretation

Production q has 257 entries, Q has 33,153 independent symmetric entries, and
there are 47,410 independent scalar decision variables including xi. M is 258x258
and its dense binary64 storage is 532,512 bytes. The centered source matrix has
at most 1,792,000 nonzeros (about 20.5 MiB in binary64/int32 CSR). Actual source
sparsity and constraint counts are reported only after authenticated construction.
If all 128 units are unstable and source intervals are non-singleton, the model
has 384 equalities and 28,386 scalar inequalities, plus one PSD cone.

These raw arrays are not a host-memory prediction: capture loading, Python
Fractions, exact JSON export, CVXPY canonicalization and solver workspaces dominate.
Budget a 16-GiB CPU environment for the first probe; practical peak RSS and runtime
are unmeasured. A dense 14,000x14,000 source normal matrix alone would require
1,568,000,000 bytes before factorization/fill, hence the manual SCS indirect path.

The report contains solver residuals when available, original-model linear,
bound and moment residuals, all moment eigenvalues, a documented numerical rank
estimate, source/first-moment reconstruction errors, and the rank-one gap.
Rank greater than one is relaxed feasibility, NOT an exact QCQP witness.
Numerical infeasibility is only go/no-go evidence, NEVER an exclusion certificate.
