#!/usr/bin/env python3
"""Token-4 separator proposal plus exact CPU support replay; no property execution."""
from fractions import Fraction
import argparse
import json
import math
import multiprocessing
from pathlib import Path
import time

import separating_variance_checker_v1 as CHECK
import cluster_common as C
import capture_benchmark24_block2_output_zero_variance_v1 as CAP

EXPECTED_ARTIFACT_SHA = "5203d9dac6dee0d0a4a1f3df4ee7ec6eb3c2237cf14753c8ac11fc0b01249e33"
CERTIFIED = "CERTIFIED_SEPARATING_VARIANCE_LOWER_BOUND"
NO_SEPARATOR = "NO_CERTIFIED_SEPARATOR"


def progress(stage, **fields):
    print(json.dumps({"event": "SEPARATING_VARIANCE_PROGRESS", "stage": stage, **fields}), flush=True)


def _solve_proposal(center, generators, low, high, timeout):
    """Untrusted max separation with ||y||_1<=1; all generators enter support.

    Let z=D(c+G midpoint), H_i=(D g_i)*halfwidth_i. Minimize
    -z.y+sum(s_i), subject to +/-H_i.y<=s_i, +/-y_j<=a_j,
    sum(a_j)<=1. Only numerical proposal copies use rounded differences.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack, eye
    started = time.monotonic()
    center, generators, low, high = map(np.asarray, (center, generators, low, high))
    n, d = generators.shape
    m = d - 1
    differences = generators[:, :m] - generators[:, -1:]
    midpoint, halfwidth = low / 2 + high / 2, high / 2 - low / 2
    z = center[:m] - center[-1] + midpoint @ differences
    H = csr_matrix(differences * halfwidth[:, None])
    zi, za, zg = csr_matrix((n, m)), csr_matrix((m, m)), csr_matrix((m, n))
    I, S = eye(m, format="csr"), eye(n, format="csr")
    matrix = vstack([hstack([H, zi, -S]), hstack([-H, zi, -S]),
                     hstack([I, -I, zg]), hstack([-I, -I, zg]),
                     coo_matrix((np.ones(m), (np.zeros(m, dtype=int), np.arange(m, 2*m))),
                                shape=(1, 2*m+n))], format="csr")
    objective = np.r_[-z, np.zeros(m), np.ones(n)]
    if not np.isfinite(matrix.data).all() or not np.isfinite(objective).all():
        return {"y": None, "reason": "NONFINITE_NUMERICAL_PROPOSAL_COPY"}
    result = linprog(objective, A_ub=matrix, b_ub=np.r_[np.zeros(matrix.shape[0]-1), 1.],
                     bounds=[(None, None)]*m + [(0., None)]*(m+n), method="highs",
                     options={"time_limit": max(.001, timeout-(time.monotonic()-started))})
    point = getattr(result, "x", None)
    usable = point is not None and len(point) == len(objective) and np.isfinite(point).all()
    return {"y": point[:m].tolist() if usable else None,
            "solver": "scipy.optimize.linprog/highs", "solver_status": int(result.status),
            "solver_message": str(result.message), "solver_success_has_no_proof_authority": bool(result.success),
            "objective_has_no_proof_authority": float(result.fun) if usable and result.fun is not None else None,
            "column_count": len(objective), "row_count": matrix.shape[0], "nnz": matrix.nnz,
            "reason": None if usable else "NO_FINITE_PROPOSED_DIRECTION"}


def _proposal_worker(connection, operands, timeout):
    try:
        connection.send(("result", _solve_proposal(*operands, timeout)))
    except Exception as error:
        connection.send(("error", error))
    finally:
        connection.close()


def bounded_proposal(operands, timeout=60.):
    if not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("numerical proposal timeout must be in (0,60] seconds")
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    worker = context.Process(target=_proposal_worker, args=(child, operands, timeout))
    started = time.monotonic()
    try:
        worker.start()
        child.close()
        if parent.poll(max(0., timeout-(time.monotonic()-started))):
            kind, result = parent.recv()
            if time.monotonic()-started <= timeout:
                if kind == "error":
                    raise result
                return {**result, "proposal_seconds": time.monotonic()-started,
                        "proposal_timeout_seconds": timeout}
        return {"y": None, "reason": "NUMERICAL_PROPOSAL_TIMEOUT",
                "proposal_seconds": time.monotonic()-started, "proposal_timeout_seconds": timeout}
    finally:
        parent.close()
        child.close()
        if worker.is_alive():
            worker.terminate()
        worker.join(.1)
        if worker.is_alive():
            worker.kill()
            worker.join(.1)


def _atomic_json(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = {**value, "record_sha256": C.canonical(value)}
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)
    return payload


def load_operands(manifest):
    state, authenticated = CAP.verify_capture(manifest)
    if authenticated["artifact_sha256"] != EXPECTED_ARTIFACT_SHA:
        raise RuntimeError("token-4 captured artifact differs from authenticated ISIS artifact")
    token = CAP.TOKEN_INDEX
    rows = state["weights"][:, token, :].tolist()
    operands = (rows[0], rows[1:], state["range_low"].tolist(),
                state["range_high"].tolist(), list(state["proof"]["ids"]))
    # Bind full capture hashes/identity, plus the exact token view separately.
    binding = {**authenticated, "state_hashes": CAP.component_hashes(state)}
    return operands, binding


def execute(manifest, output, proposal_timeout=60., certificate_path=None):
    started = time.monotonic()
    operands, authenticated = load_operands(manifest)
    protected = {Path(authenticated[k+"_path"]).resolve() for k in ("manifest", "artifact", "result")}
    output = Path(output)
    planned_certificate = output.with_suffix(".certificate.json")
    if (output.resolve() in protected or planned_certificate.resolve() in protected
            or (certificate_path is not None and output.resolve() == Path(certificate_path).resolve())
            or output.resolve() == planned_certificate.resolve()):
        raise RuntimeError("separator outputs must not overwrite authenticated inputs/certificate")
    progress("capture_authenticated", generator_count=len(operands[1]), token_index=4,
             elapsed_seconds=time.monotonic()-started)
    proposal = None
    if certificate_path is not None:
        certificate = C.verified_json(certificate_path)["certificate"]
    else:
        progress("numerical_proposal_start", wall_clock_limit_seconds=proposal_timeout)
        proposal = bounded_proposal(operands[:4], proposal_timeout)
        progress("numerical_proposal_complete", **{k:v for k,v in proposal.items() if k != "y"})
        y = proposal.get("y")
        if y is not None and (len(y) != 127 or any(not math.isfinite(x) for x in y)):
            raise RuntimeError("numerical proposed direction dimension/finite values differ")
        progress("exact_support_replay_start", generator_count=len(operands[1]))
        certificate = None if y is None or not any(y) else CHECK.construct_certificate(
            *operands, [Fraction.from_float(x) for x in y], authenticated)
    checked = None
    certificate_file = None
    replay_started = time.monotonic()
    if certificate is not None:
        # Persist, reread and independently recompute all exact scalar claims.
        certificate_file = Path(output).with_suffix(".certificate.json")
        if certificate_path is not None:
            certificate_file = Path(certificate_path)
        else:
            _atomic_json(certificate_file, {"certificate": certificate})
        checked = CHECK.verify_certificate(*operands, authenticated,
                                           C.verified_json(certificate_file)["certificate"])
    progress("independent_certificate_replay_complete", verified=checked is not None,
             elapsed_seconds=time.monotonic()-replay_started)
    # Fail closed if capture files changed during proposal or replay.
    for prefix in ("manifest", "artifact", "result"):
        if C.sha256(Path(authenticated[prefix+"_path"])) != authenticated[prefix+"_sha256"]:
            raise RuntimeError("authenticated capture changed during separator execution")
    bound = CHECK.read_rational(checked["variance_lower"]) if checked else None
    report = {"schema": "CORET_TOKEN4_SEPARATING_VARIANCE_REPORT_V1",
              "property_id": CAP.PROPERTY_ID, "stage": CAP.STAGE, "token_index": 4,
              "generator_count": len(operands[1]), "all_generators_loaded": True,
              "all_generators_included": checked is not None,
              "authenticated_capture": authenticated, "proposal": proposal,
              "final_status": CERTIFIED if checked else NO_SEPARATOR,
              "reason": None if checked else (proposal or {}).get("reason") or "NO_EXACT_STRICT_SEPARATOR",
              "exact_check": checked,
              "variance_lower_binary64_downward": CHECK.downward_binary64(bound) if checked else None,
              "certificate_path": str(certificate_file.resolve()) if checked else None,
              "certificate_file_sha256": C.sha256(certificate_file) if checked else None,
              "runtime_seconds": time.monotonic()-started,
              "scientific_queries": 0, "bound_calls": 0, "production_repair_enabled": False}
    progress("separator_complete", final_status=report["final_status"],
             runtime_seconds=report["runtime_seconds"])
    return _atomic_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--proposal-timeout-seconds", default=60., type=float)
    parser.add_argument("--verify-certificate", type=Path, help="Replay only; do not call a numerical solver")
    args = parser.parse_args()
    print(json.dumps(execute(args.capture_manifest, args.output, args.proposal_timeout_seconds,
                             args.verify_certificate), indent=2), flush=True)


if __name__ == "__main__":
    main()
