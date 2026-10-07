#!/usr/bin/env python3
"""Token-8 PRE-reduction separator: existing proposal, independent exact replay."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import cluster_common as C
import capture_s003_b2_ffn_residual_pre_reduction_v1 as PRE
import certify_block2_output_separating_variance_v1 as S
import separating_variance_checker_v1 as CHECK

CAUSAL = "REDUCTION_PRECISION_LOSS_CAUSAL"
NOT_ISOLATED = "REDUCTION_NOT_YET_CAUSALLY_ISOLATED"


def load_operands(manifest):
    state, authenticated = PRE.verify_capture(manifest)
    # No row is omitted, including numerical/zero generators and asymmetric ranges.
    rows = state["weights"][:, PRE.TOKEN_INDEX, :].tolist()
    return (rows[0], rows[1:], state["range_low"].tolist(), state["range_high"].tolist(),
            list(state["proof"]["ids"])), authenticated


def execute(manifest, output, proposal_timeout=60., certificate_path=None):
    started = time.monotonic()
    operands, authenticated = load_operands(manifest)
    output = Path(output).resolve()
    planned_certificate = output.with_suffix(".certificate.json")
    protected = {Path(binding[prefix + "_path"]).resolve()
                 for binding in (authenticated, authenticated["post_capture"])
                 for prefix in ("manifest", "artifact", "result")}
    if (output in protected or planned_certificate in protected or output == planned_certificate or
            output.exists() or (certificate_path is None and planned_certificate.exists()) or
            (certificate_path is not None and output == Path(certificate_path).resolve())):
        raise RuntimeError("separator requires new outputs; cannot overwrite authenticated comparison inputs")
    S.progress("pre_reduction_capture_authenticated", token_index=PRE.TOKEN_INDEX,
               generator_count=len(operands[1]), elapsed_seconds=time.monotonic()-started)
    proposal = None
    if certificate_path is not None:
        certificate_file = Path(certificate_path).resolve()
        certificate = C.verified_json(certificate_file)["certificate"]
    else:
        S.progress("numerical_proposal_start", wall_clock_limit_seconds=proposal_timeout)
        # Exactly the established bounded LP. No primal zero-variance solve,
        # least-squares, Bareiss, Farkas, or additional proposal method.
        proposal = S.bounded_proposal(operands[:4], proposal_timeout)
        S.progress("numerical_proposal_complete", **{k: v for k, v in proposal.items() if k != "y"})
        y = proposal.get("y")
        if y is not None and (len(y) != 127 or any(type(x) is not float or not math.isfinite(x) for x in y)):
            raise RuntimeError("separator proposed direction shape/finite values differ")
        S.progress("exact_support_replay_start", generator_count=len(operands[1]))
        certificate = None if y is None or not any(y) else CHECK.construct_certificate(
            *operands, [Fraction.from_float(x) for x in y], authenticated)
        certificate_file = planned_certificate if certificate is not None else None
        if certificate is not None:
            S._atomic_json(certificate_file, {"certificate": certificate})
    checked = None
    if certificate is not None:
        checked = CHECK.verify_certificate(*operands, authenticated,
                                           C.verified_json(certificate_file)["certificate"])
    # Authentication is not replaced by solver status or a persisted decision flag.
    for binding in (authenticated, authenticated["post_capture"]):
        for prefix in ("manifest", "artifact", "result"):
            if C.sha256(Path(binding[prefix + "_path"])) != binding[prefix + "_sha256"]:
                raise RuntimeError("authenticated pre/post input changed during separator execution")
    if PRE.identity() != authenticated["identity"]:
        raise RuntimeError("authenticated source revision changed during separator execution")
    post_failed = authenticated["post_no_separator_evidence"]
    status = S.CERTIFIED if checked else S.NO_SEPARATOR
    report = {
        "schema": "CORET_S003_PRE_REDUCTION_SEPARATING_VARIANCE_REPORT_V1",
        "property_id": PRE.PROPERTY_ID, "stage": PRE.CAP.STAGE,
        "capture_boundary": PRE.BOUNDARY, "token_index": PRE.TOKEN_INDEX,
        "radius": authenticated["identity"]["tested_radius"],
        "radius_hex": authenticated["identity"]["tested_radius_hex"],
        "generator_count": len(operands[1]), "all_generators_loaded": True,
        "all_generators_included": checked is not None,
        "authenticated_capture": authenticated, "proposal": proposal,
        "final_status": status,
        "causal_decision": CAUSAL if checked and post_failed else NOT_ISOLATED,
        "causal_scope": "positive exact pre-reduction bound versus authenticated post-reduction separator failure; not an exact post zero witness",
        "post_separator_status": "NO_CERTIFIED_SEPARATOR" if post_failed else "UNAVAILABLE",
        "reason": None if checked else (proposal or {}).get("reason") or "NO_EXACT_STRICT_SEPARATOR",
        "exact_check": checked,
        "variance_lower_binary64_downward": CHECK.downward_binary64(
            CHECK.read_rational(checked["variance_lower"])) if checked else None,
        "certificate_path": str(certificate_file) if checked else None,
        "certificate_file_sha256": C.sha256(certificate_file) if checked else None,
        "proposal_and_numerical_status_have_no_proof_authority": True,
        "runtime_seconds": time.monotonic()-started,
        "source_hashes": {"adapter_sha256": C.sha256(Path(__file__)),
                          "independent_checker_sha256": C.sha256(Path(CHECK.__file__)),
                          "existing_proposal_sha256": C.sha256(Path(S.__file__))},
        "scientific_queries": 0, "bound_calls": 0, "production_repair_enabled": False,
    }
    S.progress("pre_reduction_separator_complete", final_status=status,
               causal_decision=report["causal_decision"], runtime_seconds=report["runtime_seconds"])
    return S._atomic_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposal-timeout-seconds", type=float, default=60.)
    parser.add_argument("--verify-certificate", type=Path,
                        help="Replay existing separator certificate only; no numerical solve")
    args = parser.parse_args()
    print(json.dumps(execute(args.capture_manifest, args.output, args.proposal_timeout_seconds,
                             args.verify_certificate), indent=2), flush=True)


if __name__ == "__main__":
    main()
