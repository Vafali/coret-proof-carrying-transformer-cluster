from __future__ import annotations
import copy, hashlib, json, math, struct, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research_hab"))
import coret_trace_witness_v1 as producer
import coret_trace_checker_v1 as checker


def reseal(value):
    value.pop("canonical_sha256",None);producer.seal(value)


def rewrite(root,graph):
    producer.seal(graph);(root/"trace.json").write_bytes(producer.canonical_bytes(graph)+b"\n")


def fixture(tmp_path):
    states,graph=producer.build_minimal_trace(tmp_path)
    assert checker.check_trace(tmp_path)["accepted"]
    return states,graph


def test_minimal_trace_chain_pass(tmp_path): fixture(tmp_path)


def test_instrumentation_bitwise_transparent(tmp_path):
    instrumented,_=producer.build_minimal_trace(tmp_path,instrument=True)
    plain,_=producer.build_minimal_trace(tmp_path/"plain",instrument=False)
    def bits(states):
        return [(
            b"".join(struct.pack("<f", value)
                     for row in state.weights for value in row),
            state.generator_ids, state.ranges)
            for state in states]
    assert bits(instrumented)==bits(plain)


def test_blob_one_bit_mutation_rejects(tmp_path):
    _,g=fixture(tmp_path);rec=g["state_records"][0]["producer_tensor_content_ids"]["weights"];path=tmp_path/rec["relative_path"]
    raw=bytearray(path.read_bytes());raw[0]^=1;path.write_bytes(raw)
    with pytest.raises(AssertionError,match="SHA-256"):checker.check_trace(tmp_path)


def test_state_id_substitution_rejects(tmp_path):
    _,g=fixture(tmp_path);g["transition_records"][1]["input_state_ids"]=["s0_source"];reseal(g["transition_records"][1]);rewrite(tmp_path,g)
    with pytest.raises(AssertionError,match="state-ID"):checker.check_trace(tmp_path)


def test_reordered_generator_mapping_rejects(tmp_path):
    _,g=fixture(tmp_path);m=g["state_records"][1]["ghost_state_linkage"]["ordered_native_to_ghost"];m.reverse();reseal(g["state_records"][1]);rewrite(tmp_path,g)
    with pytest.raises(AssertionError,match="mapping/order"):checker.check_trace(tmp_path)


def test_incorrect_fresh_allocation_rejects(tmp_path):
    _,g=fixture(tmp_path);t=g["transition_records"][1];t["tau_k"]["sqrt_active_flat_indices"]=[];reseal(t);rewrite(tmp_path,g)
    with pytest.raises(AssertionError,match="allocation predicate"):checker.check_trace(tmp_path)


def test_narrowed_layernorm_sidecar_rejects(tmp_path):
    _,g=fixture(tmp_path);rec=g["state_records"][2]["producer_tensor_content_ids"]["numerical_radius"];path=tmp_path/rec["relative_path"]
    vals=[x[0] for x in struct.iter_unpack("<f",path.read_bytes())];vals=[0.0]*len(vals);raw=b"".join(struct.pack("<f",x) for x in vals)
    digest=hashlib.sha256(raw).hexdigest();new=tmp_path/"blobs"/f"{digest}.float32.le.bin";new.write_bytes(raw)
    rec.update({"sha256":digest,"relative_path":f"blobs/{new.name}","byte_count":len(raw)});reseal(rec);reseal(g["state_records"][2]);rewrite(tmp_path,g)
    with pytest.raises(AssertionError,match="sidecar too narrow"):checker.check_trace(tmp_path)


def test_incorrect_branch_rejects(tmp_path):
    _,g=fixture(tmp_path);t=g["transition_records"][2];t["tau_k"]["coordinate_cases"][0]="inactive";reseal(t);rewrite(tmp_path,g)
    with pytest.raises(AssertionError,match="branch witness"):checker.check_trace(tmp_path)


def test_incorrect_layernorm_branch_rejects(tmp_path):
    _,g=fixture(tmp_path);t=g["transition_records"][1];t["tau_k"]["branch"]="unwitnessed_branch";reseal(t);rewrite(tmp_path,g)
    with pytest.raises(AssertionError,match="LayerNorm branch witness"):checker.check_trace(tmp_path)


def test_dropped_numerical_source_rejects(tmp_path):
    _,g=fixture(tmp_path);g["state_records"][2]["producer_tensor_content_ids"].pop("numerical_radius");reseal(g["state_records"][2]);rewrite(tmp_path,g)
    with pytest.raises(KeyError):checker.check_trace(tmp_path)


def test_classifier_coefficient_mutation_rejects(tmp_path):
    _,g=fixture(tmp_path);t=g["transition_records"][3];t["operator_witness"]["matrix_hex"][0][0]=float(1.5).hex();reseal(t);rewrite(tmp_path,g)
    with pytest.raises(AssertionError,match="sidecar too narrow"):checker.check_trace(tmp_path)


def test_one_ulp_optimistic_final_margin_rejects(tmp_path):
    _,g=fixture(tmp_path);f=g["final_property_record"];x=float.fromhex(f["claimed_lower_hex"]);f["claimed_lower_hex"]=math.nextafter(x,math.inf).hex();reseal(f);rewrite(tmp_path,g)
    with pytest.raises(AssertionError,match="optimistic final margin"):checker.check_trace(tmp_path)


def test_content_deduplication_and_schema(tmp_path):
    _,g=fixture(tmp_path);refs=[]
    for s in g["state_records"]:refs.extend(x["sha256"] for x in s["producer_tensor_content_ids"].values())
    assert len(list((tmp_path/"blobs").iterdir()))==len(set(refs))
    assert g["schema"]==producer.SCHEMA and len(g["transition_records"])==4
