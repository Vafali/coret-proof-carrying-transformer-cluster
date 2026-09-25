#!/usr/bin/env python3
"""Exclusive reference freeze and read-only performance audit. No bound API."""
import argparse,ast,hashlib,json,math,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_hab/results/coret_sound_reference_v1_20260916'
MANIFEST=OUT/'coret_sound_reference_manifest_v1.json'
E192=ROOT/'research_hab/results/coret_e192_numerical_soundness_v3_20260916/coret_e192_numerical_soundness_regression_result_v3.json'
E096=ROOT/'research_hab/results/coret_e096_numerical_soundness_v1_20260916/coret_e096_numerical_soundness_regression_result_v2.json'
PARENT=E192.parent/'coret_e192_numerical_soundness_manifest_v3.json'
CORE=('coret_fused_relational_layernorm.py','coret_fused_relational_layernorm_v2_fast.py','coret_sln_cbce.py',
      'coret_numerical_soundness_v1.py','coret_numerical_soundness_v2.py','coret_numerical_soundness_v3.py',
      'coret_streaming_softmax_v3.py','coret_certified_softmax_exp_v1.py','coret_softmax_exp_forensic_v1.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def verified(path,key='record_sha256'):
    d=json.loads(Path(path).read_text());p=dict(d);s=p.pop(key)
    if canonical(p)!=s:raise RuntimeError(f'canonical identity mismatch: {path}')
    return d
def write(path,d,key='record_sha256'):
    d=dict(d);d[key]=canonical(d);Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f:json.dump(d,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(Path(path).relative_to(ROOT)),sha256=d[key],scientific_query_count=0)),flush=True)
    return d
def policy():
    node=ast.parse((ROOT/'research_hab/coret_fused_relational_layernorm_v2_fast.py').read_text())
    c=next(n for n in node.body if isinstance(n,ast.ClassDef) and n.name=='FastPolicy')
    def literal(n):
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div):return literal(n.left)/literal(n.right)
        return ast.literal_eval(n)
    return {n.target.id:literal(n.value) for n in c.body if isinstance(n,ast.AnnAssign) and n.value is not None}

def freeze():
    parent=verified(PARENT,'canonical_manifest_sha256')
    records={}
    for p in (E096,E192):
        d=verified(p)
        if d['terminal_status']!='COMPLETE' or d['containment_status']!='NO_CONTAINMENT_FAILURE_OBSERVED' or d['first_containment_failure'] is not None or d['scientific_diagnostic_query_count']!=1:
            raise RuntimeError(f'accepted soundness evidence invalid: {p}')
        if not all(x['nominal_contained'] for x in d['containment_events']):raise RuntimeError('persisted containment failure')
        records[str(p.relative_to(ROOT))]=dict(record_sha256=d['record_sha256'],file_sha256=sha(p))
    sources={str((ROOT/'research_hab'/s).relative_to(ROOT)):sha(ROOT/'research_hab'/s) for s in CORE}
    for path,h in parent['source_hashes'].items():
        if sha(ROOT/path)!=h:raise RuntimeError(f'V3 pinned source changed: {path}')
        sources[path]=h
    tests=sorted((ROOT/'research_hab/tests').glob('test_coret_*'))
    relevant=[p for p in tests if any(t in p.name for t in ('fused_relational','numerical','certified_softmax_exp','streaming','containment_trace')) and p.suffix=='.py']
    native=ROOT/'research_hab/workspaces/coret_second_checkpoint_training_v1/Robustness-Verification-for-Transformers'
    tree={str(p.relative_to(native)):sha(p) for p in sorted(native.rglob('*.py'))}
    engine=verified(ROOT/'research_hab/results/coret_v2_fast_generalization_panel_v1_20260916/coret_v2_fast_generalization_panel_manifest_v1.json','canonical_manifest_sha256')
    write(MANIFEST,dict(schema_version=1,engine_id='CoReT-Sound-v1',classification='NUMERICAL_SOUNDNESS_REPAIR_ACCEPTED',
        scientific_fallback=True,source_hashes=sources,policy=policy(),reference_evidence=records,
        parent_V3_manifest=dict(path=str(PARENT.relative_to(ROOT)),canonical_sha256=parent['canonical_manifest_sha256'],file_sha256=sha(PARENT)),
        deterministic_test_hashes={str(p.relative_to(ROOT)):sha(p) for p in relevant},
        DeepT_revision=parent['DeepT_revision'],native_source_tree_hashes=tree,native_source_tree_sha256=canonical(tree),
        environment=parent['environment'],numerical_policy=parent['numerical_policy'],
        checkpoint_sha256=parent['checkpoint_sha256'],model_config_sha256=engine['model_config_sha256'],
        dataset_sha256=engine['dataset_sha256'],tokenizer_vocab_sha256=engine['tokenizer_vocab_sha256'],
        perturbation=dict(norm='Linf',one_token=True,placement='pre-embedding-LayerNorm word embedding; fixed position/type embeddings',source_dimension=128),
        LayerNorm_epsilon=1e-12,D_DN_enabled=False,scientific_query_count_during_freeze=0,
        freeze_tool_sha256=sha(__file__)),key='canonical_manifest_sha256')

def validate(path=MANIFEST):
    d=verified(path,'canonical_manifest_sha256')
    for f,h in {**d['source_hashes'],**d['deterministic_test_hashes']}.items():
        if sha(ROOT/f)!=h:raise RuntimeError(f'reference file changed: {f}')
    for f,e in d['reference_evidence'].items():
        if sha(ROOT/f)!=e['file_sha256']:raise RuntimeError('accepted evidence changed')
    return d

def audit():
    validate();d=verified(E192);ln=d['V2_telemetry'];sm=d['streaming_softmax_telemetry']
    totals={k:sum(l['timers_seconds'][k] for l in ln) for k in ln[0]['timers_seconds']}
    unavailable=dict(seconds=None,reason='No disjoint timer/event timestamps recorded in V3; cannot reconstruct from final journal mtime')
    blocks=[];calls=0;difference_calls=0
    for b,s in enumerate(sm):
        G=s['logical_shape'][1]-1;groups=math.ceil(G/128);physical=s['generator_chunk_size']
        inner=sum(math.ceil(min(128,G-g)/physical) for g in range(0,G,128));tiles=s['tile_count']
        calls+=2*tiles*inner;difference_calls+=tiles*(1+2*inner)
        blocks.append(dict(block=b,exact_whole_block_wall=unavailable,
            recorded_softmax_seconds=s['seconds'],recorded_two_LN_seconds=sum(ln[1+2*b+i]['timers_seconds']['total_LN'] for i in (0,1)),
            generator_count=G,row_anchor_denominator_tiles=tiles,generator_groups_per_tile=groups,
            input_and_transformed_generator_loop_iterations=2*tiles*inner,
            difference_reconstructions=tiles*(1+2*inner)))
    write(OUT/'coret_e192_v3_performance_audit_v1.json',dict(schema_version=1,scientific_query_count=0,
        source_result_record_sha256=d['record_sha256'],runtime_seconds=d['runtime_seconds'],
        LayerNorm_wall_seconds=totals['total_LN'],softmax_process_values_wall_seconds=sum(s['seconds'] for s in sm),
        outside_LN_and_streaming_seconds=d['runtime_seconds']-totals['total_LN']-sum(s['seconds'] for s in sm),
        timer_semantics='LayerNorm categories nested/inclusive; do not sum support + centering/Jacobian/remainders',
        measured_inclusive_categories=totals,top_3_inclusive_categories=sorted(((k,v) for k,v in totals.items() if k not in ('total_LN',)),key=lambda x:-x[1])[:3],
        each_LN=[dict(name=l['name'],seconds=l['timers_seconds']['total_LN'],incoming_generators=l['incoming_generator_count'],calls=l['timer_call_counts']) for l in ln],
        blocks=blocks,total_softmax_tiles=sum(s['tile_count'] for s in sm),
        softmax_generator_loop_iterations=calls,softmax_score_difference_reconstructions=difference_calls,
        authoritative_concretization_calls=d['numerical_concretization_count'],certified_generator_reduction_calls=d['certified_reduction_count'],
        unavailable_wall_categories={k:unavailable for k in ('score_difference_only','certified_exp_only','denominator_only','global_support_only','generator_tiling_only','state_tiling_only','reciprocal_equality','generator_reduction_total','observer_overhead')},
        kernel_launch_count=dict(value=None,reason='No CUDA profiler/counter in V3; cannot infer exact launches from Python source'),
        implementation_findings=['GEN128 Python loops repeat support reductions/products/casts/nextafter',
            'float32 score differences reconstructed in input and transformed passes',
            'softmax per-window total buffers initialized once per generator group',
            'LayerNorm support timer dominates; no per-block timestamps or observer timer exist']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','validate','audit']);a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='audit':audit()
    else:validate();print('REFERENCE_IDENTITY_PASS; scientific_query_count=0')
