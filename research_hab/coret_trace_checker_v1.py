#!/usr/bin/env python3
"""Independent standard-library checker for CORET_TRACE_WITNESS_V1."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path

SCHEMA = "CORET_TRACE_WITNESS_V1"
BLOB_SCHEMA = "CORET_TRACE_BLOB_V1"
PINNED_REVISION = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verify_seal(value, label):
    claimed = value.get("canonical_sha256"); body = dict(value); body.pop("canonical_sha256", None)
    if claimed != hashlib.sha256(canonical_bytes(body)).hexdigest():
        raise AssertionError(f"{label} canonical hash mismatch")


def dec(value):
    if isinstance(value, Decimal): return value
    if isinstance(value, float):
        n, d = value.as_integer_ratio()
        with localcontext() as ctx:
            ctx.prec = 120
            return Decimal(n) / Decimal(d)
    return Decimal(value)


class I:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = dec(lo), dec(lo if hi is None else hi)
        if self.lo > self.hi: raise AssertionError("reversed interval")
    def __add__(self, other):
        other = as_i(other)
        with localcontext() as c: c.prec=100; c.rounding=ROUND_FLOOR; lo=self.lo+other.lo
        with localcontext() as c: c.prec=100; c.rounding=ROUND_CEILING; hi=self.hi+other.hi
        return I(lo,hi)
    def __neg__(self): return I(-self.hi,-self.lo)
    def __sub__(self,other): return self + (-as_i(other))
    def __mul__(self,other):
        other=as_i(other)
        with localcontext() as c:
            c.prec=100; c.rounding=ROUND_FLOOR
            lows=[a*b for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        with localcontext() as c:
            c.prec=100; c.rounding=ROUND_CEILING
            highs=[a*b for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        return I(min(lows),max(highs))
    def reciprocal(self):
        if self.lo <= 0 <= self.hi: raise AssertionError("reciprocal domain")
        with localcontext() as c: c.prec=100; c.rounding=ROUND_FLOOR; vals=[Decimal(1)/self.lo,Decimal(1)/self.hi]
        with localcontext() as c: c.prec=100; c.rounding=ROUND_CEILING; vals2=[Decimal(1)/self.lo,Decimal(1)/self.hi]
        return I(min(vals),max(vals2))
    def __truediv__(self,other): return self*as_i(other).reciprocal()
    def sqrt(self):
        if self.lo < 0: raise AssertionError("sqrt domain")
        with localcontext() as c: c.prec=100; c.rounding=ROUND_FLOOR; lo=self.lo.sqrt()
        with localcontext() as c: c.prec=100; c.rounding=ROUND_CEILING; hi=self.hi.sqrt()
        return I(lo,hi)
def as_i(v): return v if isinstance(v,I) else I(v)


def load_blob(root, record):
    verify_seal(record, "blob")
    if record.get("schema") != BLOB_SCHEMA or record.get("dtype") != "float32": raise AssertionError("blob schema")
    rel=Path(record["relative_path"])
    if rel.is_absolute() or ".." in rel.parts: raise AssertionError("blob path escape")
    path=(root/rel).resolve()
    try: path.relative_to(root.resolve())
    except ValueError as e: raise AssertionError("blob path escape") from e
    raw=path.read_bytes()
    if len(raw)!=record["byte_count"] or hashlib.sha256(raw).hexdigest()!=record["sha256"]: raise AssertionError("blob SHA-256 mismatch")
    shape=tuple(record["shape"]); values=[x[0] for x in struct.iter_unpack("<f",raw)]
    if math.prod(shape)!=len(values) or len(shape)!=2: raise AssertionError("blob shape")
    return [values[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]


def load_state(root, record):
    verify_seal(record, record.get("state_id","state"))
    weights=load_blob(root,record["producer_tensor_content_ids"]["weights"])
    numerical=load_blob(root,record["producer_tensor_content_ids"]["numerical_radius"])
    ids=record["generator_ids"]; ranges=[(float.fromhex(a),float.fromhex(b)) for a,b in record["explicit_ranges"]]
    if len(weights)!=1+len(ids) or len(ranges)!=len(ids) or len(numerical)!=len(weights): raise AssertionError("state dimensions")
    mapping=record["ghost_state_linkage"]["ordered_native_to_ghost"]
    if [m["native_row"] for m in mapping]!=list(range(1,len(ids)+1)) or [m["ghost_id"] for m in mapping]!=ids: raise AssertionError("generator mapping/order mismatch")
    if len(set(ids))!=len(ids): raise AssertionError("duplicate generator ID")
    if any(r<0 or not math.isfinite(r) for row in numerical for r in row): raise AssertionError("invalid numerical source")
    return {"id":record["state_id"],"weights":weights,"numerical":numerical,"ids":ids,"ranges":ranges}


def concretize_exact(coeffs,ranges):
    lo=[x for x in coeffs[0]]; hi=[x for x in coeffs[0]]
    for row,(a,b) in zip(coeffs[1:],ranges):
        for j,x in enumerate(row):
            p=x*I(a,b); lo[j]=lo[j]+I(p.lo); hi[j]=hi[j]+I(p.hi)
    return lo,hi


def relation(state, exact):
    if len(exact)!=len(state["weights"]) or any(len(a)!=len(b) for a,b in zip(exact,state["weights"])): raise AssertionError("exact/native topology mismatch")
    for r,(expected,stored,radius) in enumerate(zip(exact,state["weights"],state["numerical"])):
        for c,(e,s,n) in enumerate(zip(expected,stored,radius)):
            represented=I(dec(s)-dec(n),dec(s)+dec(n))
            if represented.lo>e.lo or represented.hi<e.hi: raise AssertionError(f"numerical sidecar too narrow at {r},{c}")


def affine_exact(coeffs,matrix,bias):
    out=[]
    for r,row in enumerate(coeffs):
        vals=[]
        for o,w in enumerate(matrix):
            acc=I(bias[o] if r==0 else 0)
            for x,k in zip(row,w): acc=acc+x*I(k)
            vals.append(acc)
        out.append(vals)
    return out


def layernorm_exact(coeffs,ids,gamma,beta,tau):
    if (tau.get("mode") != "standard"
            or tau.get("branch") != "positive_variance_standard_layernorm"):
        raise AssertionError("LayerNorm branch witness mismatch")
    d=len(coeffs[0]); g=len(ids); centered=[]
    for row in coeffs:
        mean=I(0)
        for x in row: mean=mean+x
        mean=mean/I(d); centered.append([x-mean for x in row])
    variance=[[I(0) for _ in range(d)] for _ in range(2+g)]
    v=I(0)
    for x in centered[0]: v=v+x*x
    v=v/I(d)+I(float.fromhex(tau["epsilon_hex"])); variance[0]=[v]*d
    for k in range(g):
        v=I(0)
        for c,a in zip(centered[0],centered[1+k]): v=v+c*a
        variance[1+k]=[I(2)*v/I(d)]*d
    width=I(0)
    for j in range(d):
        support=I(0)
        for k in range(g): support=support+I(0,max(abs(centered[1+k][j].lo),abs(centered[1+k][j].hi)))
        width=width+support*support
    variance[-1]=[width/I(d)]*d
    current_ids=ids+[tau["variance_fresh_ids"][0]]
    ranges=[(-1.0,1.0)]*len(current_ids)
    def unary(rows,ids,ranges,active,kind,prefix):
        lo,hi=concretize_exact(rows,ranges)
        derived=[j for j in range(d) if lo[j].lo!=hi[j].hi]
        if derived!=active: raise AssertionError(f"{kind} fresh allocation predicate mismatch")
        out=[[I(0) for _ in range(d)] for _ in rows]; fresh=[I(0) for _ in range(d)]
        for j in range(d):
            l,u=lo[j],hi[j]
            # Bounds must be point intervals here; earlier coefficient uncertainty is retained within I.
            if l.lo<=0: raise AssertionError(f"{kind} domain")
            if j not in active:
                out[0][j]=l.sqrt() if kind=="sqrt" else l.reciprocal(); continue
            if kind=="sqrt":
                sl, su=l.sqrt(),u.sqrt(); t=((u-l)/(I(2)*(su-sl))); t=t*t
                lam=(su-sl)/(u-l); x=sl-lam*l; ft=t.sqrt()
                const=I(.5)*(ft-lam*t+x); fresh[j]=I(.5)*(lam*t-ft+x)
            else:
                lam=-I(1)/(u*u); bottom=I(1)/u-lam*u; top=I(1)/l-lam*l
                const=I(.5)*(top+bottom); fresh[j]=I(.5)*(top-bottom)
            out[0][j]=lam*rows[0][j]+const
            for r in range(1,len(rows)): out[r][j]=lam*rows[r][j]
        for j in active:
            row=[I(0) for _ in range(d)]; row[j]=fresh[j]; out.append(row); ids.append(f"{prefix}.{j:06d}"); ranges.append((-1.,1.))
        return out,ids,ranges
    variance,current_ids,ranges=unary(variance,current_ids,ranges,tau["sqrt_active_flat_indices"],"sqrt","layernorm0.sqrt")
    reciprocal,current_ids,ranges=unary(variance,current_ids,ranges,tau["reciprocal_active_flat_indices"],"reciprocal","layernorm0.reciprocal")
    centered += [[I(0) for _ in range(d)] for _ in range(len(reciprocal)-len(centered))]
    product=[[I(0) for _ in range(d)] for _ in reciprocal]
    for j in range(d):
        product[0][j]=centered[0][j]*reciprocal[0][j]
        for r in range(1,len(reciprocal)): product[r][j]=centered[0][j]*reciprocal[r][j]+reciprocal[0][j]*centered[r][j]
    for j in tau["product_active_flat_indices"]:
        left=I(0);right=I(0)
        for r in range(1,len(reciprocal)):
            left=left+I(0,max(abs(centered[r][j].lo),abs(centered[r][j].hi)))
            right=right+I(0,max(abs(reciprocal[r][j].lo),abs(reciprocal[r][j].hi)))
        row=[I(0) for _ in range(d)];row[j]=left*right;product.append(row);current_ids.append(f"layernorm0.product.{j:06d}")
    if current_ids[g:]!=tau["fresh_generator_ids"]:
        raise AssertionError("LayerNorm fresh generator order mismatch")
    for r in range(len(product)):
        for j in range(d): product[r][j]=product[r][j]*I(gamma[j])+(I(beta[j]) if r==0 else I(0))
    return product,current_ids,[(-1.,1.)]*len(current_ids)


def relu_exact(coeffs,ids,ranges,tau):
    lo,hi=concretize_exact(coeffs,ranges); cases=[];active=[];out=[[I(0) for _ in lo] for _ in coeffs];fresh=[]
    for j,(l,u) in enumerate(zip(lo,hi)):
        if l.lo>=0: cases.append("active"); [out[r].__setitem__(j,coeffs[r][j]) for r in range(len(coeffs))]
        elif u.hi<=0: cases.append("inactive")
        else:
            cases.append("crossing");active.append(j);lam=u/(u-l+I(1e-12));delta=I(max((-lam*l).hi,((I(1)-lam)*u).hi))/I(2)
            out[0][j]=lam*coeffs[0][j]+delta
            for r in range(1,len(coeffs)):out[r][j]=lam*coeffs[r][j]
            fresh.append((j,delta))
    if cases!=tau["coordinate_cases"] or active!=tau["active_flat_indices"]: raise AssertionError("ReLU branch witness mismatch")
    for (j,v),identifier in zip(fresh,tau["fresh_generator_ids"]):
        row=[I(0) for _ in lo];row[j]=v;out.append(row);ids.append(identifier);ranges.append((-1.,1.))
    return out,ids,ranges


def check_trace(root):
    root=Path(root).resolve(); graph=json.loads((root/"trace.json").read_text())
    if graph.get("schema")!=SCHEMA: raise AssertionError("trace schema")
    verify_seal(graph,"trace");verify_seal(graph["run_manifest"],"run manifest");verify_seal(graph["source_domain"],"source domain")
    if graph["run_manifest"]["pinned_deept_revision"]!=PINNED_REVISION or graph["run_manifest"]["scientific_query"] is not False: raise AssertionError("run identity")
    domain=graph["source_domain"]
    if domain["p_cli"]!=100 or domain["interpreted_domain"]!="Linf" or domain["source_ranges"]!=[[(-1.).hex(),(1.).hex()]]*2: raise AssertionError("source domain mismatch")
    states={r["state_id"]:load_state(root,r) for r in graph["state_records"]}
    if graph["graph_nodes"]!=list(states): raise AssertionError("state ID/order substitution")
    source=states[graph["graph_nodes"][0]]
    if source["ids"]!=domain["source_symbol_ids"]: raise AssertionError("source IDs")
    exact=[[I(v) for v in row] for row in source["weights"]]; ids=list(source["ids"]);ranges=list(source["ranges"]);relation(source,exact)
    current=source["id"]
    for transition in graph["transition_records"]:
        verify_seal(transition,transition["transition_id"])
        if transition["input_state_ids"]!=[current]: raise AssertionError("transition state-ID substitution")
        output_id=transition["output_state_ids"][0]; output=states[output_id]; family=transition["operator_family"];w=transition["operator_witness"];tau=transition["tau_k"]
        if family in ("affine","classifier_affine"):
            matrix=[[float.fromhex(x) for x in row] for row in w["matrix_hex"]];bias=[float.fromhex(x) for x in w["bias_hex"]]
            if tau["generator_order"]!=ids: raise AssertionError("affine generator order")
            exact=affine_exact(exact,matrix,bias)
        elif family=="LayerNorm":
            exact,ids,ranges=layernorm_exact(exact,ids,[float.fromhex(x) for x in w["gamma_hex"]],[float.fromhex(x) for x in w["beta_hex"]],tau)
        elif family=="ReLU": exact,ids,ranges=relu_exact(exact,ids,ranges,tau)
        else: raise AssertionError("unsupported operator family")
        if output["ids"]!=ids: raise AssertionError("output generator identity/order")
        relation(output,exact);current=output_id
    final=graph["final_property_record"];verify_seal(final,"final property")
    if final["state_id"]!=current or len(exact)!=len(states[current]["weights"]) or len(exact[0])!=1: raise AssertionError("final state")
    lower=exact[0][0]
    for row,(lo,hi) in zip(exact[1:],ranges):
        contribution=row[0]*I(lo,hi);lower=lower+I(contribution.lo)
    claimed=dec(float.fromhex(final["claimed_lower_hex"]))
    if claimed>lower.lo: raise AssertionError("optimistic final margin")
    if final["certified"] is not True or lower.lo<=0: raise AssertionError("final margin not positive")
    return {"accepted":True,"states":len(states),"transitions":len(graph["transition_records"]),"independent_lower":str(lower.lo)}
