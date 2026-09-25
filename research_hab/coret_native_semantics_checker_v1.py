#!/usr/bin/env python3
"""Independent structural/equational checker for native-semantic certificates.

This module imports neither DeepT nor the production facade.  The numerical
equation checks operate on explicit fixture arrays supplied by the test/audit
harness; certificate checks enforce native symbol counts and range policies.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

PINNED = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_common(cert: dict[str, Any], family: str):
    _require(cert.get("revision") == PINNED, "pinned revision mismatch")
    _require(cert.get("family") == family, "operator family mismatch")
    _require(cert.get("native_result_unchanged") is True, "native result was replaced")
    _require(cert.get("generic_semantic_remainder_used") is False,
             "generic semantic remainder is forbidden")
    _require(cert["output"]["weights"]["finite"], "nonfinite output")
    _require(cert["output"]["generator_count"] >= 0, "invalid generator count")
    return True


def check_count(cert, expected):
    _require(cert["output"]["generator_count"] == int(expected),
             f"fresh-symbol count mismatch: {cert['output']['generator_count']} != {expected}")


def precise_dot_reference(a, b):
    """Independent scalar-output native precise-dot equations.

    a,b have shape [1+G,D]. Returns center, retained G-vector, fresh radius.
    """
    a = np.asarray(a, dtype=np.longdouble); b = np.asarray(b, dtype=np.longdouble)
    _require(a.shape == b.shape and a.ndim == 2, "precise-dot fixture shape")
    c = np.sum(a[0] * b[0], dtype=np.longdouble)
    linear = np.sum(a[0][None, :] * b[1:] + a[1:] * b[0][None, :], axis=1,
                    dtype=np.longdouble)
    diagonal = np.sum(a[1:] * b[1:], axis=1, dtype=np.longdouble)
    c += np.longdouble(.5) * np.sum(diagonal, dtype=np.longdouble)
    radius = np.longdouble(.5) * np.sum(np.abs(diagonal), dtype=np.longdouble)
    for i in range(a.shape[0] - 1):
        cross = np.sum(a[i + 1] * b[i + 2:], axis=1, dtype=np.longdouble)
        cross += np.sum(a[i + 2:] * b[i + 1], axis=1, dtype=np.longdouble)
        radius += np.sum(np.abs(cross), dtype=np.longdouble)
    return float(c), np.asarray(linear, dtype=np.float64), float(radius)


def relu_reference(center, coeff, lower, upper):
    center=np.asarray(center,dtype=np.float64);coeff=np.asarray(coeff,dtype=np.float64)
    lower=np.asarray(lower,dtype=np.float64);upper=np.asarray(upper,dtype=np.float64)
    overlap=lower*upper<0
    lam=upper/(upper-lower+1e-12)
    delta=np.maximum(-lam*lower,(1-lam)*upper)
    out_center=np.where(overlap,lam*center+delta/2,np.where(lower>=0,center,0))
    out_coeff=np.where(overlap[None],coeff*lam[None],np.where((lower>=0)[None],coeff,0))
    return out_center,out_coeff,delta/2,overlap


def tanh_reference(center, coeff, lower, upper):
    tlo=np.tanh(lower);thi=np.tanh(upper)
    lam=np.minimum(1-tlo*tlo,1-thi*thi);different=lower!=upper
    bias=lam*center+.5*(thi+tlo-lam*(upper+lower))
    fresh=.5*(thi-tlo-lam*(upper-lower))
    return np.where(different,bias,tlo), coeff*lam[None]*different[None], fresh, different


def reduction_reference(weights, remove):
    weights=np.asarray(weights,dtype=np.float64);remove=np.asarray(remove,dtype=np.int64)
    mask=np.ones(weights.shape[0]-1,dtype=bool);mask[remove]=False
    kept=weights[1:][mask];aggregate=np.abs(weights[1:][remove]).sum(axis=0)
    flat=aggregate.size;boxes=np.zeros((flat,)+aggregate.shape,dtype=np.float64)
    boxes.reshape(flat,flat)[np.arange(flat),np.arange(flat)]=aggregate.reshape(-1)
    return np.concatenate((weights[:1],kept,boxes),axis=0)


def check_simplex_coefficients(weights, tolerance=2e-5):
    w=np.asarray(weights,dtype=np.float64)
    summed=w.sum(axis=-1)
    _require(np.max(np.abs(summed[0]-1.0))<=tolerance,"softmax center sum mismatch")
    if w.shape[0]>1:
        _require(np.max(np.abs(summed[1:]))<=tolerance,"softmax coefficient sum mismatch")
    return True


def check_positive_domain(lower, upper):
    lower=np.asarray(lower,dtype=np.float64);upper=np.asarray(upper,dtype=np.float64)
    _require(np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)),"nonfinite domain")
    _require(np.all(lower<=upper),"reversed domain")
    _require(np.all(lower>0),"positive-domain obligation failed")
    return True


def concretize_weights(weights):
    w=np.asarray(weights,dtype=np.float64)
    radius=np.abs(w[1:]).sum(axis=0)
    return w[0]-radius,w[0]+radius


def _append_coordinate_fresh(base, values, mask=None):
    values=np.asarray(values,dtype=np.float64)
    if mask is None:mask=np.ones(values.shape,dtype=bool)
    indices=np.flatnonzero(np.asarray(mask).reshape(-1));fresh=np.zeros((len(indices),)+values.shape,dtype=np.float64)
    if len(indices):fresh.reshape(len(indices),-1)[np.arange(len(indices)),indices]=values.reshape(-1)[indices]
    return np.concatenate((base,fresh),axis=0)


def sqrt_reference(weights):
    w=np.asarray(weights,dtype=np.float64);l,u=concretize_weights(w);_require(np.all(l>0),'sqrt domain')
    different=l!=u;sl=np.sqrt(l);su=np.sqrt(u)
    t=((u-l)/(2*(su-sl)))**2
    lam=(su-sl)/(u-l);x=sl-lam*l
    const=.5*(np.sqrt(t)-lam*t+x);fresh=.5*(lam*t-np.sqrt(t)+x)
    center=np.where(different,lam*w[0]+const,sl);old=w[1:]*lam[None]*different[None]
    return _append_coordinate_fresh(np.concatenate((center[None],old)),fresh,different)


def reciprocal_reference(weights, original=True):
    w=np.asarray(weights,dtype=np.float64);l,u=concretize_weights(w);_require(np.all(l>0),'reciprocal domain')
    different=l!=u
    if original:
        lam=-1/(u*u);b=1/u-lam*u;t=1/l-lam*l
        const=.5*(t+b);fresh=.5*(t-b)
    else:
        mean=(1/u-1/l)/(u-l);critical=np.sqrt(-1/mean);point=np.maximum(critical,u/2+.01)
        lam=-1/(point*point);x=1/l-lam*l
        const=.5*(1/point-lam*point+x);fresh=.5*(lam*point-1/point+x)
    center=np.where(different,lam*w[0]+const,1/l);old=w[1:]*lam[None]*different[None]
    return _append_coordinate_fresh(np.concatenate((center[None],old)),fresh,different)


def multiply_reference(left,right):
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64);g=max(a.shape[0],b.shape[0])-1
    aa=np.zeros((g+1,)+a.shape[1:]);bb=np.zeros((g+1,)+b.shape[1:]);aa[:a.shape[0]]=a;bb[:b.shape[0]]=b
    center=aa[0]*bb[0];linear=aa[0][None]*bb[1:]+bb[0][None]*aa[1:]
    fresh=np.abs(aa[1:]).sum(0)*np.abs(bb[1:]).sum(0)
    return _append_coordinate_fresh(np.concatenate((center[None],linear)),fresh)


def layernorm_reference(weights, gamma, beta, epsilon=1e-12):
    """Pinned p=inf standard-LN equations for a 3-D native zonotope."""
    w=np.asarray(weights,dtype=np.float64);gamma=np.asarray(gamma,dtype=np.float64);beta=np.asarray(beta,dtype=np.float64)
    centered=w-w.mean(axis=-1,keepdims=True);g,l,d=centered.shape[0]-1,*centered.shape[1:]
    variance=np.zeros((g+1+l,l,d),dtype=np.float64)
    variance[0]=np.sum(centered[0]*centered[0],axis=-1,keepdims=True)/d
    variance[1:1+g]=2*np.sum(centered[0][None]*centered[1:],axis=-1,keepdims=True)/d
    support=np.abs(centered[1:]).sum(axis=0)
    fresh=np.sum(support*support,axis=-1)/d
    for token in range(l):variance[1+g+token,token,:]=fresh[token]
    variance[0]+=epsilon
    root=sqrt_reference(variance);inv=reciprocal_reference(root)
    normalized=multiply_reference(centered,inv)
    normalized=normalized*gamma
    normalized[0]+=beta
    return normalized


def exp_minimal_reference(weights):
    w=np.asarray(weights,dtype=np.float64);l,u=concretize_weights(w);different=l!=u
    with np.errstate(divide='ignore',invalid='ignore',over='raise'):
        tcrit=np.log((np.exp(u)-np.exp(l))/(u-l));tcrit=np.where(different,tcrit,np.inf)
        tcrit=np.where(np.isneginf(tcrit),.5*l+.5*u,tcrit);t=np.minimum(np.minimum(tcrit,l+.95),u);lam=np.exp(t)
        const=.5*(lam*(1-t-u))+.5*np.exp(u);fresh=.5*(lam*(t-u-1))+.5*np.exp(u)
    center=np.where(different,lam*w[0]+const,np.exp(l));old=w[1:]*lam[None]*different[None]
    return _append_coordinate_fresh(np.concatenate((center[None],old)),fresh,different)


def softmax_preconstraint_reference(weights):
    """Pinned new-softmax equations before optional equality substitution.

    Supports the one-head fixture used by the frozen parity gate.
    """
    w=np.asarray(weights,dtype=np.float64);_require(w.ndim==4 and w.shape[0]==1,'softmax reference fixture shape')
    _,e,r,n=w.shape
    vals=w[0];diff=vals[:,:,:,None].transpose(0,1,3,2)-vals[:,:,:,None]  # [E,R,i,j] = s_j-s_i
    flat=diff.reshape(e,r,n*n);l,u=concretize_weights(flat);different=l!=u
    with np.errstate(divide='ignore',invalid='ignore',over='raise'):
        tcrit=np.log((np.exp(u)-np.exp(l))/(u-l));tcrit=np.where(different,tcrit,np.inf)
        tcrit=np.where(np.isneginf(tcrit),.5*l+.5*u,tcrit);t=np.minimum(np.minimum(tcrit,l+.95),u);lam=np.exp(t)
        const=.5*(lam*(1-t-u))+.5*np.exp(u);newcoeff=.5*(lam*(t-u-1))+.5*np.exp(u)
    transformed=np.concatenate((np.where(different,lam*flat[0]+const,np.exp(l))[None],flat[1:]*lam[None]*different[None]),axis=0)
    old=transformed.reshape(e,r,n,n).sum(axis=-1)
    collapsed=np.where(different,newcoeff,0).reshape(r,n,n).sum(axis=-1)
    boxes=np.zeros((r*n,r,n),dtype=np.float64);boxes.reshape(r*n,r*n)[np.arange(r*n),np.arange(r*n)]=collapsed.reshape(-1)
    denom=np.concatenate((old,boxes),axis=0)
    recip=reciprocal_reference(denom,original=False)
    return recip[:,None,:,:].transpose(1,0,2,3)


def high_precision_unary_reference(kind, lower, upper):
    """Long-double endpoint oracle used independently of native torch code."""
    l=np.asarray(lower,dtype=np.longdouble);u=np.asarray(upper,dtype=np.longdouble)
    if kind=="sqrt":
        _require(np.all(l>0),"sqrt input not positive"); lo=np.sqrt(l);hi=np.sqrt(u)
    elif kind=="reciprocal":
        _require(np.all(l>0),"reciprocal input not positive");lo=1/u;hi=1/l
    elif kind=="exp":lo=np.exp(l);hi=np.exp(u)
    elif kind=="tanh":lo=np.tanh(l);hi=np.tanh(u)
    else:raise AssertionError("unknown unary")
    return np.asarray(lo,dtype=np.float64),np.asarray(hi,dtype=np.float64)
