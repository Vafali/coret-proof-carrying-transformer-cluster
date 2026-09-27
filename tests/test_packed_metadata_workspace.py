from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_hab"))
import coret_structural_support_precise_dot_v1 as structural


class Proxy:
    def __init__(self, weights, source=None):
        self.zonotope_w = weights
        self.error_term_range_low = None
        self.error_term_range_high = None
        self.num_input_error_terms_special_norm = 2
        self.p = 100
        self.eps = 1.0 / 1600.0
        self.perturbed_word_index = 0
    @property
    def num_error_terms(self): return self.zonotope_w.shape[1] - 1
    @property
    def num_words(self): return self.zonotope_w.shape[2]
    @property
    def word_embedding_size(self): return self.zonotope_w.shape[3]


@pytest.fixture(autouse=True)
def proxy_factory(monkeypatch):
    package = types.ModuleType("Verifiers")
    module = types.ModuleType("Verifiers.Zonotope")
    module.make_zonotope_new_weights_same_args = (
        lambda new_weights, source_zonotope, clone=False:
        Proxy(new_weights.clone() if clone else new_weights, source_zonotope))
    package.Zonotope = module
    monkeypatch.setitem(sys.modules, "Verifiers", package)
    monkeypatch.setitem(sys.modules, "Verifiers.Zonotope", module)


def legacy_qk(a, b, left, right, gmin):
    heads, _, na, dimension = a.shape
    nb = b.shape[2]
    radius = torch.zeros(heads, na, nb, device=a.device, dtype=a.dtype)
    macs = launches = 0
    inventory = []
    for query in range(na):
        for key in range(nb):
            lbit, rbit = structural.local_mask(query), structural.local_mask(key)
            la = [i for i in range(gmin) if left.masks[i] & lbit]
            ra = [i for i in range(gmin) if right.masks[i] & rbit]
            if not la or not ra:
                continue
            active = [i for i in range(gmin)
                      if (left.masks[i] & lbit) or (right.masks[i] & rbit)]
            inventory.append((query, key, tuple(active)))
            indices = torch.tensor(active, device=a.device, dtype=torch.long)
            av = a[:, 1 + indices, query, :]
            bv = b[:, 1 + indices, key, :]
            first = torch.bmm(av, bv.transpose(1, 2))
            symmetric = first + first.transpose(1, 2)
            diagonal = torch.diagonal(symmetric, dim1=1, dim2=2)
            radius[:, query, key] = (
                0.5 * symmetric.abs().sum(dim=(1, 2))
                - 0.25 * diagonal.abs().sum(dim=1))
            launches += 1
            macs += heads * len(active) * len(active) * dimension
    return radius, macs, launches, inventory


def legacy_av(a, b, left, right, ga, gb, tile):
    heads, _, queries, keys = a.shape
    features = b.shape[2]
    radius = torch.zeros(heads, queries, features, device=a.device, dtype=a.dtype)
    gmax = max(ga, gb)
    groups = {}
    for index in range(gmax):
        lm = left.masks[index] if index < ga else 0
        rm = right.masks[index] if index < gb else 0
        if lm or rm:
            groups.setdefault((lm, rm), []).append(index)
    ag = a[:, 1:1 + ga]
    if ga < gmax:
        ag = torch.cat([ag, torch.zeros(
            heads, gmax-ga, queries, keys, device=a.device, dtype=a.dtype)], 1)
    bg = b[:, 1:1 + gb]
    if gb < gmax:
        bg = torch.cat([bg, torch.zeros(
            heads, gmax-gb, features, keys, device=b.device, dtype=b.dtype)], 1)
    macs = launches = peak = 0
    inventory = []
    items = sorted(groups.items())
    for pos, ((li, ri), mi) in enumerate(items):
        keys_i = [key for key in range(keys) if ri & structural.local_mask(key)]
        for (lj, rj), mj in items[pos:]:
            keys_j = [key for key in range(keys) if rj & structural.local_mask(key)]
            same = (li, ri) == (lj, rj)
            for ii in range(0, len(mi), tile):
                inds_i = mi[ii:ii + tile]
                for jj in range(ii if same else 0, len(mj), tile):
                    inds_j = mj[jj:jj + tile]
                    if same and jj < ii:
                        continue
                    if not keys_i and not keys_j:
                        continue
                    inventory.append((tuple(inds_i), tuple(inds_j),
                                      tuple(keys_i), tuple(keys_j),
                                      same and ii == jj))
                    ti = torch.tensor(inds_i, device=a.device, dtype=torch.long)
                    tj = torch.tensor(inds_j, device=a.device, dtype=torch.long)
                    term = None
                    if keys_j:
                        kj = torch.tensor(keys_j, device=a.device, dtype=torch.long)
                        term = torch.einsum(
                            "hiqk,hjfk->hijqf",
                            ag[:, ti].index_select(3, kj),
                            bg[:, tj].index_select(3, kj))
                        macs += heads*len(inds_i)*len(inds_j)*queries*features*len(keys_j)
                    if keys_i:
                        ki = torch.tensor(keys_i, device=a.device, dtype=torch.long)
                        other = torch.einsum(
                            "hifk,hjqk->hijqf",
                            bg[:, ti].index_select(3, ki),
                            ag[:, tj].index_select(3, ki))
                        macs += heads*len(inds_i)*len(inds_j)*queries*features*len(keys_i)
                        term = other if term is None else term.add_(other)
                    launches += 1
                    peak = max(peak, term.numel()*term.element_size())
                    if same and ii == jj:
                        diagonal = torch.diagonal(term, dim1=1, dim2=2)
                        radius.add_(0.5*term.abs().sum(dim=(1, 2)))
                        radius.sub_(0.25*diagonal.abs().sum(dim=-1))
                    else:
                        radius.add_(term.abs().sum(dim=(1, 2)))
    return radius, macs, launches, peak, inventory


def proof(masks, prefix):
    return structural.proof_from_masks(masks, 3, prefix)


def test_arena_views_preserve_every_sequence_and_cap():
    builder = structural._IndexArenaBuilder(cap_bytes=64)
    values = ((4, 1, 9), (), (2,), (8, 7))
    refs = [builder.add(v) for v in values]
    arena = builder.finish(torch.device("cpu"))
    assert [tuple(arena.view(ref).tolist()) for ref in refs] == list(values)
    assert arena.device.is_contiguous()
    with pytest.raises(RuntimeError, match="exceeds cap"):
        builder.add(range(20))


def test_qk_packed_indices_are_bitwise_legacy_and_inventory_exact(monkeypatch):
    torch.manual_seed(4)
    masks = (1, 2, 4, 3, 7, 0)
    p = proof(masks, "qk")
    a = torch.randn(2, 7, 3, 5)
    b = torch.randn(2, 7, 3, 5)
    expected = legacy_qk(a, b, p, p, 6)
    recorded = []
    original_builder = structural._IndexArenaBuilder
    class RecordingBuilder(original_builder):
        def add(self, values):
            values = tuple(values); recorded.append(values)
            return super().add(values)
    monkeypatch.setattr(structural, "_IndexArenaBuilder", RecordingBuilder)
    actual = structural._ordered_qk_radius(a, b, p, p, 6)
    assert torch.equal(actual[0], expected[0])
    assert actual[1:] == expected[1:3]
    assert recorded == [task[2] for task in expected[3]]


def test_av_packed_indices_are_bitwise_legacy_and_inventory_exact(monkeypatch):
    torch.manual_seed(5)
    left = proof((7, 1, 2, 4, 3, 6, 0), "left")
    right = proof((1, 2, 4, 3, 6), "right")
    a = torch.randn(2, 8, 3, 3)
    b = torch.randn(2, 6, 4, 3)
    expected = legacy_av(a, b, left, right, 7, 5, 2)
    recorded = []
    original_builder = structural._IndexArenaBuilder
    class RecordingBuilder(original_builder):
        def add(self, values):
            values = tuple(values); recorded.append(values)
            return super().add(values)
    monkeypatch.setattr(structural, "_IndexArenaBuilder", RecordingBuilder)
    actual = structural._av_radius(a, b, left, right, 7, 5, 2)
    assert torch.equal(actual[0], expected[0])
    assert actual[1:] == expected[1:4]
    expected_sequences = []
    for task in expected[4]:
        expected_sequences.extend(task[:4])
    assert recorded == expected_sequences


def support_respecting_weights(masks, heads, rows, columns, token_axis):
    weights = torch.zeros(heads, 1 + len(masks), rows, columns)
    torch.manual_seed(6)
    weights[:, 0] = torch.randn(heads, rows, columns)
    for index, mask in enumerate(masks):
        for token in range(3):
            if mask & structural.local_mask(token):
                if token_axis == 2:
                    weights[:, 1 + index, token, :] = torch.randn(heads, columns)
                else:
                    weights[:, 1 + index, :, token] = torch.randn(heads, rows)
    return weights


def comparable_diagnostics(value):
    return {key: item for key, item in value.items()
            if key != "stage_timing_seconds"}


@pytest.mark.parametrize("mode", ["QK", "A.V"])
def test_full_transformer_output_and_provenance_bitwise(monkeypatch, mode):
    masks = (1, 2, 4, 3, 7, 0)
    p = proof(masks, "shared")
    if mode == "QK":
        left = Proxy(support_respecting_weights(masks, 2, 3, 5, 2))
        right = Proxy(support_respecting_weights(masks, 2, 3, 5, 2))
        legacy = lambda a, b, lp, rp, g: legacy_qk(a, b, lp, rp, g)[:3]
        name = "_ordered_qk_radius"
        kwargs = {"mode": mode}
    else:
        left = Proxy(support_respecting_weights(masks, 2, 3, 3, 2))
        right = Proxy(support_respecting_weights(masks, 2, 4, 3, 3))
        legacy = lambda a, b, lp, rp, ga, gb, tile: legacy_av(
            a, b, lp, rp, ga, gb, tile)[:4]
        name = "_av_radius"
        kwargs = {"mode": mode, "generator_tile": 2}
    diagnostics_old = {}
    original = getattr(structural, name)
    monkeypatch.setattr(structural, name, legacy)
    old = structural.precise_dot_structural(
        left, right, p, p, diagnostics=diagnostics_old, **kwargs)
    monkeypatch.setattr(structural, name, original)
    diagnostics_new = {}
    new = structural.precise_dot_structural(
        left, right, p, p, diagnostics=diagnostics_new, **kwargs)
    assert torch.equal(new.zonotope_w, old.zonotope_w)
    assert structural.get_support(new) == structural.get_support(old)
    assert comparable_diagnostics(diagnostics_new) == comparable_diagnostics(diagnostics_old)
