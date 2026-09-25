#!/usr/bin/env python3
"""Exact clean DeepT Table-7 3-layer SST model and deterministic inventory.

The implementation mirrors the public DeepT ``Models/modeling.py`` graph at
commit 16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf.  It intentionally exposes
the pre-embedding-LayerNorm tensor because that is where the published verifier
introduces its single-token Lp perturbation.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import subprocess
import unicodedata
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEEPT_COMMIT = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
CHECKPOINT_GIT_PATH = (
    "Robustness-Verification-for-Transformers/"
    "sst_bert_standard_layer_norm_3/ckpt-5/pytorch_model.bin"
)
CONFIG_GIT_PATH = (
    "Robustness-Verification-for-Transformers/"
    "sst_bert_standard_layer_norm_3/ckpt-5/config.json"
)
VOCAB_GIT_PATH = (
    "Robustness-Verification-for-Transformers/"
    "sst_bert_standard_layer_norm_3/ckpt-5/vocab.txt"
)
CHECKPOINT_SHA256 = "27ae76c19331bc4d83c2226f9af84650d1ca714c9d0f5f38c1439c620eecda71"
DATA_MEMBER = "data/sst/test.txt"
DATA_SHA256 = "6e54806dee95cf80cd918e7dfb3f6770f6df24bf826f289a4d1f709e1c8f6761"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_blob(repo: Path, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{DEEPT_COMMIT}:{path}"])


def _is_whitespace(char: str) -> bool:
    return char in " \t\n\r" or unicodedata.category(char) == "Zs"


def _is_control(char: str) -> bool:
    if char in "\t\n\r":
        return False
    return unicodedata.category(char) in ("Cc", "Cf")


def _is_punctuation(char: str) -> bool:
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def _whitespace_tokenize(text: str) -> list[str]:
    text = text.strip()
    return text.split() if text else []


class DeepTTokenizer:
    """BERT BasicTokenizer + greedy WordPiece semantics used by DeepT."""

    def __init__(self, vocab_text: str, do_lower_case: bool = True):
        self.vocab = OrderedDict(
            (token, index) for index, token in enumerate(vocab_text.splitlines()))
        self.unknown = "[UNK]"
        self.do_lower_case = do_lower_case

    def _clean_text(self, text: str) -> str:
        output = []
        for char in text:
            code = ord(char)
            if code == 0 or code == 0xFFFD or _is_control(char):
                continue
            output.append(" " if _is_whitespace(char) else char)
        return "".join(output)

    @staticmethod
    def _strip_accents(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn")

    @staticmethod
    def _split_punctuation(text: str) -> list[str]:
        output: list[list[str]] = []
        start_new = True
        for char in text:
            if _is_punctuation(char):
                output.append([char])
                start_new = True
            else:
                if start_new:
                    output.append([])
                output[-1].append(char)
                start_new = False
        return ["".join(item) for item in output]

    def basic_tokenize(self, text: str) -> list[str]:
        cleaned = self._clean_text(text)
        split: list[str] = []
        for token in _whitespace_tokenize(cleaned):
            if self.do_lower_case:
                token = self._strip_accents(token.lower())
            split.extend(self._split_punctuation(token))
        return _whitespace_tokenize(" ".join(split))

    def _wordpiece(self, token: str) -> list[str]:
        chars = list(token)
        if len(chars) > 100:
            return [self.unknown]
        pieces: list[str] = []
        start = 0
        while start < len(chars):
            end = len(chars)
            current = None
            while start < end:
                piece = "".join(chars[start:end])
                if start:
                    piece = "##" + piece
                if piece in self.vocab:
                    current = piece
                    break
                end -= 1
            if current is None:
                return [self.unknown]
            pieces.append(current)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        return [piece for token in self.basic_tokenize(text) for piece in self._wordpiece(token)]

    def encode_tokens(self, words: Iterable[str], max_length: int = 128) -> tuple[list[str], list[int]]:
        pieces = self.tokenize(" ".join(words))[: max_length - 2]
        tokens = ["[CLS]", *pieces, "[SEP]"]
        return tokens, [self.vocab.get(token, self.vocab[self.unknown]) for token in tokens]


class DeepTLayerNorm(nn.Module):
    def __init__(self, hidden: int, epsilon: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(-1, keepdim=True)
        variance = (value - mean).pow(2).mean(-1, keepdim=True)
        # Preserve the exact operation order in DeepT Models/modeling.py.
        normalized = (value - mean) / torch.sqrt(variance + self.epsilon)
        return self.weight * normalized + self.bias


class DeepTEmbeddings(nn.Module):
    def __init__(self, vocab: int, hidden: int, positions: int, token_types: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, hidden, padding_idx=0)
        self.position_embeddings = nn.Embedding(positions, hidden)
        self.token_type_embeddings = nn.Embedding(token_types, hidden)
        self.LayerNorm = DeepTLayerNorm(hidden)

    def pre_layernorm(self, input_ids: Tensor, token_type_ids: Tensor | None = None) -> Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(token_type_ids)
        )

    def forward(self, input_ids: Tensor, token_type_ids: Tensor | None = None,
                pre_layernorm_embeddings: Tensor | None = None) -> Tensor:
        value = self.pre_layernorm(input_ids, token_type_ids) if pre_layernorm_embeddings is None else pre_layernorm_embeddings
        return self.LayerNorm(value)


class DeepTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.attention_output = nn.Linear(hidden, hidden)
        self.attention_layernorm = DeepTLayerNorm(hidden)
        self.intermediate = nn.Linear(hidden, intermediate)
        self.output = nn.Linear(intermediate, hidden)
        self.output_layernorm = DeepTLayerNorm(hidden)

    def forward(self, hidden: Tensor, attention_mask: Tensor,
                captures: dict[str, Tensor] | None = None, prefix: str = "") -> Tensor:
        batch, length, _ = hidden.shape
        query = self.query(hidden).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(hidden).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(hidden).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        scores = query @ key.transpose(-1, -2) / np.sqrt(self.head_dim)
        scores = scores + attention_mask
        probabilities = torch.softmax(scores, dim=-1)
        context = (probabilities @ value).permute(0, 2, 1, 3).contiguous().reshape(batch, length, self.hidden)
        attention_residual = self.attention_output(context) + hidden
        attention_normalized = self.attention_layernorm(attention_residual)
        activated = torch.relu(self.intermediate(attention_normalized))
        output_residual = self.output(activated) + attention_normalized
        output = self.output_layernorm(output_residual)
        if captures is not None:
            captures.update({
                f"{prefix}.scores": scores,
                f"{prefix}.attention_residual": attention_residual,
                f"{prefix}.attention_layernorm": attention_normalized,
                f"{prefix}.relu": activated,
                f"{prefix}.output": output,
            })
        return output


class DeepTStageAModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        hidden = config["hidden_size"]
        self.embeddings = DeepTEmbeddings(
            config["vocab_size"], hidden, config["max_position_embeddings"],
            config["type_vocab_size"])
        self.blocks = nn.ModuleList([
            DeepTBlock(hidden, config["num_attention_heads"], config["intermediate_size"])
            for _ in range(config["num_hidden_layers"])
        ])
        self.pooler = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                token_type_ids: Tensor | None = None,
                pre_layernorm_embeddings: Tensor | None = None,
                return_intermediates: bool = False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended = (1.0 - attention_mask[:, None, None, :].to(self.pooler.weight.dtype)) * -10000.0
        captures: dict[str, Tensor] | None = {} if return_intermediates else None
        hidden = self.embeddings(input_ids, token_type_ids, pre_layernorm_embeddings)
        if captures is not None:
            captures["embedding_layernorm"] = hidden
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, extended, captures, f"block{index}")
        pooled_pre = self.pooler(hidden[:, 0])
        pooled = torch.tanh(pooled_pre)
        logits = self.classifier(pooled)
        if captures is not None:
            captures.update({"sequence": hidden, "pooler_pre_tanh": pooled_pre, "pooler": pooled, "logits": logits})
            return logits, captures
        return logits


def _remap_state_dict(state: dict[str, Tensor]) -> OrderedDict[str, Tensor]:
    mapped: OrderedDict[str, Tensor] = OrderedDict()
    direct = {
        "bert.embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "bert.embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "bert.embeddings.LayerNorm.weight": "embeddings.LayerNorm.weight",
        "bert.embeddings.LayerNorm.bias": "embeddings.LayerNorm.bias",
        "bert.pooler.dense.weight": "pooler.weight",
        "bert.pooler.dense.bias": "pooler.bias",
        "classifier.weight": "classifier.weight",
        "classifier.bias": "classifier.bias",
    }
    for source, target in direct.items():
        mapped[target] = state[source]
    suffixes = {
        "attention.self.query": "query",
        "attention.self.key": "key",
        "attention.self.value": "value",
        "attention.output.dense": "attention_output",
        "attention.output.LayerNorm": "attention_layernorm",
        "intermediate.dense": "intermediate",
        "output.dense": "output",
        "output.LayerNorm": "output_layernorm",
    }
    for layer in range(3):
        for old, new in suffixes.items():
            for parameter in ("weight", "bias"):
                mapped[f"blocks.{layer}.{new}.{parameter}"] = state[
                    f"bert.encoder.layer.{layer}.{old}.{parameter}"]
    return mapped


@dataclass(frozen=True)
class LoadedStageA:
    model: DeepTStageAModel
    tokenizer: DeepTTokenizer
    config: dict[str, Any]
    checkpoint_sha256: str


def load_stagea_model(root: Path | None = None, dtype: torch.dtype = torch.float32) -> LoadedStageA:
    root = repository_root() if root is None else root
    repo = root / "research_hab/public_benchmarks/DeepT"
    checkpoint = _git_blob(repo, CHECKPOINT_GIT_PATH)
    digest = hashlib.sha256(checkpoint).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("DeepT checkpoint hash mismatch")
    config = json.loads(_git_blob(repo, CONFIG_GIT_PATH))
    if config != {
        "attention_probs_dropout_prob": 0.1, "hidden_act": "relu",
        "hidden_dropout_prob": 0.1, "hidden_size": 128,
        "initializer_range": 0.02, "intermediate_size": 128,
        "layer_norm": "standard", "max_position_embeddings": 512,
        "num_attention_heads": 4, "num_hidden_layers": 3,
        "type_vocab_size": 2, "vocab_size": 30629,
    }:
        raise ValueError("unexpected DeepT Stage-A architecture")
    state = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=False)
    model = DeepTStageAModel(config).to(dtype=dtype)
    incompatibility = model.load_state_dict(_remap_state_dict(state), strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(f"checkpoint mapping mismatch: {incompatibility}")
    model.eval()
    tokenizer = DeepTTokenizer(_git_blob(repo, VOCAB_GIT_PATH).decode("utf-8"))
    return LoadedStageA(model, tokenizer, config, digest)


def deept_functional_reference(
        state: dict[str, Tensor], config: dict[str, Any], input_ids: Tensor,
        attention_mask: Tensor, token_type_ids: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    """Independent functional transcription of public DeepT ``modeling.py``."""
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    hidden = (
        F.embedding(input_ids, state["bert.embeddings.word_embeddings.weight"], padding_idx=0)
        + F.embedding(positions, state["bert.embeddings.position_embeddings.weight"])
        + F.embedding(token_type_ids, state["bert.embeddings.token_type_embeddings.weight"])
    )

    def layernorm(value: Tensor, prefix: str) -> Tensor:
        mean = value.mean(-1, keepdim=True)
        variance = (value - mean).pow(2).mean(-1, keepdim=True)
        normalized = (value - mean) / torch.sqrt(variance + 1e-12)
        return state[prefix + ".weight"] * normalized + state[prefix + ".bias"]

    hidden = layernorm(hidden, "bert.embeddings.LayerNorm")
    captures = {"embedding_layernorm": hidden}
    extended = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * -10000.0
    heads, head_dim = config["num_attention_heads"], config["hidden_size"] // config["num_attention_heads"]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"bert.encoder.layer.{layer}"
        projections = []
        for name in ("query", "key", "value"):
            value = F.linear(
                hidden, state[f"{prefix}.attention.self.{name}.weight"],
                state[f"{prefix}.attention.self.{name}.bias"])
            projections.append(value.reshape(value.shape[0], value.shape[1], heads, head_dim).permute(0, 2, 1, 3))
        query, key, value = projections
        scores = query @ key.transpose(-1, -2) / np.sqrt(head_dim) + extended
        probabilities = torch.softmax(scores, dim=-1)
        context = (probabilities @ value).permute(0, 2, 1, 3).contiguous().reshape(hidden.shape)
        attention_residual = F.linear(
            context, state[f"{prefix}.attention.output.dense.weight"],
            state[f"{prefix}.attention.output.dense.bias"]) + hidden
        attention_normalized = layernorm(attention_residual, f"{prefix}.attention.output.LayerNorm")
        activated = torch.relu(F.linear(
            attention_normalized, state[f"{prefix}.intermediate.dense.weight"],
            state[f"{prefix}.intermediate.dense.bias"]))
        output_residual = F.linear(
            activated, state[f"{prefix}.output.dense.weight"],
            state[f"{prefix}.output.dense.bias"]) + attention_normalized
        hidden = layernorm(output_residual, f"{prefix}.output.LayerNorm")
        captures[f"block{layer}.scores"] = scores
        captures[f"block{layer}.attention_residual"] = attention_residual
        captures[f"block{layer}.attention_layernorm"] = attention_normalized
        captures[f"block{layer}.relu"] = activated
        captures[f"block{layer}.output"] = hidden
    pooled_pre = F.linear(hidden[:, 0], state["bert.pooler.dense.weight"], state["bert.pooler.dense.bias"])
    pooled = torch.tanh(pooled_pre)
    logits = F.linear(pooled, state["classifier.weight"], state["classifier.bias"])
    captures.update({"sequence": hidden, "pooler_pre_tanh": pooled_pre, "pooler": pooled, "logits": logits})
    return logits, captures


def load_sst_binary_test(root: Path | None = None) -> list[dict[str, Any]]:
    root = repository_root() if root is None else root
    archive = root / "research_hab/public_benchmarks/robustness_verification_transformers/data_archive"
    with zipfile.ZipFile(archive) as zipped:
        raw = zipped.read(DATA_MEMBER)
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("SST test split hash mismatch")
    examples = []
    for source_line, line in enumerate(raw.decode("utf-8").splitlines()):
        segments = line.split(" ")
        original_label = int(segments[0][1])
        if original_label == 2:
            continue
        label = 0 if original_label < 2 else 1
        words = []
        for index in range(len(segments) - 1):
            if (len(segments[index]) > 1 and segments[index][0] == "(" and
                    segments[index][1] in "01234" and not segments[index + 1].startswith("(")):
                token = segments[index + 1].split(")", 1)[0]
                words.append("(" if token == "-LRB-" else ")" if token == "-RRB-" else token)
        examples.append({"source_line": source_line, "label": label, "words": words})
    return examples


def build_stagea_inventory(loaded: LoadedStageA, count: int = 10) -> dict[str, Any]:
    examples = load_sst_binary_test()
    rng = random.Random(0)
    selected = []
    attempts = 0
    while len(selected) < count:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("could not reconstruct Stage-A sample")
        canonical_index = rng.randint(0, len(examples) - 1)
        example = examples[canonical_index]
        tokens, ids = loaded.tokenizer.encode_tokens(example["words"], max_length=128)
        if len(tokens) > 32:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = loaded.model(input_ids, mask)[0]
        prediction = int(logits.argmax())
        if prediction != example["label"]:
            continue
        text = " ".join(example["words"])
        accepted_ordinal = len(selected)
        eligible = [
            position for position in range(1, len(tokens) - 1)
            if not tokens[position].startswith("#") and not tokens[position + 1].startswith("#")
        ]
        selected.append({
            "accepted_example_ordinal": accepted_ordinal,
            "canonical_binary_test_index": canonical_index,
            "source_test_line": example["source_line"],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "clean_label": example["label"],
            "predicted_label": prediction,
            "clean_logits": [float(value) for value in logits],
            "sequence_length": len(tokens),
            "tokens": tokens,
            "token_ids": ids,
            "eligible_token_positions": eligible,
        })
    properties = []
    for example in selected:
        for position in example["eligible_token_positions"]:
            properties.append({
                "property_id": (
                    f"deept_sst_stdln3_linf_e{example['accepted_example_ordinal']:02d}"
                    f"_line{example['source_test_line']}_tok{position:02d}"),
                "accepted_example_ordinal": example["accepted_example_ordinal"],
                "canonical_binary_test_index": example["canonical_binary_test_index"],
                "source_test_line": example["source_test_line"],
                "token_position": position,
                "token": example["tokens"][position],
                "token_id": example["token_ids"][position],
                "clean_label": example["clean_label"],
                "predicted_label": example["predicted_label"],
                "sequence_length": example["sequence_length"],
                "checkpoint_sha256": loaded.checkpoint_sha256,
                "norm": "Linf",
                "objective": "gold_logit_minus_other_logit",
            })
    payload = {
        "schema_version": 1,
        "inventory_id": "coret_deept_table7_sst_3layer_linf_stageA_inventory_v1",
        "selection_seed": 0,
        "selection_attempts": attempts,
        "examples": selected,
        "properties": properties,
    }
    payload["inventory_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    return payload
