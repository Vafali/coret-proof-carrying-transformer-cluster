#!/usr/bin/env python3
"""Depth-only binding for frozen historical standard-LN DeepT models.

This adapter changes no verifier operation.  It selects one of the immutable
historical checkpoint/config pairs and supplies the same native DeepT CLI
arguments used by the accepted three-layer adapter.
"""
from __future__ import annotations

import io
import json
from argparse import Namespace
from pathlib import Path

import torch

import coret_deept_exact_standard_ln_adapter as base


DEEPT_COMMIT = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
PREFIX = "Robustness-Verification-for-Transformers"
DEPTH_SPECS = {
    6: {
        "directory": "sst_bert_standard_layer_norm_6",
        "checkpoint_sha256": "6e75fe827f7259db65a19a1fda152eb644cb44ead6d2d5a2e92b0950f368a236",
        "config_sha256": "daf666f1fde12adc77e59883587f11f6562c735d1c8a9bd0abb080a31dd0d787",
    },
    12: {
        "directory": "sst_bert_standard_layer_norm_12",
        "checkpoint_sha256": "0686f41444b94d7f81aad874ef522a882f0a77b888e537d80d056275c6221c25",
        "config_sha256": "e2b8a46d9c52e8722ff796ea7297d4feeaa260dc4e11c7b2c2519f5d1a180533",
    },
}


def _spec(depth: int) -> dict:
    if depth not in DEPTH_SPECS:
        raise RuntimeError(f"unsupported frozen depth: {depth}")
    return DEPTH_SPECS[depth]


def _paths(depth: int) -> tuple[str, str]:
    directory = _spec(depth)["directory"]
    root = f"{PREFIX}/{directory}/ckpt-5"
    return f"{root}/pytorch_model.bin", f"{root}/config.json"


def load_native_model(modules, device: torch.device, depth: int,
                      dtype: torch.dtype = torch.float32):
    spec = _spec(depth)
    checkpoint_path, config_path = _paths(depth)
    config_blob = base._git_blob(base.DEEPT_REPOSITORY, config_path)
    checkpoint_blob = base._git_blob(base.DEEPT_REPOSITORY, checkpoint_path)
    if base.sha256_bytes(config_blob) != spec["config_sha256"]:
        raise RuntimeError("frozen depth config hash mismatch")
    if base.sha256_bytes(checkpoint_blob) != spec["checkpoint_sha256"]:
        raise RuntimeError("frozen depth checkpoint hash mismatch")
    config = json.loads(config_blob)
    if int(config.get("num_hidden_layers", -1)) != depth:
        raise RuntimeError("checkpoint configuration depth mismatch")
    model = modules.BertForSequenceClassification(
        base._native_config(config), num_labels=2)
    state = torch.load(io.BytesIO(checkpoint_blob), map_location="cpu",
                       weights_only=False)
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"native DeepT state mapping failed: {incompatibility}")
    if len(model.bert.encoder.layer) != depth:
        raise RuntimeError("loaded native model layer count mismatch")
    return model.to(device=device, dtype=dtype).eval(), state, config


def build_deept_args(modules, device: torch.device, depth: int) -> Namespace:
    spec = _spec(depth)
    arguments = modules.Parser.get_parser().parse_args([
        "--verify", "--attack-type", "lp", "--data", "sst",
        "--dir", spec["directory"],
        "--num_layers", str(depth), "--num_attention_heads", "4",
        "--hidden_size", "128", "--intermediate_size", "128",
        "--hidden_act", "relu", "--layer_norm", "standard",
        "--method", "zonotope", "--p", "100", "--perturbed_words", "1",
        "--num-input-error-terms", "128", "--empty_cache",
        "--error-reduction-method", "box", "--max-num-error-terms", "14000",
        "--add-softmax-sum-constraint",
    ])
    arguments = modules.update_arguments(arguments)
    arguments.with_lirpa_transformer = False
    arguments.device = device
    arguments.cpu = device.type == "cpu"
    if int(arguments.num_layers) != depth:
        raise RuntimeError("DeepT argument depth mismatch")
    return arguments


native_pre_layernorm_embeddings = base.native_pre_layernorm_embeddings
FrozenDeepTModules = base.FrozenDeepTModules
