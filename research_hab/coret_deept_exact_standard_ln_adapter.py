#!/usr/bin/env python3
"""Pure adapter for the frozen DeepT exact-standard-LN e003 baseline gate.

The scientific verifier is loaded byte-for-byte from DeepT commit
16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf.  This module only materializes
those frozen Python blobs, loads the already-frozen checkpoint, and constructs
the namespace and input objects expected by DeepT.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch

from deept_stagea_model import (
    CHECKPOINT_GIT_PATH,
    CHECKPOINT_SHA256,
    CONFIG_GIT_PATH,
    DEEPT_COMMIT,
    DeepTStageAModel,
    LoadedStageA,
    _git_blob,
    load_sst_binary_test,
    load_stagea_model,
)


ROOT = Path(__file__).resolve().parents[1]
DEEPT_REPOSITORY = ROOT / "research_hab/public_benchmarks/DeepT"
DEEPT_SOURCE_PREFIX = "Robustness-Verification-for-Transformers/"
PROPERTY_INVENTORY = (
    ROOT / "research_hab/results/coret_sln_stageb_v1_20260912/"
    "coret_sln_stageb_property_inventory_v1.json"
)
PROPERTY_ID = "coret_sln_stageB_e003_line894_tok06"
PROPERTY_EPSILON = 3.90625e-05
EXPECTED_NOMINAL_MARGIN = 3.071135283710289
SOURCE_DIMENSION = 128
PERTURBED_TOKEN = 6


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii"))


def _git_names() -> list[str]:
    return subprocess.check_output([
        "git", "-C", str(DEEPT_REPOSITORY), "ls-tree", "-r",
        "--name-only", DEEPT_COMMIT,
    ], text=True).splitlines()


def _frozen_python_paths() -> list[str]:
    paths = []
    for name in _git_names():
        if not name.startswith(DEEPT_SOURCE_PREFIX) or not name.endswith(".py"):
            continue
        relative = name[len(DEEPT_SOURCE_PREFIX):]
        if "/" not in relative or relative.startswith(("Models/", "Verifiers/")):
            paths.append(name)
    return sorted(paths)


def frozen_source_hashes() -> dict[str, str]:
    relevant = [
        "Models/modeling.py",
        "Parser.py",
        "Verifiers/Verifier.py",
        "Verifiers/VerifierZonotope.py",
        "Verifiers/Zonotope.py",
        "Verifiers/utils.py",
    ]
    result = {}
    for relative in relevant:
        blob = _git_blob(DEEPT_REPOSITORY, DEEPT_SOURCE_PREFIX + relative)
        result[relative] = sha256_bytes(blob)
    return result


def _append_packaging_only_dependencies() -> dict[str, str]:
    """Expose two pure-Python dependencies missing from the production env.

    They already exist in the parent Conda installation.  The path is appended,
    never prepended, so the active environment remains authoritative for torch,
    numpy, and every package it supplies.
    """
    found: dict[str, str] = {}
    for module_name in ("opt_einsum", "termcolor"):
        try:
            module = __import__(module_name)
            found[module_name] = str(Path(module.__file__).resolve())
            continue
        except ModuleNotFoundError:
            pass
        executable = Path(sys.executable).resolve()
        candidates = []
        for ancestor in executable.parents:
            candidates.extend(sorted(
                (ancestor / "lib").glob("python*/site-packages")))
        usable = next((path for path in candidates if (path / module_name).exists()), None)
        if usable is None:
            raise RuntimeError(f"missing packaging-only DeepT dependency: {module_name}")
        sys.path.append(str(usable))
        module = __import__(module_name)
        found[module_name] = str(Path(module.__file__).resolve())
    return found


class FrozenDeepTModules:
    """Keeps the temporary frozen source tree alive while DeepT executes."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="coret_deept_16ffe_")
        archive = subprocess.run([
            "git", "-C", str(DEEPT_REPOSITORY), "archive", "--format=tar",
            DEEPT_COMMIT, *_frozen_python_paths(),
        ], check=True, stdout=subprocess.PIPE).stdout
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as packed:
            packed.extractall(self._temporary.name, filter="data")
        self.source_root = (
            Path(self._temporary.name) / DEEPT_SOURCE_PREFIX
        ).resolve()
        sys.path.insert(0, str(self.source_root))
        self.packaging_dependencies = _append_packaging_only_dependencies()

        from Models.modeling import BertForSequenceClassification
        from Parser import Parser, update_arguments
        from Verifiers.VerifierZonotope import VerifierZonotope

        self.BertForSequenceClassification = BertForSequenceClassification
        self.Parser = Parser
        self.update_arguments = update_arguments
        self.VerifierZonotope = VerifierZonotope

    def module_origins(self) -> dict[str, str]:
        return {
            name: str(Path(module.__file__).resolve())
            for name, module in sorted(sys.modules.items())
            if name == "Parser" or name.startswith(("Models", "Verifiers"))
            if isinstance(module, ModuleType) and getattr(module, "__file__", None)
        }


def _native_config(config_payload: dict[str, Any]):
    from pytorch_pretrained_bert.modeling import BertConfig

    config = BertConfig(config_payload["vocab_size"])
    for key, value in config_payload.items():
        setattr(config, key, value)
    return config


def original_to_native_parameter_names() -> dict[str, str]:
    mapping = {
        "embeddings.word_embeddings.weight": "bert.embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight": "bert.embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias": "bert.embeddings.LayerNorm.bias",
        "pooler.weight": "bert.pooler.dense.weight",
        "pooler.bias": "bert.pooler.dense.bias",
        "classifier.weight": "classifier.weight",
        "classifier.bias": "classifier.bias",
    }
    suffixes = {
        "query": "attention.self.query",
        "key": "attention.self.key",
        "value": "attention.self.value",
        "attention_output": "attention.output.dense",
        "attention_layernorm": "attention.output.LayerNorm",
        "intermediate": "intermediate.dense",
        "output": "output.dense",
        "output_layernorm": "output.LayerNorm",
    }
    for layer in range(3):
        for original_suffix, native_suffix in suffixes.items():
            for parameter in ("weight", "bias"):
                mapping[f"blocks.{layer}.{original_suffix}.{parameter}"] = (
                    f"bert.encoder.layer.{layer}.{native_suffix}.{parameter}"
                )
    return mapping


def load_native_model(
    modules: FrozenDeepTModules,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
):
    config_payload = json.loads(_git_blob(DEEPT_REPOSITORY, CONFIG_GIT_PATH))
    model = modules.BertForSequenceClassification(
        _native_config(config_payload), num_labels=2)
    checkpoint = _git_blob(DEEPT_REPOSITORY, CHECKPOINT_GIT_PATH)
    if sha256_bytes(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen DeepT checkpoint hash mismatch")
    state = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=False)
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"native DeepT state mapping failed: {incompatibility}")
    return model.to(device=device, dtype=dtype).eval(), state, config_payload


def load_property() -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    inventory = json.loads(PROPERTY_INVENTORY.read_text(encoding="utf-8"))
    matches = [item for item in inventory["properties"]
               if item["property_id"] == PROPERTY_ID]
    if len(matches) != 1:
        raise RuntimeError("frozen e003 property is not unique")
    prop = matches[0]
    example = load_sst_binary_test()[prop["canonical_binary_test_index"]]
    if example["source_line"] != 894 or example["label"] != prop["clean_label"]:
        raise RuntimeError("frozen e003 example identity mismatch")
    loaded = load_stagea_model(dtype=torch.float64)
    tokens, ids = loaded.tokenizer.encode_tokens(example["words"], max_length=128)
    if (len(ids) != prop["sequence_length"] or ids[PERTURBED_TOKEN] != prop["token_id"]
            or tokens[PERTURBED_TOKEN] != prop["token"]):
        raise RuntimeError("frozen e003 tokenization mismatch")
    return prop, example, ids


def native_pre_layernorm_embeddings(native_model, input_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(
        input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).unsqueeze(0).expand_as(input_ids)
    token_types = torch.zeros_like(input_ids)
    embeddings = native_model.bert.embeddings
    return (
        embeddings.word_embeddings(input_ids)
        + embeddings.position_embeddings(positions)
        + embeddings.token_type_embeddings(token_types)
    )


def mapping_audit(original: DeepTStageAModel, native_model) -> list[dict[str, Any]]:
    original_state = original.state_dict()
    native_state = native_model.state_dict()
    mapping = original_to_native_parameter_names()
    if set(mapping) != set(original_state) or set(mapping.values()) != set(native_state):
        raise RuntimeError("parameter mapping is not total and bijective")
    report = []
    for source, destination in sorted(mapping.items()):
        left, right = original_state[source], native_state[destination]
        if left.shape != right.shape:
            raise RuntimeError(f"shape mismatch while mapping {source}")
        deviation = float((left - right).abs().max()) if left.numel() else 0.0
        report.append({
            "original_name": source,
            "original_shape": list(left.shape),
            "DeepT_destination": destination,
            "destination_shape": list(right.shape),
            "max_absolute_difference": deviation,
        })
    return report


def forward_equivalence_audit(
    original: LoadedStageA, native_model, input_ids: torch.Tensor,
    attention_mask: torch.Tensor, label: int,
) -> dict[str, Any]:
    pre = native_pre_layernorm_embeddings(native_model, input_ids)
    patterns = [
        torch.zeros(SOURCE_DIMENSION, dtype=pre.dtype, device=pre.device),
        torch.where(
            torch.arange(SOURCE_DIMENSION, device=pre.device) % 2 == 0,
            torch.ones(SOURCE_DIMENSION, dtype=pre.dtype, device=pre.device),
            -torch.ones(SOURCE_DIMENSION, dtype=pre.dtype, device=pre.device)),
        torch.linspace(-1.0, 1.0, SOURCE_DIMENSION, dtype=pre.dtype, device=pre.device),
        torch.sin(torch.arange(
            SOURCE_DIMENSION, dtype=pre.dtype, device=pre.device)).clamp(-1.0, 1.0),
    ]
    max_logits = 0.0
    max_margin = 0.0
    nominal_margin = None
    with torch.no_grad():
        original.model.to(pre.device)
        for ordinal, pattern in enumerate(patterns):
            candidate = pre.clone()
            candidate[0, PERTURBED_TOKEN] += PROPERTY_EPSILON * pattern
            original_logits = original.model(
                input_ids, attention_mask,
                pre_layernorm_embeddings=candidate)
            native_logits = native_model(
                input_ids, attention_mask=attention_mask,
                embeddings=candidate)[0]
            max_logits = max(max_logits, float(
                (original_logits - native_logits).abs().max()))
            original_margin = original_logits[0, label] - original_logits[0, 1 - label]
            native_margin = native_logits[0, label] - native_logits[0, 1 - label]
            max_margin = max(max_margin, float((original_margin - native_margin).abs()))
            if ordinal == 0:
                nominal_margin = float(original_margin)
    assert nominal_margin is not None
    return {
        "deterministic_input_count": len(patterns),
        "nonzero_perturbation_count": len(patterns) - 1,
        "maximum_logit_mismatch": max_logits,
        "maximum_margin_mismatch": max_margin,
        "nominal_margin": nominal_margin,
        "nominal_reference": EXPECTED_NOMINAL_MARGIN,
        "nominal_reference_absolute_deviation": abs(
            nominal_margin - EXPECTED_NOMINAL_MARGIN),
        "all_max_abs_xi": [float(pattern.abs().max()) for pattern in patterns],
    }


def build_deept_args(modules: FrozenDeepTModules, device: torch.device) -> Namespace:
    arguments = modules.Parser.get_parser().parse_args([
        "--verify", "--attack-type", "lp", "--data", "sst",
        "--dir", "sst_bert_standard_layer_norm_3",
        "--num_layers", "3", "--num_attention_heads", "4",
        "--hidden_size", "128", "--intermediate_size", "128",
        "--hidden_act", "relu", "--layer_norm", "standard",
        "--method", "zonotope", "--p", "100", "--perturbed_words", "1",
        "--num-input-error-terms", "128", "--empty_cache",
        "--error-reduction-method", "box", "--max-num-error-terms", "14000",
        "--add-softmax-sum-constraint",
    ])
    arguments = modules.update_arguments(arguments)
    # DeepT's original CLI injects this legacy compatibility field in
    # main.py after parsing (lines 155--158 at the frozen revision).  The
    # frozen LP attack follows that branch and therefore sets it to False.
    arguments.with_lirpa_transformer = False
    arguments.device = device
    arguments.cpu = device.type == "cpu"
    return arguments


def compatibility_audit(device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    # DeepT's published zonotope runtime is float32.  In particular, its
    # softmax equality-constraint implementation explicitly requires
    # torch.float tensors.  Keep this pure adapter on that native runtime
    # dtype rather than forcing DeepT's state to float64.
    runtime_dtype = torch.float32
    torch.set_default_dtype(runtime_dtype)
    modules = FrozenDeepTModules()
    native_model, _, config = load_native_model(
        modules, device, dtype=runtime_dtype)
    original = load_stagea_model(dtype=runtime_dtype)
    original.model.to(device)
    prop, _, ids = load_property()
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    mapping = mapping_audit(original.model, native_model)
    forward = forward_equivalence_audit(
        original, native_model, input_ids, attention_mask, prop["clean_label"])
    args = build_deept_args(modules, device)
    if any(row["max_absolute_difference"] != 0.0 for row in mapping):
        raise RuntimeError("parameter mapping is not bit-exact after dtype conversion")
    if forward["maximum_logit_mismatch"] > 1e-12:
        raise RuntimeError("DeepT-adapted forward logits are not equivalent")
    nominal_reference_tolerance = (
        8.0 * torch.finfo(runtime_dtype).eps
        * max(1.0, abs(EXPECTED_NOMINAL_MARGIN))
    )
    if (forward["nominal_reference_absolute_deviation"]
            > nominal_reference_tolerance):
        raise RuntimeError("e003 nominal margin identity mismatch")
    audit = {
        "compatibility_verdict": "COMPATIBLE_PURE_ADAPTER",
        "STANDARD_LAYERNORM_EXACT": True,
        "LayerNorm_epsilon": 1e-12,
        "DeepT_runtime_dtype": str(runtime_dtype),
        "nominal_reference_tolerance": nominal_reference_tolerance,
        "arbitrary_LayerNorm_epsilon_configurable": False,
        "target_epsilon_exactly_supported": True,
        "architecture_support": {
            "post_LN_blocks_3": True,
            "hidden_128": True,
            "attention_heads_4": True,
            "head_dimension_32": True,
            "FFN_128": True,
            "ReLU": True,
            "genuine_standard_LayerNorm": True,
            "learned_absolute_position_embeddings": True,
            "first_token_pooling": True,
            "pooler_128x128_plus_tanh": True,
            "two_logit_classifier": True,
            "one_token_embedding_Linf": True,
        },
        "LayerNorm_code_locations": [
            "Models/modeling.py:30-43 (exact forward equation)",
            "Models/modeling.py:89-92 (embedding LayerNorm instance)",
            "Models/modeling.py:194-211 (post-attention LayerNorm)",
            "Models/modeling.py:252-267 (post-FFN LayerNorm)",
            "Verifiers/Zonotope.py:1735-1753 (standard-LN abstract transformer)",
            "Verifiers/VerifierZonotope.py:122-144 (pre-embedding-LN uncertainty handoff)",
            "Verifiers/VerifierZonotope.py:235-276 (both post-LN sites per block)",
        ],
        "embedding_perturbation": {
            "placement": "pre_embedding_LayerNorm",
            "perturbed_token": PERTURBED_TOKEN,
            "source_dimension": SOURCE_DIMENSION,
            "domain": "128 independent xi_i in [-1,1], delta_i=epsilon*xi_i",
            "all_other_tokens_fixed": True,
            "construction_location": "Verifiers/Zonotope.py:116-157",
        },
        "head": {
            "token": 0,
            "pooler_weight_shape": [128, 128],
            "activation": "tanh",
            "logits": 2,
            "native_verifier_support": True,
        },
        "selected_DeepT_mode": "zonotope",
        "selected_mode_configuration": {
            "p_cli_value": 100,
            "interpreted_norm": "Linf",
            "error_reduction_method": "box",
            "max_num_error_terms": 14000,
            "softmax_sum_constraint": True,
        },
        "selection_reason": (
            "exact published standard-LN Linf script configuration and the "
            "repository's strongest practical relational zonotope mode"
        ),
        "parameter_mapping": mapping,
        "parameter_count": len(mapping),
        "maximum_parameter_difference": max(
            row["max_absolute_difference"] for row in mapping),
        "forward_equivalence": forward,
        "native_config": config,
        "DeepT_arguments": {
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in vars(args).items()
            if key in {
                "method", "p", "layer_norm", "num_layers", "hidden_size",
                "intermediate_size", "num_attention_heads", "hidden_act",
                "perturbed_words", "num_input_error_terms",
                "error_reduction_method", "max_num_error_terms",
                "add_softmax_sum_constraint", "with_lirpa_transformer",
                "device",
            }
        },
        "frozen_DeepT_revision": DEEPT_COMMIT,
        "frozen_source_hashes": frozen_source_hashes(),
        "native_module_origins": modules.module_origins(),
        "packaging_only_dependencies": modules.packaging_dependencies,
        "scientific_certification_query_count": 0,
    }
    runtime = {
        "modules": modules,
        "native_model": native_model,
        "property": prop,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pre_layernorm_embeddings": native_pre_layernorm_embeddings(
            native_model, input_ids),
        "args": args,
    }
    return audit, runtime


def build_verifier(runtime: dict[str, Any]):
    target = SimpleNamespace(model=runtime["native_model"])
    return runtime["modules"].VerifierZonotope(
        runtime["args"], target, logger=None)
