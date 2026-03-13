from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import model_addition_debugger_context

try:
    from .sae import (
        VASAE,
        BatchTopKSAE,
        KSparse,
        TopKSAE,
        VanillaSAE,
        VASAE_LearnedDecoder,
        VASAE_ReLU,
    )
except ImportError:
    pass  # Legacy models not available in new layout


def get_sae_model(model_name: str, **args) -> nn.Module:
    if model_name == "VASAE_BatchKSparse":
        model = VASAE(args["k"], args["embedding_weight"])
    elif model_name == "VASAE_KSparse":
        model = VASAE(args["k"], args["embedding_weight"], act_fn=KSparse(args["k"]))
    elif model_name == "VanillaSAE":
        model = VanillaSAE(args["dim_input"], args["dim_sparse"])
    elif model_name == "TopKSAE":
        model = TopKSAE(args["dim_input"], args["dim_sparse"], args["k"])
    elif model_name == "BatchTopKSAE":
        model = BatchTopKSAE(args["dim_input"], args["dim_sparse"], args["k"])
    elif model_name == "VASAE_ReLU":
        model = VASAE_ReLU(args["embedding_weight"])
    elif model_name == "VASAE_LearnedDecoder":
        model = VASAE_LearnedDecoder(
            args["k"], args["embedding_weight"], lambda_cos=args["lambda_cos"]
        )
    else:
        raise ValueError(f"invalid model_name {model_name}")
    return model


def get_blackbox_model(model_name, device):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device).eval()
    return model, tokenizer


def get_llava_model(model_name, device):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, processor


@dataclass
class BlackBoxModelConfig:
    name: str = "gpt2"
    dir: Path | None = None


def load_unembeding_layer(cfg: BlackBoxModelConfig) -> nn.Linear:
    return torch.load(cfg.dir / "unemb.pth", weights_only=False)


def load_embeding_layer(cfg: BlackBoxModelConfig) -> nn.Embedding:
    return torch.load(cfg.dir / "emb.pth", weights_only=False)
