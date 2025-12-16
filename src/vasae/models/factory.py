import torch.nn as nn

from .sae import VASAE, BatchTopKSAE, KSparse, TopKSAE, VanillaSAE


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
