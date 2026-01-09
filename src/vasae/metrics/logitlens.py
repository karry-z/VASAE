from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vasae.metrics.interface import IMetric


class LogitLens:
    def __init__(self, unembed_layer: nn.Linear, ln=None):
        self.unembed_layer = unembed_layer
        self.ln = ln  # TODO: check if ln is necessary

    def unembed(self, activation: torch.Tensor) -> torch.Tensor:
        activation = activation.to(self.unembed_layer.weight.device)
        if self.ln is not None:
            activation = self.ln(activation)
        with torch.no_grad():
            logits = self.unembed_layer(activation)
        return logits

    def top1(self, activation: torch.Tensor) -> Dict:
        logits = self.unembed(activation)
        probs = logits.softmax(dim=-1)
        token_probs, token_ids = probs.max(dim=-1)
        return {
            "token_ids": token_ids,
            "token_probs": token_probs,
            "token_probs": token_probs,
            "probs": probs,
        }


class LogitLensAccuracy:
    def compute(self, reconstruct_tokens, tokens):
        reconstruct_tokens = np.array(reconstruct_tokens)
        tokens = np.array(tokens)
        correct = reconstruct_tokens == tokens
        return np.mean(correct)


class LogitLensMetric:
    def __init__(self, logitlens: LogitLens, logitlens_acc: LogitLensAccuracy):
        self.logitlens = logitlens
        self.logitlens_acc = logitlens_acc

    def __call__(self, preds):
        decoded = preds["decoded"]
        data = preds["data"]

        data_ids = self.logitlens.top1(data)["token_ids"].cpu()
        recons_ids = self.logitlens.top1(decoded)["token_ids"].cpu()

        return {
            "logitlens_acc": self.logitlens_acc.compute(
                data_ids.flatten().tolist(),
                recons_ids.flatten().tolist(),
            )
        }

    def compute(self, preds):
        decoded = preds["decoded"]
        data = preds["data"]

        data_ids = self.logitlens.top1(data)["token_ids"].cpu()
        recons_ids = self.logitlens.top1(decoded)["token_ids"].cpu()

        return {
            "logitlens_acc": self.logitlens_acc.compute(
                data_ids.flatten().tolist(),
                recons_ids.flatten().tolist(),
            ),
            "data_ids": data_ids.flatten().tolist(),
            "recons_ids": recons_ids.flatten().tolist(),
        }
