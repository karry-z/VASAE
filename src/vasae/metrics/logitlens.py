import numpy as np


class LogitLens:
    def __init__(self, unembed, ln=None):
        self.unembed = unembed
        self.ln = ln  # TODO: check if ln is necessary

    def project(self, activation):
        if self.ln is not None:
            activation = self.ln(activation)
        logits = self.unembed(activation)
        return logits

    def top1(self, activation):
        logits = self.project(activation)
        return logits.argmax(dim=-1)


class LogitLensAccuracy:
    def compute(self, reconstruct_tokens, tokens):
        reconstruct_tokens = np.array(reconstruct_tokens)
        tokens = np.array(tokens)
        correct = reconstruct_tokens == tokens
        return np.mean(correct), np.std(correct, ddof=1)
