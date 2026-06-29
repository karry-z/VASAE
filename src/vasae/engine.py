from logging import Logger
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import torch
from nnsight import NNsight

from vasae.models import SAEModel, SAEOutput


class IMetric(ABC):
    @abstractmethod
    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        ...


class MetricComposer:
    def __init__(self, metrics: List[IMetric]):
        self.metrics = metrics

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        results = {}
        for metric in self.metrics:
            results.update(metric.compute(context))
        return results


class Aggregator:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, batch_metrics: Dict[str, float], batch_size: int):
        for key, value in batch_metrics.items():
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "item"):
                value = value.item()
            self.sums[key] = self.sums.get(key, 0.0) + value * batch_size
            self.counts[key] = self.counts.get(key, 0) + batch_size

    def compute(self):
        return {key: self.sums[key] / self.counts[key] for key in self.sums}


def _get_layer_proxy(model: NNsight, layer_idx: int):
    """Resolve the layer proxy inside an nnsight trace context."""
    m = model._model
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        return model.gpt_neox.layers[layer_idx]
    raise ValueError(
        f"Cannot find transformer layers for {type(m).__name__}. "
        "Add support in _get_layer_proxy()."
    )


def _as_hidden_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def extract_activations(
    model: NNsight, input_ids: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    """Extract activations from a specific transformer layer."""
    with model.trace(input_ids):
        layer = _get_layer_proxy(model, layer_idx)
        h = _as_hidden_tensor(layer.output).save()
    return h


def patch_and_forward(
    model: NNsight,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Patch activations at a specific layer and return final logits."""
    with model.trace(input_ids, attention_mask=attention_mask):
        layer = _get_layer_proxy(model, layer_idx)
        output = layer.output
        hidden = _as_hidden_tensor(output)
        layer.output = _replace_hidden(output, intervention_fn(hidden))
        logits = model.output.logits.save()
    return logits


class Trainer:
    """Training and evaluation loop for OnlineActivationSource batches."""

    def __init__(
        self,
        sae_model: SAEModel,
        optimizer: torch.optim.Optimizer,
        metrics: MetricComposer,
        eval_metrics: Optional[MetricComposer] = None,
        device: str = "cpu",
        logger: Optional[Logger] = None,
    ):
        self.sae_model = sae_model
        self.optimizer = optimizer
        self.metrics = metrics
        self.eval_metrics = eval_metrics or metrics
        self.device = device
        self.logger = logger

    def train_epoch(
        self,
        data_source,
        max_batches: int = 0,
        log_every: int = 10,
        epoch: int = 0,
        num_epochs: int = 0,
    ) -> dict:
        self.sae_model.train()
        aggregator = Aggregator()
        n_total = len(data_source) if hasattr(data_source, "__len__") else "?"

        for batch_i, batch in enumerate(data_source):
            activations = batch["activations"].to(self.device).float()
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]

            self.optimizer.zero_grad()
            output: SAEOutput = self.sae_model(activations)

            context = {
                "hidden_states": activations,
                "hidden_states_recon": output.hidden_states_recon,
                "sparse_activations": output.sparse_activations,
            }
            eval_outcomes = self.metrics.compute(context)

            aggregator.add(
                {
                    "loss": output.loss,
                    "loss_reconst": output.recon_loss,
                    "loss_anchor": output.loss_anchor,
                    **eval_outcomes,
                },
                activations.size(0),
            )

            output.loss.backward()
            self.optimizer.step()

            if self.logger is not None and (batch_i + 1) % log_every == 0:
                epoch_str = f"ep {epoch}/{num_epochs} " if num_epochs > 0 else ""
                parts = [
                    f"{epoch_str}batch {batch_i + 1}/{n_total}",
                    f"loss={output.loss.item():.4f}",
                    f"recon={output.recon_loss:.4f}",
                ]
                if output.loss_anchor is not None:
                    parts.append(f"anchor={output.loss_anchor.item():.4f}")
                for k, v in eval_outcomes.items():
                    parts.append(f"{k}={v:.4f}")
                self.logger.info("[Train] " + " | ".join(parts))

            if max_batches > 0 and (batch_i + 1) >= max_batches:
                break

        return aggregator.compute()

    @torch.no_grad()
    def evaluate(self, data_source, max_batches: int = 0, log_every: int = 0) -> dict:
        self.sae_model.eval()
        aggregator = Aggregator()
        n_total = len(data_source) if hasattr(data_source, "__len__") else "?"

        for batch_i, batch in enumerate(data_source):
            activations = batch["activations"].to(self.device).float()
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]

            output: SAEOutput = self.sae_model(activations)

            context = {
                "hidden_states": activations,
                "hidden_states_recon": output.hidden_states_recon,
                "sparse_activations": output.sparse_activations,
                "sae_model": self.sae_model,
                "input_ids": batch.get("input_ids"),
                "attention_mask": batch.get("attention_mask"),
            }
            eval_outcomes = self.eval_metrics.compute(context)

            aggregator.add(
                {
                    "loss": output.loss.detach().cpu().item(),
                    "loss_reconst": output.recon_loss,
                    **eval_outcomes,
                },
                activations.size(0),
            )

            if log_every > 0 and self.logger is not None and (batch_i + 1) % log_every == 0:
                self.logger.info(f"[Eval] batch {batch_i + 1}/{n_total}")

            if max_batches > 0 and (batch_i + 1) >= max_batches:
                break

        if self.logger is not None:
            self.logger.info(f"[Eval] {n_total} batches done")

        return aggregator.compute()
