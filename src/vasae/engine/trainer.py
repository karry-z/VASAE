from logging import Logger
from typing import Optional

import torch

from vasae.metrics.base import Aggregator, MetricComposer
from vasae.models.sae import SAEModel, SAEOutput


class Trainer:
    """Unified training and evaluation loop.

    Supports both offline (DataLoader with memmap) and online
    (OnlineActivationSource with nnsight) data sources.
    """

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
        self.eval_metrics = (
            eval_metrics or metrics
        )  # eval metric are used for eval stage, may different from metrics used in training
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
        """Train for one epoch.

        Args:
            data_source: DataLoader (offline) or OnlineActivationSource (online).
            max_batches: Stop after this many batches (0 = no limit).
            log_every: Log every N batches.
            epoch: Current epoch number (1-based).
            num_epochs: Total number of epochs.
        """
        self.sae_model.train()
        aggregator = Aggregator()
        n_total = len(data_source) if hasattr(data_source, "__len__") else "?"

        for batch_i, batch in enumerate(data_source):
            activations = batch["activations"].to(self.device).float()

            # Filter out padding positions so SAE only trains on real tokens
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]  # [N_valid, D]

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
                    "l1_loss": output.l1_loss,
                    "loss_reconst": output.recon_loss,
                    "loss_lowrank": output.loss_lowrank,
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
                if output.loss_anchor:
                    parts.append(f"anchor={output.loss_anchor:.4f}")
                if output.loss_lowrank:
                    parts.append(f"lowrank={output.loss_lowrank:.4f}")
                if output.l1_loss:
                    parts.append(f"l1={output.l1_loss:.4f}")
                for k, v in eval_outcomes.items():
                    parts.append(f"{k}={v:.4f}")
                self.logger.info("[Train] " + " | ".join(parts))

            if max_batches > 0 and batch_i >= max_batches:
                break

        return aggregator.compute()

    @torch.no_grad()
    def evaluate(self, data_source, max_batches: int = 0, log_every: int = 0) -> dict:
        """Evaluate on a data source.

        Supports both offline metrics (logitlens) and online metrics
        (CE loss recovered) via the eval_metrics composer.
        """
        self.sae_model.eval()
        aggregator = Aggregator()

        n_total = len(data_source) if hasattr(data_source, "__len__") else "?"

        for batch_i, batch in enumerate(data_source):
            activations = batch["activations"].to(self.device).float()

            # Filter out padding positions for SAE and geometric metrics
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]  # [N_valid, D]

            output: SAEOutput = self.sae_model(activations)

            context = {
                "hidden_states": activations,
                "hidden_states_recon": output.hidden_states_recon,
                "sparse_activations": output.sparse_activations,
                "sae_model": self.sae_model,
                # CE metric needs original batch structure for nnsight forward pass
                "input_ids": batch.get("input_ids"),
                "attention_mask": batch.get("attention_mask"),
            }
            eval_outcomes = self.eval_metrics.compute(context)

            aggregator.add(
                {
                    "loss": output.loss.detach().cpu().item(),
                    "loss_reconst": output.recon_loss,
                    "loss_l1": output.l1_loss,
                    "loss_lowrank": output.loss_lowrank,
                    **eval_outcomes,
                },
                activations.size(0),
            )

            if log_every > 0 and self.logger is not None and (batch_i + 1) % log_every == 0:
                self.logger.info(f"[Eval] batch {batch_i + 1}/{n_total}")

            if max_batches > 0 and batch_i >= max_batches:
                break

        if self.logger is not None:
            self.logger.info(f"[Eval] {n_total} batches done")

        return aggregator.compute()
