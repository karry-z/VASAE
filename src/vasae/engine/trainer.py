import logging
from pathlib import Path
from typing import Callable, Optional

import torch

from vasae.metrics.base import Aggregator, MetricComposer
from vasae.models.sae import SAEModel, SAEOutput

logger = logging.getLogger(__name__)


class Trainer:
    """Unified training and evaluation loop.

    Supports both offline (DataLoader with memmap) and online
    (OnlineActivationSource with nnsight) data sources.
    """

    def __init__(
        self,
        sae_model: SAEModel,
        metrics: MetricComposer,
        eval_metrics: Optional[MetricComposer] = None,
        device: str = "cpu",
    ):
        self.sae_model = sae_model
        self.metrics = metrics
        self.eval_metrics = (
            eval_metrics or metrics
        )  # eval metric are used for eval stage, may different from metrics used in training
        self.device = device

    def fit(
        self,
        train_source,
        eval_source,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        max_batches: int = 0,
        patience: int = 0,
        save_dir: str | Path | None = None,
        log_fn: Optional[Callable[[dict], None]] = None,
        load_best_model_fn: Optional[Callable[[Path], SAEModel]] = None,
    ) -> dict:
        """Train across epochs, evaluate, log, and optionally early-stop."""
        save_path = Path(save_dir) if save_dir is not None else None
        best_eval_loss = float("inf")
        patience_counter = 0
        stopped_epoch = num_epochs
        last_train: dict = {}
        last_eval: dict = {}

        for epoch in range(num_epochs):
            epoch_num = epoch + 1
            logger.info(f"=== Epoch {epoch_num}/{num_epochs} ===")

            train_out = self.train_epoch(
                train_source,
                optimizer=optimizer,
                max_batches=max_batches,
                epoch=epoch_num,
                num_epochs=num_epochs,
            )
            last_train = train_out
            logger.info(
                f"[Train] loss={train_out['loss']:.4f} "
                f"VE={train_out.get('variance_explained', 0):.4f} "
                f"logitlens={train_out.get('logitlens_acc', 0) * 100:.2f}%"
            )

            eval_out = self.evaluate(eval_source, max_batches=max_batches)
            last_eval = eval_out
            logger.info(
                f"[Eval] loss={eval_out['loss']:.4f} "
                f"VE={eval_out.get('variance_explained', 0):.4f} "
                f"logitlens={eval_out.get('logitlens_acc', 0) * 100:.2f}% "
                f"CE_recovered={eval_out.get('loss_recovered', 0):.4f}"
            )

            if log_fn is not None:
                log_fn(
                    {
                        "epoch": epoch_num,
                        **{f"train/{k}": v for k, v in train_out.items()},
                        **{f"eval/{k}": v for k, v in eval_out.items()},
                    }
                )

            if patience > 0:
                if eval_out["loss"] < best_eval_loss:
                    best_eval_loss = eval_out["loss"]
                    patience_counter = 0
                    if save_path is not None:
                        self.sae_model.save_pretrained(save_path)
                    logger.info(
                        f"Best model saved (eval_loss={best_eval_loss:.4f})"
                    )
                else:
                    patience_counter += 1
                    logger.info(f"No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    stopped_epoch = epoch_num
                    logger.info(f"Early stopping at epoch {stopped_epoch}")
                    break

        if patience > 0 and save_path is not None:
            logger.info("Loading best model for final test...")
            del optimizer
            old_model = self.sae_model
            self.sae_model = None
            del old_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.sae_model = (
                load_best_model_fn(save_path)
                if load_best_model_fn is not None
                else SAEModel.from_pretrained(save_path).to(self.device).float()
            )
        elif save_path is not None:
            self.sae_model.save_pretrained(save_path)

        return {
            "stopped_epoch": stopped_epoch,
            "best_eval_loss": best_eval_loss,
            "train": last_train,
            "eval": last_eval,
        }

    def train_epoch(
        self,
        data_source,
        optimizer: torch.optim.Optimizer,
        max_batches: int = 0,
        log_every: int = 10,
        epoch: int = 0,
        num_epochs: int = 0,
    ) -> dict:
        """Train for one epoch.

        Args:
            data_source: DataLoader (offline) or OnlineActivationSource (online).
            optimizer: Optimizer used for this training epoch.
            max_batches: Stop after this many batches (0 = no limit).
            log_every: Log every N batches.
            epoch: Current epoch number (1-based).
            num_epochs: Total number of epochs.
        """
        self.sae_model.train()
        self.metrics.reset()
        aggregator = Aggregator()
        n_total = len(data_source) if hasattr(data_source, "__len__") else "?"

        for batch_i, batch in enumerate(data_source):
            activations = batch["activations"].to(self.device).float()

            # Filter out padding positions so SAE only trains on real tokens
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]  # [N_valid, D]

            optimizer.zero_grad()
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
                    "loss_anchor": output.loss_anchor,
                    **eval_outcomes,
                },
                activations.size(0),
            )

            output.loss.backward()
            optimizer.step()

            if log_every > 0 and (batch_i + 1) % log_every == 0:
                epoch_str = f"ep {epoch}/{num_epochs} " if num_epochs > 0 else ""
                parts = [
                    f"{epoch_str}batch {batch_i + 1}/{n_total}",
                    f"loss={output.loss.item():.4f}",
                    f"recon={output.recon_loss:.4f}",
                ]
                if output.loss_anchor:
                    parts.append(f"anchor={output.loss_anchor:.4f}")
                if output.l1_loss:
                    parts.append(f"l1={output.l1_loss:.4f}")
                for k, v in eval_outcomes.items():
                    parts.append(f"{k}={v:.4f}")
                logger.info("[Train] " + " | ".join(parts))

            if max_batches > 0 and batch_i >= max_batches:
                break

        return {**aggregator.compute(), **self.metrics.finalize()}

    @torch.no_grad()
    def evaluate(self, data_source, max_batches: int = 0, log_every: int = 0) -> dict:
        """Evaluate on a data source.

        Supports both offline metrics (logitlens) and online metrics
        (CE loss recovered) via the eval_metrics composer.
        """
        self.sae_model.eval()
        self.eval_metrics.reset()
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
                    **eval_outcomes,
                },
                activations.size(0),
            )

            if log_every > 0 and (batch_i + 1) % log_every == 0:
                logger.info(f"[Eval] batch {batch_i + 1}/{n_total}")

            if max_batches > 0 and batch_i >= max_batches:
                break

        logger.info(f"[Eval] {n_total} batches done")

        return {**aggregator.compute(), **self.eval_metrics.finalize()}
