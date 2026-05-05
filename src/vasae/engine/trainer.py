import logging
import math
import time
from pathlib import Path
from typing import Callable, Optional

import torch

from vasae.metrics.base import Aggregator, MetricComposer
from vasae.models.sae import SAEModel, SAEOutput

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _source_nominal_batch_tokens(data_source) -> int | None:
    batch_size = getattr(data_source, "batch_size", None)
    max_length = getattr(data_source, "max_length", None)
    if isinstance(batch_size, int) and isinstance(max_length, int):
        if batch_size > 0 and max_length > 0:
            return batch_size * max_length
    return None


def _source_progress_totals(
    data_source,
    max_batches: int = 0,
) -> tuple[int | None, bool, int | None]:
    """Return planned batches, whether they are estimated, and planned tokens."""
    total_batches = None
    estimated_batches = False

    if hasattr(data_source, "__len__"):
        try:
            total_batches = len(data_source)
        except TypeError:
            total_batches = None

    token_budget = getattr(data_source, "total_token_budget", None)
    nominal_batch_tokens = _source_nominal_batch_tokens(data_source)
    if total_batches is None and token_budget is not None and nominal_batch_tokens:
        total_batches = math.ceil(token_budget / nominal_batch_tokens)
        estimated_batches = True

    if max_batches > 0:
        if total_batches is None:
            total_batches = max_batches
        else:
            total_batches = min(total_batches, max_batches)

    planned_tokens = token_budget if isinstance(token_budget, int) else None
    if max_batches > 0 and nominal_batch_tokens:
        max_batch_tokens = max_batches * nominal_batch_tokens
        planned_tokens = (
            min(planned_tokens, max_batch_tokens)
            if planned_tokens is not None
            else max_batch_tokens
        )

    return total_batches, estimated_batches, planned_tokens


def _count_activation_items(activations: torch.Tensor) -> int:
    if activations.ndim <= 1:
        return int(activations.size(0))
    return int(math.prod(activations.shape[:-1]))


def _count_batch_tokens(batch: dict, activations: torch.Tensor) -> int:
    mask = batch.get("attention_mask")
    if mask is not None:
        return int(mask.sum().item())
    return _count_activation_items(activations)


def _progress_line(
    *,
    label: str,
    batch_count: int,
    total_batches: int | None,
    estimated_batches: bool,
    tokens_done: int,
    total_tokens: int | None,
    start_time: float,
    extra_parts: list[str],
) -> str:
    batch_total = "?"
    if total_batches is not None:
        batch_total = f"~{total_batches:,}" if estimated_batches else f"{total_batches:,}"

    now = time.monotonic()
    elapsed = now - start_time
    batch_label = f"{label} batch" if label else "batch"
    parts = [f"{batch_label} {batch_count:,}/{batch_total}"]

    if total_tokens is not None:
        pct = min(100.0, tokens_done / total_tokens * 100) if total_tokens else 100.0
        parts.append(f"tokens={tokens_done:,}/{total_tokens:,} ({pct:.2f}%)")
    elif tokens_done:
        parts.append(f"tokens={tokens_done:,}")

    parts.append(f"elapsed={_format_duration(elapsed)}")

    if tokens_done > 0:
        tokens_per_sec = tokens_done / max(elapsed, 1e-9)
        if total_tokens is not None and tokens_per_sec > 0:
            remaining = max(total_tokens - tokens_done, 0)
            parts.append(f"eta={_format_duration(remaining / tokens_per_sec)}")
        parts.append(f"tok/s={tokens_per_sec:.1f}")
    elif total_batches is not None and batch_count > 0:
        batches_per_sec = batch_count / max(elapsed, 1e-9)
        remaining_batches = max(total_batches - batch_count, 0)
        parts.append(f"eta={_format_duration(remaining_batches / batches_per_sec)}")

    parts.extend(extra_parts)
    return " | ".join(parts)


def _summary_metric_parts(metrics: dict) -> list[str]:
    parts = [f"loss={metrics['loss']:.4f}"]
    if "variance_explained" in metrics:
        parts.append(f"VE={metrics['variance_explained']:.4f}")
    if "logitlens_acc" in metrics:
        parts.append(f"logitlens={metrics['logitlens_acc'] * 100:.2f}%")
    if "loss_recovered" in metrics:
        parts.append(f"CE_recovered={metrics['loss_recovered']:.4f}")
    return parts


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
        log_every: int = 10,
        log_interval_seconds: int = 300,
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
                log_every=log_every,
                log_interval_seconds=log_interval_seconds,
                epoch=epoch_num,
                num_epochs=num_epochs,
            )
            last_train = train_out
            logger.info("[Train] " + " ".join(_summary_metric_parts(train_out)))

            logger.info(f"[Eval] starting epoch {epoch_num}/{num_epochs}")
            eval_out = self.evaluate(
                eval_source,
                max_batches=max_batches,
                log_every=log_every,
                log_interval_seconds=log_interval_seconds,
            )
            last_eval = eval_out
            logger.info("[Eval] " + " ".join(_summary_metric_parts(eval_out)))

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
                        logger.info(f"Saving best model to {save_path}...")
                        self.sae_model.save_pretrained(save_path)
                        logger.info(f"Best model checkpoint written to {save_path}")
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
            logger.info(f"Saving final model to {save_path}...")
            self.sae_model.save_pretrained(save_path)
            logger.info(f"Final model checkpoint written to {save_path}")

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
        log_interval_seconds: int = 300,
        epoch: int = 0,
        num_epochs: int = 0,
    ) -> dict:
        """Train for one epoch.

        Args:
            data_source: DataLoader (offline) or OnlineActivationSource (online).
            optimizer: Optimizer used for this training epoch.
            max_batches: Stop after this many batches (0 = no limit).
            log_every: Log every N batches.
            log_interval_seconds: Also log at least every N seconds (0 = disabled).
            epoch: Current epoch number (1-based).
            num_epochs: Total number of epochs.
        """
        self.sae_model.train()
        self.metrics.reset()
        aggregator = Aggregator()
        total_batches, estimated_batches, total_tokens = _source_progress_totals(
            data_source,
            max_batches=max_batches,
        )
        epoch_str = f"ep {epoch}/{num_epochs} " if num_epochs > 0 else ""
        logger.info(
            "[Train] starting "
            + _progress_line(
                label=epoch_str.rstrip(),
                batch_count=0,
                total_batches=total_batches,
                estimated_batches=estimated_batches,
                tokens_done=0,
                total_tokens=total_tokens,
                start_time=time.monotonic(),
                extra_parts=[],
            )
        )

        start_time = time.monotonic()
        last_log_time = start_time
        tokens_done = 0

        for batch_i, batch in enumerate(data_source):
            batch_count = batch_i + 1
            activations = batch["activations"].to(self.device).float()
            batch_tokens = _count_batch_tokens(batch, activations)
            tokens_done += batch_tokens

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

            now = time.monotonic()
            batch_log_due = log_every > 0 and batch_count % log_every == 0
            time_log_due = (
                log_interval_seconds > 0
                and now - last_log_time >= log_interval_seconds
            )
            if batch_log_due or time_log_due:
                metric_parts = [
                    f"loss={output.loss.item():.4f}",
                    f"recon={output.recon_loss:.4f}",
                ]
                if output.loss_anchor:
                    metric_parts.append(f"anchor={output.loss_anchor:.4f}")
                if output.l1_loss:
                    metric_parts.append(f"l1={output.l1_loss:.4f}")
                for k, v in eval_outcomes.items():
                    metric_parts.append(f"{k}={v:.4f}")
                logger.info(
                    "[Train] "
                    + _progress_line(
                        label=epoch_str.rstrip(),
                        batch_count=batch_count,
                        total_batches=total_batches,
                        estimated_batches=estimated_batches,
                        tokens_done=tokens_done,
                        total_tokens=total_tokens,
                        start_time=start_time,
                        extra_parts=metric_parts,
                    )
                )
                last_log_time = now

            if max_batches > 0 and batch_count >= max_batches:
                break

        logger.info(
            "[Train] complete "
            + _progress_line(
                label=epoch_str.rstrip(),
                batch_count=batch_count if "batch_count" in locals() else 0,
                total_batches=total_batches,
                estimated_batches=estimated_batches,
                tokens_done=tokens_done,
                total_tokens=total_tokens,
                start_time=start_time,
                extra_parts=[],
            )
        )
        return {
            **aggregator.compute(),
            **self.metrics.finalize(),
            "batches_processed": batch_count if "batch_count" in locals() else 0,
            "tokens_processed": tokens_done,
        }

    @torch.no_grad()
    def evaluate(
        self,
        data_source,
        max_batches: int = 0,
        log_every: int = 0,
        log_interval_seconds: int = 300,
    ) -> dict:
        """Evaluate on a data source.

        Supports both offline metrics (logitlens) and online metrics
        (CE loss recovered) via the eval_metrics composer.
        """
        self.sae_model.eval()
        self.eval_metrics.reset()
        aggregator = Aggregator()

        total_batches, estimated_batches, total_tokens = _source_progress_totals(
            data_source,
            max_batches=max_batches,
        )
        logger.info(
            "[Eval] starting "
            + _progress_line(
                label="",
                batch_count=0,
                total_batches=total_batches,
                estimated_batches=estimated_batches,
                tokens_done=0,
                total_tokens=total_tokens,
                start_time=time.monotonic(),
                extra_parts=[],
            )
        )

        start_time = time.monotonic()
        last_log_time = start_time
        tokens_done = 0

        for batch_i, batch in enumerate(data_source):
            batch_count = batch_i + 1
            activations = batch["activations"].to(self.device).float()
            batch_tokens = _count_batch_tokens(batch, activations)
            tokens_done += batch_tokens

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

            now = time.monotonic()
            batch_log_due = log_every > 0 and batch_count % log_every == 0
            time_log_due = (
                log_interval_seconds > 0
                and now - last_log_time >= log_interval_seconds
            )
            if batch_log_due or time_log_due:
                logger.info(
                    "[Eval] "
                    + _progress_line(
                        label="",
                        batch_count=batch_count,
                        total_batches=total_batches,
                        estimated_batches=estimated_batches,
                        tokens_done=tokens_done,
                        total_tokens=total_tokens,
                        start_time=start_time,
                        extra_parts=[],
                    )
                )
                last_log_time = now

            if max_batches > 0 and batch_count >= max_batches:
                break

        logger.info(
            "[Eval] complete "
            + _progress_line(
                label="",
                batch_count=batch_count if "batch_count" in locals() else 0,
                total_batches=total_batches,
                estimated_batches=estimated_batches,
                tokens_done=tokens_done,
                total_tokens=total_tokens,
                start_time=start_time,
                extra_parts=[],
            )
        )

        return {
            **aggregator.compute(),
            **self.eval_metrics.finalize(),
            "batches_processed": batch_count if "batch_count" in locals() else 0,
            "tokens_processed": tokens_done,
        }
