from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IMetric(ABC):
    @abstractmethod
    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        context may contain:
        - "hidden_states": original activations
        - "hidden_states_recon": SAE reconstruction
        - "sparse_activations": sparse codes
        - "model": nnsight-wrapped LLM (for online metrics)
        - "input_ids", "attention_mask": original inputs (for online metrics)
        - "layer_idx": layer index
        - "sae_model": SAE model instance
        """
        ...


class MetricComposer:
    """Compose multiple IMetric instances, merging result dicts."""

    def __init__(self, metrics: List[IMetric]):
        self.metrics = metrics

    def reset(self) -> None:
        """Reset stateful metrics before a new pass.

        Notes
        -----
        Metrics may optionally define a ``reset`` method. Stateless metrics do
        not need to implement it and are skipped.
        """
        for m in self.metrics:
            reset = getattr(m, "reset", None)
            if reset is not None:
                reset()

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Compute all metrics for a batch context.

        Parameters
        ----------
        context
            Batch-level values consumed by the metrics, such as hidden states,
            reconstructions, sparse activations, or token inputs.

        Returns
        -------
        dict[str, float]
            Merged metric outputs for the current batch.
        """
        results = {}
        for m in self.metrics:
            results.update(m.compute(context))
        return results

    def finalize(self) -> Dict[str, float]:
        """Finalize stateful metrics after all batches have been processed.

        Returns
        -------
        dict[str, float]
            Merged outputs from metrics that define ``finalize``. Metrics
            without a ``finalize`` method are skipped.
        """
        results = {}
        for m in self.metrics:
            finalize = getattr(m, "finalize", None)
            if finalize is not None:
                results.update(finalize())
        return results


class Aggregator:
    """Batch-weighted averaging across batches."""

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, batch_metrics: Dict[str, float], batch_size: int):
        for k, v in batch_metrics.items():
            if v is None:
                continue
            if hasattr(v, "detach"):
                v = v.detach()
            if hasattr(v, "item"):
                v = v.item()
            self.sums[k] = self.sums.get(k, 0.0) + v * batch_size
            self.counts[k] = self.counts.get(k, 0) + batch_size

    def compute(self):
        return {k: self.sums[k] / self.counts[k] for k in self.sums}
