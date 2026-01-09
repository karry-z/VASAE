from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IMetric(ABC):
    @abstractmethod
    def compute(self, x, y):
        pass


class Metrics:
    def __init__(self, *args):
        self.metrics = []
        for metric_cls in args:
            if not issubclass(metric_cls, IMetric):
                raise TypeError(f"{metric_cls} is not supported.")
            self.metrics.append(metric_cls())

    def compute(self, x, y) -> Dict[str, Any]:
        res = {}
        for metric in self.metrics:
            res[type(metric).__name__] = metric.compute(x, y)
        return res


class MetricComposer:
    def __init__(self, metrics: List[IMetric]):
        self.metrics = metrics

    def __call__(self, eval_pred):
        preds, _ = eval_pred
        results = {}
        for metric in self.metrics:
            results.update(metric(preds))
        return results

    def compute(self, eval_pred):
        preds = eval_pred  # TODO
        results = {}
        for metric in self.metrics:
            results.update(metric.compute(preds))
        return results


class Aggregator:
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
