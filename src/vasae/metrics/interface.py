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
            results.update(metric(preds))
        return results
