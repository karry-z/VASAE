def test_import_models():
    from vasae import SAEConfig, SAEModel, SAEOutput
    from vasae.models import SAEConfig as ModelConfig
    from vasae.models.encoders import LinearEncoder
    from vasae.models.sparsity import TopKSparse

    assert SAEConfig is ModelConfig
    assert SAEModel is not None
    assert SAEOutput is not None
    assert LinearEncoder is not None
    assert TopKSparse is not None


def test_import_data_engine_metrics_analysis_utils():
    from vasae.analysis import nearest_token_alignment, nearest_token_names
    from vasae.data import OnlineActivationSource
    from vasae.engine import Trainer, extract_activations, patch_and_forward
    from vasae.metrics import (
        Aggregator,
        CELossRecovered,
        IMetric,
        LogitLens,
        LogitLensAccuracy,
        LogitLensMetric,
        MetricComposer,
        VarianceExplained,
        cross_entropy,
    )
    from vasae.utils import get_logger, set_seed

    assert nearest_token_alignment is not None
    assert nearest_token_names is not None
    assert OnlineActivationSource is not None
    assert Trainer is not None
    assert extract_activations is not None
    assert patch_and_forward is not None
    assert Aggregator is not None
    assert CELossRecovered is not None
    assert IMetric is not None
    assert LogitLens is not None
    assert LogitLensAccuracy is not None
    assert LogitLensMetric is not None
    assert MetricComposer is not None
    assert VarianceExplained is not None
    assert cross_entropy is not None
    assert get_logger is not None
    assert set_seed is not None
