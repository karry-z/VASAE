"""Import tests for all public modules.

Ensures all canonical import paths resolve correctly.
"""


# -- models ------------------------------------------------------------------

def test_import_models_canonical():
    from vasae.models.sae import SAEConfig, SAEModel, SAEOutput
    from vasae.models.sparsity import TopKSparse, BatchTopKSparse, IdentitySparsity
    from vasae.models.encoders import LinearEncoder, MLPEncoder


def test_import_models_init():
    from vasae.models import SAEConfig, SAEModel, SAEOutput


def test_import_decompose_sae():
    from vasae.models.decompose_sae import DecomposeSAEModel, DecomposeSAEOutput


def test_import_dualpath_sae():
    from vasae.models.dualpath_sae import DualPathSAE, DualPathSAEOutput


def test_import_factory():
    from vasae.models.factory import (
        load_model,
        get_layers,
        get_embedding,
        get_lm_head,
    )


def test_import_factory_legacy():
    from vasae.models.factory import (
        BlackBoxModelConfig,
        get_blackbox_model,
        load_embedding_layer,
        load_unembedding_layer,
    )


# -- data ---------------------------------------------------------------------

def test_import_data_schema():
    from vasae.data.schema import DataConfig, LayerMeta, Meta


def test_import_data_init():
    from vasae.data import DataConfig, LayerMeta, Meta


def test_import_dataset():
    from vasae.data.dataset import GPT2LayerActivations, get_dataloader, load_meta


# -- engine -------------------------------------------------------------------

def test_import_engine_config():
    from vasae.engine.config import TrainConfig


def test_import_engine_init():
    from vasae.engine import TrainConfig, train, evaluate


def test_import_engine_trainer():
    from vasae.engine.trainer import Trainer


def test_import_engine_intervention():
    from vasae.engine.intervention import extract_activations, patch_and_forward


# -- metrics ------------------------------------------------------------------

def test_import_metrics_base():
    from vasae.metrics.base import IMetric, MetricComposer, Aggregator


def test_import_metrics_init():
    from vasae.metrics import IMetric, MetricComposer, Aggregator


def test_import_logitlens():
    from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy, LogitLensMetric


def test_import_ce_loss():
    from vasae.metrics.ce_loss import CELossRecovered, cross_entropy


def test_import_activation_source():
    from vasae.data.activation_source import OnlineActivationSource


# -- utils --------------------------------------------------------------------

def test_import_utils():
    from vasae.utils.log import get_logger
    from vasae.utils.seed import set_seed
