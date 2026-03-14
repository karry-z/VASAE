"""Unit tests for VarianceExplained metric."""

import pytest
import torch

from vasae.metrics.variance_explained import VarianceExplained


class TestVarianceExplained:
    def test_perfect_reconstruction(self):
        x = torch.randn(8, 16)
        context = {"hidden_states": x, "hidden_states_recon": x}
        result = VarianceExplained().compute(context)
        assert result["variance_explained"] == pytest.approx(1.0)

    def test_zero_reconstruction(self):
        """Reconstructing with zeros should give low VE."""
        torch.manual_seed(0)
        x = torch.randn(100, 16)
        x_recon = torch.zeros_like(x)
        context = {"hidden_states": x, "hidden_states_recon": x_recon}
        result = VarianceExplained().compute(context)
        assert result["variance_explained"] < 0.1

    def test_mean_reconstruction(self):
        """Reconstructing with the mean gives VE ≈ 0."""
        torch.manual_seed(0)
        x = torch.randn(100, 16)
        x_recon = x.mean(dim=0, keepdim=True).expand_as(x)
        context = {"hidden_states": x, "hidden_states_recon": x_recon}
        result = VarianceExplained().compute(context)
        assert result["variance_explained"] == pytest.approx(0.0, abs=1e-5)

    def test_partial_reconstruction(self):
        torch.manual_seed(0)
        x = torch.randn(100, 16)
        noise = torch.randn_like(x) * 0.1
        x_recon = x + noise
        context = {"hidden_states": x, "hidden_states_recon": x_recon}
        result = VarianceExplained().compute(context)
        assert 0.0 < result["variance_explained"] < 1.0

    def test_3d_input(self):
        x = torch.randn(4, 10, 16)
        context = {"hidden_states": x, "hidden_states_recon": x}
        result = VarianceExplained().compute(context)
        assert result["variance_explained"] == pytest.approx(1.0)
