"""Integration tests for v1/v2 orchestration and feature flags.

Verifies:
- Feature flag configuration loading
- Blueprint runner mode switching
- Parallel execution and model selection
- MS-GARCH model integration
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set test environment
os.environ["BLUEPRINT_VERSION"] = "auto"

from app.models.config import FeatureFlags, BlueprintVersion
from app.models.v2.runner import BlueprintRunner, ModelType, ModelResult
from app.models.v2.ms_garch import MSGarchModel, MSGarchConfig


class TestFeatureFlags:
    """Test feature flag configuration."""
    
    def test_v1_mode(self, monkeypatch):
        """Test v1-only mode configuration."""
        monkeypatch.setenv("BLUEPRINT_VERSION", "v1")
        flags = FeatureFlags()
        
        assert flags.blueprint_version == BlueprintVersion.V1
        assert flags.enable_arima_garch is True
        assert flags.enable_ms_garch is False
        assert flags.enable_parallel_runs is False
    
    def test_v2_mode(self, monkeypatch):
        """Test v2-only mode configuration."""
        monkeypatch.setenv("BLUEPRINT_VERSION", "v2")
        flags = FeatureFlags()
        
        assert flags.blueprint_version == BlueprintVersion.V2
        assert flags.enable_arima_garch is False
        assert flags.enable_ms_garch is True
        assert flags.enable_parallel_runs is False
    
    def test_auto_mode(self, monkeypatch):
        """Test auto/parallel mode configuration."""
        monkeypatch.setenv("BLUEPRINT_VERSION", "auto")
        flags = FeatureFlags()
        
        assert flags.blueprint_version == BlueprintVersion.AUTO
        assert flags.enable_arima_garch is True
        assert flags.enable_ms_garch is True
        assert flags.enable_parallel_runs is True
    
    def test_invalid_mode_defaults_to_v1(self, monkeypatch):
        """Test invalid mode falls back to v1."""
        monkeypatch.setenv("BLUEPRINT_VERSION", "invalid")
        flags = FeatureFlags()
        
        assert flags.blueprint_version == BlueprintVersion.V1


class TestMSGarchModel:
    """Test MS-GARCH Model #2 functionality."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return series."""
        np.random.seed(42)
        n = 200
        
        # Simulate regime-switching returns
        regime = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        low_vol = np.random.normal(0, 0.01, n)
        high_vol = np.random.normal(0, 0.03, n)
        
        returns = np.where(regime == 0, low_vol, high_vol)
        return returns
    
    def test_model_initialization(self):
        """Test MS-GARCH model initializes correctly."""
        config = MSGarchConfig(n_regimes=2, max_iter=100)
        model = MSGarchModel(config)
        
        assert model.config.n_regimes == 2
        assert model.config.max_iter == 100
        assert model.is_fitted is False
    
    def test_model_fit(self, sample_returns):
        """Test model fitting process."""
        model = MSGarchModel()
        model.fit(sample_returns)
        
        assert model.is_fitted is True
        assert 'P' in model.params
        assert 'garch_0' in model.params
        assert 'garch_1' in model.params
    
    def test_model_predict(self, sample_returns):
        """Test forecast generation."""
        model = MSGarchModel()
        model.fit(sample_returns)
        
        result = model.predict(sample_returns, horizon=24)
        
        assert isinstance(result.forecasts, np.ndarray)
        assert len(result.forecasts) == 24
        assert len(result.volatility) == 24
        assert result.regime_probs.shape == (24, 2)
        assert result.aic > 0
        assert result.bic > 0
    
    def test_regime_probabilities_sum_to_one(self, sample_returns):
        """Test regime probabilities are valid."""
        model = MSGarchModel()
        model.fit(sample_returns)
        result = model.predict(sample_returns, horizon=24)
        
        for t in range(24):
            prob_sum = result.regime_probs[t].sum()
            assert abs(prob_sum - 1.0) < 1e-6


class TestBlueprintRunner:
    """Test Blueprint Runner orchestration."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 200))
        returns = np.diff(np.log(prices))
        
        df = pd.DataFrame({
            'price': prices[1:],
            'returns': returns
        }, index=dates[1:])
        
        return df
    
    @pytest.mark.asyncio
    async def test_runner_v1_mode(self, sample_data, monkeypatch):
        """Test runner in v1-only mode."""
        monkeypatch.setenv("BLUEPRINT_VERSION", "v1")
        runner = BlueprintRunner()
        
        result = await runner.run(sample_data, horizon=24)
        
        assert isinstance(result, ModelResult)
        assert result.model_type == ModelType.ARIMA_GARCH
        assert len(result.forecast) == 24
    
    @pytest.mark.asyncio
    async def test_runner_v2_mode(self, sample_data, monkeypatch):
        """Test runner in v2-only mode."""
        monkeypatch.setenv("BLUEPRINT_VERSION", "v2")
        runner = BlueprintRunner()
        
        result = await runner.run(sample_data, horizon=24)
        
        assert isinstance(result, ModelResult)
        assert result.model_type == ModelType.MS_GARCH
        assert len(result.forecast) == 24
    
    @pytest.mark.asyncio
    async def test_runner_auto_mode(self, sample_data, monkeypatch):
        """Test runner in auto/parallel mode."""
        monkeypatch.setenv("BLUEPRINT_VERSION", "auto")
        runner = BlueprintRunner()
        
        result = await runner.run(sample_data, horizon=24)
        
        assert isinstance(result, ModelResult)
        # Best model selected based on BIC
        assert result.model_type in [ModelType.ARIMA_GARCH, ModelType.MS_GARCH]
        assert len(result.forecast) == 24
        assert result.bic > 0


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_auto_mode(self):
        """Test complete pipeline in AUTO mode."""
        # Set environment
        os.environ["BLUEPRINT_VERSION"] = "auto"
        
        # Generate synthetic data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='h')
        returns = np.random.normal(0, 0.02, 500)
        df = pd.DataFrame({'returns': returns}, index=dates)
        
        # Initialize runner
        runner = BlueprintRunner()
        
        # Execute
        result = await runner.run(df, horizon=24)
        
        # Verify results
        assert result is not None
        assert hasattr(result, 'forecast')
        assert hasattr(result, 'volatility')
        assert hasattr(result, 'aic')
        assert hasattr(result, 'bic')
        assert result.execution_time_ms > 0
        
        print(f"✓ Selected model: {result.model_type.value}")
        print(f"✓ BIC: {result.bic:.2f}")
        print(f"✓ Execution time: {result.execution_time_ms:.2f}ms")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
