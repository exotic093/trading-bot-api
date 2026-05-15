"""Blueprint Runner: Orchestrates v1/v2 model execution.

Handles:
- V1-only execution (legacy ARIMA-GARCH)
- V2-only execution (MS-GARCH Model #2)
- AUTO mode (parallel execution with best model selection)
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.models.config import BlueprintVersion, feature_flags

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model type identifier."""
    ARIMA_GARCH = "arima_garch"  # v1 legacy
    MS_GARCH = "ms_garch"  # v2 Model #2


@dataclass
class ModelResult:
    """Container for model execution results."""
    model_type: ModelType
    forecast: np.ndarray
    volatility: np.ndarray
    confidence_intervals: Tuple[np.ndarray, np.ndarray]
    aic: float
    bic: float
    log_likelihood: float
    execution_time_ms: float
    metadata: Dict


class BlueprintRunner:
    """Orchestrates v1/v2 model execution based on feature flags."""
    
    def __init__(self):
        self.flags = feature_flags
        logger.info(
            f"BlueprintRunner initialized: version={self.flags.blueprint_version.value}, "
            f"ms_garch={self.flags.enable_ms_garch}, "
            f"parallel={self.flags.enable_parallel_runs}"
        )
    
    async def run(
        self,
        data: pd.DataFrame,
        horizon: int = 24,
        **kwargs
    ) -> ModelResult:
        """Execute model(s) based on feature flags.
        
        Args:
            data: Time series data with DateTimeIndex
            horizon: Forecast horizon (hours)
            **kwargs: Additional model parameters
            
        Returns:
            ModelResult: Best model result (or only result if single model)
        """
        logger.info(f"Starting blueprint run for {len(data)} data points, horizon={horizon}")
        
        if self.flags.enable_parallel_runs:
            # AUTO mode: run both and select best
            return await self._run_parallel(data, horizon, **kwargs)
        elif self.flags.enable_ms_garch:
            # V2-only mode
            return await self._run_ms_garch(data, horizon, **kwargs)
        else:
            # V1-only mode (default)
            return await self._run_arima_garch(data, horizon, **kwargs)
    
    async def _run_parallel(
        self,
        data: pd.DataFrame,
        horizon: int,
        **kwargs
    ) -> ModelResult:
        """Run v1 and v2 in parallel, select best."""
        logger.info("Executing parallel v1/v2 run")
        
        # Execute both models concurrently
        tasks = [
            self._run_arima_garch(data, horizon, **kwargs),
            self._run_ms_garch(data, horizon, **kwargs)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed executions
        valid_results = [
            r for r in results
            if isinstance(r, ModelResult)
        ]
        
        if not valid_results:
            raise RuntimeError("All models failed in parallel execution")
        
        # Select best model (lowest BIC for model comparison)
        best = min(valid_results, key=lambda r: r.bic)
        logger.info(
            f"Selected {best.model_type.value} as best model: "
            f"BIC={best.bic:.2f}, AIC={best.aic:.2f}"
        )
        
        return best
    
    async def _run_arima_garch(self, data: pd.DataFrame, horizon: int, **kwargs) -> ModelResult:
        """Execute v1 ARIMA-GARCH model."""
        import time
        logger.info("Executing v1: ARIMA-GARCH")
        start = time.perf_counter()
        
        # TODO: Import and execute actual v1 model
        # from app.models.v1.arima_garch import ArimaGarchModel
        # model = ArimaGarchModel()
        # result = model.fit_predict(data, horizon, **kwargs)
        
        # Placeholder implementation
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return ModelResult(
            model_type=ModelType.ARIMA_GARCH,
            forecast=np.zeros(horizon),
            volatility=np.zeros(horizon),
            confidence_intervals=(np.zeros(horizon), np.zeros(horizon)),
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            execution_time_ms=elapsed_ms,
            metadata={"version": "v1", "status": "placeholder"}
        )
    
    async def _run_ms_garch(self, data: pd.DataFrame, horizon: int, **kwargs) -> ModelResult:
        """Execute v2 MS-GARCH Model #2."""
        import time
        logger.info("Executing v2: MS-GARCH (Model #2)")
        start = time.perf_counter()
        
        # TODO: Import and execute MS-GARCH model
        # from app.models.v2.ms_garch import MSGarchModel
        # model = MSGarchModel()
        # result = model.fit_predict(data, horizon, **kwargs)
        
        # Placeholder implementation
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return ModelResult(
            model_type=ModelType.MS_GARCH,
            forecast=np.zeros(horizon),
            volatility=np.zeros(horizon),
            confidence_intervals=(np.zeros(horizon), np.zeros(horizon)),
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            execution_time_ms=elapsed_ms,
            metadata={"version": "v2", "status": "placeholder"}
        )
