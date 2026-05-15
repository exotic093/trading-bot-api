"""FastAPI routes for v1/v2 model orchestration.

Provides endpoints for:
- Volatility forecasting
- Model status and configuration
- Feature flag introspection
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from app.models.config import feature_flags, BlueprintVersion
from app.models.v2.runner import BlueprintRunner, ModelType

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["forecasting"])

# Initialize Blueprint Runner as singleton
blueprint_runner = BlueprintRunner()


class ForecastRequest(BaseModel):
    """Request model for volatility forecasting."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., BTC/USD)")
    horizon: int = Field(24, ge=1, le=168, description="Forecast horizon in hours")
    historical_data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional historical price data (timestamp, price, volume)"
    )


class ForecastResponse(BaseModel):
    """Response model for volatility forecasting."""
    symbol: str
    model_used: str
    blueprint_version: str
    forecast: List[float]
    volatility: List[float]
    regime_probabilities: Optional[List[List[float]]] = None
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatusResponse(BaseModel):
    """Response model for system status."""
    status: str
    blueprint_version: str
    feature_flags: Dict[str, Any]
    available_models: List[str]
    system_info: Dict[str, Any]


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status and configuration.
    
    Returns:
        StatusResponse: Current feature flag configuration and available models
    """
    return StatusResponse(
        status="operational",
        blueprint_version=feature_flags.blueprint_version.value,
        feature_flags={
            "enable_ms_garch": feature_flags.enable_ms_garch,
            "enable_arima_garch": feature_flags.enable_arima_garch,
            "enable_parallel_runs": feature_flags.enable_parallel_runs
        },
        available_models=[
            ModelType.ARIMA_GARCH.value if feature_flags.enable_arima_garch else None,
            ModelType.MS_GARCH.value if feature_flags.enable_ms_garch else None
        ],
        system_info={
            "runner_initialized": blueprint_runner is not None,
            "config_source": "environment"
        }
    )


@router.post("/forecast/volatility", response_model=ForecastResponse)
async def forecast_volatility(request: ForecastRequest):
    """Generate volatility forecast using configured model(s).
    
    Args:
        request: ForecastRequest with symbol and horizon
        
    Returns:
        ForecastResponse: Forecast results with model metadata
        
    Raises:
        HTTPException: If data fetching or model execution fails
    """
    try:
        logger.info(f"Forecast request for {request.symbol}, horizon={request.horizon}")
        
        # Fetch or use provided historical data
        if request.historical_data:
            # User provided data
            df = pd.DataFrame(request.historical_data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
        else:
            # TODO: Fetch from adapter (Binance/OANDA/Polygon)
            # For now, raise error requiring data
            raise HTTPException(
                status_code=400,
                detail="Historical data must be provided or adapter configured"
            )
        
        # Validate data
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data points. Require at least 100 observations."
            )
        
        # Calculate returns if not present
        if 'returns' not in df.columns and 'price' in df.columns:
            df['returns'] = np.log(df['price']).diff()
            df = df.dropna()
        
        # Execute Blueprint Runner
        result = await blueprint_runner.run(df, horizon=request.horizon)
        
        # Build response
        response = ForecastResponse(
            symbol=request.symbol,
            model_used=result.model_type.value,
            blueprint_version=feature_flags.blueprint_version.value,
            forecast=result.forecast.tolist(),
            volatility=result.volatility.tolist(),
            confidence_intervals={
                "lower": result.confidence_intervals[0].tolist(),
                "upper": result.confidence_intervals[1].tolist()
            },
            metadata={
                "aic": float(result.aic),
                "bic": float(result.bic),
                "log_likelihood": float(result.log_likelihood),
                "execution_time_ms": float(result.execution_time_ms),
                **result.metadata
            }
        )
        
        # Add regime probabilities if MS-GARCH
        if result.model_type == ModelType.MS_GARCH and hasattr(result, 'regime_probs'):
            response.regime_probabilities = result.metadata.get(
                'regime_probs', []
            )
        
        logger.info(
            f"Forecast complete: model={result.model_type.value}, "
            f"BIC={result.bic:.2f}, time={result.execution_time_ms:.2f}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Forecast execution failed: {str(e)}"
        )


@router.get("/models/info")
async def get_model_info():
    """Get information about available models.
    
    Returns:
        Dict: Model descriptions and capabilities
    """
    models = []
    
    if feature_flags.enable_arima_garch:
        models.append({
            "id": "arima_garch",
            "name": "ARIMA-GARCH",
            "version": "v1",
            "description": "Legacy ARIMA-GARCH model",
            "capabilities": ["volatility_forecasting", "return_forecasting"],
            "status": "production"
        })
    
    if feature_flags.enable_ms_garch:
        models.append({
            "id": "ms_garch",
            "name": "MS-GARCH",
            "version": "v2",
            "description": "Markov-Switching GARCH for regime detection",
            "capabilities": [
                "volatility_forecasting",
                "regime_detection",
                "regime_probability_estimation"
            ],
            "regimes": [
                {"id": 0, "name": "low_volatility", "description": "Normal market"},
                {"id": 1, "name": "high_volatility", "description": "Crisis/turbulence"}
            ],
            "status": "beta"
        })
    
    return {
        "available_models": models,
        "selection_mode": feature_flags.blueprint_version.value,
        "selection_criteria": "BIC" if feature_flags.enable_parallel_runs else "fixed"
    }
