# V2 Model Orchestration System

## Overview

This directory contains the **v2 model orchestration infrastructure** for the trading bot, implementing:

- **Feature flag system** for controlled v1/v2 rollout
- **Blueprint Runner** for async model execution and selection
- **MS-GARCH Model #2** for volatility regime detection
- **Parallel execution** with automatic best model selection

## Architecture

```
app/models/
├── config.py              # Feature flag configuration (v1/v2/auto modes)
├── v2/
│   ├── __init__.py        # v2 package initialization
│   ├── runner.py          # Blueprint Runner orchestration
│   ├── ms_garch.py        # MS-GARCH Model #2 implementation
│   ├── bocpd.py           # BOCPD Model #1 (probabilistic layer)
│   ├── test_integration.py # Integration tests
│   └── README.md          # This file
```

## Feature Flag Modes

### V1 Mode (Production Default)
```bash
export BLUEPRINT_VERSION=v1
```
- Runs **legacy ARIMA-GARCH only**
- Safest option for production
- No MS-GARCH execution

### V2 Mode (Testing)
```bash
export BLUEPRINT_VERSION=v2
```
- Runs **MS-GARCH Model #2 only**
- For testing new model in isolation
- Skips ARIMA-GARCH entirely

### AUTO Mode (Validation)
```bash
export BLUEPRINT_VERSION=auto
```
- Runs **both models in parallel**
- Automatically selects best model via BIC
- Recommended for v1/v2 validation phase
- Graceful fallback if one model fails

## Quick Start

### 1. Basic Usage

```python
import pandas as pd
from app.models.v2.runner import BlueprintRunner

# Initialize runner (reads BLUEPRINT_VERSION automatically)
runner = BlueprintRunner()

# Prepare your data (time series with DateTimeIndex)
data = pd.DataFrame({
    'price': [...],
    'returns': [...]
}, index=pd.date_range('2024-01-01', periods=500, freq='h'))

# Execute forecast
result = await runner.run(data, horizon=24)

print(f"Model selected: {result.model_type}")
print(f"BIC score: {result.bic}")
print(f"Forecast: {result.forecast}")
print(f"Volatility: {result.volatility}")
```

### 2. MS-GARCH Model Direct Usage

```python
from app.models.v2.ms_garch import MSGarchModel, MSGarchConfig
import numpy as np

# Configure model
config = MSGarchConfig(
    n_regimes=2,
    max_iter=500,
    tol=1e-4
)

# Initialize and fit
model = MSGarchModel(config)
model.fit(returns)  # numpy array of log returns

# Generate forecast
result = model.predict(returns, horizon=24)

print(f"Regime probabilities: {result.regime_probs}")
print(f"Transition matrix: {result.transition_matrix}")
```

## Model Components

### MS-GARCH Model #2

**Purpose**: Detect and forecast volatility regimes

**Key Features**:
- Two-regime Markov-Switching GARCH(1,1)
- Hamilton filter for regime probability estimation
- EM algorithm for parameter learning
- Regime-conditional volatility forecasting

**Regimes**:
- **Regime 0**: Low volatility (normal market conditions)
- **Regime 1**: High volatility (crisis/turbulent periods)

**Parameters**:
- Transition matrix `P`: Regime switching probabilities
- GARCH parameters per regime: `ω`, `α`, `β`

### Blueprint Runner

**Purpose**: Orchestrate v1/v2 execution based on feature flags

**Execution Flow**:
1. Check `BLUEPRINT_VERSION` environment variable
2. Route to appropriate execution path:
   - **v1**: Execute ARIMA-GARCH
   - **v2**: Execute MS-GARCH
   - **auto**: Execute both in parallel, select best
3. Return standardized `ModelResult`

**Model Selection** (AUTO mode):
- Both models run concurrently using `asyncio.gather`
- Best model selected by **lowest BIC**
- Graceful handling of failed executions

## Railway Deployment

### Environment Variables

Configure in Railway dashboard → Variables tab:

```
BLUEPRINT_VERSION=v1     # Start with v1 (safest)
```

### Rollout Strategy

**Phase 1: Validation** (Week 1-2)
```
BLUEPRINT_VERSION=auto   # Run both models in parallel
```
- Monitor both model performance
- Compare BIC scores
- Verify v2 stability

**Phase 2: Gradual Rollout** (Week 3-4)
```
BLUEPRINT_VERSION=v2     # Switch to v2 only
```
- Monitor production performance
- Ready to rollback to v1 if issues arise

**Phase 3: Production** (Week 5+)
```
BLUEPRINT_VERSION=v2     # v2 becomes primary
```
- v2 fully deployed
- v1 remains available for emergencies

## Testing

### Run Integration Tests

```bash
# Run all v2 tests
pytest app/models/v2/test_integration.py -v

# Run specific test class
pytest app/models/v2/test_integration.py::TestMSGarchModel -v

# Run with coverage
pytest app/models/v2/test_integration.py --cov=app.models.v2
```

### Test Coverage

- ✅ Feature flag configuration (v1/v2/auto)
- ✅ MS-GARCH model fit/predict
- ✅ Regime probability estimation
- ✅ Blueprint Runner orchestration
- ✅ Parallel execution and selection
- ✅ End-to-end pipeline

## Performance Characteristics

### MS-GARCH Model #2

**Computational Complexity**:
- **Fitting**: O(T × K² × I) where:
  - T = number of observations
  - K = number of regimes (2)
  - I = EM iterations (~50-200)
- **Forecasting**: O(H × K) where H = horizon

**Memory Usage**:
- ~50MB for 1000 observations
- Scales linearly with data size

**Typical Execution Time** (500 obs, 24h horizon):
- Fitting: ~200-500ms
- Forecasting: ~10-20ms

### Parallel Execution (AUTO mode)

**Expected Overhead**:
- ~20-30% compared to single model
- Dominated by slower model (usually v1 ARIMA-GARCH)
- Concurrent execution reduces total time vs sequential

## Monitoring & Debugging

### Logging

All components use Python's `logging` module:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)

# View feature flag initialization
logger = logging.getLogger('app.models.config')

# View runner execution
logger = logging.getLogger('app.models.v2.runner')

# View MS-GARCH model
logger = logging.getLogger('app.models.v2.ms_garch')
```

### Key Metrics to Monitor

1. **Model Selection** (AUTO mode)
   - Which model wins (v1 vs v2)?
   - BIC difference between models
   - Selection consistency over time

2. **Performance**
   - Execution time per model
   - Convergence iterations (MS-GARCH)
   - Forecast quality (MAE, RMSE)

3. **Regime Detection** (MS-GARCH)
   - Regime probability distribution
   - Regime transition frequency
   - High volatility regime duration

## Troubleshooting

### Issue: Feature flag not working

**Symptoms**: Always runs v1 regardless of environment variable

**Solution**:
```python
# Verify environment variable is set
import os
print(os.getenv('BLUEPRINT_VERSION'))

# Check feature flags singleton
from app.models.config import feature_flags
print(feature_flags.blueprint_version)
print(feature_flags.enable_ms_garch)
```

### Issue: MS-GARCH not converging

**Symptoms**: Model fitting takes too long or fails

**Solution**:
```python
# Reduce max iterations
config = MSGarchConfig(max_iter=100)

# Increase tolerance
config = MSGarchConfig(tol=1e-3)

# Check data quality
assert len(returns) >= 100  # Need sufficient data
assert returns.std() > 0    # Need variation
```

### Issue: Parallel execution failing

**Symptoms**: AUTO mode always returns same model or errors

**Solution**:
```python
# Check both models can run independently
os.environ['BLUEPRINT_VERSION'] = 'v1'
result_v1 = await runner.run(data, horizon=24)

os.environ['BLUEPRINT_VERSION'] = 'v2'
result_v2 = await runner.run(data, horizon=24)

# Verify error handling
try:
    os.environ['BLUEPRINT_VERSION'] = 'auto'
    result = await runner.run(data, horizon=24)
except Exception as e:
    print(f"Error: {e}")
```

## Next Steps

### Step 2: Calibration Layer (Pending)

After validating v1/v2 orchestration:

1. **Implement calibration router**
   - Gate 21: Pre-signal validation
   - Gate 22: Post-execution checks
   - Gate 14: Position sizing

2. **Wire to API endpoints**
   - Update forecast endpoints
   - Add v2 metadata to responses

3. **Performance benchmarking**
   - Backtest on historical data
   - Compare v1 vs v2 accuracy

### Step 1: Integration (Current Phase)

✅ Feature flags implemented  
✅ MS-GARCH Model #2 complete  
✅ Blueprint Runner orchestration  
✅ Integration tests  
⏳ Awaiting API keys for deployment

---

## Questions?

Refer to inline code documentation or integration tests for detailed examples.
