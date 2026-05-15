"""MS-GARCH Model #2: Markov-Switching GARCH for volatility regime detection.

Implements a two-regime Markov-Switching GARCH model for:
- High volatility regime detection
- Low volatility regime detection
- Regime-conditional volatility forecasting
- Smooth regime probability estimation
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class MSGarchConfig:
    """Configuration for MS-GARCH model."""
    n_regimes: int = 2  # Number of volatility regimes
    max_iter: int = 500  # Maximum EM iterations
    tol: float = 1e-4  # Convergence tolerance
    random_seed: int = 42


@dataclass
class MSGarchResult:
    """Results from MS-GARCH model fitting."""
    forecasts: np.ndarray  # Point forecasts
    volatility: np.ndarray  # Conditional volatility
    regime_probs: np.ndarray  # Regime probabilities (n_obs x n_regimes)
    transition_matrix: np.ndarray  # Regime transition probabilities
    parameters: Dict  # Model parameters
    aic: float
    bic: float
    log_likelihood: float


class MSGarchModel:
    """Markov-Switching GARCH model for regime-dependent volatility.
    
    Implements a two-regime MS-GARCH(1,1) model:
    - Regime 0: Low volatility (normal market)
    - Regime 1: High volatility (crisis/turbulence)
    """
    
    def __init__(self, config: Optional[MSGarchConfig] = None):
        self.config = config or MSGarchConfig()
        self.is_fitted = False
        self.params = {}
        np.random.seed(self.config.random_seed)
    
    def fit(self, returns: np.ndarray) -> 'MSGarchModel':
        """Fit MS-GARCH model to return series.
        
        Args:
            returns: Array of log returns
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting MS-GARCH with {len(returns)} observations")
        
        # Initialize parameters
        self._initialize_parameters(returns)
        
        # Run Expectation-Maximization algorithm
        log_lik_old = -np.inf
        for iteration in range(self.config.max_iter):
            # E-step: Calculate regime probabilities
            regime_probs, filtered_probs = self._expectation_step(returns)
            
            # M-step: Update parameters
            self._maximization_step(returns, regime_probs)
            
            # Check convergence
            log_lik = self._compute_log_likelihood(returns, regime_probs)
            
            if abs(log_lik - log_lik_old) < self.config.tol:
                logger.info(f"Converged at iteration {iteration}, log-likelihood: {log_lik:.2f}")
                break
            
            log_lik_old = log_lik
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        returns: np.ndarray,
        horizon: int = 24
    ) -> MSGarchResult:
        """Generate forecasts for future volatility.
        
        Args:
            returns: Historical returns for initialization
            horizon: Forecast horizon
            
        Returns:
            MSGarchResult: Forecasts and regime information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-step ahead forecasts")
        
        # Get current regime probabilities
        _, filtered_probs = self._expectation_step(returns)
        current_regime_probs = filtered_probs[-1]
        
        # Multi-step ahead forecasting
        forecasts = np.zeros(horizon)
        volatilities = np.zeros(horizon)
        regime_probs_forecast = np.zeros((horizon, self.config.n_regimes))
        
        # Initialize with last observed values
        last_return = returns[-1]
        last_variance = self._estimate_variance(returns[-20:])  # Use recent data
        
        for h in range(horizon):
            # Propagate regime probabilities
            if h == 0:
                regime_probs_forecast[h] = current_regime_probs
            else:
                regime_probs_forecast[h] = regime_probs_forecast[h-1] @ self.params['P']
            
            # Forecast volatility for each regime
            vol_regime_0 = self._forecast_variance_regime(
                last_variance, last_return, regime=0, steps=h+1
            )
            vol_regime_1 = self._forecast_variance_regime(
                last_variance, last_return, regime=1, steps=h+1
            )
            
            # Weighted average across regimes
            volatilities[h] = np.sqrt(
                regime_probs_forecast[h, 0] * vol_regime_0 +
                regime_probs_forecast[h, 1] * vol_regime_1
            )
        
        # Point forecast (mean is 0 under GARCH assumption)
        forecasts = np.zeros(horizon)
        
        # Calculate information criteria
        n_params = self._count_parameters()
        n_obs = len(returns)
        ll = self._compute_log_likelihood(returns, filtered_probs)
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(n_obs)
        
        return MSGarchResult(
            forecasts=forecasts,
            volatility=volatilities,
            regime_probs=regime_probs_forecast,
            transition_matrix=self.params['P'],
            parameters=self.params,
            aic=aic,
            bic=bic,
            log_likelihood=ll
        )
    
    def _initialize_parameters(self, returns: np.ndarray):
        """Initialize model parameters."""
        # Transition matrix (P): prob of moving between regimes
        self.params['P'] = np.array([
            [0.95, 0.05],  # Low vol regime persistence
            [0.10, 0.90]   # High vol regime persistence
        ])
        
        # GARCH parameters for each regime
        # Regime 0: Low volatility (omega, alpha, beta)
        self.params['garch_0'] = {'omega': 0.01, 'alpha': 0.05, 'beta': 0.90}
        
        # Regime 1: High volatility
        self.params['garch_1'] = {'omega': 0.05, 'alpha': 0.10, 'beta': 0.85}
    
    def _expectation_step(
        self,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """E-step: Calculate regime probabilities using Hamilton filter."""
        n = len(returns)
        n_regimes = self.config.n_regimes
        
        # Filtered probabilities
        filtered = np.zeros((n, n_regimes))
        smoothed = np.zeros((n, n_regimes))
        
        # Initialize with stationary distribution
        filtered[0] = self._stationary_distribution()
        
        # Forward pass (Hamilton filter)
        for t in range(1, n):
            # Predict regime probabilities
            predicted = filtered[t-1] @ self.params['P']
            
            # Update with observation likelihood
            likelihoods = np.array([
                self._regime_likelihood(returns[t], regime)
                for regime in range(n_regimes)
            ])
            
            # Normalize
            filtered[t] = predicted * likelihoods
            filtered[t] /= filtered[t].sum()
        
        # Backward pass (Kim smoother) - simplified
        smoothed = filtered.copy()  # Use filtered as approximation
        
        return smoothed, filtered
    
    def _regime_likelihood(self, return_val: float, regime: int) -> float:
        """Calculate likelihood of observation given regime."""
        params = self.params[f'garch_{regime}']
        # Simplified: use unconditional variance
        variance = params['omega'] / (1 - params['alpha'] - params['beta'])
        return norm.pdf(return_val, 0, np.sqrt(variance))
    
    def _maximization_step(self, returns: np.ndarray, regime_probs: np.ndarray):
        """M-step: Update parameters given regime probabilities."""
        # Update transition matrix
        self._update_transition_matrix(regime_probs)
        
        # Update GARCH parameters for each regime (simplified)
        # In full implementation, use MLE with regime probabilities as weights
        pass
    
    def _update_transition_matrix(self, regime_probs: np.ndarray):
        """Update transition matrix using regime probabilities."""
        n_regimes = self.config.n_regimes
        P_new = np.zeros((n_regimes, n_regimes))
        
        for i in range(n_regimes):
            for j in range(n_regimes):
                numerator = (regime_probs[:-1, i] * regime_probs[1:, j]).sum()
                denominator = regime_probs[:-1, i].sum()
                P_new[i, j] = numerator / (denominator + 1e-10)
        
        # Normalize rows
        P_new = P_new / (P_new.sum(axis=1, keepdims=True) + 1e-10)
        self.params['P'] = P_new
    
    def _stationary_distribution(self) -> np.ndarray:
        """Calculate stationary distribution of Markov chain."""
        P = self.params['P']
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        stationary = np.real(eigenvectors[:, np.argmax(eigenvalues)])
        return stationary / stationary.sum()
    
    def _forecast_variance_regime(
        self,
        last_variance: float,
        last_return: float,
        regime: int,
        steps: int
    ) -> float:
        """Forecast variance for specific regime."""
        params = self.params[f'garch_{regime}']
        omega, alpha, beta = params['omega'], params['alpha'], params['beta']
        
        # Multi-step GARCH forecast
        h_var = last_variance
        for _ in range(steps):
            h_var = omega + alpha * last_return**2 + beta * h_var
        
        return h_var
    
    def _estimate_variance(self, returns: np.ndarray) -> float:
        """Estimate current variance from recent returns."""
        return np.var(returns)
    
    def _compute_log_likelihood(
        self,
        returns: np.ndarray,
        regime_probs: np.ndarray
    ) -> float:
        """Compute log-likelihood of data."""
        ll = 0
        for t, ret in enumerate(returns):
            likelihood = sum(
                regime_probs[t, r] * self._regime_likelihood(ret, r)
                for r in range(self.config.n_regimes)
            )
            ll += np.log(likelihood + 1e-10)
        return ll
    
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        # Transition matrix (n_regimes * (n_regimes - 1)) + GARCH params per regime
        n_transition = self.config.n_regimes * (self.config.n_regimes - 1)
        n_garch_per_regime = 3  # omega, alpha, beta
        return n_transition + self.config.n_regimes * n_garch_per_regime
