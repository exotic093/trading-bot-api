"""Bayesian Online Changepoint Detection (BOCPD) for regime shift detection.

Implements online Bayesian changepoint detection with constant hazard rate.
Detects regime flips in real-time with formal probabilistic guarantees.

References:
- Adams & MacKay (2007): "Bayesian Online Changepoint Detection"
- Library: bayesian-changepoint-detection
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BOCPD:
    """Bayesian Online Changepoint Detection.
    
    Maintains a run-length distribution and detects regime changes online.
    Returns calibrated probability of a changepoint in the last K bars.
    
    Attributes:
        hazard_rate: Prior probability of changepoint at each timestep
        obs_likelihood: Observation likelihood model (default: Student-t)
        max_lag: Maximum run length to track (memory constraint)
        run_length_dist: Posterior distribution over run lengths
        growth_probs: P(r_t | r_{t-1}, data)
    """
    
    def __init__(
        self,
        hazard_rate: float = 1/250,  # ~1 changepoint per year for daily data
        obs_model: str = "student_t",
        max_lag: int = 1000,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        prior_dof: float = 3.0
    ):
        """Initialize BOCPD detector.
        
        Args:
            hazard_rate: Constant hazard H, P(changepoint) = H at each step
            obs_model: Observation likelihood ("student_t" or "gaussian")
            max_lag: Maximum run length to maintain
            prior_mean: Prior mean for returns
            prior_var: Prior variance scale
            prior_dof: Prior degrees of freedom (Student-t)
        """
        if not 0 < hazard_rate < 1:
            raise ValueError(f"hazard_rate must be in (0,1), got {hazard_rate}")
        
        self.hazard_rate = hazard_rate
        self.obs_model = obs_model
        self.max_lag = max_lag
        
        # Initialize run-length distribution (all mass at r=0)
        self.run_length_dist = np.zeros(max_lag)
        self.run_length_dist[0] = 1.0
        
        # Sufficient statistics for Student-t (mean, precision, dof, count)
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.prior_dof = prior_dof
        
        # Track statistics per run length
        self.sufficient_stats = self._init_sufficient_stats()
        
        # For debugging/analysis
        self.changepoint_history = []
        self.max_run_length_history = []
        
    def _init_sufficient_stats(self) -> np.ndarray:
        """Initialize sufficient statistics array."""
        # Each row: [mean, variance, dof, count]
        stats = np.zeros((self.max_lag, 4))
        stats[:, 0] = self.prior_mean
        stats[:, 1] = self.prior_var
        stats[:, 2] = self.prior_dof
        stats[:, 3] = 0
        return stats
    
    def update(self, observation: float) -> Tuple[float, np.ndarray]:
        """Update with new observation and return changepoint probability.
        
        Args:
            observation: New return/price change
            
        Returns:
            Tuple of (P(changepoint), run_length_distribution)
        """
        # Step 1: Calculate predictive probability for each run length
        pred_probs = self._predictive_probability(observation)
        
        # Step 2: Calculate growth probabilities (continue run)
        growth_probs = self.run_length_dist * pred_probs * (1 - self.hazard_rate)
        
        # Step 3: Calculate changepoint probability (restart at r=0)
        cp_prob = np.sum(self.run_length_dist * pred_probs * self.hazard_rate)
        
        # Step 4: Update run-length distribution
        new_dist = np.zeros(self.max_lag)
        new_dist[0] = cp_prob  # Mass at r=0 from changepoints
        new_dist[1:] = growth_probs[:-1]  # Shifted growth
        
        # Normalize
        evidence = np.sum(new_dist)
        if evidence > 0:
            new_dist /= evidence
        else:
            # Numerical issue - reset
            logger.warning("BOCPD evidence = 0, resetting distribution")
            new_dist[0] = 1.0
        
        self.run_length_dist = new_dist
        
        # Step 5: Update sufficient statistics
        self._update_sufficient_stats(observation)
        
        # Track history
        self.changepoint_history.append(cp_prob)
        self.max_run_length_history.append(np.argmax(self.run_length_dist))
        
        return cp_prob, self.run_length_dist
    
    def _predictive_probability(self, x: float) -> np.ndarray:
        """Calculate P(x | run_length) for all run lengths.
        
        Uses Student-t predictive distribution (Bayesian posterior predictive).
        """
        if self.obs_model == "student_t":
            return self._student_t_predictive(x)
        else:
            return self._gaussian_predictive(x)
    
    def _student_t_predictive(self, x: float) -> np.ndarray:
        """Student-t predictive probability."""
        probs = np.zeros(self.max_lag)
        
        for r in range(self.max_lag):
            mean = self.sufficient_stats[r, 0]
            var = self.sufficient_stats[r, 1]
            dof = self.sufficient_stats[r, 2]
            count = self.sufficient_stats[r, 3]
            
            # Posterior predictive variance
            pred_var = var * (1 + 1 / (count + 1)) if count > 0 else var
            
            # Student-t probability
            scale = np.sqrt(pred_var)
            probs[r] = stats.t.pdf(
                x, df=dof, loc=mean, scale=scale
            )
        
        # Prevent numerical underflow
        probs = np.maximum(probs, 1e-50)
        return probs
    
    def _gaussian_predictive(self, x: float) -> np.ndarray:
        """Gaussian predictive probability (simpler, less robust)."""
        probs = np.zeros(self.max_lag)
        
        for r in range(self.max_lag):
            mean = self.sufficient_stats[r, 0]
            var = self.sufficient_stats[r, 1]
            
            probs[r] = stats.norm.pdf(x, loc=mean, scale=np.sqrt(var))
        
        probs = np.maximum(probs, 1e-50)
        return probs
    
    def _update_sufficient_stats(self, x: float):
        """Update sufficient statistics for each run length.
        
        Uses online update rules for mean and variance.
        """
        for r in range(self.max_lag - 1):
            # Get current stats
            mean = self.sufficient_stats[r, 0]
            var = self.sufficient_stats[r, 1]
            count = self.sufficient_stats[r, 3]
            
            # Update for next run length (r+1)
            new_count = count + 1
            delta = x - mean
            new_mean = mean + delta / new_count
            
            # Welford's online variance
            M2 = var * count
            M2 += delta * (x - new_mean)
            new_var = M2 / new_count if new_count > 0 else var
            
            # Store updated stats
            self.sufficient_stats[r + 1, 0] = new_mean
            self.sufficient_stats[r + 1, 1] = max(new_var, 1e-6)  # Floor variance
            self.sufficient_stats[r + 1, 3] = new_count
        
        # Reset r=0 to prior
        self.sufficient_stats[0, 0] = self.prior_mean
        self.sufficient_stats[0, 1] = self.prior_var
        self.sufficient_stats[0, 2] = self.prior_dof
        self.sufficient_stats[0, 3] = 0
    
    def prob_changepoint_last_k(self, k: int = 10) -> float:
        """Calculate P(changepoint occurred in last K bars).
        
        This is the key output for Gate 21.
        
        Args:
            k: Lookback window
            
        Returns:
            Probability that a changepoint occurred in [t-k, t]
        """
        if len(self.changepoint_history) < k:
            # Not enough history - return 0 (safe default)
            return 0.0
        
        recent_probs = self.changepoint_history[-k:]
        
        # P(at least one changepoint) = 1 - P(no changepoints)
        # P(no changepoints) = prod(1 - p_i)
        prob_no_cp = np.prod([1 - p for p in recent_probs])
        prob_at_least_one = 1 - prob_no_cp
        
        return prob_at_least_one
    
    def get_current_regime_length(self) -> int:
        """Return most probable run length (regime age)."""
        return int(np.argmax(self.run_length_dist))
    
    def reset(self):
        """Reset detector to initial state."""
        self.run_length_dist = np.zeros(self.max_lag)
        self.run_length_dist[0] = 1.0
        self.sufficient_stats = self._init_sufficient_stats()
        self.changepoint_history = []
        self.max_run_length_history = []


def gate_21_check(bocpd: BOCPD, k: int = 10, threshold: float = 0.30) -> Tuple[bool, float]:
    """Gate 21: Check if changepoint probability is below threshold.
    
    Args:
        bocpd: BOCPD detector instance
        k: Lookback window for changepoint check
        threshold: Maximum allowed P(changepoint in last K)
        
    Returns:
        Tuple of (passes_gate, probability)
    """
    prob_cp = bocpd.prob_changepoint_last_k(k)
    passes = prob_cp < threshold
    
    if not passes:
        logger.info(
            f"Gate 21 FAIL: P(changepoint last {k}) = {prob_cp:.3f} >= {threshold}"
        )
    
    return passes, prob_cp
