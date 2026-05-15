"""V2 Probabilistic Layer Models.

This package contains the 10 models for the v2 probabilistic upgrade:
1. BOCPD - Bayesian Online Changepoint Detection
2. Markov-Switching GARCH
3. VAR with macro spillovers
4. Kalman filter for trend slope
5. LightGBM + isotonic calibration
6. Conformal prediction wrapper
7. Bayesian Model Averaging ensemble
8. CVaR-based position sizing
9. Hawkes process (crypto only)
10. Order book microstructure features
"""

from .bocpd import BOCPD, gate_21_check

__all__ = ['BOCPD', 'gate_21_check']
__version__ = '2.0.0'
