"""Feature flag configuration for v1/v2 model orchestration.

Enables parallel CI runs and controlled rollout of MS-GARCH (Model #2).
"""

import os
from enum import Enum
from typing import Literal


class BlueprintVersion(str, Enum):
    """Blueprint version control."""
    V1 = "v1"  # Legacy ARIMA-GARCH only
    V2 = "v2"  # MS-GARCH (Model #2) only
    AUTO = "auto"  # Parallel execution, best model selection


class FeatureFlags:
    """Centralized feature flag manager."""
    
    def __init__(self):
        self._blueprint_version = self._load_blueprint_version()
        self._enable_ms_garch = self._blueprint_version in (BlueprintVersion.V2, BlueprintVersion.AUTO)
        self._enable_parallel_runs = self._blueprint_version == BlueprintVersion.AUTO
    
    @staticmethod
    def _load_blueprint_version() -> BlueprintVersion:
        """Load BLUEPRINT_VERSION from environment."""
        raw = os.getenv("BLUEPRINT_VERSION", "v1").lower()
        try:
            return BlueprintVersion(raw)
        except ValueError:
            return BlueprintVersion.V1
    
    @property
    def blueprint_version(self) -> BlueprintVersion:
        return self._blueprint_version
    
    @property
    def enable_ms_garch(self) -> bool:
        """Gate for MS-GARCH Model #2."""
        return self._enable_ms_garch
    
    @property
    def enable_parallel_runs(self) -> bool:
        """Gate for parallel v1/v2 execution."""
        return self._enable_parallel_runs
    
    @property
    def enable_arima_garch(self) -> bool:
        """Gate for legacy ARIMA-GARCH."""
        return self._blueprint_version in (BlueprintVersion.V1, BlueprintVersion.AUTO)


# Global singleton
feature_flags = FeatureFlags()
