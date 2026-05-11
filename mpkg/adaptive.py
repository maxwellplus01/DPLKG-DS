from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _State:
    x: float       # estimate of tau_w
    P: float       # estimate variance


class AdaptiveRefreshController:

    def __init__(self, trigger_tau: float = 0.55, horizon_s: float = 5.0,
                 alpha: float = 0.0042, beta: float = 0.00045,
                 sigma_meas: float = 0.04) -> None:
        self.trigger_tau = float(trigger_tau)
        self.horizon_s = float(horizon_s)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma_meas = float(sigma_meas)
        self._st = _State(x=0.0, P=0.01)
        self._t_since_refresh = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._st = _State(x=0.0, P=0.01)
        self._t_since_refresh = 0.0

    def predict(self, dt: float, velocity: float) -> float:
        """Advance the estimate by ``dt`` seconds at ``velocity`` m/s.

        Returns the *current* tau_w estimate (post-step).
        """
        drift = (self.alpha * velocity + self.beta) * dt
        self._st.x = self._st.x + drift
        q = (0.5 * (self.alpha * velocity + self.beta) * dt) ** 2 + 1e-6
        self._st.P = self._st.P + q
        self._t_since_refresh += dt
        return self._st.x

    def update(self, measured_tau: float) -> None:
        """Bayesian update with a fresh ``tau_w`` measurement."""
        K = self._st.P / (self._st.P + self.sigma_meas ** 2)
        self._st.x = self._st.x + K * (measured_tau - self._st.x)
        self._st.P = (1.0 - K) * self._st.P

    def should_refresh(self, velocity: float) -> bool:
        """Forecast ``horizon_s`` seconds ahead and decide."""
        forecast = self._st.x + (self.alpha * velocity
                                  + self.beta) * self.horizon_s
        return forecast > self.trigger_tau

    def mark_refreshed(self) -> None:
        self._st.x = 0.0
        self._st.P = 0.01
        self._t_since_refresh = 0.0

    @property
    def estimate(self) -> float:
        return self._st.x

    @property
    def time_since_refresh(self) -> float:
        return self._t_since_refresh
