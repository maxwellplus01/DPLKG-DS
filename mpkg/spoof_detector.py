
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class SpoofReport:
    is_spoofed: bool
    anomaly_score: float           # higher = more likely spoofed
    mean_observed: float           # observed mean HD/weight
    mean_predicted: float          # predicted mean HD/weight from geometry


class SpoofDetector:


    def __init__(self,
                 distance_to_tau=None,
                 sigma: float = 0.04,
                 threshold: float = 3.0) -> None:
        if distance_to_tau is None:
            distance_to_tau = lambda d: float(np.clip(0.30 + 0.0040 * d, 0.30, 0.85))
        self.distance_to_tau = distance_to_tau
        self.sigma = float(sigma)
        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    @staticmethod
    def _hd_ratio(a: np.ndarray, b: np.ndarray) -> float:
        w = max(int(np.count_nonzero(a)), 1)
        return float(np.count_nonzero(a ^ b)) / w

    def detect(self, gnss_bits: Sequence[np.ndarray],
               uwb_distances: Sequence[Sequence[float]]) -> SpoofReport:
        """Return a :class:`SpoofReport` for the current round.

        ``gnss_bits[i]`` is the GNSS-only code-bit vector of drone ``i``;
        ``uwb_distances[i][j]`` is the UWB-measured distance from drone
        ``i`` to drone ``j`` (``i == j`` ignored).
        """
        n = len(gnss_bits)
        if n < 2:
            return SpoofReport(False, 0.0, 0.0, 0.0)
        observed: List[float] = []
        predicted: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                observed.append(self._hd_ratio(gnss_bits[i], gnss_bits[j]))
                predicted.append(self.distance_to_tau(uwb_distances[i][j]))
        obs_mean = float(np.mean(observed))
        pred_mean = float(np.mean(predicted))
        z = (pred_mean - obs_mean) / max(self.sigma, 1e-6)
        return SpoofReport(
            is_spoofed=bool(z > self.threshold),
            anomaly_score=float(z),
            mean_observed=obs_mean,
            mean_predicted=pred_mean,
        )
