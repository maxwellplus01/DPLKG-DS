from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np


DEFAULT_NUM_SATELLITES: int = 100

# SNR range used by the paper.  Section VII-B-1: "the maximum SNR value
# received by the drones is no more than 50.  Hence, we set Max=50, Min=0".
DEFAULT_SNR_MIN: float = 0.0
DEFAULT_SNR_MAX: float = 50.0


@dataclass
class SCSI:
    """A single SCSI snapshot.

    Attributes
    ----------
    snr:
        Vector of length ``num_satellites``.  ``snr[i]`` is the measured SNR
        of satellite ``i`` (in the paper notation ``s_i``).  The value is
        only meaningful when ``i`` is in :attr:`visible`; otherwise it is
        ignored.
    visible:
        The set ``VS = {v_1, ..., v_T}`` of indices of currently visible
        satellites.
    """

    snr: np.ndarray
    visible: set

    def __post_init__(self) -> None:
        self.snr = np.asarray(self.snr, dtype=float)
        if self.snr.ndim != 1:
            raise ValueError("snr must be a 1-D vector")
        self.visible = set(int(i) for i in self.visible)
        for i in self.visible:
            if not (0 <= i < self.snr.size):
                raise ValueError(f"visible index {i} out of range")

    @property
    def num_satellites(self) -> int:
        return int(self.snr.size)

    def is_visible(self, i: int) -> bool:
        return int(i) in self.visible


@dataclass
class SCSISimulator:


    num_satellites: int = DEFAULT_NUM_SATELLITES
    visible_prob: float = 0.4
    snr_min: float = DEFAULT_SNR_MIN
    snr_max: float = DEFAULT_SNR_MAX
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    # internal "global" state of the satellite cluster
    _global_visible: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _global_snr: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.visible_prob <= 1.0:
            raise ValueError("visible_prob must be in (0, 1]")
        if self.snr_min >= self.snr_max:
            raise ValueError("snr_min must be smaller than snr_max")
        self._regenerate_global_state()

    # ------------------------------------------------------------------
    # global state management
    # ------------------------------------------------------------------
    def _regenerate_global_state(self) -> None:
        """Re-draw the global satellite cluster state."""
        self._global_visible = self.rng.random(self.num_satellites) < self.visible_prob
        self._global_snr = self.rng.uniform(
            self.snr_min + 1.0, self.snr_max - 1.0, size=self.num_satellites
        )

    def step_global_state(self, snr_drift_std: float = 0.5,
                          visibility_flip_prob: float = 0.01) -> None:
        """Advance the global state by one time step (Section IV-B,
        "Random variability")."""
        assert self._global_snr is not None and self._global_visible is not None
        self._global_snr = np.clip(
            self._global_snr + self.rng.normal(0.0, snr_drift_std,
                                               size=self.num_satellites),
            self.snr_min, self.snr_max,
        )
        flips = self.rng.random(self.num_satellites) < visibility_flip_prob
        self._global_visible = np.where(flips, ~self._global_visible,
                                        self._global_visible)

    def sample(self, distance: float = 0.0,
               distance_scale: float = 25.0,
               base_snr_noise: float = 0.4,
               base_flip_prob: float = 0.0) -> SCSI:
        """Return one SCSI snapshot for a drone at ``distance`` metres from
        the leading drone.

        The noise grows linearly with ``distance / distance_scale`` so that
        two drones located close together produce very similar SCSIs while
        farther drones diverge -- this mirrors the behaviour reported in
        Fig. 11 of the paper, where ``25 m`` is the radius at which the key
        matching probability starts to drop.
        """
        assert self._global_snr is not None and self._global_visible is not None
        scale = max(distance, 0.0) / max(distance_scale, 1e-9)
        snr_noise_std = base_snr_noise + 1.5 * scale
        flip_prob = min(0.5, base_flip_prob + 0.15 * scale)

        snr = self._global_snr + self.rng.normal(
            0.0, snr_noise_std, size=self.num_satellites)
        snr = np.clip(snr, self.snr_min, self.snr_max)

        flips = self.rng.random(self.num_satellites) < flip_prob
        visible_mask = np.where(flips, ~self._global_visible, self._global_visible)
        visible = set(np.flatnonzero(visible_mask).tolist())
        return SCSI(snr=snr, visible=visible)

    def sample_many(self, distances: Sequence[float], **kwargs) -> List[SCSI]:
        return [self.sample(distance=d, **kwargs) for d in distances]


def scsi_from_visible(snr_values: Iterable[float],
                       visible_indices: Iterable[int],
                       num_satellites: int = DEFAULT_NUM_SATELLITES) -> SCSI:
    """Convenience helper: build an :class:`SCSI` from a sparse SNR list.

    ``snr_values[k]`` is the measured SNR of ``visible_indices[k]``.  All
    other channels default to ``0`` and are marked invisible.
    """
    snr_values = list(snr_values)
    visible_indices = [int(i) for i in visible_indices]
    if len(snr_values) != len(visible_indices):
        raise ValueError("snr_values and visible_indices must match in length")
    snr = np.zeros(num_satellites, dtype=float)
    for v, idx in zip(snr_values, visible_indices):
        if not 0 <= idx < num_satellites:
            raise ValueError(f"visible index {idx} out of range")
        snr[idx] = float(v)
    return SCSI(snr=snr, visible=set(visible_indices))
