from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fuxian.scsi import SCSI, SCSISimulator


@dataclass
class DroneState:
    """Position and heading of one drone (metres / radians)."""

    drone_id: int
    pos: np.ndarray            # shape (3,) in metres
    heading_rad: float         # 0 = North, increasing clockwise
    velocity: float = 0.0      # m/s, only used by the adaptive layer


@dataclass
class MultiSourceFeature:
    """Joint sample for one drone at one timestep."""

    drone_id: int
    scsi: SCSI                              # GNSS source
    uwb_ranges: Dict[int, float]            # neighbour_id -> measured distance
    heading_rad: float                       # IMU reading

    @property
    def num_satellites(self) -> int:
        return self.scsi.num_satellites


class MultiSourceSampler:


    def __init__(self, sim: SCSISimulator, sigma_uwb: float = 0.30,
                 sigma_heading: float = 0.05,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.sim = sim
        self.sigma_uwb = float(sigma_uwb)
        self.sigma_heading = float(sigma_heading)
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    def sample_swarm(self, drones: Sequence[DroneState],
                     base_snr_noise: float = 0.4,
                     base_flip_prob: float = 0.0,
                     spoofed: bool = False
                     ) -> List[MultiSourceFeature]:
        """Return one feature per drone.

        ``spoofed=True`` injects an external coherent GNSS signal: every
        drone observes the *same* SCSI regardless of position (the
        spoof detector in :mod:`mpkg.spoof_detector` is designed to
        catch exactly this).
        """
        # Optional: fix a single SCSI for spoofed scenario
        spoof_scsi: Optional[SCSI] = None
        if spoofed:
            spoof_scsi = self.sim.sample(distance=0.0,
                                          base_snr_noise=base_snr_noise,
                                          base_flip_prob=base_flip_prob)

        out: List[MultiSourceFeature] = []
        positions = {d.drone_id: d.pos for d in drones}
        for d in drones:
            # ---- GNSS / SCSI ----
            if spoofed:
                # Tiny per-drone receiver noise so ``hash(w)`` differs
                # but bit majority still collapses.
                scsi = SCSI(
                    snr=spoof_scsi.snr + self.rng.normal(
                        0.0, 0.05, size=spoof_scsi.snr.size),
                    visible=set(spoof_scsi.visible),
                )
            else:
                # Use distance-from-formation-centre as the noise driver
                centre = np.mean([p for p in positions.values()], axis=0)
                dist_from_centre = float(np.linalg.norm(d.pos - centre))
                scsi = self.sim.sample(distance=dist_from_centre,
                                        base_snr_noise=base_snr_noise,
                                        base_flip_prob=base_flip_prob)

            # ---- UWB ranges to the other drones ----
            ranges: Dict[int, float] = {}
            for other in drones:
                if other.drone_id == d.drone_id:
                    continue
                true = float(np.linalg.norm(other.pos - d.pos))
                ranges[other.drone_id] = true + self.rng.normal(
                    0.0, self.sigma_uwb)

            # ---- IMU heading ----
            heading = d.heading_rad + self.rng.normal(
                0.0, self.sigma_heading)

            out.append(MultiSourceFeature(
                drone_id=d.drone_id, scsi=scsi,
                uwb_ranges=ranges, heading_rad=heading))
        return out
