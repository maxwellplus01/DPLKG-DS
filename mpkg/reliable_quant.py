
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .multimodal import MultiSourceFeature


@dataclass
class QuantizedBlock:
    bits: np.ndarray            # uint8, shape (n_bits,)
    mask: np.ndarray            # uint8, shape (n_bits,)
    confidence: np.ndarray      # float, shape (n_bits,)
    drone_id: int

    @property
    def num_reliable(self) -> int:
        return int(self.mask.sum())


class ReliableQuantizer:

    SNR_BITS_PER_SAT = 4
    UWB_BITS_PER_NEIGHBOUR = 4
    HEADING_BITS = 4

    def __init__(self, num_satellites: int, num_drones: int,
                 snr_min: float = 0.0, snr_max: float = 50.0,
                 uwb_max: float = 200.0,
                 rel_threshold: float = 0.18,
                 use_gnss: bool = True,
                 use_uwb: bool = True,
                 use_heading: bool = True,
                 n_uwb_slots: Optional[int] = None) -> None:
        self.num_satellites = int(num_satellites)
        self.num_drones = int(num_drones)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.uwb_max = float(uwb_max)
        self.rel_threshold = float(rel_threshold)
        self.use_gnss = bool(use_gnss)
        self.use_uwb = bool(use_uwb)
        self.use_heading = bool(use_heading)
        # number of UWB scalar slots encoded per drone.  Defaults to
        # ``num_drones - 1`` (i.e. one bin per neighbour, paper-style).
        # In ``shared`` mode the swarm passes ``n*(n-1)/2`` (one bin per
        # canonical upper-triangle pair).
        self.n_uwb_slots = (int(n_uwb_slots)
                            if n_uwb_slots is not None
                            else max(0, self.num_drones - 1))

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    @property
    def n_bits(self) -> int:
        n = 0
        if self.use_gnss:
            n += self.num_satellites * self.SNR_BITS_PER_SAT
        if self.use_uwb:
            n += self.n_uwb_slots * self.UWB_BITS_PER_NEIGHBOUR
        if self.use_heading:
            n += self.HEADING_BITS
        return n

    # ------------------------------------------------------------------
    # Core quantiser
    # ------------------------------------------------------------------
    @staticmethod
    def _quantise_scalar(x: float, lo: float, hi: float, n_bits: int
                          ) -> Tuple[np.ndarray, float]:
        """Equal-width quantiser with confidence ``in [0, 1]``."""
        bins = 1 << n_bits
        x = float(np.clip(x, lo, hi))
        bin_width = (hi - lo) / bins
        idx = int(min(bins - 1, (x - lo) / bin_width))
        # distance to nearest boundary, normalised by bin half-width
        centre = lo + (idx + 0.5) * bin_width
        conf = 1.0 - abs(x - centre) / (bin_width / 2.0)
        conf = max(0.0, min(1.0, conf))
        # Gray-coded bits
        gray = idx ^ (idx >> 1)
        bits = np.array([(gray >> (n_bits - 1 - i)) & 1
                         for i in range(n_bits)], dtype=np.uint8)
        return bits, conf

    # ------------------------------------------------------------------
    def quantise(self, feat: MultiSourceFeature,
                 neighbour_ids: Sequence[int]) -> QuantizedBlock:
        bits = np.zeros(self.n_bits, dtype=np.uint8)
        mask = np.zeros(self.n_bits, dtype=np.uint8)
        conf = np.zeros(self.n_bits, dtype=float)
        cursor = 0

        # ---- GNSS / SCSI ----
        if self.use_gnss:
            for i in range(self.num_satellites):
                if i in feat.scsi.visible:
                    b, c = self._quantise_scalar(
                        float(feat.scsi.snr[i]), self.snr_min,
                        self.snr_max, self.SNR_BITS_PER_SAT)
                    bits[cursor:cursor + self.SNR_BITS_PER_SAT] = b
                    conf[cursor:cursor + self.SNR_BITS_PER_SAT] = c
                    if c >= self.rel_threshold:
                        mask[cursor:cursor + self.SNR_BITS_PER_SAT] = 1
                # else: leave zeros, mask stays 0 (drop these bits)
                cursor += self.SNR_BITS_PER_SAT

        # ---- UWB ranges ----
        if self.use_uwb:
            slots_used = 0
            for nid in neighbour_ids:
                if slots_used >= self.n_uwb_slots:
                    break
                if nid == feat.drone_id:
                    continue
                d = feat.uwb_ranges.get(nid, None)
                if d is None:
                    cursor += self.UWB_BITS_PER_NEIGHBOUR
                    slots_used += 1
                    continue
                b, c = self._quantise_scalar(
                    float(d), 0.0, self.uwb_max,
                    self.UWB_BITS_PER_NEIGHBOUR)
                bits[cursor:cursor + self.UWB_BITS_PER_NEIGHBOUR] = b
                conf[cursor:cursor + self.UWB_BITS_PER_NEIGHBOUR] = c
                if c >= self.rel_threshold:
                    mask[cursor:cursor + self.UWB_BITS_PER_NEIGHBOUR] = 1
                cursor += self.UWB_BITS_PER_NEIGHBOUR
                slots_used += 1
            # if fewer slots filled than configured, pad cursor
            while slots_used < self.n_uwb_slots:
                cursor += self.UWB_BITS_PER_NEIGHBOUR
                slots_used += 1

        # ---- IMU heading ----
        if self.use_heading:
            heading_norm = float(feat.heading_rad % (2 * np.pi))
            b, c = self._quantise_scalar(
                heading_norm, 0.0, 2 * np.pi, self.HEADING_BITS)
            bits[cursor:cursor + self.HEADING_BITS] = b
            conf[cursor:cursor + self.HEADING_BITS] = c
            if c >= self.rel_threshold:
                mask[cursor:cursor + self.HEADING_BITS] = 1
            cursor += self.HEADING_BITS

        assert cursor == self.n_bits
        return QuantizedBlock(bits=bits, mask=mask, confidence=conf,
                              drone_id=feat.drone_id)
