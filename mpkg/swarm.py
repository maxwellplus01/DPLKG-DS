from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fuxian.scsi import SCSISimulator

from .multimodal import DroneState, MultiSourceSampler, MultiSourceFeature
from .reliable_quant import QuantizedBlock, ReliableQuantizer
from .spoof_detector import SpoofDetector, SpoofReport
from .consensus import gossip_vote


@dataclass
class RoundOutput:
    consensus_bits: np.ndarray
    consensus_mask: np.ndarray
    per_drone_blocks: List[QuantizedBlock]
    spoof_report: SpoofReport
    derived_key_hex: str
    matching_rate: float            # fraction of drones whose local view matches the consensus on reliable bits
    used_uwb: bool


class MPLKGSwarm:
    def __init__(self, sim: SCSISimulator, num_drones: int,
                 sigma_uwb: float = 0.30, sigma_heading: float = 0.05,
                 rel_threshold: float = 0.18,
                 spoof_threshold: float = 3.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.sim = sim
        self.num_drones = int(num_drones)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sampler = MultiSourceSampler(sim, sigma_uwb=sigma_uwb,
                                           sigma_heading=sigma_heading,
                                           rng=self.rng)
        # canonical upper-triangle slot count for shared UWB
        n_pairs = num_drones * (num_drones - 1) // 2
        self.quantizer_full = ReliableQuantizer(
            num_satellites=sim.num_satellites,
            num_drones=num_drones,
            rel_threshold=rel_threshold,
            use_gnss=True, use_uwb=True, use_heading=True,
            n_uwb_slots=n_pairs)
        self.quantizer_gnss_only = ReliableQuantizer(
            num_satellites=sim.num_satellites,
            num_drones=num_drones,
            rel_threshold=rel_threshold,
            use_gnss=True, use_uwb=False, use_heading=False,
            n_uwb_slots=0)
        self.quantizer_no_gnss = ReliableQuantizer(
            num_satellites=sim.num_satellites,
            num_drones=num_drones,
            rel_threshold=rel_threshold,
            use_gnss=False, use_uwb=True, use_heading=True,
            n_uwb_slots=n_pairs)
        self.spoof_detector = SpoofDetector(threshold=spoof_threshold)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_shared_features(feats):

        import numpy as _np
        n = len(feats)
        ids = sorted(f.drone_id for f in feats)
        # symmetric matrix from averaged reciprocal ranges
        ranges = {(i, j): 0.0 for i in ids for j in ids if i != j}
        counts = {(i, j): 0 for i in ids for j in ids if i != j}
        for f in feats:
            for j, d in f.uwb_ranges.items():
                ranges[(f.drone_id, j)] += d
                counts[(f.drone_id, j)] += 1
        symm = {}
        for (i, j) in ranges:
            r1 = ranges[(i, j)] / max(counts[(i, j)], 1)
            r2 = ranges[(j, i)] / max(counts[(j, i)], 1)
            symm[(i, j)] = 0.5 * (r1 + r2)

        # mean heading -- circular mean to avoid wrap-around bias
        sin_m = _np.mean([_np.sin(f.heading_rad) for f in feats])
        cos_m = _np.mean([_np.cos(f.heading_rad) for f in feats])
        mean_heading = float(_np.arctan2(sin_m, cos_m))

        # canonical upper-triangle pair codes, identical for every drone
        canonical_pairs = [(i, j) for i in ids for j in ids if i < j]
        canonical_pair_ids = [i * 10000 + j for (i, j) in canonical_pairs]
        canonical_uwb = {code: symm[(i, j)] for (i, j), code
                         in zip(canonical_pairs, canonical_pair_ids)}

        shared_per_drone = {f.drone_id: (canonical_uwb, mean_heading)
                            for f in feats}
        return shared_per_drone, canonical_pair_ids

    def round(self, drones: Sequence[DroneState],
              base_snr_noise: float = 0.6,
              base_flip_prob: float = 0.0,
              spoofed: bool = False) -> RoundOutput:
        """Run one MPLKG-DS round across the swarm."""
        feats = self.sampler.sample_swarm(
            drones, base_snr_noise=base_snr_noise,
            base_flip_prob=base_flip_prob, spoofed=spoofed)

        # ---- spoof check (uses GNSS-only bits + UWB ranges) ----
        gnss_bits = []
        uwb_dists = []
        ids = [d.drone_id for d in drones]
        for f in feats:
            blk = self.quantizer_gnss_only.quantise(f, ids)
            gnss_bits.append(blk.bits)
        for i, d_i in enumerate(drones):
            row = []
            for j, d_j in enumerate(drones):
                if i == j:
                    row.append(0.0)
                    continue
                row.append(feats[i].uwb_ranges.get(d_j.drone_id, 0.0))
            uwb_dists.append(row)
        report = self.spoof_detector.detect(gnss_bits, uwb_dists)

        # ---- inject swarm-shared UWB + heading (post-broadcast) ----
        shared, canonical_pair_ids = self._make_shared_features(feats)
        feats_shared = []
        for f in feats:
            sr, mh = shared[f.drone_id]
            from .multimodal import MultiSourceFeature as _F
            feats_shared.append(_F(drone_id=f.drone_id, scsi=f.scsi,
                                    uwb_ranges=sr, heading_rad=mh))

        # ---- choose the quantiser based on detector outcome ----
        qz = self.quantizer_no_gnss if report.is_spoofed else self.quantizer_full
        blocks = [qz.quantise(f, canonical_pair_ids) for f in feats_shared]

        # ---- leader-less gossip vote ----
        consensus_bits, consensus_mask = gossip_vote(blocks)

        # ---- compute matching rate on reliable positions ----
        match_count = 0
        for blk in blocks:
            joint_mask = (consensus_mask & blk.mask).astype(bool)
            if not joint_mask.any():
                continue
            agree = (blk.bits[joint_mask] == consensus_bits[joint_mask]).all()
            if agree:
                match_count += 1
        matching_rate = match_count / max(len(blocks), 1)

        # ---- derive the group key from the reliable consensus bits ----
        reliable_bits = consensus_bits[consensus_mask.astype(bool)]
        packed = np.packbits(reliable_bits, bitorder="big").tobytes()
        derived = hashlib.sha256(packed).hexdigest()

        return RoundOutput(
            consensus_bits=consensus_bits,
            consensus_mask=consensus_mask,
            per_drone_blocks=blocks,
            spoof_report=report,
            derived_key_hex=derived,
            matching_rate=matching_rate,
            used_uwb=not report.is_spoofed and self.quantizer_full.use_uwb,
        )
