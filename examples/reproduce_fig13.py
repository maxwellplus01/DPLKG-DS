from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from fuxian.fuzzy_extractor import BCHCode
from fuxian.quantization import Quantizer, hamming_distance
from fuxian.scsi import SCSISimulator

# Re-use the distance noise model from the Fig. 11 reproduction.
from reproduce_fig11 import (NUM_SATELLITES, VISIBLE_PROB,
                              encode_w_factory, train_quantizer)
from fuxian.scsi import SCSI


KEY_BITS = 256                  # Section VII-C: AES-256 group keys
SPEEDS = [5, 10, 25]            # Fig. 13 legend
TAU_GRID = [0.51, 0.55, 0.59, 0.63, 0.67, 0.71, 0.75, 0.79]


DISTANCE_NOISE_FIG13: Dict[float, Tuple[float, float]] = {
    5.0:    (5.2, 0.075),
    10.0:   (5.9, 0.095),
    15.0:   (6.3, 0.110),
    22.5:   (6.8, 0.130),
    35.0:   (7.2, 0.145),
    45.0:   (7.6, 0.160),
    60.0:   (8.2, 0.180),
    75.0:   (8.8, 0.205),
    90.0:   (9.5, 0.235),
    100.0:  (10.0, 0.260),
}


def _interp_noise(d: float) -> Tuple[float, float]:
    keys = sorted(DISTANCE_NOISE_FIG13.keys())
    if d <= keys[0]:
        return DISTANCE_NOISE_FIG13[keys[0]]
    if d >= keys[-1]:
        return DISTANCE_NOISE_FIG13[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= d <= keys[i + 1]:
            d1, d2 = keys[i], keys[i + 1]
            n1, f1 = DISTANCE_NOISE_FIG13[d1]
            n2, f2 = DISTANCE_NOISE_FIG13[d2]
            t = (d - d1) / (d2 - d1)
            return (n1 + t * (n2 - n1), f1 + t * (f2 - f1))
    return DISTANCE_NOISE_FIG13[keys[-1]]


def sample_pair_phys(sim, distance: float) -> Tuple[SCSI, SCSI]:
    """Like ``reproduce_fig11.sample_pair`` but accepting any physical
    distance (interpolating in :data:`DISTANCE_NOISE_FIG13`)."""
    leader = sim.sample(distance=0.0, base_snr_noise=0.3, base_flip_prob=0.0)
    snr_noise, flip_prob = _interp_noise(distance)
    follower = sim.sample(distance=0.0, base_snr_noise=snr_noise,
                           base_flip_prob=flip_prob)
    return leader, follower


# ---------------------------------------------------------------------------
# Reference values digitised from Fig. 13 (page 4476 of the PDF).
# ---------------------------------------------------------------------------
PAPER_REFERENCE: Dict[int, Dict[float, float]] = {
    5:  {0.51: 102, 0.55: 76,  0.59: 57,  0.63: 30, 0.67: 22, 0.71: 18, 0.75: 16, 0.79: 14},
    10: {0.51: 230, 0.55: 153, 0.59: 110, 0.63: 60, 0.67: 42, 0.71: 38, 0.75: 35, 0.79: 33},
    25: {0.51: 255, 0.55: 255, 0.59: 255, 0.63: 147, 0.67: 105, 0.71: 87, 0.75: 75, 0.79: 65},
}
# v = 25 m/s saturates at ~255 bps because the time-to-leave-region
# becomes < 1 s; the paper's text confirms "leave the original area
# within 1 s, resulting in a higher KGR of over 250 bps".
KGR_CAP = 256.0   # 256 bits / 1 s = 256 bps theoretical ceiling


# ---------------------------------------------------------------------------
# Calibrate tau_w as a function of distance.
# ---------------------------------------------------------------------------
def calibrate_tau_vs_distance(sim: SCSISimulator, encode_w,
                               distances: List[float],
                               n_samples: int = 200) -> Dict[float, float]:
    """Estimate ``mean tau_w(d) = mean HD/weight at distance d``."""
    out: Dict[float, float] = {}
    for d in distances:
        ratios: List[float] = []
        for _ in range(n_samples):
            sim.step_global_state(snr_drift_std=0.3,
                                   visibility_flip_prob=0.005)
            leader_scsi, follower_scsi = sample_pair_phys(sim, d)
            wL = encode_w(leader_scsi)
            wF = encode_w(follower_scsi)
            weight = int(np.count_nonzero(wL))
            if weight == 0:
                continue
            ratios.append(hamming_distance(wL, wF) / weight)
        out[d] = float(np.mean(ratios))
    return out


def threshold_distance(tau_target: float,
                        tau_table: Dict[float, float]) -> float:
    """Smallest distance d for which ``tau_w(d) >= tau_target``.

    Uses linear interpolation between the calibration grid points.
    """
    distances = sorted(tau_table.keys())
    taus = [tau_table[d] for d in distances]
    if tau_target <= taus[0]:
        # extrapolate towards 0 (assume tau_w(0) = 0)
        return tau_target * distances[0] / max(taus[0], 1e-6)
    if tau_target >= taus[-1]:
        # extrapolate the last segment linearly
        d1, d2 = distances[-2], distances[-1]
        t1, t2 = taus[-2], taus[-1]
        slope = (d2 - d1) / max(t2 - t1, 1e-6)
        return d2 + (tau_target - t2) * slope
    for i in range(len(distances) - 1):
        if taus[i] <= tau_target <= taus[i + 1]:
            d1, d2 = distances[i], distances[i + 1]
            t1, t2 = taus[i], taus[i + 1]
            return d1 + (tau_target - t1) * (d2 - d1) / max(t2 - t1, 1e-6)
    return distances[-1]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run() -> None:
    rng = np.random.default_rng(2024)
    sim = SCSISimulator(num_satellites=NUM_SATELLITES,
                         visible_prob=VISIBLE_PROB, rng=rng)
    qz = train_quantizer(sim, num_training=300, M=10)
    bch = BCHCode(t=20, m=10)
    encode_w = encode_w_factory(qz, bch.n)

    distances = sorted(DISTANCE_NOISE_FIG13.keys())
    print("Calibrating mean tau_w vs distance ...")
    print(f"  {'distance (m)':<14}{'mean tau_w':<14}")
    tau_table = calibrate_tau_vs_distance(sim, encode_w, distances,
                                           n_samples=1500)
    # enforce monotonicity (the calibration is noisy)
    keys = sorted(tau_table.keys())
    last = -1.0
    for d in keys:
        if tau_table[d] < last:
            tau_table[d] = last
        last = tau_table[d]
    for d in distances:
        print(f"  {d:<14}{tau_table[d]:<14.3f}")

    print("\nKey Generation Rate (bps):")
    header = f"{'tau_w':<8}" + "".join(f"v={v}m/s (repro/paper)".ljust(22)
                                        for v in SPEEDS)
    print(header)
    results: Dict[int, Dict[float, float]] = {v: {} for v in SPEEDS}
    for tau in TAU_GRID:
        d_thresh = threshold_distance(tau, tau_table)
        d_thresh = max(d_thresh, 0.5)  # avoid div-by-zero
        cells = [f"{tau:<8.2f}"]
        for v in SPEEDS:
            time_s = d_thresh / v
            kgr = KEY_BITS / time_s
            kgr = min(kgr, KGR_CAP)  # 1 SCSI snapshot per second cap
            results[v][tau] = kgr
            paper = PAPER_REFERENCE[v][tau]
            cells.append(f"{kgr:6.1f} / {paper:<6.0f}    ")
        print("".join(cells))

    save_results(results, tau_table)


def save_results(results, tau_table) -> None:
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, "fig13_data.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("tau_w,"
                + ",".join(f"{v}m/s_repro,{v}m/s_paper" for v in SPEEDS)
                + "\n")
        for tau in TAU_GRID:
            row = [f"{tau:.2f}"]
            for v in SPEEDS:
                row.append(f"{results[v][tau]:.1f}")
                row.append(f"{PAPER_REFERENCE[v][tau]:.0f}")
            f.write(",".join(row) + "\n")
    print(f"\nSaved CSV to {csv_path}")

    summary_path = os.path.join(outdir, "fig13_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Reproduction of Fig. 13 (KGR vs tau_w)\n")
        f.write("=" * 60 + "\n")
        f.write("Calibrated mean tau_w(d):\n")
        for d in sorted(tau_table.keys()):
            f.write(f"  d = {int(d):3d} m   tau_w = {tau_table[d]:.3f}\n")
        f.write("\nMean absolute deviation per speed:\n")
        for v in SPEEDS:
            diffs = [abs(results[v][tau] - PAPER_REFERENCE[v][tau])
                     for tau in TAU_GRID]
            f.write(f"  v = {v:2d} m/s : MAD = {np.mean(diffs):6.1f} bps, "
                    f"max |diff| = {max(diffs):6.1f} bps\n")
    print(f"Saved summary to {summary_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable, skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    markers = {5: "^", 10: "o", 25: "s"}
    colors = {5: "black", 10: "red", 25: "blue"}
    for ax, label, src in (
            (axes[0], "Reproduced",
             {v: results[v] for v in SPEEDS}),
            (axes[1], "Paper Fig. 13 (digitised)", PAPER_REFERENCE)):
        for v in SPEEDS:
            xs = TAU_GRID
            ys = [src[v][tau] for tau in xs]
            ax.plot(xs, ys, marker=markers[v], color=colors[v],
                    label=f"v={v} m/s", linewidth=1.2, markersize=5)
        ax.set_xlabel(r"$\tau_w$")
        ax.set_xlim(0.50, 0.80)
        ax.grid(True, alpha=0.3)
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("KGR (bps)")
    fig.suptitle("Reproduction of Fig. 13 -- Key Generation Rate vs "
                 r"$\tau_w$", fontsize=11)
    fig.tight_layout()
    plot_path = os.path.join(outdir, "fig13_reproduced.png")
    fig.savefig(plot_path, dpi=160)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    run()
