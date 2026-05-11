from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from fuxian.fuzzy_extractor import BCHCode
from fuxian.quantization import hamming_distance
from fuxian.scsi import SCSISimulator

from reproduce_fig11 import (NUM_SATELLITES, VISIBLE_PROB, MAX_BCH_T,
                              encode_w_factory, train_quantizer,
                              estimate_mean_weight)


PAPER_REFERENCE: Dict[int, float] = {
    0:    0.00,
    120:  0.16,
    240:  0.30,
    360:  0.40,
    480:  0.48,
    600:  0.55,
    720:  0.63,
    840:  0.71,
    960:  0.79,
    1080: 0.90,
    1200: 1.02,
    1320: 1.15,
    1440: 1.27,
    1500: 1.30,
}


# ---------------------------------------------------------------------------
# Drift parameters used by ``sim.step_global_state``.  These were tuned so
# that tau_w(720) ~ 0.63 and tau_w(960) ~ 0.79 (the two anchor points
# explicitly mentioned in the paper text).
# ---------------------------------------------------------------------------
SNR_DRIFT_STD = 0.018           # per-second SNR random walk (dBHz)
VISIBILITY_FLIP_PROB = 0.00040  # per-second per-satellite flip probability


def simulate_curve(sim: SCSISimulator, encode_w, duration: int,
                   step_drift: float, step_flip: float) -> np.ndarray:
    """Return ``tau_w(t)`` for ``t = 0, 1, ..., duration``."""
    w0 = encode_w(sim.sample(distance=0.0))
    weight0 = int(np.count_nonzero(w0))
    if weight0 == 0:
        weight0 = 1     # safety guard
    taus = np.zeros(duration + 1, dtype=float)
    for t in range(1, duration + 1):
        sim.step_global_state(snr_drift_std=step_drift,
                               visibility_flip_prob=step_flip)
        wt = encode_w(sim.sample(distance=0.0))
        taus[t] = hamming_distance(w0, wt) / weight0
    return taus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=int, default=1500,
                    help="simulation length in seconds (default: 1500)")
    ap.add_argument("--trials", type=int, default=20,
                    help="number of independent trajectories to average "
                         "(default: 20)")
    ap.add_argument("--seed", type=int, default=20240614)
    ap.add_argument("--snr-drift", type=float, default=SNR_DRIFT_STD)
    ap.add_argument("--flip-prob", type=float, default=VISIBILITY_FLIP_PROB)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sim = SCSISimulator(num_satellites=NUM_SATELLITES,
                        visible_prob=VISIBLE_PROB, rng=rng)

    print("Training quantiser ...", flush=True)
    qz = train_quantizer(sim)
    n_bch = NUM_SATELLITES * 10   # full encoded length, no BCH needed
    encode_w = encode_w_factory(qz, n_bch)
    mean_w = estimate_mean_weight(sim, encode_w)
    print(f"  n_bits = {n_bch},  mean weight = {mean_w:.1f}")

    print(f"Simulating {args.trials} trajectories of {args.duration} s ...",
          flush=True)
    taus = np.zeros(args.duration + 1, dtype=float)
    for trial in range(args.trials):
        # Each trial uses a freshly redrawn global satellite state so we
        # average over the random visible-set realisation.
        sim._regenerate_global_state()                 # noqa: SLF001
        taus += simulate_curve(sim, encode_w, args.duration,
                               args.snr_drift, args.flip_prob)
        if (trial + 1) % 5 == 0:
            print(f"  trial {trial + 1}/{args.trials}", flush=True)
    taus /= args.trials

    # ------------------------------------------------------------------
    # Save raw data
    # ------------------------------------------------------------------
    csv_path = os.path.join(out_dir, "fig14_data.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_s", "tau_w_repro", "tau_w_paper"])
        for t in range(args.duration + 1):
            paper = PAPER_REFERENCE.get(t, "")
            wr.writerow([t, f"{taus[t]:.4f}", paper])
    print(f"\nSaved CSV to {csv_path}")

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    summary_lines = ["tau_w(t) -- reproduction vs. Fig. 14 paper anchors",
                     "  t (s)   tau_w_repro   tau_w_paper"]
    for t, paper in PAPER_REFERENCE.items():
        summary_lines.append(f"  {t:>5d}   {taus[t]:>10.3f}   {paper:>10.3f}")
    txt_path = os.path.join(out_dir, "fig14_summary.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print("\n" + "\n".join(summary_lines))
    print(f"\nSaved summary to {txt_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    times_s = np.arange(args.duration + 1)
    ax.plot(times_s / 60.0, taus, color="tab:blue", lw=1.6,
            label="Reproduction ($n=50$ satellites)")
    paper_t = np.array(sorted(PAPER_REFERENCE.keys()))
    paper_v = np.array([PAPER_REFERENCE[t] for t in paper_t])
    ax.plot(paper_t / 60.0, paper_v, "o--", color="tab:red", lw=1.2,
            ms=5, label="Paper Fig. 14 (digitised)")
    ax.axhline(0.63, color="gray", ls=":", lw=1.0)
    ax.text(0.5, 0.64, r"$\tau_w = 0.63$ (PLGK threshold)",
            color="gray", fontsize=8)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$\tau_w(t) = \mathrm{HD}(w_0, w_t)\,/\,\mathrm{weight}(w_0)$")
    ax.set_title("Fig. 14 reproduction: temporal SCSI dissimilarity (stationary drone)")
    ax.set_xlim(0, args.duration / 60.0)
    ax.set_ylim(0, max(1.5, taus.max() * 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "fig14_reproduced.png")
    fig.savefig(png_path, dpi=140)
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
