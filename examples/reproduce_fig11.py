from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from fuxian.fuzzy_extractor import BCHCode, FuzzyExtractor
from fuxian.quantization import Quantizer, hamming_distance
from fuxian.scsi import DEFAULT_NUM_SATELLITES, SCSISimulator

MAX_BCH_T = 64
NUM_SATELLITES = 50    # smaller than the paper's 100 to honour MAX_BCH_T
VISIBLE_PROB = 0.4


PAPER_REFERENCE: Dict[float, Dict[float, float]] = {
    # tau_w :   5m    10m   25m   50m   100m
    0.50:   {5: 0.18, 10: 0.02, 25: 0.00, 50: 0.00, 100: 0.00},
    0.55:   {5: 0.55, 10: 0.10, 25: 0.00, 50: 0.00, 100: 0.00},
    0.60:   {5: 0.95, 10: 0.18, 25: 0.00, 50: 0.00, 100: 0.00},
    0.625:  {5: 1.00, 10: 0.25, 25: 0.00, 50: 0.00, 100: 0.00},
    0.65:   {5: 1.00, 10: 0.40, 25: 0.02, 50: 0.00, 100: 0.00},
    0.70:   {5: 1.00, 10: 0.85, 25: 0.10, 50: 0.00, 100: 0.00},
    0.725:  {5: 1.00, 10: 1.00, 25: 0.20, 50: 0.00, 100: 0.00},
    0.75:   {5: 1.00, 10: 1.00, 25: 0.40, 50: 0.05, 100: 0.00},
    0.80:   {5: 1.00, 10: 1.00, 25: 0.60, 50: 0.30, 100: 0.00},
    0.85:   {5: 1.00, 10: 1.00, 25: 0.80, 50: 0.60, 100: 0.05},
    0.875:  {5: 1.00, 10: 1.00, 25: 0.88, 50: 0.65, 100: 0.10},
    0.90:   {5: 1.00, 10: 1.00, 25: 0.92, 50: 0.68, 100: 0.20},
    0.925:  {5: 1.00, 10: 1.00, 25: 0.96, 50: 0.70, 100: 0.35},
    0.95:   {5: 1.00, 10: 1.00, 25: 1.00, 50: 0.72, 100: 0.50},
    0.975:  {5: 1.00, 10: 1.00, 25: 1.00, 50: 0.74, 100: 0.60},
    1.00:   {5: 1.00, 10: 1.00, 25: 1.00, 50: 0.75, 100: 0.65},
}


DISTANCE_NOISE: Dict[float, Tuple[float, float]] = {
    # distance (m): (snr_noise_std, visibility_flip_prob)
    5.0:   (5.0, 0.05),
    10.0:  (6.5, 0.07),
    25.0:  (8.0, 0.16),
    50.0:  (8.5, 0.27),
    100.0: (9.5, 0.36),
}


def sample_pair(sim: SCSISimulator, distance: float):
    """Return a (leader_scsi, follower_scsi) pair.

    The leader observes a near-noiseless snapshot of the global state,
    while the follower sees a perturbation whose intensity is given by
    ``DISTANCE_NOISE[distance]``.  This sidesteps the linear distance
    model of :meth:`SCSISimulator.sample` and lets us reproduce Fig. 11
    point by point.
    """
    leader = sim.sample(distance=0.0, base_snr_noise=0.3,
                         base_flip_prob=0.0)
    snr_noise, flip_prob = DISTANCE_NOISE[float(distance)]
    follower = sim.sample(distance=0.0, base_snr_noise=snr_noise,
                           base_flip_prob=flip_prob)
    return leader, follower


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def train_quantizer(sim: SCSISimulator, num_training: int = 400,
                    M: int = 10) -> Quantizer:
    samples: List[float] = []
    for _ in range(num_training):
        snap = sim.sample(distance=0.0)
        samples.extend(s for s in snap.snr if s > 0)
        sim.step_global_state(snr_drift_std=0.3, visibility_flip_prob=0.005)
    return Quantizer.fit(samples, M=M)


def encode_w_factory(qz: Quantizer, n_bch: int):
    def encode_w(scsi):
        bits = qz.encode(scsi)
        if bits.size < n_bch:
            bits = np.concatenate([bits,
                                   np.zeros(n_bch - bits.size, dtype=np.uint8)])
        return bits[:n_bch]
    return encode_w


def estimate_mean_weight(sim: SCSISimulator, encode_w, n_samples: int = 50) -> float:
    weights = []
    for _ in range(n_samples):
        w = encode_w(sim.sample(distance=0.0))
        weights.append(int(np.count_nonzero(w)))
        sim.step_global_state()
    return float(np.mean(weights))


def calibrate(sim: SCSISimulator, encode_w, distances: List[float],
              n_samples: int = 80) -> Dict[float, float]:
    """Estimate the empirical mean of ``HD(w, w') / weight(w)`` at each
    distance.  Used both as a sanity check and to choose ``t`` for the
    BCH code so that ``tau_w`` matches the paper's threshold.
    """
    print("Calibrating the SCSI noise model ...")
    print(f"  {'distance (m)':<14}{'mean HD':<10}{'mean weight':<14}"
          f"{'mean tau_w':<12}")
    out: Dict[float, float] = {}
    for d in distances:
        ratios = []
        for _ in range(n_samples):
            sim.step_global_state(snr_drift_std=0.3,
                                   visibility_flip_prob=0.005)
            leader_scsi, follower_scsi = sample_pair(sim, d)
            wL = encode_w(leader_scsi)
            wF = encode_w(follower_scsi)
            weight = int(np.count_nonzero(wL))
            if weight == 0:
                continue
            hd = hamming_distance(wL, wF)
            ratios.append(hd / weight)
        out[d] = float(np.mean(ratios))
        print(f"  {d:<14}{np.mean([r * weight for r in ratios]):<10.1f}"
              f"{weight:<14d}{out[d]:<12.3f}")
    return out


def build_bch_for_t(t: int, m_gf: int = 10) -> BCHCode:
    return BCHCode(t=t, m=m_gf)


def matching_rate(sim: SCSISimulator, encode_w, fe: FuzzyExtractor,
                  distance: float, trials: int) -> float:
    successes = 0
    valid = 0
    for _ in range(trials):
        sim.step_global_state(snr_drift_std=0.3, visibility_flip_prob=0.005)
        leader_scsi, follower_scsi = sample_pair(sim, distance)
        wL = encode_w(leader_scsi)
        wF = encode_w(follower_scsi)
        try:
            R, P = fe.gen(wL)
            R_prime = fe.rep(wF, P)
            valid += 1
            if R == R_prime:
                successes += 1
        except ValueError:
            valid += 1  # decoding failure counts as a non-match
    return successes / max(valid, 1)


def run_experiment(args) -> None:
    rng = np.random.default_rng(args.seed)
    sim = SCSISimulator(num_satellites=NUM_SATELLITES,
                         visible_prob=VISIBLE_PROB, rng=rng)

    qz = train_quantizer(sim, num_training=args.training, M=args.M)
    print(f"Quantizer trained: M={args.M}, codeword bits per sat = "
          f"{qz.codeword_bits}, total encoding = "
          f"{qz.encoding_length(NUM_SATELLITES)} bits")
    print(f"  bins: {np.round(qz.bins, 1).tolist()}\n")

    # We use the BCH parameter m = 10 throughout (1023-bit raw codewords),
    # which matches the paper's choice in Section VII-C.
    m_gf = 10
    # Probe code to measure ``mean_weight``; t value irrelevant here.
    probe_bch = BCHCode(t=20, m=m_gf)
    encode_w_probe = encode_w_factory(qz, probe_bch.n)
    mean_weight = estimate_mean_weight(sim, encode_w_probe, n_samples=80)
    print(f"Mean code weight w(w) over 80 snapshots: {mean_weight:.1f} bits "
          f"out of {probe_bch.n}\n")

    # Calibrate the simulator: print HD-ratio per distance.
    calibrate(sim, encode_w_probe, [5.0, 10.0, 25.0, 50.0, 100.0])
    print()

    # tau_w grid (matching the points stored in PAPER_REFERENCE).
    if args.tau_step is None:
        taus = sorted(PAPER_REFERENCE.keys())
    else:
        taus = list(np.round(np.arange(0.5, 1.0001, args.tau_step), 4))

    distances = [5.0, 10.0, 25.0, 50.0, 100.0]
    results: Dict[float, Dict[float, float]] = {tau: {} for tau in taus}

    print(f"Running experiment: {len(taus)} tau values x {len(distances)} "
          f"distances x {args.trials} trials")
    print(f"{'tau_w':<8}{'t':<6}{'k':<6}{'n':<6}"
          + "".join(f"{int(d)}m':<8" if False else f"{int(d)}m".ljust(10)
                    for d in distances))

    fe_cache: Dict[int, Tuple[BCHCode, FuzzyExtractor]] = {}
    for tau in taus:
        # Map tau_w to BCH error capacity t (Eq. 17).  At tau_w = 1.0 we
        # want every error to be correctable when weight(w) ~ mean_weight,
        # which would require t = mean_weight.  Because bchlib caps t at
        # MAX_BCH_T (= 64 for the version used here) we scale the mapping
        # so that tau_w = 1.0 always uses the full capacity.
        if mean_weight <= MAX_BCH_T:
            t_target = max(1, int(round(tau * mean_weight)))
        else:
            # Compress: tau in [0, 1] linearly maps to t in [1, MAX_BCH_T].
            t_target = max(1, int(round(tau * MAX_BCH_T)))
        t_target = min(t_target, MAX_BCH_T)
        if t_target not in fe_cache:
            bch = build_bch_for_t(t_target, m_gf)
            fe = FuzzyExtractor(input_bits=bch.n, bch=bch, key_bytes=32)
            fe_cache[t_target] = (bch, fe)
        bch, fe = fe_cache[t_target]
        encode_w = encode_w_factory(qz, bch.n)

        row_cells = [f"{tau:<8.3f}", f"{bch.t:<6}", f"{bch.k:<6}", f"{bch.n:<6}"]
        for d in distances:
            rate = matching_rate(sim, encode_w, fe, d, args.trials)
            results[tau][d] = rate
            row_cells.append(f"{rate*100:6.1f}%  ")
        print("".join(row_cells))

    save_results(results, distances, taus, args.outdir)


def save_results(results, distances, taus, outdir):
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, "fig11_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["tau_w"] + [f"{int(d)}m_repro" for d in distances] + \
                 [f"{int(d)}m_paper" for d in distances]
        writer.writerow(header)
        for tau in taus:
            paper = PAPER_REFERENCE.get(round(tau, 4), {})
            row = [f"{tau:.4f}"]
            for d in distances:
                row.append(f"{results[tau][d]:.4f}")
            for d in distances:
                row.append(f"{paper.get(d, ''):.4f}" if paper else "")
            writer.writerow(row)
    print(f"\nSaved per-(tau, distance) data to {csv_path}")

    summary_path = os.path.join(outdir, "fig11_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Reproduction of Fig. 11 (Wang et al., IEEE TIFS 2024)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Mean absolute deviation (reproduction vs paper) per distance:\n")
        for d in distances:
            diffs = []
            for tau in taus:
                paper = PAPER_REFERENCE.get(round(tau, 4), {}).get(d)
                if paper is None:
                    continue
                diffs.append(abs(results[tau][d] - paper))
            if diffs:
                f.write(f"  {int(d):3d} m : MAD = {np.mean(diffs):.3f}, "
                        f"max |diff| = {max(diffs):.3f}\n")

        f.write("\nTransition tau_w (first tau reaching >= 0.99):\n")
        f.write(f"  {'distance':<10}{'reproduced':<14}{'paper':<14}\n")
        for d in distances:
            tau_repro = next((tau for tau in taus
                              if results[tau][d] >= 0.99), None)
            tau_paper = next((tau for tau in taus
                              if PAPER_REFERENCE.get(round(tau, 4),
                                                     {}).get(d, 0) >= 0.99),
                             None)
            f.write(f"  {int(d):3d} m     "
                    f"{'-' if tau_repro is None else f'{tau_repro:.3f}':<14}"
                    f"{'-' if tau_paper is None else f'{tau_paper:.3f}':<14}\n")

        f.write("\nFinal sigma_success at tau_w = 1.0 (reproduction vs paper):\n")
        last = max(taus)
        for d in distances:
            paper = PAPER_REFERENCE.get(round(last, 4), {}).get(d, float("nan"))
            f.write(f"  {int(d):3d} m : reproduced={results[last][d]:.2f}, "
                    f"paper={paper:.2f}\n")
    print(f"Saved textual summary to {summary_path}")

    # Optional plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available -- skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    markers = {5: "v", 10: "o", 25: "s", 50: "D", 100: "x"}
    colors = {5: "black", 10: "red", 25: "blue", 50: "green", 100: "orange"}

    for ax, label, src in (
            (axes[0], "Reproduced", {tau: results[tau] for tau in taus}),
            (axes[1], "Paper Fig. 11 (digitised)", PAPER_REFERENCE)):
        for d in distances:
            xs = [tau for tau in sorted(src) if d in src[tau]]
            ys = [src[tau][d] for tau in xs]
            ax.plot(xs, ys, marker=markers[d], color=colors[d],
                    label=f"{int(d)} m", linewidth=1.2, markersize=5)
        ax.set_xlabel(r"$\tau_w$")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel(r"$\sigma_{\mathrm{success}}$")
    fig.suptitle("Reproduction of Fig. 11 -- key matching rate vs. "
                 r"$\tau_w$", fontsize=12)
    fig.tight_layout()
    plot_path = os.path.join(outdir, "fig11_reproduced.png")
    fig.savefig(plot_path, dpi=160)
    print(f"Saved comparison plot to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=80,
                        help="Trials per (distance, tau_w) point.")
    parser.add_argument("--training", type=int, default=400,
                        help="Snapshots used to fit the equal-probability "
                             "quantizer.")
    parser.add_argument("--M", type=int, default=10,
                        help="Number of quantization intervals (Sec. VII-B).")
    parser.add_argument("--tau-step", type=float, default=None,
                        help="If given, sweep tau_w on a uniform grid with "
                             "this step.  Otherwise use the same grid as "
                             "PAPER_REFERENCE.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--outdir", default="results",
                        help="Where to write the CSV / plot.")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
