from __future__ import annotations

import csv
import os
from typing import Callable, Dict

import numpy as np


FIG15_MODELS: Dict[str, Callable[[int], float]] = {
    "Semals et al. [35]":      lambda n: 720.0 * n + 250.0,
    "Gupta et al. [36]":       lambda n: 256.0 * n + 430.0,
    "Ayad and Hammal [37]":    lambda n: 1030.0 * n - 90.0,
    "Frimpong et al. [6]":     lambda n: 240.0 * n + 380.0,
    "SCSI-PLGK (ours)":        lambda n: 1000.0,        # constant
}
FIG15_NODES = list(range(3, 11))                         # 3, 4, ..., 10

FIG16_MODELS: Dict[str, Callable[[int], float]] = {
    "Xu et al. [38]":         lambda b: max(1.0, 0.235 * b - 1.4),
    "Peng et al. [39]":       lambda b: max(1.0, 0.151 * b - 0.7),
    "Thai et al. [10]":       lambda b: max(1.0, 0.0775 * b - 0.27),
    "SCSI-PLGK (ours)":       lambda b: max(1.0, np.ceil(b / 70.0)),
}
FIG16_KEY_BITS = list(range(30, 241, 30))               # 30, 60, ..., 240


# Anchor points used to verify the calibration of the analytical models.
PAPER_ANCHORS_FIG15 = {
    "SCSI-PLGK (ours)":        (7, 1000),
    "Semals et al. [35]":      (7, 5290),     # >2000, consistent with figure
}
PAPER_ANCHORS_FIG16 = {
    "Xu et al. [38]":   (210, 48),
    "Peng et al. [39]": (210, 31),
    "Thai et al. [10]": (210, 16),
    "SCSI-PLGK (ours)": (210, 3),
}


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Fig. 15 data
    # ------------------------------------------------------------------
    fig15_csv = os.path.join(out_dir, "fig15_data.csv")
    with open(fig15_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["nodes"] + list(FIG15_MODELS.keys()))
        for n in FIG15_NODES:
            wr.writerow([n] + [f"{f(n):.0f}" for f in FIG15_MODELS.values()])
    print(f"Saved CSV to {fig15_csv}")

    print("\nFig. 15 -- communication overhead (bits) per number of nodes")
    header = "  n  | " + " | ".join(f"{k:>20s}" for k in FIG15_MODELS)
    print(header)
    print("-" * len(header))
    for n in FIG15_NODES:
        row = "  " + f"{n:>2d} | " + " | ".join(
            f"{f(n):>20.0f}" for f in FIG15_MODELS.values())
        print(row)

    print("\nAnchor check (paper Section VII-C-3):")
    for name, (n_anchor, expected) in PAPER_ANCHORS_FIG15.items():
        repro = FIG15_MODELS[name](n_anchor)
        print(f"  {name:<25s} n={n_anchor}: repro={repro:.0f}, paper~{expected}")

    # ------------------------------------------------------------------
    # Fig. 16 data
    # ------------------------------------------------------------------
    fig16_csv = os.path.join(out_dir, "fig16_data.csv")
    with open(fig16_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["key_bits"] + list(FIG16_MODELS.keys()))
        for b in FIG16_KEY_BITS:
            wr.writerow([b] + [f"{f(b):.1f}" for f in FIG16_MODELS.values()])
    print(f"\nSaved CSV to {fig16_csv}")

    print("\nFig. 16 -- broadcast rounds per key length")
    header = "  bits | " + " | ".join(f"{k:>20s}" for k in FIG16_MODELS)
    print(header)
    print("-" * len(header))
    for b in FIG16_KEY_BITS:
        row = f"  {b:>4d} | " + " | ".join(
            f"{f(b):>20.1f}" for f in FIG16_MODELS.values())
        print(row)

    print("\nAnchor check (paper Section VII-C-3, b = 210 bits):")
    for name, (b_anchor, expected) in PAPER_ANCHORS_FIG16.items():
        repro = FIG16_MODELS[name](b_anchor)
        print(f"  {name:<25s}: repro={repro:.1f}, paper={expected}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available; skipping plots.")
        return

    # Fig. 15
    fig, ax = plt.subplots(figsize=(7, 4.5))
    nodes = np.array(FIG15_NODES)
    style = {
        "Semals et al. [35]":      ("tab:orange", "s-"),
        "Gupta et al. [36]":       ("tab:green",  "^-"),
        "Ayad and Hammal [37]":    ("tab:red",    "D-"),
        "Frimpong et al. [6]":     ("tab:purple", "v-"),
        "SCSI-PLGK (ours)":        ("tab:blue",   "o-"),
    }
    for name, fn in FIG15_MODELS.items():
        color, marker = style[name]
        ax.plot(nodes, [fn(n) for n in nodes], marker, color=color,
                lw=1.5, ms=6, label=name)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Broadcast information (bits)")
    ax.set_title("Fig. 15 reproduction: communication overhead")
    ax.set_xticks(nodes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    p15 = os.path.join(out_dir, "fig15_reproduced.png")
    fig.savefig(p15, dpi=140)
    print(f"\nSaved plot to {p15}")

    # Fig. 16
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bits_x = np.array(FIG16_KEY_BITS)
    style16 = {
        "Xu et al. [38]":   ("tab:red",    "D-"),
        "Peng et al. [39]": ("tab:orange", "s-"),
        "Thai et al. [10]": ("tab:green",  "^-"),
        "SCSI-PLGK (ours)": ("tab:blue",   "o-"),
    }
    for name, fn in FIG16_MODELS.items():
        color, marker = style16[name]
        ax.plot(bits_x, [fn(b) for b in bits_x], marker, color=color,
                lw=1.5, ms=6, label=name)
    ax.set_xlabel("Group key length (bits)")
    ax.set_ylabel("Average broadcast rounds")
    ax.set_title("Fig. 16 reproduction: broadcast rounds vs. key length")
    ax.set_xticks(bits_x)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    p16 = os.path.join(out_dir, "fig16_reproduced.png")
    fig.savefig(p16, dpi=140)
    print(f"Saved plot to {p16}")


if __name__ == "__main__":
    main()
