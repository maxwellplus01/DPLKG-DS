from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from fuxian.fuzzy_extractor import BCHCode, FuzzyExtractor, sha256
from fuxian.key_update import GroupKeyGenerator
from fuxian.quantization import Quantizer
from fuxian.scsi import SCSISimulator


NUM_KEYS = 500          # Section VII-B-3: "we ... 500 generated keys"
KEY_BITS = 256          # AES-256 keys (Section VII-C)
NUM_SATELLITES = 50     # mirrors examples/reproduce_fig11.py


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------
def generate_keys(num_keys: int, seed: int = 2024) -> np.ndarray:
    """Return ``num_keys * KEY_BITS`` keys as a ``(num_keys, KEY_BITS)``
    numpy array of bits."""
    rng = np.random.default_rng(seed)
    sim = SCSISimulator(num_satellites=NUM_SATELLITES, visible_prob=0.4, rng=rng)

    # Quick quantizer training (small footprint -- this is only needed to
    # get a non-trivial encoding ``w``).
    samples: List[float] = []
    for _ in range(120):
        samples.extend(s for s in sim.sample(distance=0.0).snr if s > 0)
        sim.step_global_state(snr_drift_std=0.4, visibility_flip_prob=0.005)
    qz = Quantizer.fit(samples, M=10)

    bch = BCHCode(t=20, m=10)
    fe = FuzzyExtractor(input_bits=bch.n, bch=bch, key_bytes=KEY_BITS // 8)

    def encode_w(scsi):
        bits = qz.encode(scsi)
        if bits.size < bch.n:
            bits = np.concatenate([bits,
                                   np.zeros(bch.n - bits.size, dtype=np.uint8)])
        return bits[: bch.n]

    leader = GroupKeyGenerator(fuzzy_extractor=fe, n=5)
    leader.initialize(encode_w(sim.sample(distance=0.0)))

    keys = np.empty((num_keys, KEY_BITS), dtype=np.uint8)
    keys[0] = np.unpackbits(np.frombuffer(leader.state.K, dtype=np.uint8))
    for i in range(1, num_keys):
        sim.step_global_state(snr_drift_std=0.4, visibility_flip_prob=0.005)
        # The hash-chain update means K_i = R_i XOR h(K_{i-1}); even when
        # the regenerated R_i would not change much, the chain guarantees
        # full 256-bit randomness.  We don't care about cross-drone
        # consistency here, so we only run the leader.
        _, _, K = leader.update(encode_w(sim.sample(distance=0.0)))
        keys[i] = np.unpackbits(np.frombuffer(K, dtype=np.uint8))
    return keys


# ---------------------------------------------------------------------------
# Randomness analyses
# ---------------------------------------------------------------------------
def zero_ratio_per_key(keys: np.ndarray) -> np.ndarray:
    """Fig. 12(a): proportion of 0-bits inside each generated key."""
    return 1.0 - keys.mean(axis=1)


def zero_ratio_per_xor_pair(keys: np.ndarray) -> np.ndarray:
    """Fig. 12(b): zero-bit ratio of K_i XOR K_{i-1} for i >= 1."""
    xors = np.bitwise_xor(keys[1:], keys[:-1])
    return 1.0 - xors.mean(axis=1)


def zero_ratio_per_bit(keys: np.ndarray) -> np.ndarray:
    """Fig. 12(c): for each bit position, proportion of 0 across keys."""
    return 1.0 - keys.mean(axis=0)


# ---------------------------------------------------------------------------
# NIST SP 800-22 statistical tests (subset)
# ---------------------------------------------------------------------------
def _erfc(x: float) -> float:
    return math.erfc(x)


def nist_frequency(bits: np.ndarray) -> float:
    """Frequency (Monobit) test."""
    n = bits.size
    s = int(bits.sum() * 2 - n)  # +/-1 sum
    s_obs = abs(s) / math.sqrt(n)
    return _erfc(s_obs / math.sqrt(2))


def nist_block_frequency(bits: np.ndarray, block_size: int = 128) -> float:
    n = bits.size
    n_blocks = n // block_size
    if n_blocks == 0:
        return float("nan")
    bits = bits[: n_blocks * block_size]
    blocks = bits.reshape(n_blocks, block_size)
    pi = blocks.mean(axis=1)
    chi2 = 4.0 * block_size * np.sum((pi - 0.5) ** 2)
    # P-value from upper-tail of chi-square w/ N degrees of freedom.
    from math import gamma
    # Use complementary regularised gamma approximation.
    a = n_blocks / 2.0
    x = chi2 / 2.0
    return _gammaincc(a, x)


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x).  Simple series/CF impl."""
    # use scipy if available
    try:
        from scipy.special import gammaincc
        return float(gammaincc(a, x))
    except Exception:
        pass
    if x < 0 or a <= 0:
        return 1.0
    if x < a + 1:
        # series
        term = 1.0 / a
        s = term
        for k in range(1, 200):
            term *= x / (a + k)
            s += term
            if abs(term) < 1e-12 * abs(s):
                break
        gam_inc = s * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - gam_inc
    else:
        # continued fraction
        b = x + 1 - a
        c = 1e30
        d = 1.0 / b
        h = d
        for k in range(1, 200):
            an = -k * (k - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < 1e-12:
                break
        return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def nist_runs(bits: np.ndarray) -> float:
    n = bits.size
    pi = bits.mean()
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    v_n = 1 + int(np.sum(bits[1:] != bits[:-1]))
    num = abs(v_n - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return _erfc(num / den)


def nist_cumulative_sums(bits: np.ndarray) -> float:
    n = bits.size
    x = 2 * bits.astype(int) - 1
    s = np.cumsum(x)
    z = int(np.max(np.abs(s)))
    if z == 0:
        return 1.0
    sqrt_n = math.sqrt(n)
    # forward direction (mode = 0)
    from math import erfc
    # P-value formula from SP 800-22.
    def phi(t):
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))
    k_low = int((-n / z + 1) / 4)
    k_high = int((n / z - 1) / 4)
    s1 = 0.0
    for k in range(k_low, k_high + 1):
        s1 += phi(((4 * k + 1) * z) / sqrt_n) - phi(((4 * k - 1) * z) / sqrt_n)
    k_low2 = int((-n / z - 3) / 4)
    s2 = 0.0
    for k in range(k_low2, k_high + 1):
        s2 += phi(((4 * k + 3) * z) / sqrt_n) - phi(((4 * k + 1) * z) / sqrt_n)
    p = 1.0 - s1 + s2
    return max(0.0, min(1.0, p))


def nist_approximate_entropy(bits: np.ndarray, m: int = 10) -> float:
    n = bits.size
    def phi(mm):
        # build window of size mm with wrap-around
        win = np.concatenate([bits, bits[: mm - 1]])
        # encode each window as int
        powers = 1 << np.arange(mm - 1, -1, -1)
        codes = np.array([int(np.dot(win[i:i + mm], powers)) for i in range(n)])
        counts = np.bincount(codes, minlength=1 << mm).astype(float) / n
        nz = counts[counts > 0]
        return float(np.sum(nz * np.log(nz)))
    apen = phi(m) - phi(m + 1)
    chi2 = 2 * n * (math.log(2) - apen)
    return _gammaincc(2 ** (m - 1), chi2 / 2.0)


def run_nist_tests(bitstream: np.ndarray) -> List[Tuple[str, float, str]]:
    """Run the implemented NIST tests and return ``(name, p_value, verdict)``."""
    tests = [
        ("Frequency",         nist_frequency(bitstream)),
        ("Block-Frequency",   nist_block_frequency(bitstream, block_size=128)),
        ("Runs",              nist_runs(bitstream)),
        ("Cumulative-Sums",   nist_cumulative_sums(bitstream)),
        ("Approximate-Entropy", nist_approximate_entropy(bitstream, m=8)),
    ]
    return [(name, p, "PASS" if p >= 0.01 else "FAIL") for name, p in tests]


# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------
def save_outputs(keys: np.ndarray, outdir: str = "results") -> None:
    os.makedirs(outdir, exist_ok=True)

    a = zero_ratio_per_key(keys)
    b = zero_ratio_per_xor_pair(keys)
    c = zero_ratio_per_bit(keys)

    # CSV with mean / std / quartiles for each panel.
    csv_path = os.path.join(outdir, "fig12_data.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("panel,n,mean,std,min,max\n")
        for label, arr in (("(a) zero ratio per key", a),
                           ("(b) zero ratio of K_i XOR K_{i-1}", b),
                           ("(c) zero ratio per bit position", c)):
            f.write(f"\"{label}\",{arr.size},{arr.mean():.4f},"
                    f"{arr.std():.4f},{arr.min():.4f},{arr.max():.4f}\n")
    print(f"Saved per-panel summary to {csv_path}")

    # NIST tests on the concatenated bit stream.
    bitstream = keys.flatten()
    print(f"\nNIST SP 800-22 (subset) on {bitstream.size} bits "
          f"({keys.shape[0]} x {keys.shape[1]}-bit keys):")
    print(f"  {'Test':<24}{'p-value':<12}{'verdict':<8}")
    nist_path = os.path.join(outdir, "table2_nist.txt")
    with open(nist_path, "w", encoding="utf-8") as f:
        f.write("Reproduction of Table II (NIST SP 800-22 subset)\n")
        f.write("Bitstream length: %d\n\n" % bitstream.size)
        f.write(f"{'Test':<24}{'p-value':<12}{'verdict':<8}\n")
        for name, p, verdict in run_nist_tests(bitstream):
            line = f"  {name:<24}{p:<12.4f}{verdict:<8}"
            print(line)
            f.write(line + "\n")
    print(f"Saved NIST results to {nist_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable, skipping plot)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    titles = ("(a)  zero ratio within key",
              "(b)  zero ratio of K_i XOR K_{i-1}",
              "(c)  zero ratio per bit position")
    xlabels = ("Key Index", "Key-pair Index", "Bit Position")
    data = (a, b, c)
    for ax, title, xl, arr in zip(axes, titles, xlabels, data):
        ax.bar(np.arange(arr.size), arr, color="#4F86C6", width=1.0)
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xl)
        ax.set_ylabel("Zero Ratio")
        ax.set_ylim(0.0, 0.6)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Reproduction of Fig. 12 -- Key randomness (proportion of "
                 "0-bits)", fontsize=11)
    fig.tight_layout()
    plot_path = os.path.join(outdir, "fig12_reproduced.png")
    fig.savefig(plot_path, dpi=160)
    print(f"Saved plot to {plot_path}")


def main() -> None:
    print(f"Generating {NUM_KEYS} keys of {KEY_BITS} bits each ...")
    keys = generate_keys(NUM_KEYS)

    a = zero_ratio_per_key(keys)
    b = zero_ratio_per_xor_pair(keys)
    c = zero_ratio_per_bit(keys)
    print(f"Fig. 12 (a) zero ratio per key:        "
          f"mean={a.mean():.3f}, std={a.std():.3f}, "
          f"range=[{a.min():.3f}, {a.max():.3f}]")
    print(f"Fig. 12 (b) zero ratio of XOR pairs:   "
          f"mean={b.mean():.3f}, std={b.std():.3f}, "
          f"range=[{b.min():.3f}, {b.max():.3f}]")
    print(f"Fig. 12 (c) zero ratio per bit:        "
          f"mean={c.mean():.3f}, std={c.std():.3f}, "
          f"range=[{c.min():.3f}, {c.max():.3f}]")

    save_outputs(keys)


if __name__ == "__main__":
    main()
