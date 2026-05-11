from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import math

import numpy as np

from .scsi import SCSI, DEFAULT_SNR_MAX, DEFAULT_SNR_MIN


# ---------------------------------------------------------------------------
# bin computation (Eq. 2)
# ---------------------------------------------------------------------------
def equal_probability_bins(snr_samples: Sequence[float], M: int,
                           snr_min: float = DEFAULT_SNR_MIN,
                           snr_max: float = DEFAULT_SNR_MAX) -> np.ndarray:
    """Return the ``M+1`` boundaries ``[b_0, b_1, ..., b_M]`` of the
    equal-probability quantization (Eq. 2).

    ``b_0 = Min`` and ``b_M = Max``; the inner boundaries are the empirical
    ``m / M`` quantiles of ``snr_samples``.  The samples should typically
    come from previously observed visible-satellite SNRs.
    """
    if M < 1:
        raise ValueError("M must be a positive integer")
    samples = np.asarray([s for s in snr_samples if s > 0], dtype=float)
    bins = np.empty(M + 1, dtype=float)
    bins[0] = snr_min
    bins[-1] = snr_max
    if samples.size == 0:
        # fall back to a uniform partitioning of [Min, Max]
        bins[1:-1] = np.linspace(snr_min, snr_max, M + 1)[1:-1]
    else:
        # m/M quantiles for m = 1, ..., M-1
        qs = np.linspace(0.0, 1.0, M + 1)[1:-1]
        bins[1:-1] = np.quantile(samples, qs)
    # ensure strictly non-decreasing for searchsorted determinism
    for k in range(1, bins.size):
        if bins[k] < bins[k - 1]:
            bins[k] = bins[k - 1]
    return bins


# ---------------------------------------------------------------------------
# quantization (Eq. 1)
# ---------------------------------------------------------------------------
def quantize(scsi: SCSI, bins: np.ndarray, M: int,
             snr_max: float = DEFAULT_SNR_MAX) -> np.ndarray:
    """Apply Eq. (1) to every component of ``scsi.snr``.

    The returned vector ``Q`` has the same length as ``scsi.snr`` and
    contains integer values in ``[0, 4 + M]``.
    """
    if bins.size != M + 1:
        raise ValueError("bins must have length M + 1")
    snr = scsi.snr
    n = snr.size
    Q = np.zeros(n, dtype=int)
    for i in range(n):
        if i not in scsi.visible:
            Q[i] = 0
            continue
        s = snr[i]
        if s == 0.0:
            Q[i] = 2
            continue
        if s > snr_max:
            Q[i] = 4 + M
            continue
        # Find m such that s in (b_m, b_{m+1}].  ``side='left'`` gives the
        # index of the first bin boundary strictly greater than s, which is
        # exactly ``m + 1``.  Edge case ``s == b_0`` is handled by clamping.
        m_plus_1 = int(np.searchsorted(bins, s, side="left"))
        m = max(0, min(M - 1, m_plus_1 - 1))
        Q[i] = 4 + m
    return Q


# ---------------------------------------------------------------------------
# Gray encoding (Eq. 3)
# ---------------------------------------------------------------------------
def _gray_encode(value: int, num_bits: int) -> List[int]:
    """Return the ``num_bits``-wide Gray code of ``value`` MSB first."""
    if value < 0 or value >= (1 << num_bits):
        raise ValueError(
            f"value {value} cannot be represented on {num_bits} bits"
        )
    g = value ^ (value >> 1)
    return [(g >> j) & 1 for j in range(num_bits - 1, -1, -1)]


def encode(quantized: Sequence[int], M: int) -> np.ndarray:
    """Apply Eq. (3) to every component of ``quantized``.

    Each integer is represented on ``u = ceil(log2(4 + M + 1))`` bits using
    its Gray code, and all codewords are concatenated MSB-first.  The total
    output length is ``N * u`` bits.
    """
    if M < 1:
        raise ValueError("M must be a positive integer")
    u = max(1, math.ceil(math.log2(4 + M + 1)))
    out = np.empty(len(quantized) * u, dtype=np.uint8)
    for i, q in enumerate(quantized):
        out[i * u : (i + 1) * u] = _gray_encode(int(q), u)
    return out


def code_word_bits(M: int) -> int:
    """Number of bits ``u`` used to encode a single quantized value."""
    return max(1, math.ceil(math.log2(4 + M + 1)))


# ---------------------------------------------------------------------------
# convenience pipeline
# ---------------------------------------------------------------------------
@dataclass
class Quantizer:
    """Bundle the quantization parameters together for convenience.

    Parameters
    ----------
    M:
        Number of partition intervals.
    bins:
        Bin boundaries of length ``M + 1`` (typically obtained from
        :func:`equal_probability_bins`).
    snr_max:
        Upper saturation value used in Eq. (1).
    """

    M: int
    bins: np.ndarray
    snr_max: float = DEFAULT_SNR_MAX

    @classmethod
    def fit(cls, training_samples: Iterable[float], M: int,
            snr_min: float = DEFAULT_SNR_MIN,
            snr_max: float = DEFAULT_SNR_MAX) -> "Quantizer":
        bins = equal_probability_bins(list(training_samples), M, snr_min, snr_max)
        return cls(M=M, bins=bins, snr_max=snr_max)

    @property
    def codeword_bits(self) -> int:
        return code_word_bits(self.M)

    def quantize(self, scsi: SCSI) -> np.ndarray:
        return quantize(scsi, self.bins, self.M, self.snr_max)

    def encode(self, scsi: SCSI) -> np.ndarray:
        return encode(self.quantize(scsi), self.M)

    def encoding_length(self, num_satellites: int) -> int:
        return num_satellites * self.codeword_bits


# ---------------------------------------------------------------------------
# similarity helpers used by tests / experiments
# ---------------------------------------------------------------------------
def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Return the Hamming distance between two equal-length bit vectors."""
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return int(np.count_nonzero(a ^ b))


def tau_w(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the parameter :math:`\\tau_w` defined in Eq. (17)."""
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    weight = int(np.count_nonzero(a))
    if weight == 0:
        return 0.0
    return hamming_distance(a, b) / weight


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a ``{0, 1}`` array (MSB-first) into a ``bytes`` object."""
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return bytes(np.packbits(bits, bitorder="big").tolist())


def bytes_to_bits(data: bytes, num_bits: int) -> np.ndarray:
    """Inverse of :func:`bits_to_bytes`; returns the first ``num_bits``."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits[:num_bits].copy()


__all__ = [
    "equal_probability_bins",
    "quantize",
    "encode",
    "code_word_bits",
    "Quantizer",
    "hamming_distance",
    "tau_w",
    "bits_to_bytes",
    "bytes_to_bits",
]
