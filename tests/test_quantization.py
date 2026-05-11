"""Unit tests for :mod:`fuxian.quantization`."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fuxian.quantization import (
    Quantizer,
    bits_to_bytes,
    bytes_to_bits,
    code_word_bits,
    encode,
    equal_probability_bins,
    hamming_distance,
    quantize,
    tau_w,
)
from fuxian.scsi import SCSI, SCSISimulator


def test_codeword_bits_match_paper():
    # Section VII-B-1: "each signal requires only 4 = log2(50/5) bits"
    # i.e. M = 10 ⇒ u = 4
    assert code_word_bits(10) == 4
    assert code_word_bits(1) == 3   # 4+1=5 -> 3 bits
    assert code_word_bits(60) == 7  # 4+60=64 -> 7 bits


def test_equal_probability_bins_partition_quantiles():
    rng = np.random.default_rng(0)
    samples = rng.uniform(1.0, 49.0, size=10000)
    M = 10
    bins = equal_probability_bins(samples, M=M, snr_min=0.0, snr_max=50.0)
    assert bins.size == M + 1
    # roughly equal mass per bin
    counts = np.histogram(samples, bins=bins)[0]
    assert counts.min() >= 0.05 * samples.size  # >5% per bin


def test_quantize_invisible_returns_zero():
    snr = np.array([10.0, 20.0, 30.0])
    scsi = SCSI(snr=snr, visible={1})
    bins = np.array([0.0, 15.0, 25.0, 50.0])
    q = quantize(scsi, bins, M=3, snr_max=50.0)
    assert q[0] == 0  # invisible
    assert q[2] == 0  # invisible
    # 20 falls in (15, 25] -> m = 1 -> 4 + 1 = 5
    assert q[1] == 5


def test_quantize_handles_saturation_and_zero():
    snr = np.array([0.0, 70.0])
    scsi = SCSI(snr=snr, visible={0, 1})
    bins = np.array([0.0, 10.0, 50.0])
    q = quantize(scsi, bins, M=2, snr_max=50.0)
    assert q[0] == 2          # visible and s == 0
    assert q[1] == 4 + 2      # saturated above Max


def test_gray_encode_preserves_adjacent_distance():
    # adjacent quantized values differ in exactly one bit after Gray coding
    M = 10
    u = code_word_bits(M)
    for v in range(4 + M):
        a = encode([v], M)
        b = encode([v + 1], M)
        assert a.size == u
        assert hamming_distance(a, b) == 1


def test_quantizer_pipeline_round_trip_for_close_drones():
    sim = SCSISimulator(num_satellites=64, visible_prob=0.45,
                        rng=np.random.default_rng(123))
    samples = [sim.sample(distance=0.0) for _ in range(50)]
    training = [s for snap in samples for s in snap.snr if s > 0]
    qz = Quantizer.fit(training, M=10)
    leader = sim.sample(distance=0.0)
    follower = sim.sample(distance=2.0)  # very close
    w = qz.encode(leader)
    w_p = qz.encode(follower)
    # close drones produce highly similar bit strings
    assert tau_w(w, w_p) < 0.3


def test_bits_bytes_round_trip():
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=400, dtype=np.uint8)
    data = bits_to_bytes(bits)
    assert len(data) == math.ceil(400 / 8)
    rebuilt = bytes_to_bits(data, 400)
    assert np.array_equal(bits, rebuilt)


def test_tau_w_matches_paper_definition():
    # tau_w = Hamming distance / weight of w
    w  = np.array([1, 1, 0, 1, 0, 1], dtype=np.uint8)
    wp = np.array([1, 0, 0, 1, 1, 1], dtype=np.uint8)
    # weight(w) = 4, hd = 2 -> tau = 0.5
    assert tau_w(w, wp) == 0.5
