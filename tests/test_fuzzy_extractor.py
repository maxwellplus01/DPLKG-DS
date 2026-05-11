"""Unit tests for :mod:`fuxian.fuzzy_extractor`."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fuxian.fuzzy_extractor import BCHCode, FuzzyExtractor, hamming_distance


def _random_bits(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def test_bch_round_trip_no_errors():
    bch = BCHCode(t=8, m=10)
    rng = np.random.default_rng(0)
    msg = _random_bits(bch.k, rng)
    cw = bch.encode(msg)
    assert cw.size == bch.n
    decoded, nerr = bch.decode(cw)
    assert nerr == 0
    assert np.array_equal(decoded, msg)


def test_bch_corrects_up_to_t_errors():
    bch = BCHCode(t=8, m=10)
    rng = np.random.default_rng(1)
    msg = _random_bits(bch.k, rng)
    cw = bch.encode(msg)
    # flip exactly t bits at random positions
    positions = rng.choice(bch.n, size=bch.t, replace=False)
    cw_noisy = cw.copy()
    cw_noisy[positions] ^= 1
    decoded, nerr = bch.decode(cw_noisy)
    assert nerr == bch.t
    assert np.array_equal(decoded, msg)


def test_bch_decoding_failure_above_capacity():
    bch = BCHCode(t=4, m=10)
    rng = np.random.default_rng(2)
    msg = _random_bits(bch.k, rng)
    cw = bch.encode(msg)
    # flip many bits — likely beyond the correction capability
    cw_noisy = cw.copy()
    flip = rng.choice(bch.n, size=120, replace=False)
    cw_noisy[flip] ^= 1
    with pytest.raises(ValueError):
        bch.decode(cw_noisy)


def test_fuzzy_extractor_recovers_with_few_errors():
    bch = BCHCode(t=15, m=10)
    fe = FuzzyExtractor(input_bits=bch.n, bch=bch, key_bytes=32)
    rng = np.random.default_rng(3)
    w = _random_bits(fe.input_bits, rng)
    R, helper = fe.gen(w)

    # exact reproduction
    assert fe.rep(w, helper) == R

    # within capability
    w_prime = w.copy()
    flip = rng.choice(fe.input_bits, size=10, replace=False)
    w_prime[flip] ^= 1
    R_prime = fe.rep(w_prime, helper)
    assert R_prime == R


def test_fuzzy_extractor_helper_size_matches_paper():
    """``|P| = 2|w|`` (Section V-A.3)."""
    bch = BCHCode(t=8, m=10)
    fe = FuzzyExtractor(input_bits=bch.n, bch=bch)
    rng = np.random.default_rng(4)
    w = _random_bits(fe.input_bits, rng)
    _R, helper = fe.gen(w)
    assert helper.length == 2 * fe.input_bits


def test_fuzzy_extractor_fails_when_too_many_errors():
    bch = BCHCode(t=4, m=10)
    fe = FuzzyExtractor(input_bits=bch.n, bch=bch)
    rng = np.random.default_rng(5)
    w = _random_bits(fe.input_bits, rng)
    _R, helper = fe.gen(w)
    w_prime = w.copy()
    flip = rng.choice(fe.input_bits, size=200, replace=False)
    w_prime[flip] ^= 1
    with pytest.raises(ValueError):
        fe.rep(w_prime, helper)
