"""Unit tests for :mod:`fuxian.key_update`."""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fuxian.fuzzy_extractor import BCHCode, FuzzyExtractor, sha256, _xor
from fuxian.key_update import (
    GroupKeyFollower,
    GroupKeyGenerator,
    update_L,
)


def _xor_many(*items: bytes) -> bytes:
    out = items[0]
    for it in items[1:]:
        out = _xor(out, it)
    return out


def test_update_L_matches_paper_formula():
    """After ``t`` calls (with ``t >= n``), entry at position ``q`` (0 = oldest)
    must equal ``h(A_{t-n+1+q}) XOR R_{t-n+2+q} XOR ... XOR R_t``.
    """
    n = 4
    L = deque([sha256(b"seed-%d" % i) for i in range(n)], maxlen=n)
    A = [b"\x00" * 32]  # A[0] = A_0
    R = [b"\x00" * 32]  # R[0] not used
    rng = np.random.default_rng(0)
    T = 7
    for t in range(1, T + 1):
        R_t = bytes(rng.integers(0, 256, size=32, dtype=np.uint8))
        A_t = _xor(R_t, A[-1])
        update_L(L, A_t, R_t)
        A.append(A_t)
        R.append(R_t)

    # check the latest entry: should be h(A_T) (no XORs yet)
    assert L[-1] == sha256(A[T])
    # second-to-last: h(A_{T-1}) XOR R_T
    assert L[-2] == _xor(sha256(A[T - 1]), R[T])
    # third-to-last: h(A_{T-2}) XOR R_{T-1} XOR R_T
    assert L[-3] == _xor_many(sha256(A[T - 2]), R[T - 1], R[T])
    # oldest: h(A_{T-n+1}) XOR R_{T-n+2} ... R_T
    expected_oldest = sha256(A[T - n + 1])
    for s in range(T - n + 2, T + 1):
        expected_oldest = _xor(expected_oldest, R[s])
    assert L[0] == expected_oldest


def _make_extractor(t: int = 8) -> FuzzyExtractor:
    bch = BCHCode(t=t, m=10)
    return FuzzyExtractor(input_bits=bch.n, bch=bch)


def test_leader_and_follower_share_the_same_key_chain():
    fe = _make_extractor()
    rng = np.random.default_rng(0)

    leader = GroupKeyGenerator(fuzzy_extractor=fe, n=5)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=5)

    def fresh_w() -> np.ndarray:
        return rng.integers(0, 2, size=fe.input_bits, dtype=np.uint8)

    # ---- initialisation ----
    w0 = fresh_w()
    P0, h_R0 = leader.initialize(w0)
    follower.initialize(w0.copy(), P0, expected_R0_digest=h_R0)
    assert leader.state.K == follower.state.K
    assert leader.state.A == follower.state.A
    K_history = [leader.state.K]

    # ---- several updates ----
    for _ in range(6):
        w_i = fresh_w()
        P_i, digest, K_leader = leader.update(w_i)
        K_follower = follower.update(w_i.copy(), P_i, digest)
        assert K_leader == K_follower
        # key actually changes
        assert K_leader != K_history[-1]
        K_history.append(K_leader)


def test_follower_rejects_tampered_digest():
    fe = _make_extractor()
    rng = np.random.default_rng(1)

    leader = GroupKeyGenerator(fuzzy_extractor=fe, n=3)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=3)

    w0 = rng.integers(0, 2, size=fe.input_bits, dtype=np.uint8)
    P0, h_R0 = leader.initialize(w0)
    follower.initialize(w0.copy(), P0, expected_R0_digest=h_R0)

    w1 = rng.integers(0, 2, size=fe.input_bits, dtype=np.uint8)
    P1, digest, _ = leader.update(w1)
    bad_digest = bytes((digest[0] ^ 0x01,)) + digest[1:]
    try:
        follower.update(w1.copy(), P1, bad_digest)
    except ValueError:
        return
    raise AssertionError("follower should have rejected the tampered digest")


def test_follower_rejects_bad_initial_R0_digest():
    fe = _make_extractor()
    rng = np.random.default_rng(2)
    leader = GroupKeyGenerator(fuzzy_extractor=fe, n=3)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=3)

    w0 = rng.integers(0, 2, size=fe.input_bits, dtype=np.uint8)
    P0, h_R0 = leader.initialize(w0)
    bad = bytes((h_R0[0] ^ 0xff,)) + h_R0[1:]
    try:
        follower.initialize(w0.copy(), P0, expected_R0_digest=bad)
    except ValueError:
        return
    raise AssertionError("follower must reject a wrong initial R0 hash")
