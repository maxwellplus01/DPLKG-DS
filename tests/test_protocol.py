"""Unit tests for :mod:`fuxian.protocol`."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fuxian.fuzzy_extractor import BCHCode, FuzzyExtractor, sha256
from fuxian.key_update import GroupKeyGenerator, GroupKeyFollower
from fuxian.protocol import (
    ReturningDrone,
    aes256_decrypt,
    aes256_encrypt,
    leading_drone_response,
    run_robust_agreement,
    get_string_C,
)


def _make_extractor(t: int = 8) -> FuzzyExtractor:
    bch = BCHCode(t=t, m=10)
    return FuzzyExtractor(input_bits=bch.n, bch=bch)


def test_aes256_round_trip():
    key = os.urandom(32)
    msg = b"hello drone swarm" * 5
    ct = aes256_encrypt(key, msg)
    assert aes256_decrypt(key, ct) == msg


def test_get_string_C_returns_random_when_not_found():
    M = sha256(b"some Ai")
    L = [os.urandom(32) for _ in range(4)]
    A_jm1 = os.urandom(32)
    C, found = get_string_C(M, L, A_jm1)
    assert not found
    assert len(C) == len(M)


def _drive_leader(leader: GroupKeyGenerator, follower: GroupKeyFollower,
                   rng: np.random.Generator, num_updates: int):
    """Run one initialise + ``num_updates`` update rounds."""
    fe = leader.fuzzy_extractor
    w0 = rng.integers(0, 2, size=fe.input_bits, dtype=np.uint8)
    P0, h_R0 = leader.initialize(w0)
    follower.initialize(w0.copy(), P0, expected_R0_digest=h_R0)
    for _ in range(num_updates):
        w_i = rng.integers(0, 2, size=fe.input_bits, dtype=np.uint8)
        P_i, digest, _ = leader.update(w_i)
        follower.update(w_i.copy(), P_i, digest)


def test_robust_agreement_recovers_h_K_jm1():
    fe = _make_extractor()
    rng = np.random.default_rng(0)
    n = 5
    leader = GroupKeyGenerator(fuzzy_extractor=fe, n=n)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=n)

    # Synchronise leader & follower for a few stages.
    _drive_leader(leader, follower, rng, num_updates=2)

    # The follower goes off-mission at this point: snapshot its (K_i, A_i).
    drone = ReturningDrone(Ki=follower.state.K, Ai=follower.state.A)

    # The swarm continues for a few more updates...
    for _ in range(n - 2):
        w_i = rng.integers(0, 2, size=fe.input_bits, dtype=np.uint8)
        P_i, digest, _ = leader.update(w_i)
        follower.update(w_i.copy(), P_i, digest)

    h_Kjm1 = run_robust_agreement(drone, leader)
    assert h_Kjm1 == sha256(leader.state.K)
    assert drone.recovered_A_jm1 == leader.state.A


def test_robust_agreement_rejects_unknown_drone():
    """A drone whose ``A_i`` is not in ``L`` should be rejected."""
    fe = _make_extractor()
    rng = np.random.default_rng(1)
    n = 3
    leader = GroupKeyGenerator(fuzzy_extractor=fe, n=n)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=n)
    _drive_leader(leader, follower, rng, num_updates=4)

    fake = ReturningDrone(Ki=os.urandom(32), Ai=os.urandom(32))
    with pytest.raises(ValueError):
        run_robust_agreement(fake, leader)


def test_returning_drone_detects_tampered_response():
    fe = _make_extractor()
    rng = np.random.default_rng(2)
    n = 4
    leader = GroupKeyGenerator(fuzzy_extractor=fe, n=n)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=n)
    _drive_leader(leader, follower, rng, num_updates=2)

    drone = ReturningDrone(Ki=follower.state.K, Ai=follower.state.A)
    M, Q = drone.build_request()
    S, EK, tag, accepted = leading_drone_response(leader.state, M, Q)
    assert accepted
    bad_tag = bytes((tag[0] ^ 0xff,)) + tag[1:]
    with pytest.raises(ValueError):
        drone.finalize_response(S, EK, bad_tag)


def test_robust_agreement_outside_window_fails():
    """If the returning drone has been gone for more than ``n`` stages, the
    leader's queue ``L`` no longer contains a matching entry."""
    fe = _make_extractor()
    rng = np.random.default_rng(3)
    n = 2
    leader = GroupKeyGenerator(fuzzy_extractor=fe, n=n)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=n)
    _drive_leader(leader, follower, rng, num_updates=1)
    drone = ReturningDrone(Ki=follower.state.K, Ai=follower.state.A)
    # advance way beyond n stages
    for _ in range(n + 5):
        w_i = rng.integers(0, 2, size=fe.input_bits, dtype=np.uint8)
        P_i, digest, _ = leader.update(w_i)
        follower.update(w_i.copy(), P_i, digest)
    with pytest.raises(ValueError):
        run_robust_agreement(drone, leader)
