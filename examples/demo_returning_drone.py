"""
Run with::

    python examples/demo_returning_drone.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fuxian.fuzzy_extractor import BCHCode, FuzzyExtractor, sha256, _xor
from fuxian.key_update import GroupKeyFollower, GroupKeyGenerator
from fuxian.protocol import (
    ReturningDrone,
    leading_drone_response,
    run_robust_agreement,
)


def _random_w(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def main() -> None:
    rng = np.random.default_rng(42)
    bch = BCHCode(t=8, m=10)
    fe  = FuzzyExtractor(input_bits=bch.n, bch=bch, key_bytes=32)

    L_capacity = 5
    leader   = GroupKeyGenerator(fuzzy_extractor=fe, n=L_capacity)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=L_capacity)

    # ---- 1. shared initialisation ----
    w0 = _random_w(rng, fe.input_bits)
    P0, h_R0 = leader.initialize(w0)
    follower.initialize(w0.copy(), P0, expected_R0_digest=h_R0)
    print(f"After init: K_0 = {leader.state.K[:8].hex()}...  "
          f"match={leader.state.K == follower.state.K}")

    # ---- 2. two synchronised key updates ----
    for i in range(1, 3):
        wi = _random_w(rng, fe.input_bits)
        Pi, digest, K = leader.update(wi)
        follower.update(wi.copy(), Pi, digest)
        print(f"Stage {i}:  K_{i} = {K[:8].hex()}...")

    # ---- 3. follower goes off-mission ----
    snapshot_K = follower.state.K
    snapshot_A = follower.state.A
    print("\nFollower drone leaves the swarm with state:")
    print(f"  K_i = {snapshot_K[:8].hex()}...")
    print(f"  A_i = {snapshot_A[:8].hex()}...")

    # ---- 4. swarm continues for a few more stages ----
    for i in range(3, 3 + L_capacity - 1):
        wi = _random_w(rng, fe.input_bits)
        Pi, digest, K = leader.update(wi)
        print(f"Stage {i} (without lost follower): K_{i} = {K[:8].hex()}...")

    # ---- 5. drone returns and runs the protocol ----
    print("\nReturning drone runs the robust key agreement protocol ...")
    returning = ReturningDrone(Ki=snapshot_K, Ai=snapshot_A)

    # Step 1
    M, Q = returning.build_request()
    print(f"  -> sends (M={M[:6].hex()}..., Q={Q[:6].hex()}...)")

    # Step 2
    S, EK, tag, accepted = leading_drone_response(leader.state, M, Q)
    print(f"  <- leader: accepted={accepted}, "
          f"tag={tag[:6].hex()}..., |EK|={len(EK)} bytes")

    # Step 3
    h_K_jm1 = returning.finalize_response(S, EK, tag)
    print(f"  recovered h(K_{{j-1}}) = {h_K_jm1[:8].hex()}...")
    print(f"  matches leader's h(K)?  {h_K_jm1 == sha256(leader.state.K)}")

    # Equivalent one-liner:
    h2 = run_robust_agreement(
        ReturningDrone(Ki=snapshot_K, Ai=snapshot_A), leader)
    print(f"\nrun_robust_agreement() yields the same result: "
          f"{h2 == h_K_jm1}")


if __name__ == "__main__":
    main()
