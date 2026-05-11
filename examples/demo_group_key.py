"""
Run with::
    python examples/demo_group_key.py
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from fuxian.fuzzy_extractor import BCHCode, FuzzyExtractor
from fuxian.key_update import GroupKeyFollower, GroupKeyGenerator
from fuxian.quantization import Quantizer, hamming_distance
from fuxian.scsi import DEFAULT_NUM_SATELLITES, SCSISimulator


def main() -> None:
    rng = np.random.default_rng(2024)
    sim = SCSISimulator(num_satellites=DEFAULT_NUM_SATELLITES,
                         visible_prob=0.4, rng=rng)

    # ---- 1.  Quantizer training (Eq. 2: equal-probability bins) ----
    print("Training the equal-probability quantizer ...")
    training_samples = []
    for _ in range(200):
        snap = sim.sample(distance=0.0)
        training_samples.extend(s for s in snap.snr if s > 0)
        sim.step_global_state(snr_drift_std=0.3, visibility_flip_prob=0.005)
    qz = Quantizer.fit(training_samples, M=10)
    print(f"  bin boundaries (M=10): {np.round(qz.bins, 1).tolist()}")
    print(f"  per-satellite codeword: {qz.codeword_bits} bits")
    print(f"  total encoding length:   {qz.encoding_length(DEFAULT_NUM_SATELLITES)} bits")

    # ---- 2.  Choose BCH parameters that match the encoding length ----
    bch = BCHCode(t=20, m=10)
    fe = FuzzyExtractor(input_bits=bch.n, bch=bch, key_bytes=32)
    enc_bits = qz.encoding_length(DEFAULT_NUM_SATELLITES)
    print(f"\nBCH(t={bch.t}, m={bch.m}): n={bch.n}, k={bch.k}, "
          f"ecc_bits={bch.ecc_bits}")
    if enc_bits < bch.n:
        print(f"  encoded SCSI ({enc_bits} bits) is shorter than n; "
              f"zero-padding to {bch.n} bits.")

    def encode_w(scsi) -> np.ndarray:
        bits = qz.encode(scsi)
        if bits.size < bch.n:
            bits = np.concatenate([bits, np.zeros(bch.n - bits.size, dtype=np.uint8)])
        return bits[: bch.n]

    # ---- 3.  Single-stage matching probability vs. distance ----
    distances = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    trials = 30
    print(f"\nGroup key matching rate over {trials} trials per distance:")
    print(f"  {'distance (m)':<14}{'success':<10}")
    for d in distances:
        successes = 0
        for _ in range(trials):
            sim.step_global_state(snr_drift_std=0.3,
                                   visibility_flip_prob=0.005)
            leader_scsi = sim.sample(distance=0.0)
            follower_scsi = sim.sample(distance=d)
            w  = encode_w(leader_scsi)
            wp = encode_w(follower_scsi)
            R, P = fe.gen(w)
            try:
                R_prime = fe.rep(wp, P)
                if R_prime == R:
                    successes += 1
            except ValueError:
                pass  # decoding failure
        print(f"  {d:<14}{successes / trials:.2%}")

    # ---- 4.  Full group-key chain over a few stages ----
    print("\nRunning a 5-stage key update chain (close-range scenario)")
    leader  = GroupKeyGenerator(fuzzy_extractor=fe, n=5)
    follower = GroupKeyFollower(fuzzy_extractor=fe, n=5)

    sim.step_global_state()
    w0_leader  = encode_w(sim.sample(distance=0.0))
    # close follower observes a near-identical SCSI
    w0_follower = encode_w(sim.sample(distance=2.0))
    P0, h_R0 = leader.initialize(w0_leader)
    follower.initialize(w0_follower, P0, expected_R0_digest=h_R0)
    print(f"  K_0 (hex prefix): {leader.state.K[:8].hex()}  "
          f"(follower match: {leader.state.K == follower.state.K})")

    for i in range(1, 6):
        sim.step_global_state()
        wL = encode_w(sim.sample(distance=0.0))
        wF = encode_w(sim.sample(distance=2.0))
        P_i, digest, K_leader = leader.update(wL)
        try:
            K_follower = follower.update(wF, P_i, digest)
        except ValueError as exc:
            print(f"  stage {i}: follower failed: {exc}")
            continue
        print(f"  K_{i} (hex prefix): {K_leader[:8].hex()}  "
              f"(follower match: {K_leader == K_follower})")


if __name__ == "__main__":
    main()
