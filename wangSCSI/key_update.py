from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from .fuzzy_extractor import FuzzyExtractor, HelperData, sha256, _xor


# ---------------------------------------------------------------------------
# Algorithm 1 -- Update L
# ---------------------------------------------------------------------------
def update_L(L: Deque[bytes], A_i: bytes, R_i: bytes) -> None:
    if len(L) == 0:
        raise ValueError("L must have a non-zero capacity (n >= 1)")
    L.popleft()
    for k in range(len(L)):
        L[k] = _xor(L[k], R_i)
    L.append(sha256(A_i))


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------
@dataclass
class GroupKeyState:
    """Mutable per-drone group-key state.

    Attributes
    ----------
    i:
        Current stage index (``0`` after :func:`initialize`).
    K:
        Current group key ``K_i`` (bytes).
    A:
        Current accumulator ``A_i`` (bytes).
    L:
        Queue of the last ``n`` accumulated hashes (Algorithm 1).
    n:
        Maximum length of ``L``.
    """

    n: int
    i: int = -1
    K: bytes = b""
    A: bytes = b""
    L: Deque[bytes] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be a positive integer")
        if not self.L:
            self.L = deque([b"\x00" * 32 for _ in range(self.n)], maxlen=self.n)
        else:
            self.L = deque(self.L, maxlen=self.n)

    def initialize(self, R0: bytes) -> None:
        """Set ``K_0 = A_0 = R_0`` and reset ``L``."""
        self.i = 0
        self.K = R0
        self.A = R0
        # paper does not specify L's initial content; use h(A_0) entries to
        # mirror Algorithm 1's behaviour after the first key update.
        seed = sha256(R0)
        self.L = deque([seed for _ in range(self.n)], maxlen=self.n)


# ---------------------------------------------------------------------------
# Group key generator (the "leading drone")
# ---------------------------------------------------------------------------
@dataclass
class GroupKeyGenerator:
    """Implements the actions of the leading drone of the swarm.

    The leading drone owns the public ``FuzzyExtractor`` parameters and is
    responsible for generating ``R_i`` and broadcasting
    ``(P_i, h(A_i || R_i))`` at every key update (Section V-B.2).
    """

    fuzzy_extractor: FuzzyExtractor
    n: int = 5  # default queue size used by the experiments (Section VII-C)
    state: GroupKeyState = field(init=False)

    def __post_init__(self) -> None:
        self.state = GroupKeyState(n=self.n)

    # ------------------------------------------------------------------
    # initialisation -- Section V-B.1
    # ------------------------------------------------------------------
    def initialize(self, w0: np.ndarray) -> Tuple[HelperData, bytes]:
        """Initial broadcast::

            (R_0, P_0) <- Gen(w_0)
            broadcast  P_0, h(R_0)
            K_0 = A_0 = R_0
        """
        R0, P0 = self.fuzzy_extractor.gen(w0)
        self.state.initialize(R0)
        return P0, sha256(R0)

    # ------------------------------------------------------------------
    # periodic update -- Section V-B.2
    # ------------------------------------------------------------------
    def update(self, w_i: np.ndarray) -> Tuple[HelperData, bytes, bytes]:
        """Generate the next group key and the broadcast tuple.

        Returns
        -------
        P_i:
            Helper string produced by the fuzzy extractor.
        digest:
            ``h(A_i || R_i)`` -- used by the receivers to verify ``R_i``
            after they recover it via ``Rep``.
        K_i:
            The new group key (kept locally for convenience).
        """
        if self.state.i < 0:
            raise RuntimeError("call initialize() before update()")
        R_i, P_i = self.fuzzy_extractor.gen(w_i)
        new_A = _xor(R_i, self.state.A)
        new_K = _xor(R_i, sha256(self.state.K))
        update_L(self.state.L, new_A, R_i)
        self.state.A = new_A
        self.state.K = new_K
        self.state.i += 1
        digest = sha256(new_A + R_i)
        return P_i, digest, new_K


# ---------------------------------------------------------------------------
# Group key follower (the "other drones")
# ---------------------------------------------------------------------------
@dataclass
class GroupKeyFollower:
    """Implements the actions of every non-leading drone.

    Apart from the initial ``Rep`` step, the follower performs exactly the
    same hash-chain / queue update as the leading drone.
    """

    fuzzy_extractor: FuzzyExtractor
    n: int = 5
    state: GroupKeyState = field(init=False)

    def __post_init__(self) -> None:
        self.state = GroupKeyState(n=self.n)

    # ------------------------------------------------------------------
    def initialize(self, w0_prime: np.ndarray, P0: HelperData,
                    expected_R0_digest: Optional[bytes] = None) -> bytes:
        """Recover ``R_0`` from the broadcast helper and check its hash."""
        R0 = self.fuzzy_extractor.rep(w0_prime, P0)
        if expected_R0_digest is not None and sha256(R0) != expected_R0_digest:
            raise ValueError("R_0 hash mismatch during initialization")
        self.state.initialize(R0)
        return self.state.K

    # ------------------------------------------------------------------
    def update(self, w_i_prime: np.ndarray, P_i: HelperData,
                expected_digest: bytes) -> bytes:
        """Receive a key update broadcast from the leading drone.

        The follower verifies ``h(A_i || R_i) == expected_digest`` after
        recovering ``R_i``.
        """
        if self.state.i < 0:
            raise RuntimeError("call initialize() before update()")
        R_i = self.fuzzy_extractor.rep(w_i_prime, P_i)
        new_A = _xor(R_i, self.state.A)
        if sha256(new_A + R_i) != expected_digest:
            raise ValueError("digest verification failed (R_i mismatch)")
        new_K = _xor(R_i, sha256(self.state.K))
        update_L(self.state.L, new_A, R_i)
        self.state.A = new_A
        self.state.K = new_K
        self.state.i += 1
        return new_K


__all__ = [
    "update_L",
    "GroupKeyState",
    "GroupKeyGenerator",
    "GroupKeyFollower",
]
