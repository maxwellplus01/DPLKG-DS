from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Tuple

from Crypto.Cipher import AES

from .fuzzy_extractor import sha256, _xor
from .key_update import GroupKeyState, GroupKeyGenerator


# ---------------------------------------------------------------------------
# Algorithm 2 -- Get string C
# ---------------------------------------------------------------------------
def get_string_C(M: bytes, L: Iterable[bytes], A_jm1: bytes) -> Tuple[bytes, bool]:
    """Implementation of Algorithm 2 (page 4471).

    Returns ``(C, found)``.  ``found`` is ``True`` iff a matching ``item``
    was located in ``L``; otherwise ``C`` is a fresh random string of the
    same length as ``M``.
    """
    for item in L:
        C = _xor(_xor(M, item), A_jm1)
        if sha256(C) == M:
            return C, True
    return os.urandom(len(M)), False


# ---------------------------------------------------------------------------
# AES-256 helpers (deterministic IV-less mode for protocol verification)
# ---------------------------------------------------------------------------
def _aes_key(material: bytes) -> bytes:
    """Derive a 32-byte AES-256 key from arbitrary-length material."""
    return sha256(material)


def aes256_encrypt(key_material: bytes, plaintext: bytes) -> bytes:
    """AES-256 in CTR mode with a random 8-byte nonce, prepended to the
    ciphertext.  AES-CTR matches the "secure symmetric encryption
    algorithm" requirement of the paper while also providing IND-CPA
    security."""
    cipher = AES.new(_aes_key(key_material), AES.MODE_CTR, nonce=os.urandom(8))
    return cipher.nonce + cipher.encrypt(plaintext)


def aes256_decrypt(key_material: bytes, blob: bytes) -> bytes:
    if len(blob) < 8:
        raise ValueError("ciphertext too short")
    nonce, ciphertext = blob[:8], blob[8:]
    cipher = AES.new(_aes_key(key_material), AES.MODE_CTR, nonce=nonce)
    return cipher.decrypt(ciphertext)


# ---------------------------------------------------------------------------
# Returning drone (the side that lost synchronisation)
# ---------------------------------------------------------------------------
@dataclass
class ReturningDrone:
    """Stateful representation of a returning drone.

    The drone remembers the ``(K_i, A_i)`` it had right before leaving the
    swarm at stage ``i``.  After the protocol completes,
    :attr:`recovered_h_K_jm1` holds ``h(K_{j-1})`` which can be combined
    with the next ``R_j`` (received via a regular key-update broadcast) to
    re-derive ``K_j``.
    """

    Ki: bytes
    Ai: bytes
    nonce_bytes: int = 32

    # populated by run/build_request and run/finalize_response
    _N: bytes = b""
    recovered_A_jm1: bytes = b""
    recovered_h_K_jm1: bytes = b""

    # ------------------------------------------------------------------
    # step 1
    # ------------------------------------------------------------------
    def build_request(self) -> Tuple[bytes, bytes]:
        """Returns the pair ``(M, Q)`` sent to the leading drone."""
        if len(self.Ai) == 0:
            raise RuntimeError("ReturningDrone.Ai must be non-empty")
        self._N = os.urandom(len(self.Ai))
        M = sha256(self.Ai)
        Q = _xor(self.Ai, self._N)
        return M, Q

    # ------------------------------------------------------------------
    # step 3
    # ------------------------------------------------------------------
    def finalize_response(self, S: bytes, EK: bytes, tag: bytes) -> bytes:
        """Verify and decrypt the leading drone's reply.

        Returns ``h(K_{j-1})`` on success, raises :class:`ValueError`
        otherwise.
        """
        if not self._N:
            raise RuntimeError("call build_request() before finalize_response()")
        A = _xor(S, self._N)
        expected_tag = sha256(A + EK + S)
        if expected_tag != tag:
            raise ValueError("authentication tag mismatch (replay or forgery)")
        h_K_jm1 = aes256_decrypt(A, EK)
        self.recovered_A_jm1 = A
        self.recovered_h_K_jm1 = h_K_jm1
        return h_K_jm1


# ---------------------------------------------------------------------------
# Leading drone responder
# ---------------------------------------------------------------------------
def leading_drone_response(state: GroupKeyState, M: bytes,
                            Q: bytes) -> Tuple[bytes, bytes, bytes, bool]:
    """Step 2 of the protocol.

    ``state`` is the leading drone's current :class:`GroupKeyState`
    (typically ``GroupKeyGenerator.state``).  ``state.A`` is treated as
    ``A_{j-1}`` and ``state.K`` as ``K_{j-1}``.

    Returns ``(S, EK, tag, accepted)`` where ``accepted`` is ``True`` iff
    Algorithm 2 successfully matched ``M`` against the queue ``L`` (i.e.
    the request looks legitimate).
    """
    A_jm1 = state.A
    K_jm1 = state.K
    if len(M) != len(A_jm1):
        raise ValueError("M and A_{j-1} must have the same length")
    if len(Q) != len(A_jm1):
        raise ValueError("Q and A_{j-1} must have the same length")

    C, accepted = get_string_C(M, list(state.L), A_jm1)
    # Recover the nonce N from C and Q (works whenever C == A_i; otherwise
    # this just produces a random-looking value that will fail the
    # verification on the returning-drone side).
    N = _xor(Q, C)
    S = _xor(N, A_jm1)
    EK = aes256_encrypt(A_jm1, sha256(K_jm1))
    tag = sha256(A_jm1 + EK + S)
    return S, EK, tag, accepted


# ---------------------------------------------------------------------------
# High-level orchestration helper used by the demos and tests
# ---------------------------------------------------------------------------
def run_robust_agreement(returning: ReturningDrone,
                          leader: GroupKeyGenerator) -> bytes:
    """Run the three-pass protocol end-to-end and return ``h(K_{j-1})``.

    The function does not perform the subsequent key-update step; the
    returning drone uses the recovered ``A_{j-1}`` and ``h(K_{j-1})``
    together with the next broadcast ``(P_j, h(A_j || R_j))`` to compute
    its own ``K_j``.
    """
    M, Q = returning.build_request()
    S, EK, tag, accepted = leading_drone_response(leader.state, M, Q)
    if not accepted:
        raise ValueError("leading drone rejected the request "
                         "(A_i not found in L)")
    return returning.finalize_response(S, EK, tag)


__all__ = [
    "get_string_C",
    "aes256_encrypt",
    "aes256_decrypt",
    "ReturningDrone",
    "leading_drone_response",
    "run_robust_agreement",
]
