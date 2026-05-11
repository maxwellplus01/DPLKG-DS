from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import bchlib  # type: ignore
except (ImportError, UnicodeDecodeError) as exc:  # pragma: no cover
    raise ImportError(
        "fuxian.fuzzy_extractor requires the 'bchlib' package "
        "(pip install bchlib)"
    ) from exc

from .quantization import bits_to_bytes, bytes_to_bits, hamming_distance


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256(data: bytes) -> bytes:
    """SHA-256 hash, matching Section VII-C of the paper."""
    return hashlib.sha256(data).digest()


def _xor(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("xor operands must have the same length")
    return bytes(x ^ y for x, y in zip(a, b))


def _xor_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("xor operands must have the same shape")
    return np.bitwise_xor(a.astype(np.uint8), b.astype(np.uint8))


# ---------------------------------------------------------------------------
# BCH wrapper
# ---------------------------------------------------------------------------
@dataclass
class BCHCode:
    """Thin wrapper around :class:`bchlib.BCH`.

    Parameters
    ----------
    t:
        Number of correctable bit errors.
    m:
        Galois-field exponent; the resulting codeword length is
        ``n = 2**m - 1`` bits.
    """

    t: int = 8
    m: int = 10

    def __post_init__(self) -> None:
        self._bch = bchlib.BCH(t=self.t, m=self.m)
        # Raw library parameters.  ``bchlib`` operates on whole-byte
        # messages, so we restrict the code to the largest data length that
        # leaves the codeword (data + ecc) within ``2**m - 1`` bits.
        raw_n = int(self._bch.n)
        ecc_bits = int(self._bch.ecc_bits)
        max_data_bits = raw_n - ecc_bits
        self.message_bytes = max_data_bits // 8        # whole bytes only
        self.k = self.message_bytes * 8                 # message length (bits)
        self.ecc_bits = ecc_bits
        self.ecc_bytes = int(self._bch.ecc_bytes)
        self.n = self.k + self.ecc_bits                 # effective codeword length

    # ------------------------------------------------------------------
    def encode(self, message_bits: np.ndarray) -> np.ndarray:
        """Encode a ``k``-bit message into an ``n``-bit codeword.

        The BCH code we use is *systematic*: the codeword is the message
        concatenated with the ECC bits.
        """
        if message_bits.size != self.k:
            raise ValueError(
                f"message must be exactly {self.k} bits long "
                f"(got {message_bits.size})"
            )
        msg_bytes = bits_to_bytes(message_bits)
        # right-pad to message_bytes if the bit length is not a multiple of 8
        msg_bytes = msg_bytes.ljust(self.message_bytes, b"\x00")
        ecc = self._bch.encode(msg_bytes)
        ecc_bits = bytes_to_bits(bytes(ecc), self.ecc_bits)
        return np.concatenate([message_bits.astype(np.uint8), ecc_bits])

    def decode(self, codeword_bits: np.ndarray) -> Tuple[np.ndarray, int]:
        """Decode a possibly-corrupted ``n``-bit codeword.

        Returns
        -------
        message_bits:
            The decoded ``k``-bit message.
        nerr:
            Number of corrected bit errors.  ``-1`` (returned by ``bchlib``)
            means decoding failure: the function then raises
            :class:`ValueError`.
        """
        if codeword_bits.size != self.n:
            raise ValueError(
                f"codeword must be exactly {self.n} bits long "
                f"(got {codeword_bits.size})"
            )
        msg_bits = codeword_bits[: self.k]
        ecc_bits = codeword_bits[self.k :]
        data = bytearray(bits_to_bytes(msg_bits).ljust(self.message_bytes, b"\x00"))
        ecc = bytearray(bits_to_bytes(ecc_bits).ljust(self.ecc_bytes, b"\x00"))
        nerr = self._bch.decode(bytes(data), bytes(ecc))
        if nerr < 0:
            raise ValueError("BCH decoding failed (too many errors)")
        # Apply the in-place correction.
        self._bch.correct(data, ecc)
        corrected = bytes_to_bits(bytes(data), self.k)
        return corrected, int(nerr)


# ---------------------------------------------------------------------------
# Fuzzy extractor
# ---------------------------------------------------------------------------
@dataclass
class HelperData:
    """Public auxiliary string ``P = (m, r1)`` produced by ``Gen``.

    Both ``m`` and ``r1`` are bit vectors of length ``n = |w|``, so
    ``|P| = 2|w|`` (matching the bound stated in Section V-A.3).
    """

    m: np.ndarray
    r1: np.ndarray

    def __post_init__(self) -> None:
        self.m = np.asarray(self.m, dtype=np.uint8)
        self.r1 = np.asarray(self.r1, dtype=np.uint8)
        if self.m.shape != self.r1.shape:
            raise ValueError("|m| must equal |r1|")

    @property
    def length(self) -> int:
        return int(self.m.size + self.r1.size)


@dataclass
class FuzzyExtractor:
    """Fuzzy extractor :math:`F\\!F_E(\\mathcal{M}, l, t)`.

    Parameters
    ----------
    input_bits:
        Length of the input string ``w``.  Must equal the BCH codeword
        length ``n``.
    bch:
        A :class:`BCHCode` instance.  The number of correctable errors
        ``t`` is read from this object.
    key_bytes:
        Length ``l`` of the extracted key, in bytes (default 32, i.e. 256
        bits, as in the paper).
    rng:
        A function returning ``num_bytes`` random bytes; defaults to
        :func:`os.urandom`.
    """

    input_bits: int
    bch: BCHCode
    key_bytes: int = 32  # 256-bit key as in the paper
    rng: Optional[callable] = None

    def __post_init__(self) -> None:
        if self.input_bits != self.bch.n:
            raise ValueError(
                f"input_bits ({self.input_bits}) must equal the BCH "
                f"codeword length n = {self.bch.n}"
            )
        if self.rng is None:
            self.rng = os.urandom

    # convenience read-only attributes
    @property
    def t(self) -> int:
        return self.bch.t

    @property
    def message_bits(self) -> int:
        return self.bch.k

    # ------------------------------------------------------------------
    # Gen
    # ------------------------------------------------------------------
    def gen(self, w: np.ndarray) -> Tuple[bytes, HelperData]:
        """Run the ``Gen(w) -> (R, P)`` procedure.

        Returns the ``key_bytes``-byte secret ``R`` and the public helper
        ``P``.
        """
        w = np.asarray(w, dtype=np.uint8)
        if w.size != self.input_bits:
            raise ValueError(
                f"w must be exactly {self.input_bits} bits long "
                f"(got {w.size})"
            )

        r1 = bytes_to_bits(self.rng((self.input_bits + 7) // 8),
                           self.input_bits)
        # r2 is a random message for the BCH encoder (k bits)
        r2 = bytes_to_bits(self.rng((self.bch.k + 7) // 8), self.bch.k)
        codeword = self.bch.encode(r2)
        m = _xor_bits(w, codeword)

        secret_bits = _xor_bits(w, r1)
        R = sha256(bits_to_bytes(secret_bits))[: self.key_bytes]
        return R, HelperData(m=m, r1=r1)

    # ------------------------------------------------------------------
    # Rep
    # ------------------------------------------------------------------
    def rep(self, w_prime: np.ndarray, helper: HelperData) -> bytes:
        """Run the ``Rep(w', P) -> R'`` procedure.

        Raises :class:`ValueError` when the BCH decoder fails (i.e. when
        ``D_{dis}(w, w') > t``).
        """
        w_prime = np.asarray(w_prime, dtype=np.uint8)
        if w_prime.size != self.input_bits:
            raise ValueError(
                f"w' must be exactly {self.input_bits} bits long "
                f"(got {w_prime.size})"
            )
        if helper.m.size != self.input_bits:
            raise ValueError("helper.m has the wrong length")

        # Decode w' xor m to recover r2, then re-encode and xor with m to
        # recover w.
        noisy_codeword = _xor_bits(w_prime, helper.m)
        r2_recovered, _ = self.bch.decode(noisy_codeword)
        codeword = self.bch.encode(r2_recovered)
        w = _xor_bits(helper.m, codeword)

        secret_bits = _xor_bits(w, helper.r1)
        return sha256(bits_to_bytes(secret_bits))[: self.key_bytes]


__all__ = [
    "sha256",
    "BCHCode",
    "HelperData",
    "FuzzyExtractor",
    "hamming_distance",
]
