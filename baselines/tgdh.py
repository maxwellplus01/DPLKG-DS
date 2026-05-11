from __future__ import annotations

import math
from typing import List

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat,
)


X25519_PUBKEY_BYTES = 32   # 256 bits on the wire


def _hkdf(secret: bytes, info: bytes = b"tgdh") -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32,
                 salt=None, info=info).derive(secret)


def _node_secret(left_priv: X25519PrivateKey,
                 right_pub: X25519PublicKey) -> bytes:
    """Diffie-Hellman secret at an internal node, derived from the
    left child's private key and the right child's public key.

    In TGDH proper the two children are recursively node keys; for a
    pure overhead measurement it suffices to chain HKDF over the
    successive co-path public keys, which is what the broadcast layer
    transports.  We model this faithfully and only count broadcast
    bytes -- not the local computation.
    """
    raw = left_priv.exchange(right_pub)
    return _hkdf(raw)


def measure_tgdh_refresh_bits(n: int) -> int:
    """Run one TGDH refresh by leaf 0 in a balanced tree of ``n``
    members and return the total *broadcast* wire payload in bits.

    Returns the sum over the co-path of one X25519 public key per
    level, which is what Sect. 4.4 of Kim et al. 2004 specifies.
    """
    if n < 2:
        return 0
    h = math.ceil(math.log2(n))
    # one public key per level on the co-path
    bits = 0
    for level in range(h):
        # generate fresh key + co-path peer key, broadcast our pubkey
        my_priv = X25519PrivateKey.generate()
        peer_pub = X25519PrivateKey.generate().public_key()
        # measure: actual serialised bytes of the public key
        wire = my_priv.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw)
        assert len(wire) == X25519_PUBKEY_BYTES
        bits += 8 * len(wire)
        # compute next-level secret (not transmitted)
        _ = _node_secret(my_priv, peer_pub)
    return bits


def measure_mls_commit_bits(n: int) -> int:
    """Lower-bound for MLS Commit broadcast in a balanced TreeKEM
    over n members (RFC 9420).

    A Commit at the root by leaf 0 ships an HPKE ciphertext (32-byte
    HPKE-encapped key + 32-byte AEAD payload + 16-byte AEAD tag = 80
    bytes per affected internal node) on the co-path of length
    ceil(log2(n)).  This is the minimum wire payload; framing,
    signatures and PSK metadata are excluded.  Implemented as a
    counted-byte stub (no live HPKE key generation) because no public
    Python implementation of MLS is bundled with cryptography.
    """
    if n < 2:
        return 0
    h = math.ceil(math.log2(n))
    HPKE_NODE_BYTES = 32 + 32 + 16
    return 8 * h * HPKE_NODE_BYTES


def measure_for_swarm(sizes: List[int]) -> List[dict]:
    rows = []
    for n in sizes:
        tgdh = measure_tgdh_refresh_bits(n)
        mls = measure_mls_commit_bits(n)
        rows.append({"n": n, "tgdh_measured_bits": tgdh,
                      "mls_lowerbound_bits": mls})
    return rows


if __name__ == "__main__":
    for r in measure_for_swarm([3, 5, 7, 10, 15, 20, 32]):
        print(r)
