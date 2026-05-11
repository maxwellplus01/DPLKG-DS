from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .reliable_quant import QuantizedBlock


def gossip_vote(blocks: Sequence[QuantizedBlock],
                quorum: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if not blocks:
        raise ValueError("gossip_vote: empty input")
    n = len(blocks)
    n_bits = blocks[0].bits.size
    for b in blocks:
        if b.bits.size != n_bits or b.mask.size != n_bits:
            raise ValueError("blocks have inconsistent length")
    if quorum is None:
        quorum = n // 2 + 1
    quorum = max(1, int(quorum))

    bit_stack = np.stack([b.bits.astype(np.int32) for b in blocks])     # (n, n_bits)
    mask_stack = np.stack([b.mask.astype(np.int32) for b in blocks])

    # Per-position weighted votes: ones counted only where mask=1.
    ones_votes = np.sum(bit_stack * mask_stack, axis=0)
    valid_votes = np.sum(mask_stack, axis=0)

    consensus_bits = (ones_votes * 2 >= valid_votes).astype(np.uint8)
    # Tie-break (ones == zeros): prefer 0 -- more conservative.
    consensus_bits[(valid_votes == 0)] = 0

    consensus_mask = (valid_votes >= quorum).astype(np.uint8)
    return consensus_bits, consensus_mask
