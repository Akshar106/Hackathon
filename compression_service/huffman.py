"""
Huffman Encoding — 2-pass input-adaptive algorithm.

First pass:  count byte frequencies in the input (O(n)).
Second pass: build the optimal Huffman tree, encode every symbol (O(n log n)).

This approach adapts the code table to each specific input's frequency
distribution, achieving near-Shannon-entropy compression — far better than
a cold-start adaptive (FGK) coder on the short texts produced by OCR.

No zlib, bz2, or any external compression library is used.

Public API:
    encode(data: bytes) -> (bitstring: str, stats: dict)
    decode(bitstring: str, original_length: int, freqs: dict) -> bytes
    compress(data: bytes) -> dict
    decompress(payload: dict) -> bytes
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from typing import Dict, Optional, Tuple


# ── Tree node ─────────────────────────────────────────────────────────────────

class _HNode:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(
        self,
        freq: int,
        symbol: Optional[int] = None,
        left:  "Optional[_HNode]" = None,
        right: "Optional[_HNode]" = None,
    ):
        self.freq   = freq
        self.symbol = symbol   # int byte value for leaves, None for internal nodes
        self.left   = left
        self.right  = right

    def __lt__(self, other: "_HNode") -> bool:
        # Primary: lower frequency first (min-heap)
        if self.freq != other.freq:
            return self.freq < other.freq
        # Tie-break: leaves before internal nodes, then by symbol value
        a = self.symbol if self.symbol is not None else -1
        b = other.symbol if other.symbol is not None else -1
        return a < b


# ── Tree construction ─────────────────────────────────────────────────────────

def _build_tree(freqs: Dict[int, int]) -> _HNode:
    """Build an optimal Huffman tree from a byte → count frequency dict."""
    heap: list = [_HNode(f, sym) for sym, f in freqs.items()]
    heapq.heapify(heap)

    # Edge case: only one unique symbol in the data
    if len(heap) == 1:
        only = heapq.heappop(heap)
        return _HNode(only.freq, left=only)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        heapq.heappush(heap, _HNode(lo.freq + hi.freq, left=lo, right=hi))

    return heap[0]


def _build_codebook(root: _HNode) -> Dict[int, str]:
    """DFS traversal to produce symbol → bitstring codebook."""
    codebook: Dict[int, str] = {}

    def _dfs(node: Optional[_HNode], bits: str) -> None:
        if node is None:
            return
        if node.symbol is not None:          # leaf node
            codebook[node.symbol] = bits or "0"   # single-symbol edge case
            return
        _dfs(node.left,  bits + "0")
        _dfs(node.right, bits + "1")

    _dfs(root, "")
    return codebook


# ── Core encode / decode ──────────────────────────────────────────────────────

def encode(data: bytes) -> Tuple[str, dict]:
    """
    Encode bytes using 2-pass adaptive Huffman.

    Returns:
        bitstring  — compressed bit sequence as a string of '0'/'1'
        stats      — encoding metrics dict (includes 'freqs' for decoding)
    """
    if not data:
        return "", {
            "original_bytes": 0, "compressed_bits": 0, "compressed_bytes": 0,
            "compression_ratio": 1.0, "entropy_bits_per_symbol": 0.0,
            "avg_bits_per_symbol": 0.0, "encoding_efficiency": 1.0,
            "freqs": {},
        }

    freqs    = dict(Counter(data))
    root     = _build_tree(freqs)
    codebook = _build_codebook(root)

    bitstring = "".join(codebook[b] for b in data)

    # ── Metrics ───────────────────────────────────────────────────────────────
    n               = len(data)
    original_bits   = n * 8
    compressed_bits = len(bitstring)
    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 1.0

    entropy = 0.0
    for cnt in freqs.values():
        p = cnt / n
        entropy -= p * math.log2(p)

    avg_bits   = compressed_bits / n
    efficiency = (entropy / avg_bits) if avg_bits > 0 else 1.0

    stats = {
        "original_bytes":          n,
        "compressed_bits":         compressed_bits,
        "compressed_bytes":        math.ceil(compressed_bits / 8),
        "compression_ratio":       round(compression_ratio, 4),
        "entropy_bits_per_symbol": round(entropy, 4),
        "avg_bits_per_symbol":     round(avg_bits, 4),
        "encoding_efficiency":     round(min(efficiency, 1.0), 4),
        "freqs":                   freqs,    # carried for decompression
    }
    return bitstring, stats


def decode(bitstring: str, original_length: int, freqs: Dict[int, int]) -> bytes:
    """
    Decode a bitstring produced by encode().

    freqs must be the exact byte-frequency dict used during encoding.
    """
    if original_length == 0 or not bitstring:
        return b""

    root   = _build_tree(freqs)
    result = bytearray()
    cur    = root

    for bit in bitstring:
        if cur.symbol is not None:           # arrived at a leaf
            result.append(cur.symbol)
            if len(result) == original_length:
                break
            cur = root                        # restart from root
        cur = cur.left if bit == "0" else cur.right

    # Flush the final leaf (happens when the last bit lands exactly on a leaf)
    if len(result) < original_length and cur is not None and cur.symbol is not None:
        result.append(cur.symbol)

    return bytes(result)


# ── High-level compress / decompress ─────────────────────────────────────────

def compress(data: bytes) -> dict:
    """
    Compress bytes and return a payload dict suitable for storage/transport.

    Returned dict keys:
        bitstring       — raw '0'/'1' string
        packed_b64      — base64-encoded packed bytes
        pad_bits        — zero-padding bits added at end
        original_length — original byte count
        freqs           — frequency table needed for decompression
        stats           — compression metrics
    """
    import base64

    bitstring, stats = encode(data)
    freqs = stats.pop("freqs")              # lift out of stats to top level

    pad    = (8 - len(bitstring) % 8) % 8
    padded = bitstring + "0" * pad
    packed = bytearray(
        int(padded[i : i + 8], 2) for i in range(0, len(padded), 8)
    )

    return {
        "bitstring":       bitstring,
        "packed_b64":      base64.b64encode(bytes(packed)).decode(),
        "pad_bits":        pad,
        "original_length": len(data),
        "freqs":           freqs,
        "stats":           stats,
    }


def decompress(payload: dict) -> bytes:
    """Decompress a payload dict produced by compress()."""
    return decode(
        payload["bitstring"],
        payload["original_length"],
        payload["freqs"],
    )
