from __future__ import annotations

import heapq
import math
from typing import Dict, Optional, Tuple

_NYT = 256     

class _HNode:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(
        self,
        freq:   int,
        symbol: Optional[int]    = None,
        left:   "Optional[_HNode]" = None,
        right:  "Optional[_HNode]" = None,
    ):
        self.freq   = freq
        self.symbol = symbol  
        self.left   = left
        self.right  = right

    def __lt__(self, other: "_HNode") -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        # Deterministic tie-break: lower symbol first; internal nodes last (-1)
        a = self.symbol if self.symbol is not None else 257
        b = other.symbol if other.symbol is not None else 257
        return a < b


def _build_tree(freqs: Dict[int, int]) -> _HNode:
    """Build Huffman tree from symbol→count dict. Deterministic."""
    heap: list = [_HNode(f, sym) for sym, f in freqs.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        only = heapq.heappop(heap)
        return _HNode(only.freq, left=only)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        heapq.heappush(heap, _HNode(lo.freq + hi.freq, left=lo, right=hi))
    return heap[0]


def _build_codebook(root: _HNode) -> Dict[int, str]:
    """DFS traversal → symbol-to-bitstring codebook."""
    book: Dict[int, str] = {}

    def _dfs(node: Optional[_HNode], bits: str) -> None:
        if node is None:
            return
        if node.symbol is not None:
            book[node.symbol] = bits or "0"
            return
        _dfs(node.left,  bits + "0")
        _dfs(node.right, bits + "1")

    _dfs(root, "")
    return book

def _encode_adaptive(data: bytes) -> str:
    """
    Adaptive Huffman encode.

    State shared with decoder (maintained identically on both sides):
      freqs  — symbol frequency table (real bytes only, not NYT)

    Protocol for each byte b:
      • b already in freqs  → output codebook[b]
      • b is new            → output codebook[NYT] + format(b, '08b')
      Then increment freqs[b] and rebuild tree.

    First byte special case: no tree exists yet → transmit as raw 8 bits.
    """
    freqs: Dict[int, int] = {}
    parts: list[str] = []

    for byte in data:
        if not freqs:
            parts.append(format(byte, "08b"))
        else:
            tree_freqs = dict(freqs)
            tree_freqs[_NYT] = 1
            codebook = _build_codebook(_build_tree(tree_freqs))

            if byte in freqs:
                parts.append(codebook[byte])
            else:
                parts.append(codebook[_NYT])
                parts.append(format(byte, "08b"))

        freqs[byte] = freqs.get(byte, 0) + 1

    return "".join(parts)


def _decode_adaptive(bitstring: str, original_length: int) -> bytes:
    """
    Adaptive Huffman decode — mirrors _encode_adaptive exactly.

    Maintains identical freq state; rebuilds the same tree before each symbol.
    """
    freqs: Dict[int, int] = {}
    result = bytearray()
    pos = 0

    while len(result) < original_length and pos < len(bitstring):
        if not freqs:
            # First symbol: raw 8 bits
            byte = int(bitstring[pos : pos + 8], 2)
            pos += 8
        else:
            # Rebuild same tree as encoder
            tree_freqs = dict(freqs)
            tree_freqs[_NYT] = 1
            root = _build_tree(tree_freqs)

            # Walk the tree bit by bit
            node = root
            while node.symbol is None:
                if pos >= len(bitstring):
                    break
                node = node.left if bitstring[pos] == "0" else node.right
                pos += 1

            if node.symbol == _NYT:
                # Read raw 8-bit new symbol
                byte = int(bitstring[pos : pos + 8], 2)
                pos += 8
            else:
                byte = node.symbol  # type: ignore[assignment]

        result.append(byte)
        freqs[byte] = freqs.get(byte, 0) + 1

    return bytes(result)

def _compute_metrics(data: bytes, bitstring: str) -> dict:
    from collections import Counter
    import base64

    n = len(data)
    if n == 0:
        return {
            "original_bytes": 0, "compressed_bytes": 0,
            "compression_ratio": 1.0,
            "entropy_bits_per_symbol": 0.0,
            "avg_bits_per_symbol": 0.0,
            "encoding_efficiency": 1.0,
        }

    original_bits   = n * 8
    compressed_bits = len(bitstring)
    compressed_bytes = math.ceil(compressed_bits / 8)
    compression_ratio = original_bits / compressed_bits if compressed_bits else 1.0

    freqs = Counter(data)
    entropy = -sum((c / n) * math.log2(c / n) for c in freqs.values())
    avg_bits   = compressed_bits / n
    efficiency = (entropy / avg_bits) if avg_bits > 0 else 1.0

    return {
        "original_bytes":          n,
        "compressed_bytes":        compressed_bytes,
        "compression_ratio":       round(compression_ratio, 4),
        "entropy_bits_per_symbol": round(entropy, 4),
        "avg_bits_per_symbol":     round(avg_bits, 4),
        "encoding_efficiency":     round(min(efficiency, 1.0), 4),
    }

def compress(data: bytes) -> dict:
    """
    Compress bytes using adaptive Huffman and return a payload dict.

    Because this is adaptive, the decoder does NOT need a pre-built frequency
    table — it reconstructs the model on-the-fly from the bitstream itself.
    The 'freqs' field in the payload is kept for API compatibility but is
    derived post-hoc and is not used during decompression.

    Returned dict keys:
        bitstring       — raw '0'/'1' string
        packed_b64      — base64-encoded packed bytes (for transport)
        pad_bits        — zero-padding bits appended to fill last byte
        original_length — original byte count (needed to know when to stop)
        freqs           — post-encoding frequency table (API compatibility)
        stats           — compression metrics
    """
    import base64
    from collections import Counter

    if len(data) == 0:
        return {
            "bitstring": "", "packed_b64": "", "pad_bits": 0,
            "original_length": 0, "freqs": {},
            "stats": {
                "original_bytes": 0, "compressed_bytes": 0,
                "compression_ratio": 1.0,
                "entropy_bits_per_symbol": 0.0,
                "avg_bits_per_symbol": 0.0,
                "encoding_efficiency": 1.0,
            },
        }

    bitstring = _encode_adaptive(data)
    stats     = _compute_metrics(data, bitstring)

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
        "freqs":           dict(Counter(data)),
        "stats":           stats,
    }


def decompress(payload: dict) -> bytes:
    """
    Decompress a payload produced by compress().

    Only 'bitstring' and 'original_length' are required.
    The 'freqs' field is ignored — the adaptive decoder reconstructs
    the model from the bitstream itself.
    """
    return _decode_adaptive(
        payload["bitstring"],
        payload["original_length"],
    )
