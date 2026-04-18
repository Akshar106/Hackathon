"""
Adaptive Huffman Encoding — FGK Algorithm (Faller-Gallager-Knuth)

Encodes/decodes a byte string using an adaptively-updated Huffman tree.
No zlib, bz2, or any external compression library is used.

Key concepts:
  - NYT (Not Yet Transmitted) node represents all symbols not yet seen
  - Tree satisfies the sibling property: nodes listed in non-decreasing
    weight order have sibling pairs adjacent
  - On each new symbol: emit NYT code + 8-bit literal, then update tree
  - On repeated symbol: emit its current codeword, then update tree

Public API:
    encode(data: bytes) -> (bitstring: str, tree_stats: dict)
    decode(bitstring: str, original_length: int) -> bytes
    compress(data: bytes) -> dict   — encode + metrics
    decompress(payload: dict) -> bytes
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math


# ── Tree node ─────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ("weight", "symbol", "parent", "left", "right", "order")

    def __init__(
        self,
        weight: int = 0,
        symbol: Optional[int] = None,
        order: int = 0,
    ):
        self.weight = weight
        self.symbol = symbol          # None for internal nodes; int (0-255) for leaves; -1 for NYT
        self.parent: Optional[_Node] = None
        self.left:   Optional[_Node] = None
        self.right:  Optional[_Node] = None
        self.order = order            # higher order = higher in the sibling list


# ── Adaptive Huffman tree (FGK) ───────────────────────────────────────────────

class AdaptiveHuffmanTree:
    """
    A single FGK adaptive Huffman tree that can encode and decode
    one symbol at a time while updating itself.
    """

    MAX_ORDER = 512  # enough for 256 leaves + 255 internal + NYT

    def __init__(self):
        self._order_counter = self.MAX_ORDER
        self.nyt = _Node(weight=0, symbol=-1, order=self._next_order())
        self.root = self.nyt
        self._symbol_to_node: Dict[int, _Node] = {}

    # ── Order counter ─────────────────────────────────────────────────────────

    def _next_order(self) -> int:
        o = self._order_counter
        self._order_counter -= 1
        return o

    # ── Codeword for a node ───────────────────────────────────────────────────

    def get_code(self, node: _Node) -> str:
        bits = []
        cur = node
        while cur.parent is not None:
            if cur.parent.left is cur:
                bits.append("0")
            else:
                bits.append("1")
            cur = cur.parent
        return "".join(reversed(bits))

    # ── Sibling swap (FGK step 1) ─────────────────────────────────────────────

    def _find_highest_in_block(self, node: _Node) -> _Node:
        """Return the node with highest order among all nodes of the same weight."""
        highest = node
        self._collect_block(self.root, node.weight, highest_ref=[highest])
        return self._collect_block_result

    _collect_block_result: _Node = None  # type: ignore

    def _collect_block(self, cur: Optional[_Node], weight: int, highest_ref: list):
        if cur is None:
            return
        if cur.weight == weight and cur.order > highest_ref[0].order:
            highest_ref[0] = cur
            self._collect_block_result = cur
        else:
            if not hasattr(self, "_collect_block_result") or self._collect_block_result is None:
                self._collect_block_result = highest_ref[0]
        self._collect_block(cur.left,  weight, highest_ref)
        self._collect_block(cur.right, weight, highest_ref)
        self._collect_block_result = highest_ref[0]

    def _swap_nodes(self, a: _Node, b: _Node):
        """Swap a and b in the tree (not the root, not parent-child pairs)."""
        if a is b or a is self.root or b is self.root:
            return
        if a.parent is b or b.parent is a:
            return

        pa, pb = a.parent, b.parent

        # Swap child pointers in parents
        if pa.left is a:
            pa.left = b
        else:
            pa.right = b

        if pb.left is b:
            pb.left = a
        else:
            pb.right = a

        a.parent, b.parent = pb, pa
        a.order, b.order = b.order, a.order

    # ── Update tree after seeing symbol ──────────────────────────────────────

    def _increment_and_slide(self, node: _Node):
        """Walk from node to root, sliding and incrementing (FGK update)."""
        cur: Optional[_Node] = node
        while cur is not None:
            # Find the leader of the current block
            leader = self._find_highest_order_in_block(cur)
            if leader is not cur and leader is not cur.parent:
                self._swap_nodes(cur, leader)
                cur = leader   # after swap cur is now where leader was
            cur.weight += 1
            cur = cur.parent

    def _find_highest_order_in_block(self, node: _Node) -> _Node:
        """BFS over tree to find node with same weight and highest order number."""
        best = node
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is None:
                continue
            if n.weight == node.weight and n.order > best.order:
                best = n
            stack.append(n.left)
            stack.append(n.right)
        return best

    # ── Encode one symbol ─────────────────────────────────────────────────────

    def encode_symbol(self, symbol: int) -> str:
        """
        Returns the bitstring for this symbol and updates the tree.
        For a new symbol: NYT code + 8-bit literal.
        For a seen symbol: its current codeword.
        """
        if symbol in self._symbol_to_node:
            node = self._symbol_to_node[symbol]
            code = self.get_code(node)
            self._increment_and_slide(node)
        else:
            # New symbol: transmit NYT path + 8-bit literal
            nyt_code = self.get_code(self.nyt)
            literal = format(symbol, "08b")
            code = nyt_code + literal
            self._add_symbol(symbol)

        return code

    def _add_symbol(self, symbol: int):
        """Split NYT into new internal node with NYT child and new leaf."""
        new_internal = _Node(weight=0, order=self._next_order())
        new_leaf      = _Node(weight=0, symbol=symbol, order=self._next_order())
        old_nyt       = self.nyt

        new_internal.parent = old_nyt.parent
        if old_nyt.parent is not None:
            if old_nyt.parent.left is old_nyt:
                old_nyt.parent.left = new_internal
            else:
                old_nyt.parent.right = new_internal
        else:
            self.root = new_internal

        new_internal.left  = old_nyt
        new_internal.right = new_leaf
        old_nyt.parent     = new_internal
        new_leaf.parent    = new_internal

        self._symbol_to_node[symbol] = new_leaf
        self.nyt = old_nyt

        self._increment_and_slide(new_leaf)

    # ── Decode one symbol ─────────────────────────────────────────────────────

    def decode_symbol(self, bits: str, pos: int) -> Tuple[int, int]:
        """
        Decode one symbol starting at bit position pos.
        Returns (symbol, new_pos).
        """
        cur = self.root

        while True:
            if cur is self.nyt:
                # Read next 8 bits as literal
                symbol = int(bits[pos : pos + 8], 2)
                pos += 8
                self._add_symbol(symbol)
                return symbol, pos

            if cur.left is None and cur.right is None:
                # Leaf node
                symbol = cur.symbol
                self._increment_and_slide(cur)
                return symbol, pos  # type: ignore

            if pos >= len(bits):
                raise ValueError("Bitstring ended unexpectedly during decode")

            bit = bits[pos]
            pos += 1
            cur = cur.left if bit == "0" else cur.right


# ── Public encode / decode functions ─────────────────────────────────────────

def encode(data: bytes) -> Tuple[str, dict]:
    """
    Encode bytes using FGK adaptive Huffman.

    Returns:
        bitstring  — the full compressed bit sequence as a string of '0'/'1'
        stats      — encoding metrics dict
    """
    tree = AdaptiveHuffmanTree()
    parts = []
    for byte in data:
        parts.append(tree.encode_symbol(byte))
    bitstring = "".join(parts)

    # Metrics
    original_bits = len(data) * 8
    compressed_bits = len(bitstring)
    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 1.0

    # Shannon entropy of source
    from collections import Counter
    counts = Counter(data)
    n = len(data)
    entropy = 0.0
    if n > 0:
        for c in counts.values():
            p = c / n
            entropy -= p * math.log2(p)

    # Encoding efficiency = entropy / avg_bits_per_symbol
    avg_bits = compressed_bits / n if n > 0 else 0.0
    efficiency = (entropy / avg_bits) if avg_bits > 0 else 1.0

    stats = {
        "original_bytes": len(data),
        "compressed_bits": compressed_bits,
        "compressed_bytes": math.ceil(compressed_bits / 8),
        "compression_ratio": round(compression_ratio, 4),
        "entropy_bits_per_symbol": round(entropy, 4),
        "avg_bits_per_symbol": round(avg_bits, 4),
        "encoding_efficiency": round(min(efficiency, 1.0), 4),
    }
    return bitstring, stats


def decode(bitstring: str, original_length: int) -> bytes:
    """
    Decode a bitstring produced by encode().
    original_length is the number of bytes expected.
    """
    tree = AdaptiveHuffmanTree()
    result = bytearray()
    pos = 0
    while len(result) < original_length:
        symbol, pos = tree.decode_symbol(bitstring, pos)
        result.append(symbol)
    return bytes(result)


# ── High-level compress / decompress ─────────────────────────────────────────

def compress(data: bytes) -> dict:
    """
    Compress bytes and return a payload dict suitable for JSON serialisation
    (bitstring stored as base64-packed bytes).

    Returned dict keys:
        bitstring       — raw '0'/'1' string (for metrics / debugging)
        packed_b64      — base64-encoded packed bytes (for transport)
        original_length — original byte count
        stats           — compression metrics
    """
    import base64

    bitstring, stats = encode(data)

    # Pack bits into bytes (pad with zeros at end)
    pad = (8 - len(bitstring) % 8) % 8
    padded = bitstring + "0" * pad
    packed = bytearray()
    for i in range(0, len(padded), 8):
        packed.append(int(padded[i : i + 8], 2))

    return {
        "bitstring": bitstring,
        "packed_b64": base64.b64encode(bytes(packed)).decode(),
        "pad_bits": pad,
        "original_length": len(data),
        "stats": stats,
    }


def decompress(payload: dict) -> bytes:
    """Decompress a payload dict produced by compress()."""
    return decode(payload["bitstring"], payload["original_length"])
