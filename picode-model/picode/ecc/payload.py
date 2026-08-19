"""Pack short text messages into a fixed number of payload bits.

The first bit selects the character encoding:

* ``0`` — compact 6-bit alphabet (``a-z``, ``A-Z``, ``0-9``, ``-``), which fits
  short-link style IDs: 6 characters in the 38-bit payload of LDPC(72, 38).
* ``1`` — raw UTF-8 bytes, 8 bits each, for anything outside that alphabet.

Unused trailing bits are zero, and a zero symbol terminates the text.
"""

from __future__ import annotations

import string

ALPHABET = "\0" + string.ascii_lowercase + string.ascii_uppercase + string.digits + "-"
assert len(ALPHABET) == 64
_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def _to_bits(value: int, width: int) -> list[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _from_bits(bits: list[int]) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def capacity(num_bits: int, text: str = "") -> int:
    """Maximum number of characters of ``text``'s kind that fit in ``num_bits``."""
    compact = all(c in _INDEX and c != "\0" for c in text)
    return (num_bits - 1) // (6 if compact else 8)


def encode_text(text: str, num_bits: int) -> tuple[list[int], str]:
    """Encode text into exactly ``num_bits`` bits.

    Args:
        text: Message to encode.
        num_bits: Payload size in bits (must be at least 7).

    Returns:
        Tuple of (bits, stored_text); ``stored_text`` is shorter than ``text`` when the
        message had to be truncated to fit.

    Raises:
        ValueError: If ``num_bits`` is too small for a single character.
    """
    if num_bits < 7:
        raise ValueError("payload must hold at least one 6-bit character plus the mode bit")
    compact = all(c in _INDEX and c != "\0" for c in text)
    if compact:
        stored = text[: (num_bits - 1) // 6]
        bits = [0]
        for c in stored:
            bits += _to_bits(_INDEX[c], 6)
    else:
        data = text.encode("utf-8")[: (num_bits - 1) // 8]
        stored = data.decode("utf-8", errors="ignore")  # never split a multi-byte char
        data = stored.encode("utf-8")
        bits = [1]
        for byte in data:
            bits += _to_bits(byte, 8)
    bits += [0] * (num_bits - len(bits))
    return bits, stored


def decode_text(bits: list[int]) -> str:
    """Inverse of :func:`encode_text`."""
    if not bits:
        return ""
    if bits[0] == 0:
        chars = []
        for i in range(1, len(bits) - 5, 6):
            idx = _from_bits(bits[i : i + 6])
            if idx == 0:
                break
            chars.append(ALPHABET[idx])
        return "".join(chars)
    data = bytearray()
    for i in range(1, len(bits) - 7, 8):
        byte = _from_bits(bits[i : i + 8])
        if byte == 0:
            break
        data.append(byte)
    return data.decode("utf-8", errors="replace")
