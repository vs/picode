"""Tests for text payload packing and the CLI's LDPC wiring."""

import pytest
import torch

from picode.cli import LDPC_SEED, ldpc_decode, resolve_ldpc
from picode.ecc.ldpc import LDPC
from picode.ecc.payload import capacity, decode_text, encode_text


class TestPayload:
    @pytest.mark.parametrize("text", ["hello", "pc-Q7x", "A1", ""])
    def test_compact_round_trip(self, text: str) -> None:
        bits, stored = encode_text(text, 38)
        assert len(bits) == 38
        assert bits[0] == 0
        assert stored == text
        assert decode_text(bits) == text

    def test_utf8_round_trip(self) -> None:
        bits, stored = encode_text("hi!", 38)
        assert bits[0] == 1
        assert decode_text(bits) == stored == "hi!"

    def test_truncation(self) -> None:
        _, stored = encode_text("abcdefghij", 38)
        assert stored == "abcdef"  # 6 x 6 bits + mode bit fit in 38
        _, stored = encode_text("Привет", 38)
        assert stored == "Пр"  # 4 bytes; never splits a multi-byte character

    def test_capacity(self) -> None:
        assert capacity(38, "abc") == 6
        assert capacity(38, "a b") == 4

    def test_too_small(self) -> None:
        with pytest.raises(ValueError):
            encode_text("a", 6)


class TestCliLdpc:
    def test_auto_uses_production_code(self) -> None:
        ldpc = resolve_ldpc("auto", "picotrust", 72)
        assert ldpc is not None
        assert (ldpc.codeword_length, ldpc.message_length) == (72, 38)

    def test_none_and_unsupported(self) -> None:
        assert resolve_ldpc("none", "picotrust", 72) is None
        assert resolve_ldpc("auto", "picotier", 96) is None
        assert resolve_ldpc("auto", "picotrust", 31) is None  # no LDPC code for 31 bits
        with pytest.raises(SystemExit):
            resolve_ldpc("ldpc", "picotrust", 31)

    def test_decode_corrects_flipped_bits(self) -> None:
        ldpc = LDPC(n=72, seed=LDPC_SEED)
        payload, _ = encode_text("pic0de", ldpc.message_length)
        codeword = ldpc.encode(torch.tensor([payload], dtype=torch.float32))
        logits = (codeword * 2 - 1) * 4.0
        logits[0, [3, 17, 40]] *= -0.25  # three weakly wrong bits
        bits, ok, snr = ldpc_decode(ldpc, logits)
        assert ok and snr is not None
        assert decode_text(bits) == "pic0de"
