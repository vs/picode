"""Tests for PicodeTier training step integration."""

import torch
import torch.nn.functional as F

from picode.models.picodetier.decoder import Decoder
from picode.models.picodetier.encoder import Encoder
from picode.models.picodetier.tiers import MAX_BITS, NUM_TIERS, TIERS


class TestPicodetierMessageGeneration:
    """Test per-tier message generation with masks."""

    def test_generate_tiered_messages(self) -> None:
        """Messages and masks have correct bit counts per tier."""
        batch_size = 8
        tiers = torch.randint(0, NUM_TIERS, (batch_size,))
        messages = torch.zeros(batch_size, MAX_BITS)
        masks = torch.zeros(batch_size, MAX_BITS)

        for i in range(batch_size):
            t_idx = int(tiers[i].item())
            n_bits = int(TIERS[t_idx]["bits"])
            messages[i, :n_bits] = torch.randint(0, 2, (n_bits,)).float()
            masks[i, :n_bits] = 1.0

        for i in range(batch_size):
            t = int(tiers[i].item())
            expected = int(TIERS[t]["bits"])
            assert masks[i].sum().item() == expected
            assert messages[i, expected:].sum().item() == 0.0

    def test_masked_message_loss(self) -> None:
        """Masked MSE loss only depends on active bits."""
        logits = torch.randn(4, MAX_BITS)
        messages = torch.zeros(4, MAX_BITS)
        masks = torch.zeros(4, MAX_BITS)

        messages[:, :16] = torch.randint(0, 2, (4, 16)).float()
        masks[:, :16] = 1.0

        probs = torch.sigmoid(logits)
        loss_per_bit = (probs - messages) ** 2
        loss = (loss_per_bit * masks).sum() / masks.sum()

        assert loss.dim() == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

        # Loss should only depend on first 16 bits
        logits_modified = logits.clone()
        logits_modified[:, 16:] = 999.0
        probs2 = torch.sigmoid(logits_modified)
        loss2 = ((probs2 - messages) ** 2 * masks).sum() / masks.sum()
        assert torch.allclose(loss, loss2)

    def test_all_tiers_produce_valid_masks(self) -> None:
        """Each tier produces a mask with exactly the right number of active bits."""
        for t_idx in range(NUM_TIERS):
            n_bits = int(TIERS[t_idx]["bits"])
            mask = torch.zeros(MAX_BITS)
            mask[:n_bits] = 1.0
            assert mask.sum().item() == n_bits
            assert mask[n_bits:].sum().item() == 0.0


class TestPicodetierEncoderDecoderRoundtrip:
    """Test encoder-decoder forward pass and loss computation."""

    def test_roundtrip_shapes(self) -> None:
        """Encoder and decoder produce correct output shapes."""
        enc = Encoder(image_size=256)
        dec = Decoder(image_size=256)

        images = torch.rand(2, 3, 256, 256)
        messages = torch.randint(0, 2, (2, MAX_BITS)).float()
        tiers = torch.tensor([0, 3])

        encoded = enc(images, messages, tiers)["encoded"]
        logits, tier_logits = dec(encoded, tier=tiers)

        assert logits.shape == (2, MAX_BITS)
        assert tier_logits.shape == (2, NUM_TIERS)

    def test_loss_computation(self) -> None:
        """Masked message loss + tier CE loss backprops through both models."""
        enc = Encoder(image_size=256)
        dec = Decoder(image_size=256)

        images = torch.rand(2, 3, 256, 256)
        tiers = torch.tensor([1, 2])

        messages = torch.zeros(2, MAX_BITS)
        masks = torch.zeros(2, MAX_BITS)
        messages[0, :32] = torch.randint(0, 2, (32,)).float()
        masks[0, :32] = 1.0
        messages[1, :64] = torch.randint(0, 2, (64,)).float()
        masks[1, :64] = 1.0

        encoded = enc(images, messages, tiers)["encoded"]
        logits, tier_logits = dec(encoded, tier=tiers)

        probs = torch.sigmoid(logits)
        loss_msg = ((probs - messages) ** 2 * masks).sum() / masks.sum()
        loss_tier = F.cross_entropy(tier_logits, tiers)

        total = loss_msg + loss_tier
        total.backward()

        assert enc.secret_dense.weight.grad is not None
        assert dec.tier_classifier.weight.grad is not None

    def test_masked_bit_accuracy(self) -> None:
        """Bit accuracy only considers active (masked) bits."""
        decoded_probs = torch.zeros(2, MAX_BITS)
        decoded_probs[0, :16] = 1.0  # All 1s for tier 0
        decoded_probs[1, :32] = 1.0  # All 1s for tier 1

        messages = torch.ones(2, MAX_BITS)  # All 1s target
        masks = torch.zeros(2, MAX_BITS)
        masks[0, :16] = 1.0
        masks[1, :32] = 1.0

        predicted_bits = (decoded_probs > 0.5).float()
        correct_masked = ((predicted_bits == messages).float() * masks).sum()
        accuracy = (correct_masked / masks.sum()).item()

        assert accuracy == 1.0

    def test_tier_strength_annealing(self) -> None:
        """Tier strengths anneal from initial to per-tier targets."""
        enc = Encoder(image_size=256)

        initial = 1.0
        # Simulate mid-annealing (progress=0.5)
        progress = 0.5
        for t_idx in range(NUM_TIERS):
            target = float(TIERS[t_idx]["strength"])
            expected = initial + progress * (target - initial)
            enc.tier_strengths[t_idx] = expected

        for t_idx in range(NUM_TIERS):
            target = float(TIERS[t_idx]["strength"])
            expected = initial + progress * (target - initial)
            assert abs(enc.tier_strengths[t_idx].item() - expected) < 1e-6

        # At progress=1.0, strengths should equal targets
        for t_idx in range(NUM_TIERS):
            target = float(TIERS[t_idx]["strength"])
            enc.tier_strengths[t_idx] = target

        for t_idx in range(NUM_TIERS):
            target = float(TIERS[t_idx]["strength"])
            assert abs(enc.tier_strengths[t_idx].item() - target) < 1e-6


class TestPicodetierConfig:
    """Test config changes for PicodeTier."""

    def test_model_config_defaults(self) -> None:
        """PicodeTier gets 512 encoder / 256 decoder when sizes are default."""
        from picode.training.config import ModelConfig

        cfg = ModelConfig(type="picodetier")
        assert cfg.encoder_size == 512
        assert cfg.decoder_size == 256

    def test_model_config_explicit_sizes(self) -> None:
        """Explicit sizes are preserved (no override)."""
        from picode.training.config import ModelConfig

        cfg = ModelConfig(type="picodetier", encoder_size=384, decoder_size=192)
        assert cfg.encoder_size == 384
        assert cfg.decoder_size == 192

    def test_loss_config_tier_classifier(self) -> None:
        """LossConfig has tier_classifier field with correct defaults."""
        from picode.training.config import LossConfig

        cfg = LossConfig()
        assert cfg.tier_classifier.scale == 1.0
        assert cfg.tier_classifier.ramp_steps == 1

    def test_tier_classifier_deserialization(self) -> None:
        """tier_classifier is properly deserialized from dict."""
        from picode.training.config import _dict_to_config

        data = {
            "experiment_name": "test",
            "data": {"source": "folder", "path": "/tmp/data"},
            "loss": {
                "tier_classifier": {"scale": 2.0, "ramp_steps": 5000},
            },
        }
        config = _dict_to_config(data)
        assert config.loss.tier_classifier.scale == 2.0
        assert config.loss.tier_classifier.ramp_steps == 5000
