"""PicoTrust: StegaStamp U-Net encoder + TrustMark enhancements.

Combines the proven StegaStamp U-Net architecture with three targeted
improvements from TrustMark (Adobe, ICCV 2025):
- E_post (post-processing network) on the encoder
- Focal Frequency Loss (FFL) for frequency-domain quality
- WGAN discriminator for perceptual realism

Trained at 256x256 with residual upscaling for arbitrary resolution inference.
"""

from picode.models.picotrust.decoder import Decoder
from picode.models.picotrust.encoder import Encoder

__all__ = ["Encoder", "Decoder"]
