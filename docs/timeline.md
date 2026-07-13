# PicoTrust Evolution Timeline

## v1-v4: Foundation (100 bits, 512→256)
- **v1**: StegaStamp baseline. 256×256, unbounded residual. 26.5 dB, colour shifts.
- **v2**: Grayscale residual (no colour shifts), softsign bound, strength annealing 1.0→0.03. 32.8 dB.
- **v3**: Pushed to s=0.010 → accuracy collapsed (67.6%). Lesson: bounded residuals from step 0 kill bootstrap.
- **v4**: s=0.020 sweet spot. 35.6 dB, 97.8%. Solid baseline.

## v5-v6: De-annealing discovery (80 bits)
- **v5**: 80 bits, s=0.012. High PSNR (39 dB) but low accuracy (85.5%).
- **v6a→v6c**: Invented de-annealing — resume at higher strength. v5(0.012)→v6a(0.013)→v6b(0.014)→v6c(0.015). Maps PSNR-accuracy curve without retraining.

## v7-v8: Decoder resolution experiments
- **v7**: 512→512 decoder, 128 bits. Higher PSNR but visible HF artifacts. Key lesson: downsampling is a feature (forces LF patterns).
- **v8**: 512→416 decoder + LPIPS 1.5 + GAN 1.5. First content-adaptive encoding — residuals concentrate in textured regions. But 96 bits exceeded capacity.

## v9-v10: Sweet spot search (32 bits)
- **v9**: 32 bits, 416 decoder. 99.2% accuracy. Bootstraps instantly. Content-adaptive. HF artifacts visible.
- **v10**: 256 decoder + Laplacian loss + bilinear upsample + dilated E_post. Smoothest residuals ever — but NOT content-adaptive (256 too coarse).
- Key tension identified: **256 = smooth but uniform. 416 = adaptive but HF artifacts.**

## v11-v12: Decoder-side blur breakthrough (64 bits)
- **v11**: Encoder-side blur σ=1.0 on residual. Content-adaptive but encoded images look soft/blurred. Encoder-side blur during bootstrap kills it (0/15 attempts).
- **v12**: **blur(encoded)** for decoder + LPIPS/GAN. L2 on real output. Clean encoded images + content-adaptive. The blur trick: LPIPS sees blurred large-scale patterns → guides spatial placement. Encoder output stays clean. σ=0.8, 416 decoder.
- Post-annealed to s=0.010: 40.7 dB, 94.0%.

## v13: Learned mask experiment (failed)
- Learned spatial mask (4.6k params) on U-Net features. No blurred LPIPS.
- 98.5% accuracy but diagonal curve artifacts. Mask controls WHERE but not WHAT SHAPE. Without blurred LPIPS, encoder creates structured HF patterns.

## v14: 512 decoder + blur(encoded) σ=1.0 (64 bits)
- 512→512 decoder (full resolution) + blur(encoded) σ=1.0. Best content-adaptivity: TRC=0.591.
- Key finding: `blur(encoded)` works but `original + blur(residual)` doesn't — blur must smooth image+residual together for LPIPS to guide patterns effectively.
- Post-annealed to s=0.015: 37.3 dB, 97.3%.

## v15: Lower sigma for more capacity (72 bits, LDPC)
- σ=0.5 instead of 1.0. More capacity (decoder uses higher frequencies) at the cost of weaker LPIPS guidance.
- 72 bits for LDPC(72,38) = 38 payload bits.
- Post-annealed to s=0.010: 40.7 dB, 93.4%.
- **Adaptive encoding scheme**: per-image strength + texture mask + LDPC. 49/50 images at 100% message recovery, 68% at 43+ dB PSNR.

## v16: FiLM at E_post only (failed)
- Strength conditioning via FiLM (2.2k params) modulating features before E_post.
- Residual patterns identical across strengths — just scaled. FiLM at E_post is too late, spatial strategy already committed by U-Net upstream.

## v17: Full U-Net FiLM (production model)
- FiLM at ALL 9 U-Net conv layers (198k params). Random exponential annealing: two curves define widening strength range [1.0,1.0]→[0.010,0.025].
- **Auto-adaptive**: encoder automatically suppresses smooth regions at higher strength. Like a learned texture mask built in.
- Beats v15 at every strength: 95.2%/41.0 dB at s=0.010, 99.5%/34.8 dB at s=0.025.
- One checkpoint replaces multiple post-annealed models.
- σ=0.5 for both decoder and LPIPS/GAN.

## v18: Split blur experiment (in progress)
- Decoder σ=1.0 + LPIPS/GAN σ=1.5. Full U-Net FiLM.
- Hypothesis: higher LPIPS blur → stronger adaptivity, separate decoder blur → independent capacity control.
- Training from scratch, 200k steps.

---

## Key Ideas in Order of Discovery

1. **Strength annealing** (v2) — start unbounded, squeeze to target
2. **Grayscale residual** (v2) — architectural guarantee of no colour shifts
3. **De-annealing** (v6) — relax strength from converged model
4. **Downsampling as quality control** (v7) — 256 decoder forces LF patterns
5. **LPIPS 1.5 + GAN 1.5 for content-adaptivity** (v8) — perceptual losses guide spatial placement
6. **Decoder-side blur** (v12) — blur(encoded) for training, clean output at inference
7. **Blurred LPIPS/GAN** (v12) — LPIPS sees large-scale patterns → content-adaptive gradients
8. **Exponential annealing** (v11) — log-linear interpolation for multiplicative parameters
9. **Texture mask post-processing** (v15) — attenuate residual in smooth regions at inference
10. **Per-image adaptive strength** (v15) — encode at low strength, escalate if LDPC fails
11. **Full U-Net FiLM conditioning** (v17) — encoder adapts spatial strategy based on strength
12. **Random exponential annealing** (v17) — two curves define widening strength range
13. **Split blur** (v18) — separate σ for decoder vs LPIPS/GAN

## Parameter Evolution

| Parameter | v1-v8 | v9-v10 | v11-v12 | v14 | v15 | v17 | v18 |
|-----------|-------|--------|---------|-----|-----|-----|-----|
| Bits | 80-128 | 32 | 64 | 64 | 72 | 72 | 72 |
| Decoder | 256 | 256-416 | 416 | 512 | 512 | 512 | 512 |
| Decoder σ | — | — | 0.8 | 1.0 | 0.5 | 0.5 | 1.0 |
| LPIPS σ | — | — | 0.8 | 1.0 | 0.5 | 0.5 | 1.5 |
| Strength target | 0.010-0.030 | 0.010-0.014 | 0.014 | 0.025 | 0.020 | 0.010-0.025 | 0.010-0.025 |
| Annealing | linear | linear | exponential | exponential | exponential | random exp | random exp |
| LPIPS scale | 1.0 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| GAN scale | 1.0 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| FiLM | — | — | — | — | — | 198k params | 198k params |
| ECC | BCH | LDPC | LDPC | LDPC | LDPC(72,38) | LDPC(72,38) | LDPC(72,38) |
