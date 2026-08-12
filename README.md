<p align="center">
  <img src="docs/assets/logo.svg" width="96" height="96" alt="Picode logo">
</p>

<h1 align="center">Picode</h1>

<p align="center">
  <b>Neural image steganography that hides a short ID inside a photo — invisible to people, readable by a phone camera.</b>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg">
  <img alt="iOS 16+" src="https://img.shields.io/badge/iOS-16%2B-lightgrey.svg">
  <a href="https://huggingface.co/vadishev/picotrust"><img alt="Weights on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-picotrust-yellow.svg"></a>
</p>

![Original, encoded and residual images for three sample photos](docs/assets/hero.png)

<sub>72-bit payload encoded with the production model (b72s20m85) using Sobel adaptive encoding.
Each image is encoded at the lowest strength on the ladder at which the payload decodes exactly;
overlaid numbers are raw bit accuracy before error correction. Residual amplified ×10.
Source photos: public-domain / CC0 samples from scikit-image. Rendered by
<code>picode-model/scripts/readme_figures.py</code>.</sub>

## Why

QR codes work, but they are ugly and take up space. Picode puts a short, machine-readable ID
*inside* the photo itself. You can print or display the image, point a camera at it, and look the
ID up to get a link or metadata.

The hard part is making the signal both **invisible** and **robust**. It has to survive JPEG,
blur, noise, perspective and print-to-photo capture. Picode starts from
[StegaStamp](https://github.com/tancik/StegaStamp) (Tancik et al., CVPR 2020) and evolves it
into **PicoTrust**, an encoder that puts its signal into textured regions and leaves smooth
areas (sky, skin, walls) almost untouched.

## Features

- **Content-adaptive encoder**: a U-Net with a grayscale, softsign-bounded residual. A Sobel
  texture mask used during training teaches it to hide the signal in texture (TRC ≈ 0.58).
- **Adaptive inference**: the encoder runs once at full strength, and the residual is then
  scaled by the Sobel mask down to the weakest strength that still decodes. Choosing the best of
  many candidate IDs in one batch avoids "hard" payloads. The result averages about 41 dB PSNR.
- **Error correction**: soft-decision LDPC (belief propagation over decoder logits) and BCH.
- **Distortion library**: 19 differentiable distortions (JPEG, blur, noise, perspective,
  print-to-photo, and more), written in pure PyTorch. Most also have a Kornia twin. They are
  used for curriculum training and robustness sweeps.
- **Detection**: a fast binary detector plus a multi-scale detector with perspective
  rectification. Both export to Core ML and TFLite.
- **iOS scanner**: a SwiftUI camera app that runs the detector and decoder on-device with
  Core ML.
- **Dataset scraper**: a distributed, Postgres-coordinated scraper for real
  original/re-photographed image pairs.
- **Experiment log**: more than 20 model generations, with the results and lessons written up
  in [docs](#documentation).

## Results

The production model is **PicoTrust b72s20m85**: 72 channel bits with LDPC(72,38), giving a
38-bit payload. It uses a 512×512 encoder and decoder and a Sobel mask floor of 0.85.

| Mode | PSNR | Bit accuracy | JPEG Q10 | LDPC success |
|------|------|--------------|----------|--------------|
| Fixed strength s=0.020 | 35.90 dB | 98.5% | 98.6% | — |
| Sobel adaptive (composite distortions, 5 trials/image) | 41.0 dB avg | — | — | 93.6% |

Earlier generations are compared in [docs/picotrust.md](docs/picotrust.md#results) and
[docs/sobel_adaptive.md](docs/sobel_adaptive.md#results).

## Quick start

### Encode and decode (local)

```bash
git clone https://github.com/vs/picode.git && cd picode
python -m venv venv && source venv/bin/activate
pip install -e "./picode-model[dev,lpips]"

# Download the production checkpoint (public, ~215 MB)
pip install huggingface_hub
hf download vadishev/picotrust b72s20m85_mirflickr/picotrust_b72s20m85_mirflickr_best.pt \
    --local-dir picode-model/checkpoints

cd picode-model
CKPT=checkpoints/b72s20m85_mirflickr/picotrust_b72s20m85_mirflickr_best.pt
picode -c $CKPT encode photo.jpg encoded.png -m "hello" --save-residual residual.png
picode -c $CKPT decode encoded.png
```

> The `picode` CLI encodes raw bits at the checkpoint's training strength, with no error
> correction, so an occasional flipped bit can corrupt one character of a text message. The
> production path (Sobel mask, strength ladder, batch ID selection and LDPC) is described in
> [docs/sobel_adaptive.md](docs/sobel_adaptive.md). `scripts/readme_figures.py` is a compact,
> runnable reference implementation.

### Dataset scraper (Docker)

```bash
cd picode-scraper
docker compose --profile status run --rm status              # starts Postgres, creates tables, prints stats
docker compose --profile discover run --rm discover           # lists source plugins
docker compose --profile harvest up -d --scale worker=3       # runs 3 harvest workers
```

### iOS app

```bash
cd picode-ios
xcodegen generate
open picode-ios.xcodeproj
```

The app needs Core ML models in `picode-ios/picode-ios/Models/`. Export them with
`detect-export` and `picode-model/scripts/export_decoder.py`
(see [picode-ios/README.md](picode-ios/README.md)).

## Configuration

Model training is configured with YAML files in `picode-model/configs/`. Any key can be
overridden on the command line (`picode-train configs/….yaml training.lr=0.0002`). The scraper reads YAML
too, and every key can be overridden with an environment variable.

| Variable | Used by | Default | Description |
|----------|---------|---------|-------------|
| `PICODE_CONFIG` | scraper CLI | *(unset)*; `/app/configs/default.yaml` in the Docker image | Config file path (same as `-c/--config`) |
| `PICODE_DATABASE__URL` | scraper | `postgresql://picode:picode@localhost:5432/picode_scraper` (from `configs/default.yaml`) | PostgreSQL or SQLite URL. docker-compose sets it to host `db` |
| `PICODE_DATABASE__POOL_SIZE` | scraper | `5` | SQLAlchemy pool size |
| `PICODE_STORAGE__BACKEND` | scraper | `local` | `local` or `s3` |
| `PICODE_STORAGE__LOCAL_PATH` | scraper | `/data/images` | Image directory for the `local` backend |
| `PICODE_STORAGE__S3_BUCKET` | scraper | *(none)* | Bucket for the `s3` backend |
| `PICODE_STORAGE__S3_PREFIX` | scraper | `images` | Key prefix for the `s3` backend |
| `PICODE_STORAGE__S3_ENDPOINT_URL` | scraper | *(none)* | Custom S3 endpoint (MinIO, R2, …) |
| `PICODE_SCRAPING__USER_AGENT` | scraper | `PicodeDatasetCollector/1.0 (research; …)` | HTTP User-Agent |
| `PICODE_SCRAPING__REQUEST_DELAY` | scraper | `1.0` | Seconds between requests |
| `PICODE_SCRAPING__MAX_RETRIES` | scraper | `3` | Retries before a task is dead-lettered |
| `PICODE_SCRAPING__TIMEOUT` | scraper | `30` | HTTP timeout, seconds |
| `PICODE_SCRAPING__PROXY_URL` | scraper | *(none)* | Optional HTTP proxy |
| `PICODE_VALIDATION__MIN_IMAGE_SIZE` | scraper | `256` | Minimum image side, px |
| `PICODE_VALIDATION__MIN_SIMILARITY` | scraper | `0.6` | Pair-matching threshold |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | scraper (`s3` backend) | *(none)* | Standard boto3 credentials |
| `LIGHTNING_USERNAME`, `LIGHTNING_API_KEY` | `scripts/lightning_runner.py` | *(none)* | Lightning.ai credentials |
| `MODEL_NAME` | `scripts/kaggle/kaggle_train.py` | `picodeframe` | Which config the Kaggle kernel trains |

[`.env.example`](.env.example) lists every variable with placeholder values. Cloud training
on Modal, GCE and Kaggle authenticates through each provider's own CLI login, not through
environment variables.

## Architecture

```
Image + ID ─▶ LDPC encode ─▶ Encoder (U-Net + E_post) ─▶ residual × Sobel mask × strength ─▶ Encoded image
                                                                                                │
                              print / screen / JPEG / blur / perspective / camera ◀─────────────┘
                                                                                                │
ID ◀─ LDPC soft decode ◀─ Decoder (CNN + STN) ◀─ rectified crop ◀─ Detector ◀────────────────────┘
```

| Directory | What it is |
|-----------|------------|
| [`picode-model/`](picode-model/) | PyTorch package: models (PicoTrust, StegaStamp and experimental variants), distortions, ECC, detection, training, CLIs |
| [`picode-ios/`](picode-ios/) | SwiftUI + Core ML scanner app (XcodeGen project) |
| [`picode-scraper/`](picode-scraper/) | Distributed scraper for original/capture image pairs (Postgres, S3, Docker) |
| [`docs/`](docs/) | Architecture notes, experiment results and training lessons |

## Development

```bash
source venv/bin/activate

# Model package
cd picode-model
pytest picode/tests/ -q          # ~880 tests, CPU-friendly
ruff check picode/

# Scraper
cd ../picode-scraper
pip install -e ".[dev]"
pytest tests/ -q
ruff check picode_scraper/ && mypy picode_scraper/

# iOS
cd ../picode-ios
xcodegen generate
xcodebuild test -scheme picode-ios -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

## Documentation

- [PicoTrust architecture and results](docs/picotrust.md): every model generation, its results,
  and training lessons 1–34
- [Sobel adaptive encoding](docs/sobel_adaptive.md): the production inference strategy
- [Evolution timeline](docs/timeline.md): v1 to v18, one paragraph per step (later models are in
  [sobel_adaptive.md](docs/sobel_adaptive.md))
- [Training best practices](docs/training_best_practices.md): a distilled training recipe
- [Training wisdom](docs/wisdom.md): hard-won lessons, including dead ends
- [PicoTier](docs/picotier.md): a multi-capacity model experiment
- [Scraper guide](picode-scraper/README.md) and [iOS app](picode-ios/README.md)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, conventions and the leak guard,
and [SECURITY.md](SECURITY.md) to report a vulnerability.

## Disclaimer

Picode is a research project. It is **not** a security, authentication or copyright-protection
mechanism. The payload is not encrypted or signed, and a determined party can detect, remove or
forge it. Use it only on images you have the right to modify. Robustness figures come from the
evaluation sets described in the docs, and results on your own images and cameras will vary.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attributions, including the MIT-licensed
StegaStamp work that the baseline models re-implement.
