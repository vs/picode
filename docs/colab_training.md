# Google Colab Training Guide

Train steganography models on Google Colab using free or Pro GPUs, with Google Drive for persistent checkpoints. This is a cost-free alternative to [Modal cloud training](../picode-model/configs/modal_training.yaml).

## Prerequisites

1. **Google account** with Google Drive space (~20GB for COCO + checkpoints)
2. **Colab runtime set to GPU**: Runtime > Change runtime type > T4 GPU
3. **Training images** — COCO train2017 recommended (~118K images, 18GB)

### Colab Tiers

| Tier | GPU | Session Limit | Cost |
|------|-----|---------------|------|
| Free | T4 (16GB) | ~12 hours, may disconnect | Free |
| Pro | T4/A100 | ~24 hours, priority access | $12/mo |
| Pro+ | A100 (40GB) | ~24 hours, background execution | $50/mo |

Free tier is sufficient for full training (140K steps takes ~10-20 hours depending on disconnects).

## Quick Start

Open the notebook in Colab:

```
picode-model/scripts/colab_train.ipynb
```

Upload to Colab via: File > Upload notebook, or push to GitHub and open with the "Open in Colab" button.

The notebook has 8 cells that walk through the full workflow:

1. **Setup** — Mount Drive, clone repo, install package
2. **GPU Check** — Verify GPU is available
3. **Data Setup** — Create directories, check for images
4. **Train** — Start training from scratch
5. **Resume** — Resume from latest checkpoint after disconnect
6. **TensorBoard** — Monitor metrics inline
7. **Evaluate** — Run evaluation on best checkpoint
8. **Download** — Download checkpoint to local machine

## Getting Training Data onto Drive

### Option A: Download COCO Directly in Colab (Recommended)

The notebook includes a cell to download COCO train2017. Uncomment and run:

```python
!wget -q http://images.cocodataset.org/zips/train2017.zip -O /content/train2017.zip
!unzip -q /content/train2017.zip -d /content/drive/MyDrive/picode/data/
!rm /content/train2017.zip
```

This takes ~20 minutes but only needs to be done once — data persists on Drive.

### Option B: Upload via Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Create folder: `My Drive/picode/data/train2017/`
3. Upload your training images (any `.jpg` or `.png` files)

### Option C: Existing Data

If you already have images on Drive from a previous session, just verify the path matches:

```
/content/drive/MyDrive/picode/data/train2017/
```

## Training

### Fresh Start

```python
from picode.training.trainer import Trainer

trainer = Trainer.from_config("configs/colab_training.yaml")
trainer.fit()
```

This uses `picode-model/configs/colab_training.yaml`, which is identical to the Modal config except:
- Data path points to Google Drive
- `num_workers: 2` (Colab has fewer CPUs)
- Checkpoint/log dirs point to Google Drive

### Resuming After Disconnect

Colab sessions disconnect after inactivity or hitting time limits. Since checkpoints save to Google Drive every 5000 steps, you can resume:

```python
import glob
from picode.training.trainer import Trainer

checkpoints = sorted(glob.glob("/content/drive/MyDrive/picode/checkpoints/*/checkpoint_*.pt"))
if checkpoints:
    trainer = Trainer.from_checkpoint(checkpoints[-1])
    trainer.fit()
```

The trainer continues from the saved step — no progress is lost.

### Config Overrides

To adjust hyperparameters without editing the YAML:

```python
trainer = Trainer.from_config("configs/colab_training.yaml", overrides={
    "training.lr": 0.0002,
    "data.batch_size": 8,  # If using A100 with more VRAM
})
trainer.fit()
```

## Monitoring

### TensorBoard

Run inline in the notebook:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/picode/checkpoints/runs
```

### Console Output

The trainer logs to console every 50 steps (configurable via `logging.log_every_steps`). Key metrics to watch:

| Metric | Description | Good Sign |
|--------|-------------|-----------|
| `bit_accuracy` | Fraction of bits decoded correctly | Rises toward 1.0 |
| `loss_msg` | Message BCE loss | Decreases steadily |
| `loss_l2` | Image reconstruction loss | Decreases after warmup |
| `loss_lpips` | Perceptual loss | Decreases after warmup |
| `residual_abs_max` | Max encoder perturbation | Stays below ~0.15 |

## Evaluation

After training completes (or on a saved checkpoint):

```python
from picode.training.trainer import Trainer

trainer = Trainer.from_checkpoint("/content/drive/MyDrive/picode/checkpoints/picode_coco/best.pt")
metrics = trainer.evaluate()

print(f"Bit accuracy:     {metrics.bit_accuracy:.4f}")
print(f"Message accuracy: {metrics.message_accuracy:.4f}")
print(f"PSNR:             {metrics.psnr:.2f} dB")
print(f"SSIM:             {metrics.ssim:.4f}")
```

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Bit accuracy | > 0.95 | Per-bit correctness |
| Message accuracy | > 0.80 | Fully correct 100-bit messages |
| PSNR | > 33 dB | Image quality (higher = less visible) |
| SSIM | > 0.95 | Structural similarity |

## Downloading Checkpoints

### From Colab to Local Machine

```python
from google.colab import files
files.download("/content/drive/MyDrive/picode/checkpoints/picode_coco/best.pt")
```

### From Google Drive Directly

Checkpoints are in `My Drive/picode/checkpoints/`. Download via the Drive web UI or desktop sync.

## Directory Structure on Drive

After setup and training, your Drive will contain:

```
My Drive/picode/
├── data/
│   └── train2017/          # Training images (~18GB)
│       ├── 000000000009.jpg
│       ├── 000000000025.jpg
│       └── ...
└── checkpoints/
    ├── picode_coco/        # Experiment directory
    │   ├── checkpoint_00005000.pt
    │   ├── checkpoint_00010000.pt
    │   ├── ...
    │   └── best.pt
    └── runs/               # TensorBoard logs
        └── picode_coco/
```

## Troubleshooting

### "No GPU found"

Colab sometimes assigns a CPU runtime. Go to Runtime > Change runtime type and select T4 GPU. If GPUs are unavailable (free tier), try again later or upgrade to Pro.

### Session Disconnects

This is normal on free tier. Just rerun cells 1 (Setup), 2 (GPU Check), then 5 (Resume). Checkpoints on Drive are safe.

### "CUDA out of memory"

Reduce batch size in the config or via override:

```python
trainer = Trainer.from_config("configs/colab_training.yaml", overrides={
    "data.batch_size": 2,
})
```

### Slow Drive I/O

Google Drive can be slow for random reads. If training is I/O bound:

1. Copy data to the Colab local disk first:
   ```python
   !cp -r /content/drive/MyDrive/picode/data/train2017 /content/train2017
   ```
2. Override the data path:
   ```python
   trainer = Trainer.from_config("configs/colab_training.yaml", overrides={
       "data.path": "/content/train2017",
   })
   ```

Checkpoints should still save to Drive for persistence.

### "ModuleNotFoundError: No module named 'picode'"

Re-run the setup cell — the package needs to be reinstalled each session:

```python
%cd /content/picode/picode-model
!pip install -q -e '.[lpips,kornia]'
```

## Colab vs Modal Comparison

| | Google Colab (Free) | Google Colab Pro | Modal (T4) |
|---|---|---|---|
| **Cost** | Free | $12/mo | ~$0.59/hr |
| **GPU** | T4 (16GB) | T4 or A100 | T4/A10G/A100/H100 |
| **Session limit** | ~12 hrs | ~24 hrs | 12 hrs |
| **Persistence** | Google Drive | Google Drive | Modal volumes |
| **Setup** | Notebook UI | Notebook UI | CLI + scripts |
| **Disconnect risk** | High (inactivity) | Lower | None (serverless) |

Colab is best for experimentation and budget-conscious training. Modal is best for unattended, reliable runs.

## References

- [Colab notebook](../picode-model/scripts/colab_train.ipynb) — The training notebook
- [Colab config](../picode-model/configs/colab_training.yaml) — Colab-specific training config
- [Modal training config](../picode-model/configs/modal_training.yaml) — Modal equivalent for comparison
- [Model architecture](./model_implementation.md) — StegaStamp encoder/decoder details
