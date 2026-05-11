# Lightning.ai Training Guide

Train Picode models on Lightning.ai using **80 free GPU hours/month**. Lightning.ai offers persistent storage, access to powerful GPUs (T4/L4/A100/H100), and a Python SDK for remote automation.

## Why Lightning.ai?

| Feature | Lightning.ai (Free) | Google Colab (Free) |
|---------|---------------------|---------------------|
| GPU Hours | 80/month | ~12/day (unreliable) |
| GPUs Available | T4, L4, A100, H100 | T4 only |
| Storage | 50GB persistent | Google Drive |
| Session Limit | 4 hours (restart) | 12 hours (disconnect) |
| API/SDK | Yes ✓ | No |
| Background Jobs | Yes ✓ | No |

## Quick Start

### 1. Create Lightning.ai Account

1. Sign up at [lightning.ai/sign-up](https://lightning.ai/sign-up)
2. Wait for account activation (~1 hour)
3. Get your API key: [lightning.ai/\<username\>/home?settings=keys](https://lightning.ai)

### 2. Set Environment Variables

```bash
export LIGHTNING_API_KEY="your-api-key"
export LIGHTNING_USERNAME="your-username"
```

Add to `~/.bashrc` or `~/.zshrc` for persistence.

### 3. Install Lightning SDK

```bash
pip install lightning-sdk
```

### 4. Start Training

```bash
cd picode-model

# Start training (creates studio automatically)
python scripts/lightning_runner.py train

# Resume after restart
python scripts/lightning_runner.py train --resume

# Use a more powerful GPU
python scripts/lightning_runner.py train --gpu a100
```

## Remote Runner Commands

The `lightning_runner.py` script provides full remote control:

```bash
# Start/resume training
python scripts/lightning_runner.py train
python scripts/lightning_runner.py train --resume
python scripts/lightning_runner.py train --gpu l4 --batch-size 8

# Check status
python scripts/lightning_runner.py status

# View recent logs
python scripts/lightning_runner.py logs

# Download checkpoint
python scripts/lightning_runner.py download
python scripts/lightning_runner.py download -o ./my-checkpoints

# Stop studio (preserves data)
python scripts/lightning_runner.py stop

# Download COCO training data (one-time, ~20 min)
python scripts/lightning_runner.py setup-data

# Delete studio (WARNING: deletes all data!)
python scripts/lightning_runner.py delete
```

## Manual Studio Usage

You can also use Lightning.ai through the web UI:

### 1. Create a Studio

1. Go to [lightning.ai/studios](https://lightning.ai/studios)
2. Click "New Studio"
3. Name it `picode-training`
4. Select a GPU machine (T4 for free tier)

### 2. Setup Environment

In the studio terminal:

```bash
# Clone repo
git clone https://github.com/vs/code.git ~/code

# Install package
cd ~/code/picode-model
pip install -e '.[lpips,kornia]'
```

### 3. Get Training Data

Option A - Download COCO directly (recommended):
```bash
mkdir -p ~/data
cd ~/data
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip
```

Option B - Upload your own images to `~/data/train2017/`

### 4. Train

```bash
cd ~/code/picode-model

# Fresh start
python scripts/lightning_train.py --data-path ~/data/train2017

# Resume
python scripts/lightning_train.py --resume
```

## Directory Structure

On Lightning.ai, data persists in `/teamspace/studios/this_studio/`:

```
/teamspace/studios/this_studio/
├── code/                    # Cloned repository
│   └── picode-model/
├── data/
│   └── train2017/           # Training images
└── checkpoints/
    ├── picode_lightning/    # Experiment checkpoints
    │   ├── checkpoint_00005000.pt
    │   ├── checkpoint_00010000.pt
    │   └── best.pt
    └── runs/                # TensorBoard logs
```

## GPU Selection

| GPU | VRAM | Batch Size | Speed | Free Tier |
|-----|------|------------|-------|-----------|
| T4 | 16GB | 4 | 1x | ✓ |
| L4 | 24GB | 8 | 1.5x | ✓ |
| A10G | 24GB | 8 | 2x | Limited |
| A100 | 40GB | 16 | 4x | Limited |
| H100 | 80GB | 32 | 8x | Pro only |

Free tier has access to T4 and L4. A100/H100 may require Pro subscription.

```bash
# Use L4 for faster training (if available)
python scripts/lightning_runner.py train --gpu l4 --batch-size 8
```

## Monitoring

### TensorBoard (in Studio)

```bash
tensorboard --logdir ~/checkpoints/runs --host 0.0.0.0
```

Access via the studio's "Open Port" feature.

### Console Output

Training logs to console every 50 steps:
- `bit_accuracy` — target: > 0.95
- `loss_msg` — should decrease
- `psnr` — target: > 33 dB

## Troubleshooting

### "No GPU found"

Switch to GPU machine in studio UI or use:
```bash
python scripts/lightning_runner.py train --gpu t4
```

### Session Expired

Free tier restarts every 4 hours. Just resume:
```bash
python scripts/lightning_runner.py train --resume
```

### "ModuleNotFoundError"

Reinstall package after restart:
```bash
cd ~/code/picode-model && pip install -e '.[lpips,kornia]'
```

### Slow Training

1. Check GPU is active: `nvidia-smi`
2. Increase batch size if VRAM allows
3. Upgrade to L4 or A100

## Cost Comparison

| Platform | Free Tier | Paid |
|----------|-----------|------|
| Lightning.ai | 80 GPU-hrs/mo | $0.60/hr (T4) |
| Google Colab | ~12 hrs/day | $12/mo (Pro) |
| Modal | - | $0.59/hr (T4) |
| AWS | - | $0.53/hr (g4dn) |

Lightning.ai free tier is the best value for training runs under 80 hours/month.

## References

- [Lightning.ai Docs](https://lightning.ai/docs)
- [Lightning SDK](https://pypi.org/project/lightning-sdk/)
- [Training Config](../picode-model/configs/lightning_training.yaml)
- [Runner Script](../picode-model/scripts/lightning_runner.py)
- [Colab Guide](./colab_training.md) — alternative using Google Colab
