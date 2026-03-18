# picode-scraper

A distributed, containerized CLI tool for collecting paired image datasets (original + real-world capture) to train steganography models. Multiple worker instances coordinate via PostgreSQL to scrape photography forums for comparison posts.

## Features

- **Distributed Workers**: Scale horizontally with multiple workers across VPS instances
- **PostgreSQL Coordination**: Row-level locking (`FOR UPDATE SKIP LOCKED`) for race-free task claiming
- **SIFT Validation**: Automatic pair detection using feature matching and homography estimation
- **Image Deduplication**: Content-based (SHA-256) and perceptual (pHash) deduplication
- **Graceful Shutdown**: SIGTERM handling with task release for clean container stops
- **Structured Logging**: JSON logs for containers, pretty console for development
- **Export Pipeline**: Generate train/val/test splits with stratified sampling

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI / Orchestrator                           │
│  discover → status → harvest (N workers) → export                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │   PostgreSQL    │             │     Storage     │
          │  (coordination) │             │  (local / S3)   │
          └─────────────────┘             └─────────────────┘
                    │
     ┌──────────────┼──────────────┬──────────────┐
     ▼              ▼              ▼              ▼
┌─────────┐  ┌─────────┐    ┌─────────┐    ┌─────────┐
│Worker 1 │  │Worker 2 │    │Worker 3 │    │Worker N │
│ (VPS A) │  │ (VPS A) │    │ (VPS B) │    │ (VPS C) │
└─────────┘  └─────────┘    └─────────┘    └─────────┘
```

## Installation

### Requirements

- Python 3.10+
- PostgreSQL 15+
- OpenCV dependencies (for SIFT)

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/picode-scraper.git
cd picode-scraper

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Docker

```bash
# Build the image
docker build -t picode-scraper:latest .

# Or use pre-built (if published)
docker pull ghcr.io/your-org/picode-scraper:latest
```

## Quick Start

### 1. Start PostgreSQL

```bash
# Using Docker Compose (includes database)
docker compose up -d db

# Wait for healthy
docker compose ps
```

### 2. Initialize Database

```bash
# Using Docker Compose
docker compose up migrate

# Or manually
picode-scraper -c configs/default.yaml status  # Creates tables on first run
```

### 3. Discover Sources

```bash
# List available source plugins
picode-scraper discover --list

# Run discovery with search terms
picode-scraper -c configs/default.yaml discover \
  -t "monitor vs print comparison" \
  -t "screen calibration photo"
```

### 4. Run Workers

```bash
# Single worker
picode-scraper -c configs/default.yaml harvest

# Multiple workers with Docker Compose
docker compose --profile harvest up -d --scale worker=5
```

### 5. Monitor Progress

```bash
# Check status
picode-scraper -c configs/default.yaml status

# Output:
# Tasks:
#   pending: 142
#   claimed: 3
#   completed: 55
#   failed: 2
#
# Pairs collected: 23
# Images stored: 89
# Active workers: 3
```

### 6. Export Dataset

```bash
picode-scraper -c configs/default.yaml export \
  --output ./dataset \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --min-quality 0.6 \
  --seed 42
```

## Configuration

### Configuration File

Create a YAML configuration file (see `configs/default.yaml`):

```yaml
database:
  url: postgresql://picode:picode@localhost:5432/picode_scraper
  pool_size: 5

storage:
  backend: local                    # 'local' or 's3'
  local_path: /data/images          # For local backend
  # S3 settings (for production)
  # s3_bucket: picode-dataset
  # s3_prefix: images
  # s3_endpoint_url: null           # Set for MinIO

scraping:
  user_agent: "PicodeDatasetCollector/1.0 (research)"
  request_delay: 1.0                # Seconds between requests per domain
  max_retries: 3                    # Retries before dead_letter
  timeout: 30
  proxy_url: null                   # Optional: http://proxy:8080
  proxy_rotation: false

validation:
  min_image_size: 256               # Minimum dimension in pixels
  min_corner_confidence: 0.7        # SIFT inlier ratio threshold
  min_coverage: 0.1                 # Minimum area ratio in capture
  min_similarity: 0.6               # Perceptual similarity threshold
  proxy_resolution: 800             # Max dimension for initial SIFT check
```

### Environment Variables

All configuration can be overridden via environment variables using the `PICODE_` prefix with `__` for nesting:

```bash
# Database URL (recommended for secrets)
export PICODE_DATABASE__URL="postgresql://user:pass@host:5432/db"

# Storage backend
export PICODE_STORAGE__BACKEND="s3"
export PICODE_STORAGE__S3_BUCKET="my-bucket"

# Scraping settings
export PICODE_SCRAPING__REQUEST_DELAY="2.0"
export PICODE_SCRAPING__MAX_RETRIES="5"
```

### Configuration Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `database.url` | string | - | PostgreSQL connection URL |
| `database.pool_size` | int | 5 | Connection pool size |
| `storage.backend` | string | "local" | Storage backend: "local" or "s3" |
| `storage.local_path` | path | /data/images | Local storage directory |
| `storage.s3_bucket` | string | null | S3 bucket name |
| `storage.s3_prefix` | string | "images" | S3 key prefix |
| `storage.s3_endpoint_url` | string | null | S3 endpoint (for MinIO) |
| `scraping.user_agent` | string | - | HTTP User-Agent header |
| `scraping.request_delay` | float | 1.0 | Delay between requests (seconds) |
| `scraping.max_retries` | int | 3 | Max retries before dead_letter |
| `scraping.timeout` | int | 30 | HTTP timeout (seconds) |
| `scraping.proxy_url` | string | null | HTTP proxy URL |
| `validation.min_image_size` | int | 256 | Minimum image dimension |
| `validation.min_corner_confidence` | float | 0.7 | SIFT confidence threshold |
| `validation.min_coverage` | float | 0.1 | Minimum coverage ratio |
| `validation.min_similarity` | float | 0.6 | Perceptual similarity threshold |

## CLI Reference

### Global Options

```bash
picode-scraper [OPTIONS] COMMAND [ARGS]

Options:
  -c, --config PATH    Path to config file
  --json-logs          Output JSON formatted logs (for containers)
  --log-level TEXT     Log level: DEBUG, INFO, WARNING, ERROR [default: INFO]
  --version            Show version and exit
  --help               Show help and exit
```

### Commands

#### `discover` - Find sources and populate harvest queue

```bash
picode-scraper discover [OPTIONS]

Options:
  -s, --source TEXT    Source types to search (repeatable)
  -t, --terms TEXT     Search terms (repeatable)
  --list               List available source plugins
```

#### `harvest` - Run a worker to collect image pairs

```bash
picode-scraper harvest [OPTIONS]

Options:
  --worker-id TEXT     Unique worker identifier (auto-generated if omitted)
  -s, --source TEXT    Limit to specific sources (repeatable)
```

#### `status` - Show harvest progress and statistics

```bash
picode-scraper status
```

#### `export` - Export dataset to training format

```bash
picode-scraper export [OPTIONS]

Options:
  -o, --output PATH    Output directory [required]
  --train-ratio FLOAT  Training set ratio [default: 0.8]
  --val-ratio FLOAT    Validation set ratio [default: 0.1]
  --test-ratio FLOAT   Test set ratio [default: 0.1]
  --min-quality FLOAT  Minimum quality score filter [default: 0.0]
  --no-symlinks        Skip creating by_type symlinks
  --seed INTEGER       Random seed for reproducible splits
```

#### `requeue` - Requeue failed tasks for retry

```bash
picode-scraper requeue [OPTIONS]

Options:
  -s, --status TEXT    Status to requeue: failed, dead_letter [default: failed]
  -l, --limit INTEGER  Max tasks to requeue
```

## Deployment

### Single Machine (Docker Compose)

```bash
# Start everything
docker compose up -d db
docker compose up migrate
docker compose --profile harvest up -d --scale worker=3

# Monitor
docker compose logs -f worker
docker compose --profile status up
```

### Multi-VPS Deployment

For production deployments across multiple VPS instances, you need:
1. A centralized PostgreSQL database (accessible from all VPS)
2. Shared storage (S3-compatible or NFS)
3. Workers deployed on each VPS

#### Step 1: Set Up Central Database

On your database server (or use managed PostgreSQL like AWS RDS, DigitalOcean):

```bash
# Create database
createdb picode_scraper

# Create user with password
psql -c "CREATE USER picode WITH PASSWORD 'secure_password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE picode_scraper TO picode;"
```

Ensure the database is accessible from your VPS instances (configure `pg_hba.conf` and firewall rules).

#### Step 2: Set Up S3 Storage

For multi-VPS deployments, use S3-compatible storage:

```yaml
# configs/production.yaml
database:
  url: postgresql://picode:secure_password@db.example.com:5432/picode_scraper
  pool_size: 10

storage:
  backend: s3
  s3_bucket: picode-dataset
  s3_prefix: images
  s3_endpoint_url: null  # Use AWS S3
  # Or for MinIO:
  # s3_endpoint_url: https://minio.example.com:9000
```

#### Step 3: Deploy Workers on Each VPS

**Option A: Docker (Recommended)**

On each VPS:

```bash
# Pull the image
docker pull ghcr.io/your-org/picode-scraper:latest

# Create environment file
cat > /etc/picode/.env << 'EOF'
PICODE_DATABASE__URL=postgresql://picode:secure_password@db.example.com:5432/picode_scraper
PICODE_STORAGE__BACKEND=s3
PICODE_STORAGE__S3_BUCKET=picode-dataset
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
EOF

# Run workers (adjust replica count based on VPS resources)
docker run -d \
  --name picode-worker-1 \
  --env-file /etc/picode/.env \
  --restart unless-stopped \
  picode-scraper:latest \
  harvest -c /app/configs/default.yaml --worker-id "vps1-worker-1"

docker run -d \
  --name picode-worker-2 \
  --env-file /etc/picode/.env \
  --restart unless-stopped \
  picode-scraper:latest \
  harvest -c /app/configs/default.yaml --worker-id "vps1-worker-2"
```

**Option B: Systemd Service**

On each VPS:

```bash
# Install picode-scraper
pip install picode-scraper

# Create config
mkdir -p /etc/picode
cat > /etc/picode/config.yaml << 'EOF'
database:
  url: postgresql://picode:secure_password@db.example.com:5432/picode_scraper
  pool_size: 5
storage:
  backend: s3
  s3_bucket: picode-dataset
scraping:
  request_delay: 1.5
  max_retries: 3
EOF

# Create systemd service
cat > /etc/systemd/system/picode-worker@.service << 'EOF'
[Unit]
Description=Picode Scraper Worker %i
After=network.target

[Service]
Type=simple
User=picode
Environment="AWS_ACCESS_KEY_ID=your_access_key"
Environment="AWS_SECRET_ACCESS_KEY=your_secret_key"
ExecStart=/usr/local/bin/picode-scraper -c /etc/picode/config.yaml harvest --worker-id %H-worker-%i
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start workers
systemctl daemon-reload
systemctl enable picode-worker@{1..4}
systemctl start picode-worker@{1..4}
```

#### Step 4: Run Discovery (Once)

From any machine with database access:

```bash
picode-scraper -c /etc/picode/config.yaml discover \
  -t "monitor vs print calibration" \
  -t "screen paper color comparison"
```

#### Step 5: Monitor from Any Machine

```bash
# Check overall status
picode-scraper -c /etc/picode/config.yaml status

# View JSON logs (container)
docker logs picode-worker-1 --tail 100

# View systemd logs
journalctl -u picode-worker@1 -f
```

### Example: 3-VPS Production Setup

```
                    ┌─────────────────────┐
                    │   PostgreSQL RDS    │
                    │  db.example.com     │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│    VPS 1      │      │    VPS 2      │      │    VPS 3      │
│  (US East)    │      │  (EU West)    │      │  (Asia)       │
│               │      │               │      │               │
│  4 workers    │      │  4 workers    │      │  4 workers    │
│  2 vCPU, 4GB  │      │  2 vCPU, 4GB  │      │  2 vCPU, 4GB  │
└───────────────┘      └───────────────┘      └───────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      S3 Bucket      │
                    │  picode-dataset     │
                    └─────────────────────┘
```

**VPS 1 Setup Script:**

```bash
#!/bin/bash
# vps1-setup.sh

VPS_NAME="vps1-useast"
WORKERS=4

for i in $(seq 1 $WORKERS); do
  docker run -d \
    --name picode-worker-$i \
    --env-file /etc/picode/.env \
    --restart unless-stopped \
    picode-scraper:latest \
    harvest -c /app/configs/default.yaml \
    --worker-id "${VPS_NAME}-worker-${i}" \
    --json-logs
done
```

## Operations

### Monitoring

```bash
# Task queue status
picode-scraper -c config.yaml status

# Active workers (from database)
psql -c "SELECT claimed_by, COUNT(*) FROM harvest_tasks WHERE status='claimed' GROUP BY claimed_by;"

# Failed tasks
psql -c "SELECT error_message, COUNT(*) FROM harvest_tasks WHERE status='failed' GROUP BY error_message LIMIT 10;"
```

### Requeuing Failed Tasks

```bash
# Requeue all failed tasks
picode-scraper -c config.yaml requeue --status failed

# Requeue dead_letter tasks (exceeded max_retries)
picode-scraper -c config.yaml requeue --status dead_letter --limit 100
```

### Graceful Shutdown

Workers handle SIGTERM gracefully:

```bash
# Stop with grace period (releases claimed task)
docker stop --time 30 picode-worker-1

# Systemd
systemctl stop picode-worker@1
```

### Troubleshooting

**Workers not claiming tasks:**
```bash
# Check for stuck claimed tasks (>5 min old)
psql -c "UPDATE harvest_tasks SET status='pending', claimed_by=NULL WHERE status='claimed' AND claimed_at < NOW() - INTERVAL '10 minutes';"
```

**High failure rate:**
```bash
# Check error patterns
psql -c "SELECT error_message, COUNT(*) as cnt FROM harvest_tasks WHERE status='failed' GROUP BY error_message ORDER BY cnt DESC LIMIT 10;"

# Increase retry limit
export PICODE_SCRAPING__MAX_RETRIES=5
```

**Database connection issues:**
```bash
# Test connection
psql $PICODE_DATABASE__URL -c "SELECT 1;"

# Check pool exhaustion (increase pool_size)
psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname='picode_scraper';"
```

## Export Format

The `export` command generates the following structure:

```
dataset/
├── pairs/
│   ├── a1b2c3d4/
│   │   ├── original.png
│   │   ├── capture.jpg
│   │   ├── corners.json
│   │   └── metadata.json
│   └── e5f6g7h8/
│       └── ...
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── by_type/
│   ├── screen/
│   │   └── a1b2c3d4 -> ../../pairs/a1b2c3d4
│   └── photo/
│       └── e5f6g7h8 -> ../../pairs/e5f6g7h8
└── index.csv
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check picode_scraper/ tests/
mypy picode_scraper/

# Run with debug logging
picode-scraper --log-level DEBUG -c config.yaml harvest
```

## License

Apache 2.0 - see [LICENSE](../LICENSE) for details.
