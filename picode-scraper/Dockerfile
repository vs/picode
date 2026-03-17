FROM python:3.11-slim

WORKDIR /app

# Install OpenCV dependencies and other system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml .
COPY picode_scraper/__init__.py picode_scraper/

# Install dependencies (without dev)
RUN pip install --no-cache-dir -e .

# Copy source code
COPY picode_scraper/ picode_scraper/
COPY configs/ configs/

# Default config path
ENV PICODE_CONFIG=/app/configs/default.yaml

# Health check - verify CLI works
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD picode-scraper --version || exit 1

ENTRYPOINT ["picode-scraper"]
CMD ["harvest"]
