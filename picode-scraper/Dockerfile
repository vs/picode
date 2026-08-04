FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless only needs glib at runtime (no libGL/X11)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml README.md ./
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
