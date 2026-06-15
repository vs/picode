#!/bin/bash
# Modal training with automatic bootstrap retry.
# The training code itself detects collapse at step 2000 and exits with error.
# This script simply retries on failure.
#
# Usage: bash scripts/modal_train_autoretry.sh <config> [override]

CONFIG="${1:?Usage: $0 <config> [override]}"
OVERRIDE="${2:-}"
MAX_ATTEMPTS=10

echo "=== Modal Bootstrap Retry Launcher ==="
echo "Config: $CONFIG"
echo "Max attempts: $MAX_ATTEMPTS"
echo ""

EXPERIMENT=$(grep 'experiment_name:' "$CONFIG" | awk '{print $2}')

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "--- Attempt $attempt/$MAX_ATTEMPTS ---"

    # Clean checkpoints from failed attempts
    if [ -n "$EXPERIMENT" ]; then
        modal volume rm picode-checkpoints "$EXPERIMENT/" 2>/dev/null || true
    fi

    # Run training (blocks until complete or crash)
    if [ -n "$OVERRIDE" ]; then
        modal run --detach scripts/modal_train.py::train \
            --config "$CONFIG" --override "$OVERRIDE" 2>&1
    else
        modal run --detach scripts/modal_train.py::train \
            --config "$CONFIG" 2>&1
    fi

    # Wait for bootstrap check (~3.5 min for 2000 steps on A10G)
    echo "Waiting for bootstrap check (step 2000)..."
    sleep 210

    # Check if app is still running (= bootstrap passed)
    APP_ID=$(modal app list 2>&1 | grep "ephemeral" | grep "detached" | head -1 | awk '{print $2}')

    if [ -z "$APP_ID" ]; then
        echo "App not running — collapsed or failed to launch."
        sleep 5
        continue
    fi

    # Check if it's still alive (tasks > 0)
    TASKS=$(modal app list 2>&1 | grep "$APP_ID" | grep -o 'Tasks.*' || echo "")
    STATE=$(modal app list 2>&1 | grep "$APP_ID" || echo "stopped")

    if echo "$STATE" | grep -q "stopped"; then
        echo "App stopped — bootstrap collapsed."
        sleep 5
        continue
    fi

    echo ""
    echo "=== BOOTSTRAP SUCCEEDED (attempt $attempt) ==="
    echo "App: $APP_ID (running detached)"
    echo "Monitor: modal app logs $APP_ID"
    exit 0
done

echo ""
echo "=== FAILED after $MAX_ATTEMPTS attempts ==="
exit 1
