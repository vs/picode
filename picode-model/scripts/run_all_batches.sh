#!/bin/bash
# Run all remaining batches sequentially

cd ~/workspace/code/picode-model
source .venv/bin/activate

START_BATCH=${1:-7}
END_BATCH=${2:-25}

echo "=== Running batches $START_BATCH to $END_BATCH ==="

for i in $(seq $START_BATCH $END_BATCH); do
    echo ""
    echo "=========================================="
    echo "Starting batch $i at $(date)"
    echo "=========================================="
    
    ./scripts/batch_generate.sh $i
    
    if [ $? -eq 0 ]; then
        echo "Batch $i completed successfully"
    else
        echo "Batch $i FAILED with exit code $?"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All batches complete!"
echo "Total samples: $((END_BATCH * 2000))"
echo "=========================================="
