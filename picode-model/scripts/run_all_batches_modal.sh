#!/bin/bash
# Run all batches with Modal upload and local cleanup

cd ~/workspace/code/picode-model
source .venv/bin/activate

START_BATCH=${1:-11}
END_BATCH=${2:-25}

echo "=== Running batches $START_BATCH to $END_BATCH with Modal upload ==="
echo "Free space before: $(df -h / | tail -1 | awk '{print $4}')"

for i in $(seq $START_BATCH $END_BATCH); do
    echo ""
    echo "=========================================="
    echo "Batch $i starting at $(date)"
    echo "Free space: $(df -h / | tail -1 | awk '{print $4}')"
    echo "=========================================="
    
    ./scripts/batch_generate_and_upload.sh $i
    
    if [ $? -eq 0 ]; then
        echo "Batch $i: generated, uploaded, cleaned"
    else
        echo "Batch $i: FAILED"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All batches complete!"
echo "Free space after: $(df -h / | tail -1 | awk '{print $4}')"
echo "=========================================="
