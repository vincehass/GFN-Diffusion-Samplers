#!/bin/bash

# Configuration for test run
N=12                  # Sequence length
K=3                   # Bits per word
D_MODEL=64            # Model dimension
D_HID=256             # Hidden dimension
NHEAD=8               # Number of attention heads
NLAYERS=3             # Number of transformer layers
BATCH_SIZE=16         # Batch size
NUM_ITERATIONS=20     # Number of training iterations for testing
DEVICE="cpu"          # Device to use (cpu/cuda)
PRINT_EVERY=10        # Print metrics every N iterations

# Make sure N is divisible by K
if (( N % K != 0 )); then
    # Adjust N to be divisible by K
    N=$(( (N / K + 1) * K ))
    echo "Adjusted N to $N to ensure it's divisible by K=$K"
fi

# Enable OpenMP fix if needed
export KMP_DUPLICATE_LIB_OK=TRUE

# Run a quick test with Forward KL
echo "Running quick test with Forward KL divergence"
python bitseq/run.py \
    --objective vi \
    --divergence fkl \
    --n $N \
    --k $K \
    --d_model $D_MODEL \
    --d_hid $D_HID \
    --nhead $NHEAD \
    --nlayers $NLAYERS \
    --batch_size $BATCH_SIZE \
    --lr 0.001 \
    --z_lr 0.001 \
    --num_iterations $NUM_ITERATIONS \
    --device $DEVICE \
    --no_wandb \
    --run_name "test_vi_fkl" \
    --print_every $PRINT_EVERY

echo "Test completed." 