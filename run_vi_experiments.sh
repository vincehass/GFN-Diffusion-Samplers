#!/bin/bash

# Configuration for all experiments
PROJECT_NAME="gflownet-vi"
N=12                  # Sequence length
K=3                   # Bits per word
D_MODEL=64            # Model dimension
D_HID=256             # Hidden dimension
NHEAD=8               # Number of attention heads
NLAYERS=3             # Number of transformer layers
BATCH_SIZE=16         # Batch size
LR=0.001              # Learning rate
Z_LR=0.001            # Partition function learning rate
NUM_ITERATIONS=50000  # Number of training iterations
DEVICE="cpu"          # Device to use (cpu/cuda)
WANDB_ENTITY=""       # Your wandb entity name (leave empty if not needed)
PRINT_EVERY=500       # Print metrics every N iterations
VALIDATE_EVERY=2000   # Validate every N iterations

# Make sure N is divisible by K
if (( N % K != 0 )); then
    # Adjust N to be divisible by K
    N=$(( (N / K + 1) * K ))
    echo "Adjusted N to $N to ensure it's divisible by K=$K"
fi

# Create experiment directory
EXPERIMENT_DIR="experiments/vi_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXPERIMENT_DIR

# Enable OpenMP fix if needed
export KMP_DUPLICATE_LIB_OK=TRUE

# Function to run an experiment with given divergence
run_experiment() {
    local div=$1
    local alpha=${2:-2.0}
    local cv_method=${3:-lsd}
    
    echo "Running experiment with divergence=$div, alpha=$alpha, cv_method=$cv_method"
    
    # Only include wandb_entity if it's not empty
    WANDB_ARGS=""
    if [ ! -z "$WANDB_ENTITY" ]; then
        WANDB_ARGS="--wandb_entity $WANDB_ENTITY"
    fi
    
    python bitseq/run.py \
        --objective vi \
        --divergence $div \
        --alpha $alpha \
        --cv_method $cv_method \
        --n $N \
        --k $K \
        --d_model $D_MODEL \
        --d_hid $D_HID \
        --nhead $NHEAD \
        --nlayers $NLAYERS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --z_lr $Z_LR \
        --num_iterations $NUM_ITERATIONS \
        --device $DEVICE \
        --wandb_project $PROJECT_NAME \
        $WANDB_ARGS \
        --run_name "vi_${div}_${N}_${K}" \
        --print_every $PRINT_EVERY \
        --validate_every $VALIDATE_EVERY \
        --log_grad_norm \
        2>&1 | tee "$EXPERIMENT_DIR/${div}_${alpha}_${cv_method}.log"
}

# Run Forward KL experiment
run_experiment "fkl"

# Run Reverse KL experiment
run_experiment "rkl"

# Run Tsallis divergence experiments with different alpha values
run_experiment "tsallis" 1.5 "lsd"
run_experiment "tsallis" 2.0 "lsd"
run_experiment "tsallis" 2.5 "lsd"

# Optionally run experiments with REINFORCE instead of LSD
# run_experiment "tsallis" 2.0 "reinforce"

echo "All experiments completed. Logs are in $EXPERIMENT_DIR" 