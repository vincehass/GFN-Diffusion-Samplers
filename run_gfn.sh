#!/bin/bash

# Specify our project
#SBATCH --account=your_project

# Request resources
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Specify the partition
#SBATCH --partition=gpu

# Start from clean slate
#SBATCH --export=None

# Place the slurm output file in a specific directory
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

# Job name
#SBATCH --job-name=gfn_train

# Create output directory if it doesn't exist
mkdir -p slurm_logs

# Echo job information
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load modules
module load Python/3.11.3-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a matplotlib/3.7.2-gfbf-2023a
module list

# Set up environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional: Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Set default parameters (can be overridden via command line)
ENERGY=${ENERGY:-"9gmm"}
T=${T:-100}
BATCH_SIZE=${BATCH_SIZE:-300}
EPOCHS=${EPOCHS:-10000}
LOCAL_SEARCH=${LOCAL_SEARCH:-"--local_search"}
MAX_ITER_LS=${MAX_ITER_LS:-200}
SEED=${SEED:-12345}
USE_AMP=${USE_AMP:-"--use_amp"}

# Print configuration
echo "Configuration:"
echo "  Energy: $ENERGY"
echo "  Trajectory Length (T): $T"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Local Search: $LOCAL_SEARCH"
echo "  Max Iterations LS: $MAX_ITER_LS"
echo "  Seed: $SEED"
echo "  Use AMP: $USE_AMP"

# Run the script
echo "Starting training..."
python energy_sampling/train.py \
  --energy $ENERGY \
  --T $T \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  $LOCAL_SEARCH \
  --max_iter_ls $MAX_ITER_LS \
  --seed $SEED \
  $USE_AMP

# Print completion information
echo "Job completed at: $(date)"
echo "Elapsed time: $SECONDS seconds"

# Optional: Copy output files to a specific location
# mkdir -p /path/to/results/$SLURM_JOB_ID
# cp -r output/* /path/to/results/$SLURM_JOB_ID/ 