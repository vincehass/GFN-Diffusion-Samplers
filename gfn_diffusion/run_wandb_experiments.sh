#!/bin/bash

# Exit on error
set -e

# Print commands before executing
set -x

# Default settings
EXPERIMENT="both"
WANDB_ENTITY="nadhirvincenthassen"
DEVICE="cpu"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WANDB_PROJECT="gfn-diffusion-experiments"
OFFLINE_FLAG=""
NUM_LAYERS=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --wandb_entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --offline)
      OFFLINE_FLAG="--offline"
      # Set environment variable for child processes
      export WANDB_MODE="offline"
      shift
      ;;
    --num_layers)
      NUM_LAYERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--experiment <energy|vae|both>] [--wandb_entity <username>] [--device <cpu|cuda>] [--project <project_name>] [--offline] [--num_layers <n>]"
      exit 1
      ;;
  esac
done

# Create logs directory
mkdir -p logs

echo "Running experiments with settings:"
echo "  Experiment type: $EXPERIMENT"
echo "  Wandb entity: $WANDB_ENTITY"
echo "  Wandb project: $WANDB_PROJECT"
echo "  Device: $DEVICE"
echo "  Timestamp: $TIMESTAMP"
if [[ -n "$OFFLINE_FLAG" ]]; then
  echo "  Offline mode: yes"
else
  echo "  Offline mode: no"
fi

# Check if wandb is installed and install if needed
pip list | grep -q wandb || pip install -q wandb

# Check torch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Only attempt login if WANDB_API_KEY is set and not in offline mode
if [[ -z "$OFFLINE_FLAG" ]] && [[ -n "$WANDB_API_KEY" ]]; then
  echo "Using provided WANDB_API_KEY for authentication"
  wandb login "$WANDB_API_KEY"
elif [[ -n "$OFFLINE_FLAG" ]]; then
  echo "Running in offline mode, no authentication required"
else
  echo "No WANDB_API_KEY provided, using existing authentication if available"
fi

# Ensure the vae directory exists
if [ ! -d "gfn_diffusion/vae" ]; then
  echo "Creating vae directory structure..."
  mkdir -p gfn_diffusion/vae/energies
fi

# Install required dependencies if needed
pip install -q wandb matplotlib tqdm

run_energy_experiment() {
  echo "Running energy sampling experiment..."
  
  python gfn_diffusion/energy_sampling/train.py \
    --wandb \
    --wandb_project=$WANDB_PROJECT \
    --wandb_entity=$WANDB_ENTITY \
    --device=$DEVICE \
    --seed=42 \
    --hidden_dim=128 \
    --energy=25gmm \
    --epochs=1000 \
    --batch_size=64 \
    --guidance_scale=1.0 \
    --run_name="energy_sampling_${TIMESTAMP}" \
    --num_layers=${NUM_LAYERS} \
    $OFFLINE_FLAG \
    2>&1 | tee logs/energy_experiment_${TIMESTAMP}.log
    
  echo "Energy sampling experiment completed."
}

run_vae_experiment() {
  echo "Running VAE experiment..."
  
  # First train the VAE model
  echo "Training VAE model..."
  python gfn_diffusion/vae/train.py \
    --train_vae \
    --wandb \
    --wandb_project=$WANDB_PROJECT \
    --wandb_entity=$WANDB_ENTITY \
    --device=$DEVICE \
    --seed=42 \
    --hidden_dim=128 \
    --epochs=20 \
    --batch_size=128 \
    --run_name="vae_training_${TIMESTAMP}" \
    $OFFLINE_FLAG \
    2>&1 | tee logs/vae_train_${TIMESTAMP}.log
  
  # Then train the GFN-Diffusion model
  echo "Training GFN-Diffusion with VAE energy..."
  python gfn_diffusion/vae/train.py \
    --train_gfn \
    --wandb \
    --wandb_project=$WANDB_PROJECT \
    --wandb_entity=$WANDB_ENTITY \
    --device=$DEVICE \
    --seed=42 \
    --hidden_dim=128 \
    --epochs=50 \
    --batch_size=128 \
    --guidance_scale=1.0 \
    --run_name="vae_gfn_diffusion_${TIMESTAMP}" \
    --langevin \
    $OFFLINE_FLAG \
    2>&1 | tee logs/vae_gfn_${TIMESTAMP}.log
    
  echo "VAE experiment completed."
}

# Run the specified experiment(s)
case "$EXPERIMENT" in
  "energy")
    run_energy_experiment
    ;;
  "vae")
    run_vae_experiment
    ;;
  "both")
    run_energy_experiment
    run_vae_experiment
    ;;
  *)
    echo "Invalid experiment type: $EXPERIMENT"
    echo "Please use 'energy', 'vae', or 'both'"
    exit 1
    ;;
esac

echo "All experiments completed!" 