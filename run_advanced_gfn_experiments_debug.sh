#!/bin/bash

# This is a debug version of the GFN-Diffusion experiments script
# with enhanced wandb logging to help troubleshoot connectivity issues

# Default values
EXPERIMENT="energy"  # Start with just the energy experiment
WANDB_ENTITY="nadhirvincenthassen"
WANDB_PROJECT="gfn-diffusion-test"  # Changed to a distinct test project
DEVICE="cpu"  # Default to CPU for testing
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OFFLINE_FLAG=""
NUM_LAYERS=1  # Simplified model for quicker testing

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --wandb_entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num_layers)
      NUM_LAYERS="$2"
      shift 2
      ;;
    --offline)
      OFFLINE_FLAG="--offline"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--experiment energy|vae|both] [--wandb_entity ENTITY] [--wandb_project PROJECT] [--device DEVICE] [--num_layers LAYERS] [--offline]"
      exit 1
      ;;
  esac
done

# Create logs directory
mkdir -p logs

echo "=== Running advanced GFN-Diffusion experiments (DEBUG MODE) ==="
echo "  Experiment type: $EXPERIMENT"
echo "  Wandb entity: $WANDB_ENTITY"
echo "  Wandb project: $WANDB_PROJECT (SPECIAL TEST PROJECT)"
echo "  Device: $DEVICE"
echo "  Number of layers: $NUM_LAYERS"
echo "  Timestamp: $TIMESTAMP"
if [[ -n "$OFFLINE_FLAG" ]]; then
  echo "  Offline mode: yes"
else
  echo "  Offline mode: no"
fi

# Check if wandb is installed and install if needed
echo "Checking wandb installation..."
pip list | grep wandb
if [ $? -ne 0 ]; then
  echo "Installing wandb..."
  pip install -q wandb
fi

# Check wandb version
echo "Checking wandb version..."
wandb --version

# Check wandb login status
echo "Checking wandb login status..."
wandb login --cloud
echo "Wandb login check complete. You should be logged in as shown above ^^^ "
echo "This run will be logged to: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "Please verify you can access this URL in your browser."

# Only attempt login if WANDB_API_KEY is set and not in offline mode
if [[ -z "$OFFLINE_FLAG" ]] && [[ -n "$WANDB_API_KEY" ]]; then
  echo "Using provided WANDB_API_KEY for authentication"
  wandb login "$WANDB_API_KEY"
elif [[ -n "$OFFLINE_FLAG" ]]; then
  echo "Running in offline mode, no authentication required"
else
  echo "No WANDB_API_KEY provided, using existing authentication if available"
fi

# Make sure directories exist
echo "Creating necessary directories..."
mkdir -p gfn_diffusion/vae/energies
mkdir -p data
mkdir -p models/vae_experiment
mkdir -p results/vae_experiment

# Install required dependencies if needed
echo "Installing required dependencies..."
pip install -q matplotlib tqdm

run_energy_experiment_debug() {
  echo "=== Running energy sampling experiment (DEBUG MODE) ==="
  echo "IMPORTANT: This run will be logged to: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
  
  # Run with fewer epochs for faster testing
  python gfn_diffusion/energy_sampling/train.py \
    --wandb \
    --wandb_project=$WANDB_PROJECT \
    --wandb_entity=$WANDB_ENTITY \
    --device=$DEVICE \
    --seed=42 \
    --hidden_dim=64 \
    --energy=25gmm \
    --epochs=3 \
    --batch_size=32 \
    --guidance_scale=1.0 \
    --run_name="debug_test_${TIMESTAMP}" \
    --num_layers=${NUM_LAYERS} \
    $OFFLINE_FLAG \
    2>&1 | tee logs/debug_energy_experiment_${TIMESTAMP}.log
    
  echo "Energy sampling experiment completed."
  echo "Check for this run at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
}

# Run the test energy experiment only
echo "Starting debug experiment..."
run_energy_experiment_debug

# Check wandb directory for runs
echo "Checking for wandb runs..."
if [ -d "wandb" ]; then
  echo "Local wandb directory found. Contents:"
  ls -la wandb/
  echo "Latest run folder contents:"
  LATEST_RUN=$(ls -t wandb | grep "run-" | head -1)
  if [ -n "$LATEST_RUN" ]; then
    echo "Latest run: $LATEST_RUN"
    ls -la "wandb/$LATEST_RUN"
    
    # Check for log files
    echo "Checking log files..."
    if [ -d "wandb/$LATEST_RUN/logs" ]; then
      echo "Log content:"
      cat "wandb/$LATEST_RUN/logs/debug.log" | grep -E "URL|entity|nadhirvincenthassen"
      echo "Full logs available at: wandb/$LATEST_RUN/logs/debug.log"
    else
      echo "No log directory found."
    fi
  else
    echo "No run directories found."
  fi
else
  echo "No local wandb directory found."
fi

echo "All debug experiments completed!"
echo "IMPORTANT: This run should be visible at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "Please check this URL in your browser." 