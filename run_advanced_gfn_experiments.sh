#!/bin/bash

# This script runs advanced experiments for GFlowNet guided diffusion
# We've updated the code to include consistent metrics across all experiments:
# - avg_reward (negative energy for energy-based experiments)
# - diversity (uniqueness of generated samples)
# - novelty (difference from reference distributions)
# - unique_positions (count of unique discrete grid positions covered)
# These metrics will allow for consistent comparisons across different types of experiments.

# Check if experiment type is provided
if [ -z "$1" ]; then
    echo "Please specify an experiment type (energy, images, etc.)"
    exit 1
fi

# Parse arguments
experiment_type=""
device="cuda"
loss_type="tb_avg"  # Default to using the TB-Avg loss

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --experiment)
        experiment_type="$2"
        shift
        shift
        ;;
        --device)
        device="$2"
        shift
        shift
        ;;
        --loss)
        loss_type="$2"
        shift
        shift
        ;;
        *)
        # Unknown option
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# If experiment_type is still empty, use the first positional argument
if [ -z "$experiment_type" ]; then
    experiment_type="$1"
    shift
fi

# Run appropriate experiment
if [ "$experiment_type" = "energy" ]; then
    echo "Running energy-based sampling experiments with loss_type=$loss_type..."
    
    # First, run the standard experiment
    python -m gfn_diffusion.energy_sampling.train \
        --epochs 1000 \
        --batch_size 64 \
        --wandb \
        --latent_dim 2 \
        --energy 25gmm \
        --guidance_scale 1.0 \
        --device $device \
        --loss_type $loss_type \
        --output_dir results/energy_sampling_standard_advanced_${loss_type}
    
    # Then run the conditional experiment
    python -m gfn_diffusion.energy_sampling.train \
        --epochs 1000 \
        --batch_size 64 \
        --wandb \
        --latent_dim 2 \
        --conditional \
        --num_conditions 4 \
        --guidance_scale 1.0 \
        --device $device \
        --loss_type $loss_type \
        --output_dir results/energy_sampling_conditional_advanced_${loss_type}
    
elif [ "$experiment_type" = "images" ]; then
    echo "Running image-based experiments with loss_type=$loss_type..."
    
    # Train VAE on MNIST
    python -m gfn_diffusion.vae.train \
        --dataset mnist \
        --epochs 50 \
        --batch_size 128 \
        --latent_dim 16 \
        --wandb \
        --device $device \
        --output_dir results/vae_mnist_advanced
    
    # Train diffusion model on VAE latent space
    python -m gfn_diffusion.energy_sampling.train \
        --latent_dim 16 \
        --epochs 1000 \
        --batch_size 64 \
        --energy vae \
        --vae_path results/vae_mnist_advanced/model.pt \
        --vae_type mnist \
        --guidance_scale 1.0 \
        --wandb \
        --device $device \
        --loss_type $loss_type \
        --output_dir results/vae_diffusion_advanced_${loss_type}
        
else
    echo "Unknown experiment type: $experiment_type"
    exit 1
fi

echo "All advanced experiments completed successfully!"

# Check for wandb runs
if [ -d "wandb" ]; then
    echo "Checking for wandb runs..."
    echo "Local wandb directory found. Contents:"
    # ls -la wandb
fi 