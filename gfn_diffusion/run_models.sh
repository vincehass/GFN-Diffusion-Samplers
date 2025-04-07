#!/bin/bash

# GFN-Diffusion Models Runner
# This script runs the different GFN-Diffusion models implemented in the package

# Set execution directory to the script location
cd "$(dirname "$0")"

# Create output directory
mkdir -p results

echo "=================================================="
echo "GFN-Diffusion Models Runner"
echo "=================================================="

# =============== PART 1: Energy Sampling ===============
echo "Running Energy Sampling experiments..."

# 1.1 Run the basic example script for demo purposes
echo "1.1 Running example script..."
python example.py
echo "Example script completed. Visualizations saved to the results directory."

# 1.2 Run Unconditional GMM Energy Sampling
echo "1.2 Running Unconditional GMM Energy Sampling..."
python -m energy_sampling.train \
    --energy gmm \
    --output_dir results/energy_sampling/gmm \
    --epochs 500 \
    --batch_size 64 \
    --num_samples 100 \
    --sample_interval 100 \
    --guidance_scale 2.0

# 1.3 Run 'Many Well' Energy Sampling with TB + Exploratory
echo "1.3 Running 'Many Well' Energy with TB + Exploratory..."
python -m energy_sampling.train \
    --t_scale 1.0 \
    --energy multi_modal \
    --pis_architectures \
    --zero_init \
    --clipping \
    --mode_fwd tb \
    --lr_policy 1e-3 \
    --lr_flow 1e-1 \
    --exploratory \
    --exploration_wd \
    --exploration_factor 0.2 \
    --output_dir results/energy_sampling/multi_modal_tb_expl \
    --epochs 500

# 1.4 Run Conditional Model
echo "1.4 Running Conditional Energy Sampling..."
python -m energy_sampling.train \
    --model_type conditional \
    --output_dir results/energy_sampling/conditional \
    --data_dim 2 \
    --condition_dim 4 \
    --hidden_dim 64 \
    --epochs 500 \
    --batch_size 64

echo "Energy Sampling experiments completed!"

# =============== PART 2: VAE Experiment ===============
echo "Running VAE experiments..."

# 2.1 Create VAE directories if they don't exist
mkdir -p vae/models
mkdir -p vae/results

# 2.2 Train VAE model (if not already trained)
echo "2.1 Training VAE model..."
cd vae
python energies/vae.py \
    --epochs 20 \
    --latent_dim 2 \
    --hidden_dims 512 256 128 64 \
    --beta 1.0
echo "VAE training completed."

# 2.3 Run VAE-based GFN Diffusion (basic version for quick testing)
echo "2.2 Running VAE-based GFN-Diffusion (basic version)..."
python train.py \
    --energy vae \
    --vae_path models/vae_best.pt \
    --pis_architectures \
    --zero_init \
    --mode_fwd tb \
    --lr_policy 1e-3 \
    --lr_flow 1e-1 \
    --epochs 200 \
    --batch_size 64 \
    --sample_interval 50 \
    --output_dir results/vae_basic \
    --model_dir models/vae_basic

# 2.4 Run VAE-based GFN Diffusion with TB + Exploratory + LS (more complete version)
echo "2.3 Running VAE-based GFN-Diffusion with TB + Exploratory + LS..."
python train.py \
    --energy vae \
    --pis_architectures \
    --zero_init \
    --clipping \
    --mode_fwd cond-tb-avg \
    --mode_bwd cond-tb-avg \
    --lr_policy 1e-3 \
    --lr_flow 1e-1 \
    --lr_back 1e-3 \
    --exploratory \
    --exploration_wd \
    --exploration_factor 0.1 \
    --both_ways \
    --local_search \
    --max_iter_ls 500 \
    --burn_in 200 \
    --buffer_size 9000 \
    --prioritized rank \
    --rank_weight 0.01 \
    --ld_step 0.001 \
    --ld_schedule \
    --target_acceptance_rate 0.574 \
    --epochs 200 \
    --batch_size 64 \
    --sample_interval 50 \
    --output_dir results/vae_full \
    --model_dir models/vae_full

cd ..
echo "VAE experiments completed!"

echo "=================================================="
echo "All GFN-Diffusion models have been run!"
echo "Results are saved in the 'results' directory"
echo "==================================================" 