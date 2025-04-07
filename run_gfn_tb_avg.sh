#!/bin/bash

# This script runs the energy experiment with the TB-Avg loss

echo "Running energy experiment with TB-Avg loss..."

# Create a default model_dir value in the output directory
python -m gfn_diffusion.energy_sampling.train \
    --epochs 1000 \
    --batch_size 64 \
    --wandb \
    --latent_dim 2 \
    --energy 25gmm \
    --guidance_scale 1.0 \
    --device cpu \
    --loss_type tb_avg \
    --project_name "gfn-diffusion-tb-avg" \
    --run_name "energy_tb_avg_test" \
    --output_dir results/energy_tb_avg

echo "Experiment completed!" 