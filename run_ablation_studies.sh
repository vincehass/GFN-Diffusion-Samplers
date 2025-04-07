#!/bin/bash

# GFlowNet-guided Diffusion Ablation Studies
# This script runs a series of experiments to analyze the effect of different parameters
# on the GFlowNet-guided diffusion model for energy-based sampling.

# Set common parameters
EPOCHS=2000  # Increased to 2000 epochs for longer training
BATCH_SIZE=64
HIDDEN_DIM=128
LATENT_DIM=2
NUM_TIMESTEPS=1000  # 1000 diffusion timesteps per sampling operation
DEVICE="cpu"
SCHEDULE_TYPE="linear"
LOSS_TYPE="tb_avg"
BASE_OUTPUT_DIR="results/ablation_studies"

# Create base output directory if it doesn't exist
mkdir -p $BASE_OUTPUT_DIR

echo "Starting GFlowNet-guided Diffusion Ablation Studies"
echo "====================================================="
echo "Common parameters:"
echo "- Epochs: $EPOCHS"
echo "- Batch size: $BATCH_SIZE"
echo "- Hidden dimensions: $HIDDEN_DIM"
echo "- Latent dimensions: $LATENT_DIM"
echo "- Number of diffusion timesteps: $NUM_TIMESTEPS"
echo "- Device: $DEVICE"
echo "- Schedule type: $SCHEDULE_TYPE"
echo "- Loss type: $LOSS_TYPE"
echo "====================================================="

###########################################
# Part 1: Energy Function Ablation
###########################################
echo "Part 1: Energy Function Ablation"
echo "-------------------------------"

# Run experiments with different energy functions
energy_functions=("gmm" "ring" "diagonal" "complex_mixture")
guidance_scale=2.5

for energy in "${energy_functions[@]}"; do
    output_dir="${BASE_OUTPUT_DIR}/energy_ablation/${energy}"
    run_name="ablation_energy_${energy}"
    
    echo "Running experiment with energy function: $energy"
    
    python -m gfn_diffusion.energy_sampling.train \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --latent_dim $LATENT_DIM \
        --num_timesteps $NUM_TIMESTEPS \
        --device $DEVICE \
        --schedule_type $SCHEDULE_TYPE \
        --energy $energy \
        --guidance_scale $guidance_scale \
        --loss_type $LOSS_TYPE \
        --output_dir $output_dir \
        --wandb \
        --run_name $run_name \
        --eval_interval 100
        
    echo "Completed experiment with energy function: $energy"
    echo ""
done

###########################################
# Part 2: Guidance Scale Ablation
###########################################
echo "Part 2: Guidance Scale Ablation"
echo "-------------------------------"

# Run experiments with different guidance scales
guidance_scales=(0.5 1.0 2.5 5.0 10.0)
energy="complex_mixture"

for scale in "${guidance_scales[@]}"; do
    output_dir="${BASE_OUTPUT_DIR}/guidance_ablation/scale_${scale}"
    run_name="ablation_guidance_${scale}"
    
    echo "Running experiment with guidance scale: $scale"
    
    python -m gfn_diffusion.energy_sampling.train \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --latent_dim $LATENT_DIM \
        --num_timesteps $NUM_TIMESTEPS \
        --device $DEVICE \
        --schedule_type $SCHEDULE_TYPE \
        --energy $energy \
        --guidance_scale $scale \
        --loss_type $LOSS_TYPE \
        --output_dir $output_dir \
        --wandb \
        --run_name $run_name \
        --eval_interval 100
        
    echo "Completed experiment with guidance scale: $scale"
    echo ""
done

###########################################
# Part 3: Loss Type Ablation
###########################################
echo "Part 3: Loss Type Ablation"
echo "-------------------------------"

# Run experiments with different loss types
loss_types=("tb" "tb_avg" "db" "fm")
energy="complex_mixture"
guidance_scale=2.5

for loss in "${loss_types[@]}"; do
    output_dir="${BASE_OUTPUT_DIR}/loss_ablation/${loss}"
    run_name="ablation_loss_${loss}"
    
    echo "Running experiment with loss type: $loss"
    
    python -m gfn_diffusion.energy_sampling.train \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --latent_dim $LATENT_DIM \
        --num_timesteps $NUM_TIMESTEPS \
        --device $DEVICE \
        --schedule_type $SCHEDULE_TYPE \
        --energy $energy \
        --guidance_scale $guidance_scale \
        --loss_type $loss \
        --output_dir $output_dir \
        --wandb \
        --run_name $run_name \
        --eval_interval 100
        
    echo "Completed experiment with loss type: $loss"
    echo ""
done

###########################################
# Part 4: Network Architecture Ablation
###########################################
echo "Part 4: Network Architecture Ablation"
echo "-------------------------------"

# Run experiments with different hidden dimensions
hidden_dims=(64 128 256 512)
energy="complex_mixture"
guidance_scale=2.5

for dim in "${hidden_dims[@]}"; do
    output_dir="${BASE_OUTPUT_DIR}/architecture_ablation/hidden_${dim}"
    run_name="ablation_hidden_dim_${dim}"
    
    echo "Running experiment with hidden dimensions: $dim"
    
    python -m gfn_diffusion.energy_sampling.train \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $dim \
        --latent_dim $LATENT_DIM \
        --num_timesteps $NUM_TIMESTEPS \
        --device $DEVICE \
        --schedule_type $SCHEDULE_TYPE \
        --energy $energy \
        --guidance_scale $guidance_scale \
        --loss_type $LOSS_TYPE \
        --output_dir $output_dir \
        --wandb \
        --run_name $run_name \
        --eval_interval 100
        
    echo "Completed experiment with hidden dimensions: $dim"
    echo ""
done

###########################################
# Part 5: Diffusion Schedule Ablation
###########################################
echo "Part 5: Diffusion Schedule Ablation"
echo "-------------------------------"

# Run experiments with different schedule types
schedule_types=("linear" "cosine" "quadratic")
energy="complex_mixture"
guidance_scale=2.5

for schedule in "${schedule_types[@]}"; do
    output_dir="${BASE_OUTPUT_DIR}/schedule_ablation/${schedule}"
    run_name="ablation_schedule_${schedule}"
    
    echo "Running experiment with schedule type: $schedule"
    
    python -m gfn_diffusion.energy_sampling.train \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --latent_dim $LATENT_DIM \
        --num_timesteps $NUM_TIMESTEPS \
        --device $DEVICE \
        --schedule_type $schedule \
        --energy $energy \
        --guidance_scale $guidance_scale \
        --loss_type $LOSS_TYPE \
        --output_dir $output_dir \
        --wandb \
        --run_name $run_name \
        --eval_interval 100
        
    echo "Completed experiment with schedule type: $schedule"
    echo ""
done

###########################################
# Part 6: Timestep Ablation
###########################################
echo "Part 6: Timestep Ablation"
echo "-------------------------------"

# Run experiments with different numbers of timesteps
timesteps=(100 500 1000 2000 5000)
energy="complex_mixture"
guidance_scale=2.5

for steps in "${timesteps[@]}"; do
    output_dir="${BASE_OUTPUT_DIR}/timestep_ablation/steps_${steps}"
    run_name="ablation_timesteps_${steps}"
    
    echo "Running experiment with timesteps: $steps"
    
    python -m gfn_diffusion.energy_sampling.train \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --latent_dim $LATENT_DIM \
        --num_timesteps $steps \
        --device $DEVICE \
        --schedule_type $SCHEDULE_TYPE \
        --energy $energy \
        --guidance_scale $guidance_scale \
        --loss_type $LOSS_TYPE \
        --output_dir $output_dir \
        --wandb \
        --run_name $run_name \
        --eval_interval 100
        
    echo "Completed experiment with timesteps: $steps"
    echo ""
done

echo "All ablation studies completed!"
echo "Results are available in: $BASE_OUTPUT_DIR"
echo "Check WandB for detailed metrics and visualizations." 