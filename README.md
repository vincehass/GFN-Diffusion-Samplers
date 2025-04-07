# GFlowNet-guided Diffusion for Energy-Based Sampling

This repository contains an implementation of energy-based sampling using GFlowNet-guided diffusion models. The repository demonstrates how GFlowNet training can guide a diffusion model toward sampling from low-energy regions of complex energy landscapes.

## Overview

The framework combines diffusion models with GFlowNet training to create a guided sampling process that:

1. Learns to sample from complex multimodal energy landscapes
2. Balances exploration and exploitation of low-energy regions
3. Can be configured with different guidance scales to control the strength of energy guidance

## Implemented Energy Functions

The repository includes several energy functions for experimentation:

1. **Simple Gaussian Mixture Model (GMM)**

   - Basic 2D Gaussian mixture with 4 modes
   - Implemented in `gmm_energy` function
   - Good for basic validation of the method

2. **Ring Energy**

   - Creates a ring-shaped low-energy region
   - Implemented in `ring_energy` function
   - Tests the model's ability to sample from non-Gaussian distributions

3. **Diagonal Energy**

   - Creates diagonal valleys of low energy
   - Implemented in `diagonal_energy` function
   - Tests directional bias in sampling

4. **Complex Mixture Energy** (Advanced)
   - Combines multiple patterns: 5x5 GMM grid, ring overlay, and diagonal valley
   - Implemented in `complex_mixture_energy` function
   - Most challenging landscape for testing sampling capabilities
   - Created to test the model's ability to handle highly multimodal and complex energy landscapes

## Installation

- Create conda environment:

```sh
conda create -n gfn-diff python=3.10
conda activate gfn-diff
```

- Install dependencies for bitseq and hypergrid experiments:

```sh
pip install -r requirements.txt
```

- Install dependencies for molecular experiments:

```sh
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

## Experiments and Ablation Studies

We conducted several experiments to evaluate the GFlowNet-guided diffusion approach:

### Advanced Experiment: Complex Mixture Energy with Different Guidance Scales

We tested the complex mixture energy function with different guidance scales to understand how strongly the GFlowNet guides the diffusion process:

1. **Small Guidance** (guidance_scale=1.0)

   - Less aggressive guidance toward low-energy regions
   - Higher diversity in samples
   - Output in `results/complex_mixture_small`
   - WandB run: `complex_mixture_small_guidance`

2. **Medium Guidance** (guidance_scale=2.5)

   - Balanced approach between exploration and exploitation
   - Moderate concentration on low-energy regions
   - Output in `results/complex_mixture_medium`
   - WandB run: `complex_mixture_medium_guidance`

3. **High Guidance** (guidance_scale=5.0)
   - Strong bias toward low-energy regions
   - More concentrated sampling in energy minima
   - Output in `results/complex_mixture_high`
   - WandB run: `complex_mixture_high_guidance`

### Comprehensive Ablation Studies

For a more systematic evaluation, we created an ablation study script (`run_ablation_studies.sh`) that tests various parameters with extended training (2000 epochs) and 1000 diffusion timesteps:

1. **Energy Function Ablation**

   - Tests all implemented energy functions (GMM, Ring, Diagonal, Complex Mixture)
   - Compares how well the model adapts to different energy landscapes

2. **Guidance Scale Ablation**

   - Tests guidance scales ranging from 0.5 to 10.0
   - Analyzes the trade-off between exploration and exploitation

3. **Loss Type Ablation**

   - Tests different GFlowNet loss functions (TB, TB-Avg, DB, FM)
   - Compares training stability and final performance

4. **Network Architecture Ablation**

   - Tests different hidden dimensions (64, 128, 256, 512)
   - Analyzes how model capacity affects sampling quality

5. **Diffusion Schedule Ablation**

   - Tests different noise schedules (Linear, Cosine, Quadratic)
   - Analyzes the impact of the noise schedule on sample quality

6. **Timestep Ablation**
   - Tests diffusion timesteps from 100 to 5000
   - Analyzes the trade-off between computation time and sample quality
   - Each experiment is trained for the full 2000 epochs for thorough evaluation

## Training Parameters

Our ablation studies use the following training parameters:

- **Training Epochs**: 2000 epochs (iterations of the GFlowNet training process)
- **Diffusion Timesteps**: 1000 steps per sampling operation (controls noise addition/removal granularity)
- **Batch Size**: 64 samples per training batch
- **Evaluation Interval**: Every 100 epochs (for efficiency while still tracking progress)
- **Full Metrics**: Computed at evaluation intervals and at the end of training
- **Latent Dimensions**: 2D for easy visualization

## Evaluation Metrics

We evaluate the quality of samples using several metrics:

1. **Average Reward**: Negative energy of samples (higher is better)
2. **Diversity**: Entropy of sample distribution
3. **Novelty**: How different samples are from the training data
4. **Clustering**: Analysis of sample clustering at different tolerance levels
   - At tolerance 0.1: Measures coarse-grained mode coverage
   - At tolerance 0.01-0.0001: Measures fine-grained sample diversity

### Metrics Logging and Charts

The metrics in our experiments are logged at two different intervals:

1. **Training Metrics** (every epoch):

   - Loss/Average Reward: Logged at every epoch to show training progress
   - These are continuously updated in the WandB charts

2. **Evaluation Metrics** (every 100 epochs):
   - Diversity: Measures how spread out the samples are
   - Novelty: Measures how different samples are from reference points
   - Unique Positions: Counts distinct sample clusters at different tolerance levels
   - These metrics are more expensive to compute and are calculated periodically
   - The final values are shown in the WandB charts and detailed in the logs

When looking at the charts, note that:

- Continuous metrics (loss, reward) show the complete training trajectory across all 2000 epochs
- Advanced metrics (diversity, unique positions) are updated every 100 epochs
- The final evaluation at the end of training includes detailed metrics at multiple tolerance levels

## Visualization

The framework includes visualization capabilities:

- Energy landscape visualization
- Sample overlay on energy landscapes
- Comparative visualization between standard and GFN-guided samples
- Saved in the `visualizations` subdirectory within each experiment's output directory

## Usage

To run a single experiment, use the provided training script with appropriate parameters:

```bash
python -m gfn_diffusion.energy_sampling.train --epochs 2000 \
    --batch_size 64 \
    --hidden_dim 128 \
    --latent_dim 2 \
    --num_timesteps 1000 \
    --device cpu \
    --schedule_type linear \
    --energy <energy_function> \
    --guidance_scale <guidance_scale> \
    --loss_type tb_avg \
    --output_dir <output_directory> \
    --wandb \
    --run_name <experiment_name> \
    --eval_interval 100
```

To run comprehensive ablation studies:

```bash
./run_ablation_studies.sh
```

This will execute a series of experiments testing different configurations and save results to `results/ablation_studies/`.

## Results

The GFlowNet-guided diffusion approach shows promising results:

- Successfully concentrates samples in low-energy regions
- Scales effectively with guidance strength
- Handles complex multimodal energy landscapes
- Maintains sample diversity even with strong guidance

Key findings from our guidance scale experiments:

- Small guidance (1.0): 30 unique positions at tolerance 0.1, higher diversity
- Medium guidance (2.5): 21 unique positions at tolerance 0.1, balanced diversity
- High guidance (5.0): 17 unique positions at tolerance 0.1, concentrated on minima

Results are logged to Weights & Biases for comprehensive tracking and visualization.
