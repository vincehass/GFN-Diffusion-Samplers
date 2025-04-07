# GFN-Diffusion

Implementation of a diffusion model guided by GFlowNets (GFN-Diffusion) for energy-based sampling and conditional generation.

## Overview

GFN-Diffusion combines the strengths of:

- **Diffusion Models**: For high-quality sample generation
- **GFlowNets**: For targeting specific energy functions or distributions

This implementation provides both:

1. **Unconditional Sampling**: For energy-based sampling problems
2. **Conditional Sampling**: For tasks like variational inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gfn-diffusion.git
cd gfn-diffusion

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
gfn_diffusion/
├── __init__.py
├── energy_sampling/
│   ├── __init__.py
│   ├── models.py      # UNet, ConditionalUNet, GFNDiffusion classes
│   └── train.py       # Training loops for different scenarios
├── vae/
│   ├── energies/      # VAE-based energy functions
│   └── train.py       # VAE experiment training
├── utils/
│   └── wandb_utils.py # Utilities for Weights & Biases logging
├── example.py         # Example usage script
├── run_with_wandb.py  # Script for running experiments with wandb
└── README.md          # Documentation
```

## Usage

### Basic Usage

```python
from gfn_diffusion.energy_sampling.models import UNet, DiffusionSchedule, GFNDiffusion

# Define energy function (example using GMM energy function from the package)
from gfn_diffusion.energy_sampling.models import gmm_energy

# Setup energy function (e.g., Gaussian Mixture Model)
means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
weights = torch.tensor([0.5, 0.5])
energy_fn = lambda x: gmm_energy(x, means, weights, std=0.5)

# Setup diffusion model
model = UNet(input_dim=2, hidden_dim=64, output_dim=2)
schedule = DiffusionSchedule(num_timesteps=100, schedule_type="linear")

diffusion = GFNDiffusion(
    model=model,
    dim=2,
    schedule=schedule,
    energy_fn=energy_fn,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Sample from the model
samples = diffusion.sample(
    num_samples=100,
    use_gfn=True,  # Set to False for standard diffusion
    guidance_scale=1.0,
    num_steps=50
)
```

### Running the Example

The example script demonstrates both unconditional and conditional sampling:

```bash
python -m gfn_diffusion.example
```

This will:

1. Run an unconditional example for sampling from a 25-component GMM energy function
2. Run a conditional example showing how GFN guidance affects sampling for different conditioning values
3. Generate visualization plots for both examples

## Experiment Tracking with Weights & Biases

This implementation includes comprehensive support for tracking experiments with [Weights & Biases](https://wandb.ai/) (wandb), a platform for experiment tracking, model visualization, and collaboration.

### Setup

To use Weights & Biases with this project:

1. **Install wandb** (if not already installed):

```bash
pip install wandb
```

2. **Login to wandb**:

```bash
wandb login
```

### Running All Diffusion Models with wandb

The simplest way to run all diffusion models with wandb tracking is using the provided bash script:

```bash
# Run all models (both energy-based and VAE-based) with wandb tracking
./gfn_diffusion/run_wandb_experiments.sh --experiment both
```

> **Important**: Run this command from the root directory of the project, not from inside the gfn_diffusion directory.

This script automatically:

- Initializes wandb for each experiment
- Tracks metrics during training
- Logs visualizations of energy functions and samples
- Compares results with and without GFN guidance

### Advanced Features with run_advanced_gfn_experiments.sh

For advanced GFN-Diffusion experiments with enhanced features, use:

```bash
./run_advanced_gfn_experiments.sh --experiment both --device cpu
```

This script enables several advanced features that improve sample quality and training stability:

#### Advanced Features Included:

- **Langevin dynamics**: Combines gradient information with random noise to generate better samples
- **Local search**: Performs gradient-based optimization to find lower energy states
- **Prioritized replay buffer**: Stores samples with priorities based on energy values for more efficient training
- **Exploratory sampling**: Encourages the model to explore different regions of the state space
- **Learning rate scheduling**: Adaptively adjusts Langevin dynamics step size based on acceptance rate
- **Gradient clipping**: Improves training stability by preventing gradient explosion

#### Command-line Options:

```
Usage: ./run_advanced_gfn_experiments.sh [options]

Options:
  --experiment energy|vae|both    Choose which experiment to run
  --device DEVICE                 Specify the computing device (cuda, cpu)
  --num_layers LAYERS             Set the number of UNet layers
  --wandb_entity ENTITY           Set Weights & Biases username
  --wandb_project PROJECT         Set Weights & Biases project name
  --offline                       Run in offline mode (no wandb connection)
```

#### Run Naming Convention

The scripts automatically generate descriptive run names that encode key experiment parameters:

**For Energy-based experiments:**

```
energy_gmm_uncond_gs1.0_gpu_20240505_123456
```

This naming convention includes:

- Experiment type: `energy`
- Energy function: `gmm` (Gaussian Mixture Model)
- Conditioning: `uncond` (or `cond4` for conditional with 4 conditions)
- Guidance scale: `gs1.0`
- Device: `gpu` or `cpu`
- Timestamp: `YYYYMMDD_HHMMSS`

**For VAE experiments:**

```
vae_training_h128_gpu_20240505_123456  # VAE training run
vae_gfn_gs1.0_h128_gpu_20240505_123456  # GFN-guided VAE sampling
```

These descriptive names make it easy to identify and compare different experiments in the wandb dashboard.

#### Default User Configuration

The scripts are configured with the following defaults:

- **Username (Entity)**: `nadhirvincenthassen`
- **Project Name**: `gfn-diffusion` (and `gfn-diffusion-vae` for VAE experiments)

These defaults ensure that all experiments are organized under the same user account and projects.

#### Customizing Your Runs

The bash script offers many customization options:

```bash
# Run energy models with stronger GFN guidance
./gfn_diffusion/run_wandb_experiments.sh --experiment energy --guidance-scale 2.5

# Run VAE models with more epochs and a custom run name prefix
./gfn_diffusion/run_wandb_experiments.sh --experiment vae --epochs 10000 --prefix "vae_long_run"

# Run conditional models with 8 different conditions
./gfn_diffusion/run_wandb_experiments.sh --experiment energy --conditional --num-conditions 8

# Run all models with GPU selection and custom batch size
./gfn_diffusion/run_wandb_experiments.sh --experiment both --device cuda:0 --batch-size 256
```

#### First-Time Setup Options

For first-time users, these options can be helpful:

```bash
# Install wandb if not already installed and update requirements.txt
./gfn_diffusion/run_wandb_experiments.sh --experiment both --install-wandb --update-requirements

# Run on CPU explicitly (for systems without CUDA)
./gfn_diffusion/run_wandb_experiments.sh --experiment both --device cpu
```

> **Note**: You don't need to specify the wandb entity or project name in the command line. The scripts automatically use `nadhirvincenthassen` as the entity and `gfn-diffusion` as the project name.

> **Note**: The scripts automatically detect if CUDA is available and will fall back to CPU if it's not. You'll see a warning message if this happens.

#### Available Parameters

View all available parameters with:

```bash
./gfn_diffusion/run_wandb_experiments.sh --help
```

Key parameters include:

- `--experiment`: Choose `energy`, `vae`, or `both`
- `--device`: Select computing device (`cuda`, `cuda:0`, `cpu`, etc.)
- `--energy`: Energy function type:
  - `gmm` - 25-component Gaussian Mixture Model (maps to '25gmm' internally)
  - `rings` - Concentric rings distribution (maps to 'many_well' internally)
  - `moons` - Two moons distribution (maps to '3gmm' internally)
- `--conditional`: Enable conditional modeling
- `--guidance-scale`: Control the strength of GFN guidance
- `--epochs` and `--batch-size`: Adjust training parameters
- `--hidden-dim`: Model capacity

### Direct Python Script Usage

For more advanced customization, you can use the Python script directly:

```bash
# Run energy sampling models
python gfn_diffusion/run_with_wandb.py --experiment_type energy --wandb_project gfn-diffusion \
    --energy_type gmm --guidance_scale 2.0 --epochs 5000

# Run VAE models with specific settings
python gfn_diffusion/run_with_wandb.py --experiment_type vae --wandb_project gfn-diffusion-vae \
    --hidden_dim 256 --batch_size 64
```

### Tracked Metrics and Visualizations

When using wandb, the following metrics and visualizations are tracked:

1. **Training Metrics**:

   - Loss curves
   - Learning rates

2. **Energy Function Visualizations**:

   - Contour plots of energy functions
   - GMM components

3. **Sample Visualizations**:

   - Samples with and without GFN guidance
   - Comparison plots showing the effect of guidance

4. **VAE Experiment Visualizations**:
   - Latent space samples
   - Reconstructed images
   - Comparison of VAE samples with and without GFN guidance

### Customizing Wandb Logging

You can customize wandb logging by modifying the `utils/wandb_utils.py` file, which contains functions for initializing wandb, logging energy functions, samples, and comparisons.

## Key Components

### Models

- **UNet**: Base model for unconditional diffusion
- **ConditionalUNet**: Extended model that supports conditioning
- **DiffusionSchedule**: Controls the noise schedule during diffusion
- **GFNDiffusion**: Main class that implements GFN-guided diffusion sampling

### Energy Functions

The package includes a `gmm_energy` function for Gaussian Mixture Models, but you can define any energy function that takes tensor inputs and returns scalar energy values.

## GFN Guidance

The GFN guidance is controlled by:

- `use_gfn`: Enable/disable GFN guidance
- `guidance_scale`: Controls the strength of guidance (higher values = stronger guidance)

The diffusion process is guided by computing gradients of the energy function with respect to the current state, allowing the sampler to target specific energy functions or distributions.

## VAE Experiment

The package also includes implementation of the VAE experiment from the GFN-Diffusion paper. This experiment involves training a diffusion model to sample from the latent space of a variational autoencoder (VAE).

### VAE Pretraining

First, you need to pretrain a VAE model:

```bash
cd gfn_diffusion/vae
python energies/vae.py
```

This will train a VAE on MNIST and save the model to `models/vae_best.pt`.

### VAE-based GFN-Diffusion

After pretraining the VAE, you can train a GFN-Diffusion model to sample from the VAE's latent space:

```bash
# TB + Exploratory + Local Search
python train.py \
  --energy vae --pis_architectures --zero_init --clipping \
  --mode_fwd cond-tb-avg --mode_bwd cond-tb-avg --repeats 5 \
  --lr_policy 1e-3 --lr_flow 1e-1 --lr_back 1e-3 \
  --exploratory --exploration_wd --exploration_factor 0.1 \
  --both_ways --local_search \
  --max_iter_ls 500 --burn_in 200 \
  --buffer_size 90000 --prioritized rank --rank_weight 0.01 \
  --ld_step 0.001 --ld_schedule --target_acceptance_rate 0.574
```

The VAE experiment demonstrates how GFN-Diffusion can be applied to sample from complex posterior distributions, which is useful for tasks like variational inference.

### Comparing VAE-based and General Conditional Sampling

There are important differences between the VAE-based conditional sampling and the general conditional sampling approach in the energy sampling module:

#### VAE-based Conditional Sampling

- **Conditioning Mechanism**: Implicit conditioning through the VAE's learned manifold
- **Energy Definition**: Based on VAE reconstruction loss and KL divergence (negative ELBO)
- **Purpose**: Focused on variational inference applications (sampling from posteriors)
- **Architecture**: Uses standard UNet with VAE-specific energy functions

#### Energy Sampling Conditional Model

- **Conditioning Mechanism**: Explicit conditioning through direct input concatenation
- **Energy Definition**: Can be any arbitrary energy function (GMM, funnel, etc.)
- **Purpose**: General purpose, applicable to a wide range of conditional generation tasks
- **Architecture**: Uses `ConditionalUNet` with specialized condition input handling

The VAE experiment represents a specific application of GFN-Diffusion for posterior sampling, while the energy sampling conditional approach provides a more general framework for conditioning the generation process on arbitrary inputs.

The conditional sampling approaches in the GFN-Diffusion implementation differ significantly between the VAE experiment and the energy sampling module, though they might seem similar at first.

### VAE-based Conditional Sampling

In the VAE experiment (`gfn_diffusion/vae/`):

- It focuses on sampling from the VAE's latent space, which is constrained by the learned distribution
- The conditioning is based on the VAE's encoder mapping high-dimensional data to the latent space
- The energy function is defined as the negative ELBO (evidence lower bound)
- It uses the pretrained VAE's parameters to define the energy landscape
- The goal is to sample from complex posterior distributions in the latent space

From the implementation in `vae_energy.py`, the energy is computed as:

```python
# Compute energy (negative ELBO)
energy = recon_loss + self.beta * kl_div
```

### Energy Sampling Conditional Model

In the energy sampling module (`energy_sampling/`):

- The conditional model directly conditions the UNet on external data
- It uses a `ConditionalUNet` architecture that concatenates condition vectors with input
- The energy function can be any function (GMM, funnel, etc.), not necessarily VAE-based
- Conditions are explicitly provided inputs (like class labels or attributes)
- The goal is to guide sampling toward specific regions based on the condition

From the `ConditionalUNet` implementation, conditioning is applied directly:

```python
# Encode condition
cond_emb = self.condition_encoder(condition)

# Concatenate input and condition
x = torch.cat([x, cond_emb], dim=-1)
```

### Key Differences

1. **Conditioning Mechanism**:

   - VAE: Implicit conditioning through the VAE's learned manifold
   - Energy Sampling: Explicit conditioning through input concatenation

2. **Energy Definition**:

   - VAE: Based on VAE reconstruction and KL divergence
   - Energy Sampling: Can be any arbitrary energy function

3. **Purpose**:

   - VAE: Focused on variational inference applications (sampling posteriors)
   - Energy Sampling: More general purpose, applicable to a wider range of conditional generation tasks

4. **Architecture**:
   - VAE: Uses standard UNet but with VAE-specific energy
   - Energy Sampling: Uses a specialized architecture with condition inputs

The VAE experiment demonstrates a specific application of GFN-Diffusion for sampling from complex posteriors in the latent space of a VAE, while the energy sampling conditional approach is more general and can be applied to various conditional generation tasks.

## References

- [GFlowNets for Generative Energy-Based Modeling](https://arxiv.org/abs/2201.13259)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

## Experimentation

The repository includes two bash scripts to help you run and test the GFN-Diffusion implementation. These scripts provide a structured way to validate the implementation and run comprehensive experiments.

### Testing the Implementation

The `test_models.sh` script performs lightweight tests on all components to ensure they work correctly:

```bash
# Run the test script
./gfn_diffusion/test_models.sh
```

This script:

1. **Tests Energy Sampling Components**:

   - UNet and DiffusionSchedule classes
   - GFNDiffusion sampling and training
   - ConditionalUNet architecture
   - Example script with minimal configuration

2. **Tests VAE Experiment Components**:
   - VAE model encoding, decoding, and sampling
   - VAE energy function
   - Visualization utilities
   - Replay buffer, Langevin dynamics, and local search

The tests use minimal configurations for quick validation and generate visualizations in the `test_results` directory to verify the output.

### Running Comprehensive Experiments

The `run_models.sh` script runs all the GFN-Diffusion models with comprehensive configurations:

```bash
# Run the full experiments
./gfn_diffusion/run_models.sh
```

This script includes:

1. **Energy Sampling Experiments**:

   - Basic example script
   - Unconditional GMM energy sampling
   - Multi-modal energy with TB + Exploratory
   - Conditional energy sampling

2. **VAE Experiments**:
   - VAE pretraining on MNIST
   - Basic VAE-based GFN-Diffusion
   - Advanced VAE-based GFN-Diffusion with TB + Exploratory + LS

Results from these experiments are saved in the `results` directory with appropriate subdirectories for each experiment type.

### Experiment Parameters

The experiments in `run_models.sh` use various configurations:

#### Energy Sampling Parameters:

- **GMM Energy**: Simple Gaussian mixture model with guidance scale of 2.0
- **Multi-modal Energy**: Uses TB (Trajectory Balance) with Exploratory objectives
- **Conditional Model**: Generates samples conditioned on 4-dimensional vectors

#### VAE Experiment Parameters:

- **VAE Pretraining**: 20 epochs with 2D latent space and β=1.0
- **Basic GFN-Diffusion**: Uses TB with minimal settings
- **Advanced GFN-Diffusion**: Includes:
  - Trajectory Balance (TB) with exploratory objectives
  - Local Search (LS) for finding lower energy states
  - Prioritized replay buffer for sampling
  - Langevin dynamics with adaptive step size scheduling

### Advanced Experiments with run_advanced_gfn_experiments.sh

For the most advanced configurations, use the `run_advanced_gfn_experiments.sh` script:

```bash
./run_advanced_gfn_experiments.sh --experiment both
```

This script enables all the advanced features discussed in the paper:

1. **Langevin dynamics with adaptive step size**: Adjusts the step size based on acceptance rate
2. **Local search optimization**: Finds lower energy states by gradient descent
3. **Prioritized replay buffer**: Focuses training on interesting samples
4. **Exploratory objective**: Prevents mode collapse and improves sample diversity
5. **Trajectory Balance**: Ensures consistent forward and backward flows
6. **Gradient clipping**: Improves training stability

The script provides detailed logs and visualizations to track the performance of these advanced methods. For optimal results, use multiple UNet layers:

```bash
./run_advanced_gfn_experiments.sh --experiment both --num_layers 4
```

The combination of these advanced techniques significantly improves the quality of samples, especially for complex multimodal distributions.

### Customizing Experiments

You can modify these scripts to:

1. Adjust hyperparameters for different energy functions
2. Change the VAE architecture or training settings
3. Experiment with different GFN guidance techniques
4. Test the effect of various sampling strategies

Looking at the code, when the `--experiment both` option is specified, it runs two distinct diffusion model experiments:

1. **Energy Sampling Diffusion** - This is the model we just fixed, which uses GFlowNet guidance over energy functions (like the 25-mode Gaussian Mixture Model) to guide the diffusion process. This model works directly in data space.

2. **VAE-based Diffusion** - This is a more advanced model that works with a pre-trained Variational Autoencoder (VAE). It operates in the VAE's latent space rather than directly in data space. This approach allows the model to handle more complex distributions and is closer to the advanced models described in the README.

The main differences between the two are:

- The energy sampling model operates directly on 2D data points
- The VAE-based model operates in the latent space of a pre-trained VAE, which allows it to handle more complex distributions
- The VAE model can be used for conditional generation based on attributes

Both models implement the GFN (GFlowNet) guidance mechanism, but they differ in where they operate (data space vs latent space) and their complexity.

The run_wandb_experiments.sh script only includes implementations for:
Energy-based diffusion (direct diffusion in data space)
VAE-based diffusion (diffusion in latent space)
The advanced features like Trajectory Balance, Local Search, prioritized replay, and Langevin dynamics would likely be in a separate implementation or need to be added. These are research-level enhancements to the basic GFN-Diffusion algorithm.
