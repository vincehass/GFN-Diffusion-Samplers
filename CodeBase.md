<think>
Let me analyze the code in the given files to understand the system and create a comprehensive documentation.

From a first glance, this looks like an implementation of a Generative Flow Network (GFN) approach to energy-based sampling. GFNs are a class of generative models that learn to sample from a target distribution by learning flow networks. They seem to be applying this approach to energy-based models.

The key components appear to be:

1. **Architecture models** (architectures.py): Contains neural network architectures for various components of the system.
2. **GFN model** (gfn.py): Implements the main GFN model that handles trajectory generation and sampling.
3. **GFlowNet losses** (gflownet_losses.py): Different loss functions used to train the GFN.
4. **Evaluation functions** (evaluations.py): Functions to evaluate the performance of the model.
5. **Training loop** (train.py): Main training code that ties everything together.
6. **Sample metrics** (sample_metrics.py): Metrics to compare the quality of samples.

From the paper link, it seems this is related to the paper "Energy-Based Sampling with Generative Flow Networks" which looks at using GFNs for energy-based sampling.

Let me go through each file in more detail to understand their functionality and how they relate to each other.
</think>

# Energy-Based Sampling with Generative Flow Networks

This repository implements the approach described in the paper [Energy-Based Sampling with Generative Flow Networks](https://arxiv.org/html/2402.05098v4). The codebase provides a framework for sampling from energy-based models using Generative Flow Networks (GFNs).

## Overview

Generative Flow Networks (GFNs) are a class of generative models that learn to sample from complex target distributions by parameterizing a stochastic process that incrementally builds samples. This implementation focuses on using GFNs for sampling from energy-based models, where the target distribution is defined by an energy function.

The key components of the codebase are:

1. **Neural Network Architectures** (`architectures.py`): Various neural network components used in the GFN
2. **GFN Model** (`gfn.py`): The core implementation of the Generative Flow Network
3. **Training Losses** (`gflownet_losses.py`): Different loss functions for training the GFN
4. **Evaluation Metrics** (`evaluations.py`): Functions to evaluate model performance
5. **Training Loop** (`train.py`): Main training script
6. **Sample Metrics** (`sample_metrics.py`): Metrics to evaluate sample quality

## Detailed Documentation

### Neural Network Architectures (`architectures.py`)

This file contains various neural network modules used throughout the codebase:

- **Time Encoding Models** (`TimeEncoding`, `TimeEncodingPIS`): Encode time steps into embeddings
- **State Encoding Models** (`StateEncoding`, `StateEncodingPIS`): Encode state vectors into embeddings
- **Policy Models** (`JointPolicy`, `JointPolicyPIS`): Predict actions/transitions given states and times
- **Flow Models** (`FlowModel`, `FlowModelPIS`): Model flow components for the GFN
- **Langevin Scaling Models** (`LangevinScalingModel`, `LangevinScalingModelPIS`): Scale Langevin dynamics

There are two variants of each architecture - standard and PIS (Parameter-Independent Sampling) versions. The PIS architectures use more parameter-efficient designs.

### GFN Model (`gfn.py`)

The `GFN` class is the core of the implementation, implementing a Generative Flow Network for energy-based sampling. Key features include:

- **Forward and Backward Trajectories**: Generate trajectories from initial state to target distribution and vice versa
- **Langevin Dynamics Integration**: Optional integration of Langevin dynamics for improved sampling
- **Parameterized Backward Process**: Options to learn the backward process versus using a fixed analytical form
- **Time-Dependent Flow Components**: Flow components that vary with the time step
- **Exploratory Sampling**: Support for exploratory sampling with controllable noise levels

Key methods:

- `predict_next_state`: Predicts the next state given current state and time
- `get_trajectory_fwd`: Generates a forward trajectory from initial state
- `get_trajectory_bwd`: Generates a backward trajectory from a target state
- `sample`: Samples from the learned distribution

### Training Losses (`gflownet_losses.py`)

This file implements various loss functions for training GFNs:

- **Trajectory Balance (TB)**: `fwd_tb`, `bwd_tb` - Ensures consistency between forward and backward trajectories
- **Detailed Balance (DB)**: `db` - Enforces detailed balance conditions between adjacent states
- **Subtrajectory Balance (SubTB)**: `subtb` - Applies trajectory balance to all subtrajectories
- **Maximum Likelihood Estimation**: `bwd_mle` - Maximizes likelihood of samples
- **Parameter-Independent Sampling (PIS)**: `pis` - Special training objective for PIS variants

Each loss function has specific properties and trade-offs in terms of stability, sample efficiency, and convergence speed.

### Evaluation Functions (`evaluations.py`)

Functions to evaluate model performance:

- **log_partition_function**: Estimates the log partition function (normalization constant)
- **mean_log_likelihood**: Computes mean log-likelihood of data under the model
- **get_sample_metrics**: Computes various metrics comparing generated samples to ground truth

### Training Loop (`train.py`)

The main training script that:

1. Initializes the energy model, GFN model, and optimizers
2. Implements training loops for forward and/or backward training
3. Handles evaluation, visualization, and logging
4. Supports various training configurations and hyperparameters

Key components:

- **Energy Models**: Different energy functions to sample from (GMMs, funnels, many-well)
- **Replay Buffer**: For storing and reusing generated samples
- **Local Search**: Optional local search with Langevin dynamics to improve samples
- **Visualization**: Tools for visualizing the learned distribution

### Sample Metrics (`sample_metrics.py`)

Implements various metrics to evaluate sample quality:

- **Wasserstein Distance**: Measures transport distance between distributions
- **Maximum Mean Discrepancy (MMD)**: Kernel-based distance between distributions
- **Mean/Median Statistics**: Simpler statistics on means and medians

## Key Concepts

### Generative Flow Networks (GFNs)

GFNs learn to sample from a target distribution by parameterizing a stochastic process that incrementally builds samples. They're trained to satisfy certain flow-matching conditions that ensure the correct target distribution.

### Energy-Based Sampling

Energy-based sampling involves sampling from a distribution defined by an energy function:
p(x) ∝ exp(-E(x)) or equivalently p(x) ∝ R(x) where R(x) = exp(-E(x)) is the reward function.

### Trajectory Balance (TB)

A training objective for GFNs that enforces consistency between forward and backward trajectories:
log P_F(τ) + log F_0 = log P_B(τ) + log R(x)

### Langevin Dynamics

A sampling technique that uses gradient information of the target density to guide the sampling process.

## Usage

The codebase supports various training configurations through command-line arguments:

```bash
python train.py --energy 9gmm --mode_fwd tb --mode_bwd tb --both_ways --local_search
```

Key parameters include:

- `--energy`: Target energy function (9gmm, 25gmm, hard_funnel, easy_funnel, many_well)
- `--mode_fwd`: Forward training mode (tb, tb-avg, db, subtb, pis)
- `--mode_bwd`: Backward training mode (tb, tb-avg, mle)
- `--both_ways`: Train in both forward and backward directions
- `--local_search`: Use local search with Langevin dynamics
- `--langevin`: Incorporate Langevin dynamics into the GFN
- `--pis_architectures`: Use Parameter-Independent Sampling architectures

## Conclusion

This codebase provides a flexible framework for energy-based sampling using Generative Flow Networks. It implements several training objectives, model architectures, and sampling strategies, allowing for experimentation with different approaches to this challenging problem.

## Energy Sampling

This is a codebase for energy sampling with generative flow networks.

### Buffer

The buffer is a class that stores the generated samples and the rewards.

### Training Loop

The training loop is a class that implements the training loop for the energy sampling.

### Evaluation

The evaluation is a class that implements the evaluation for the energy sampling.

### Utils

The utils are a class that implements the utils for the energy sampling.

### Energy Sampling

The energy sampling is a class that implements the energy sampling.

## running the code

```bash
python energy_sampling/train.py --energy 9gmm --mode_fwd tb --mode_bwd tb --both_ways --local_search --batch_size 300 --buffer_size 100000 --epochs 10000
```

<think>
I'll create a comprehensive command to train the model with optimal parameters for speed and quality. I'll include key parameters for:

1. Energy model selection
2. Training modes
3. Buffer parameters
4. Performance optimizations
5. Logging settings

Let's make sure to include the newly added arguments like --use_amp, --num_workers, etc.
</think>

# Command to Run Training with Optimized Settings

Here's a comprehensive command to train the model with optimized settings for both performance and quality:

```bash
python energy_sampling/train.py \
  --energy 9gmm \
  --mode_fwd tb \
  --mode_bwd tb \
  --both_ways \
  --local_search \
  --batch_size 1024 \
  --buffer_size 100000 \
  --epochs 20000 \
  --lr_policy 1e-4 \
  --lr_flow 1e-4 \
  --lr_back 1e-4 \
  --use_weight_decay \
  --weight_decay 1e-5 \
  --use_amp \
  --num_workers 4 \
  --grad_clip 1.0 \
  --log_interval 50 \
  --exploratory \
  --exploration_factor 1.0 \
  --alpha 0.5 \
  --prioritized reward \
  --langevin \
  --learned_variance
```

## Variations for Different Use Cases:

### For Faster Training (Lower Quality)

```bash
python energy_sampling/train.py \
  --energy 9gmm \
  --mode_fwd tb \
  --mode_bwd tb \
  --both_ways \
  --batch_size 2048 \
  --buffer_size 50000 \
  --epochs 10000 \
  --use_amp \
  --num_workers 8 \
  --log_interval 100
```

### For Higher Quality (Slower Training)

```bash
python energy_sampling/train.py \
  --energy 9gmm \
  --mode_fwd subtb \
  --mode_bwd tb \
  --both_ways \
  --local_search \
  --batch_size 512 \
  --buffer_size 200000 \
  --epochs 30000 \
  --lr_policy 5e-5 \
  --lr_flow 5e-5 \
  --lr_back 5e-5 \
  --use_weight_decay \
  --weight_decay 1e-6 \
  --langevin \
  --learned_variance \
  --t_scale 10.0 \
  --beta 5.0
```

### For More Complex Energy Landscapes

```bash
python energy_sampling/train.py \
  --energy many_well \
  --mode_fwd subtb \
  --mode_bwd tb \
  --both_ways \
  --local_search \
  --batch_size 1024 \
  --buffer_size 150000 \
  --epochs 25000 \
  --lr_policy 1e-4 \
  --lr_flow 1e-4 \
  --lr_back 1e-4 \
  --use_amp \
  --num_workers 4 \
  --grad_clip 1.0 \
  --langevin \
  --learned_variance \
  --exploratory \
  --exploration_factor 1.5 \
  --alpha 0.7 \
  --prioritized reward
```

These commands should give you a good starting point for training with different priorities. The optimized code should now run significantly faster while maintaining or improving quality.
