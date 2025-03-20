# GFN-Diffusion Samplers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Implementation of Improved Off-Policy Training of Diffusion Samplers with Generative Flow Networks (GFNs), supporting both conditional and unconditional sampling paradigms.

![Manywell Samples](assets/manywell_samples.png)

## Key Features

- 🌀 Integration of GFN trajectory balance with diffusion models
- ⚡ Efficient off-policy training with replay buffers
- 🔀 Support for both conditional and unconditional sampling
- 📈 Multiple energy functions and exploration strategies
- 🧪 Reproducible experiments on 25GMM, Manywell, and VAE tasks

## Installation

```bash
git clone https://github.com/vincehass/gfn-diffusion-samplers
cd gfn-diffusion-samplers
pip install -r requirements.txt
```

## Quick Start

### Unconditional Sampling (25GMM)
```python
from src.models import GFNDiffusion
from src.sampling import unconditional

# Initialize model with energy function
model = GFNDiffusion(energy_fn=unconditional.gmm25_energy)

# Sample from trained model
samples = model.sample(batch_size=256, steps=1000)
```

### Conditional Sampling (VAE)
```python
from src.sampling.conditional import vae_energy

# Initialize with VAE components
energy_fn = lambda z: vae_energy(z, x, decoder, prior)
conditional_model = GFNDiffusion(energy_fn=energy_fn)

# Sample latent vectors conditioned on input x
latent_samples = conditional_model.sample(condition=x)
```

## Configuration

Experiment configurations are managed through YAML files:

```yaml
# experiments/configs/manywell.yaml
energy_type: manywell
diffusion_steps: 1000
batch_size: 256
gfn:
  use_trajectory_balance: true
  exploration: local_search
  replay_buffer_size: 100000
```


## Introduction

This repository provides a mathematical and computational framework integrating **Generative Flow Networks (GFNs)** with **Diffusion Models** to enable energy-guided generative modeling. By leveraging **trajectory balance**, **energy-based guidance**, and **diffusion-based denoising**, this framework provides a structured approach for learning generative policies that can produce high-quality samples in an energy-efficient manner.

This work aims to bridge the gap between two powerful paradigms:
- **Diffusion Models:** Which learn to generate samples by gradually denoising random noise.
- **Generative Flow Networks (GFNs):** Which model probabilistic trajectories for generating structured objects proportionally to a reward function.

This repository provides theoretical formulations, algorithmic implementations, and sampling strategies that combine the best of both worlds.

## Background

### Diffusion Models

Diffusion models are a class of probabilistic generative models that learn to reverse a Markovian noise process. They consist of two key phases:

1. **Forward Process (Noise Addition):**
   - A data sample **x₀** is gradually transformed into pure noise **x_T** through a sequence of Gaussian transitions:
     \[
     q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I)
     \]
   - This process ensures that after **T** steps, the data distribution transforms into an isotropic Gaussian **N(0, I)**.

2. **Backward Process (Denoising):**
   - A neural network **p_θ(x_{t-1} | x_t)** is trained to reverse this process and recover **x₀** from noisy samples.
   - The model learns to predict either the noise **ϵ** or the original data point **x₀**.

Diffusion models achieve high-quality sample generation but often require a large number of inference steps. **Energy-based guidance** can help steer this generation process toward desirable outcomes, which is where GFNs play a role.

### Generative Flow Networks (GFNs)

GFNs provide a framework for learning probability distributions over structured objects by treating generation as a sequential decision-making process. Key principles include:

- **Trajectory Balance:** Ensuring consistency between forward (generation) and backward (denoising) probabilities.
- **Energy-Based Guidance:** Using reward functions to bias sampling towards desirable samples.
- **Flow Conservation:** Enforcing conservation of probability mass across transitions.

In the context of diffusion models, GFNs modify the backward process to introduce an **energy-based reward function** that influences sample selection.


## Conditional and Unconditional Sampling

### Unconditional Sampling
In unconditional sampling, the model generates samples purely based on the data distribution learned during training. The backward process follows standard noise prediction:

\[
\tilde{ϵ}_θ(x_t, t) = ϵ_θ(x_t, t)
\]

If energy-based guidance is applied, the predicted noise is modified:
\[
\tilde{ϵ}_θ(x_t, t) = ϵ_θ(x_t, t) - λ ∇_x E(x_t)
\]

This biases the sample generation towards **low-energy (desirable) states** while maintaining the underlying learned distribution.

### Conditional Sampling
In conditional sampling, an external signal **c** (such as a class label) is used to guide the generation. This is done using **classifier-free guidance**, where:

1. Two noise predictions are obtained:
   - **Conditional Prediction:** ϵ_θ(x_t, t, c)
   - **Unconditional Prediction:** ϵ_θ(x_t, t)
2. The two predictions are merged using:
   \[
   \tilde{ϵ}_θ(x_t, t) = ϵ_θ(x_t, t, c) + λ(ϵ_θ(x_t, t, c) - ϵ_θ(x_t, t))
   \]

This allows the model to generate samples that satisfy the conditional constraints while leveraging the robustness of the unconditional model.

## Training

The training objective integrates both the diffusion model loss and the GFN trajectory balance loss:

\[
L_{total} = L_{simple} + \lambda_{GFN} L_{GFN}
\]

where:
- **L_simple**: Standard noise prediction loss.
- **L_GFN**: GFN-based trajectory balance loss.
- **λ_GFN**: Controls the trade-off between denoising accuracy and energy-based guidance.

## Sampling Algorithm

The guided sampling process follows these steps:

1. **Initialize** with pure noise **x_T ~ N(0, I)**.
2. For **t = T, T-1, ..., 1**:
   - Predict noise: **ϵ_θ(x_t, t)**
   - If using energy guidance, compute **∇ₓE(x_t)** and modify **ϵ**.
   - Compute **μ_θ(x_t, t)** based on the modified noise prediction.
   - Sample **x_{t-1}** using the reverse process distribution.
3. **Output** final sample **x₀**.

## Extensions
- **Multi-Objective Energy Functions:** Combining multiple reward criteria.
- **Adaptive Guidance Strength:** Modulating guidance over timesteps.
- **Classifier-Free Guidance:** Using unconditional models for flexibility.


## Key Differences: Conditional vs Unconditional

| Feature | Unconditional | Conditional |
|---------|---------------|-------------|
| Energy Function | E(x) | E(z; x) |
| Target Distribution | p(x) | p(z\|x) |
| Partition Function | Global Z | Per-instance Z(x) |
| Training Objective | Standard TB | Conditional TB |


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

For questions or collaborations, please contact [Your Name] - your.email@institution.edu

## Acknowledgements

- Original paper authors: Marcin Sendera et al.
- Mila Institute for foundational research
- Compute resources provided by [Compute Canada/AIML]

## Citation
If you use this work, please cite:

@article{Hassen2024,
  author    = {Nadhir Hassen},
  title     = {Foundations of Generative Flow Networks for Diffusion Models},
  year      = {2024},
}


