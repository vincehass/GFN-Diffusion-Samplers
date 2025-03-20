# DiffusionRecapGFN

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

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DiffusionRecapGFN.git
   cd DiffusionRecapGFN
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

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

## Conclusion
This repository presents a **GFN-enhanced diffusion model** for energy-based generative modeling. By integrating energy functions, trajectory balance, and flow conservation, this framework enables controlled, high-quality generation with applications in structured design, molecular synthesis, and image generation.

For more details, refer to **DiffusionRecapGFN.pdf** included in this repository.

---

## Citation
If you use this work, please cite:
```
@article{Hassen2024,
  author    = {Nadhir Hassen},
  title     = {Mathematical Formulation of Generative Flow Networks for Diffusion Models},
  year      = {2024},
}
```

