"""
Core models for GFN-guided diffusion.
This module implements:
1. UNet and ConditionalUNet for noise prediction
2. DiffusionSchedule for controlling the diffusion process
3. GFNDiffusion for running the GFN-guided diffusion sampling
4. gmm_energy and other energy functions for unconditional diffusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Callable


def gmm_energy(x, means, weights, std=0.5):
    """
    Gaussian Mixture Model energy function.
    
    Args:
        x: Input tensor of shape [batch_size, dim]
        means: Tensor of means for each Gaussian component [num_components, dim]
        weights: Tensor of weights for each component [num_components]
        std: Standard deviation for each component (scalar or tensor)
    
    Returns:
        energy: Negative log probability (energy) for each input [batch_size]
    """
    # Handle edge cases with input shape
    if x is None:
        return torch.tensor([100.0], device=means.device)  # Default high energy
    
    # Reshape x if necessary to ensure it's 2D [batch_size, dim]
    orig_shape = x.shape
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension if missing
    elif len(x.shape) > 2:
        # If we have a higher-dimensional tensor, reshape to [batch_size, dim]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        print(f"Reshaped gmm_energy input from {orig_shape} to {x.shape}")
    
    # Handle dimension mismatch
    if means.shape[1] != x.shape[1]:
        min_dim = min(means.shape[1], x.shape[1])
        means = means[:, :min_dim]
        x = x[:, :min_dim]
        print(f"Dimension mismatch in gmm_energy. Using first {min_dim} dimensions.")
    
    # Check for NaN/Inf and replace them
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("NaN/Inf detected in gmm_energy input. Cleaning...")
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
    
    # Reshape for broadcasting
    x_expanded = x.unsqueeze(1)  # [batch_size, 1, dim]
    means_expanded = means.unsqueeze(0)  # [1, num_components, dim]
    
    # Calculate squared distances
    squared_dists = ((x_expanded - means_expanded) ** 2).sum(dim=-1)  # [batch_size, num_components]
    
    # Calculate log probabilities for each component
    log_probs = -0.5 * squared_dists / (std ** 2) - math.log(std) * means.shape[1]
    
    # Add log weights and use logsumexp for numerical stability
    log_probs = log_probs + torch.log(weights).unsqueeze(0)
    log_total_prob = torch.logsumexp(log_probs, dim=1)
    
    # Return negative log probability (energy)
    energy = -log_total_prob
    
    # Ensure energy is not NaN or Inf
    if torch.isnan(energy).any() or torch.isinf(energy).any():
        print("Warning: NaN/Inf in gmm_energy output. Replacing with high energy.")
        nan_inf_mask = torch.isnan(energy) | torch.isinf(energy)
        energy[nan_inf_mask] = 100.0  # Default high energy
    
    return energy


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """
    Convolutional block with residual connection.
    """
    def __init__(self, in_channels, out_channels, time_embedding_dim=None):
        super().__init__()
        self.time_mlp = (
            nn.Linear(time_embedding_dim, out_channels)
            if time_embedding_dim is not None
            else None
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
    def forward(self, x, time_emb=None):
        residual = self.residual_conv(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = time_emb[..., None, None]
            x = x + time_emb
            
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        
        return x + residual


class MLPBlock(nn.Module):
    """
    MLP block for low-dimensional data.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, time_embedding_dim=None):
        super().__init__()
        
        self.time_mlp = (
            nn.Linear(time_embedding_dim, hidden_dim)
            if time_embedding_dim is not None
            else None
        )
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Store dimensions for debugging
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Create residual connection if dimensions don't match
        self.residual_fc = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity()
        )
        
    def forward(self, x, time_emb=None):
        # Store original dimensions for debugging
        if hasattr(torch, 'jit') and not torch.jit.is_scripting():
            print(f"MLPBlock input shape: {x.shape}, time_emb shape: {None if time_emb is None else time_emb.shape}")
            print(f"MLPBlock dims: in={self.in_dim}, hidden={self.hidden_dim}, out={self.out_dim}")
        
        # Apply residual connection with proper dimensions
        residual = self.residual_fc(x)
        
        # First layer
        x = self.fc1(x)
        x = self.norm(x)
        x = F.silu(x)
        
        # Add time embedding if provided
        if time_emb is not None and self.time_mlp is not None:
            # Transform time embedding
            time_emb = self.time_mlp(time_emb)
            
            # Make sure time_emb has the same batch dimension as x
            if x.shape[0] != time_emb.shape[0]:
                # If there's a batch size mismatch, broadcast time_emb
                if time_emb.shape[0] == 1:
                    time_emb = time_emb.repeat(x.shape[0], 1)
                else:
                    raise ValueError(f"Cannot broadcast time embedding of shape {time_emb.shape} to match input of shape {x.shape}")
            
            # Make sure dimensions align properly for addition
            if len(x.shape) > len(time_emb.shape):
                # Add dimensions to time_emb to match x
                for _ in range(len(x.shape) - len(time_emb.shape)):
                    time_emb = time_emb.unsqueeze(-1)
            
            # Add time embedding to x
            x = x + time_emb
            
        # Second layer
        x = self.fc2(x)
        
        # Add residual connection
        return x + residual


class UNet(nn.Module):
    """
    UNet model for diffusion models with low-dimensional data.
    
    Args:
        input_dim: Dimension of input data
        hidden_dim: Base hidden dimension
        output_dim: Dimension of output (usually same as input_dim)
        time_dim: Dimension of time embeddings
        num_layers: Number of down/up layers
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, time_dim=128, num_layers=4):
        super().__init__()
        
        # Time embedding
        self.time_embeddings = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Store network dimensions for later debugging
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        current_dim = hidden_dim
        
        # Pre-compute dimensions for down and up paths
        self.down_dims = [current_dim]
        
        # Scale factor for how much dimension increases at each layer
        scale_factor = 2
        
        # Calculate down dimensions
        for i in range(num_layers):
            next_dim = current_dim * scale_factor
            self.down_dims.append(next_dim)
            current_dim = next_dim
        
        # Create down blocks
        for i in range(num_layers):
            in_dim = self.down_dims[i]
            out_dim = self.down_dims[i+1]
            self.down_blocks.append(MLPBlock(in_dim, out_dim, out_dim, time_dim))
            
        # Middle block
        middle_in_dim = self.down_dims[-1]
        middle_hidden_dim = middle_in_dim * scale_factor
        self.middle_block = MLPBlock(middle_in_dim, middle_hidden_dim, middle_in_dim, time_dim)
            
        # Calculate up dimensions
        self.up_dims = []
        for i in range(num_layers):
            # Skip connection + previous layer output
            skip_dim = self.down_dims[-(i+2)]  # Get corresponding dimension from down path
            in_dim = self.down_dims[-(i+1)] + skip_dim  # Current + skip connection
            out_dim = skip_dim
            self.up_dims.append((in_dim, out_dim))
            
        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_dim, out_dim = self.up_dims[i]
            self.up_blocks.append(MLPBlock(in_dim, in_dim, out_dim, time_dim))
            
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Log dimensions
        print(f"UNet dimensions:")
        print(f"  Down dims: {self.down_dims}")
        print(f"  Up dims: {self.up_dims}")
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Down path
        residuals = [x]
        for i, block in enumerate(self.down_blocks):
            x = block(x, t_emb)
            residuals.append(x)
            
        # Middle block
        x = self.middle_block(x, t_emb)
        
        # Up path with skip connections
        for i, block in enumerate(self.up_blocks):
            residual = residuals.pop()
            x = torch.cat([x, residual], dim=-1)
            x = block(x, t_emb)
            
        # Output projection
        x = self.output_proj(x)
        
        return x


class ConditionalUNet(nn.Module):
    """
    UNet model with conditioning for conditional diffusion models.
    
    Args:
        input_dim: Dimension of input data (usually latent space)
        condition_dim: Dimension of condition data
        hidden_dim: Base hidden dimension
        output_dim: Dimension of output (usually same as input_dim)
        time_dim: Dimension of time embeddings
        num_layers: Number of down/up layers
    """
    def __init__(self, input_dim=2, condition_dim=10, hidden_dim=64, output_dim=2, time_dim=128, num_layers=4):
        super().__init__()
        
        # Time embedding
        self.time_embeddings = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Store network dimensions for later debugging
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        
        # Initial projection - includes both input and condition
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        current_dim = hidden_dim
        
        # Pre-compute dimensions for down and up paths
        self.down_dims = [current_dim]
        
        # Scale factor for how much dimension increases at each layer
        scale_factor = 2
        
        # Calculate down dimensions
        for i in range(num_layers):
            next_dim = current_dim * scale_factor
            self.down_dims.append(next_dim)
            current_dim = next_dim
        
        # Create down blocks
        for i in range(num_layers):
            in_dim = self.down_dims[i]
            out_dim = self.down_dims[i+1]
            self.down_blocks.append(MLPBlock(in_dim, out_dim, out_dim, time_dim))
            
        # Middle block
        middle_in_dim = self.down_dims[-1]
        middle_hidden_dim = middle_in_dim * scale_factor
        self.middle_block = MLPBlock(middle_in_dim, middle_hidden_dim, middle_in_dim, time_dim)
            
        # Calculate up dimensions
        self.up_dims = []
        for i in range(num_layers):
            # Skip connection + previous layer output
            skip_dim = self.down_dims[-(i+2)]  # Get corresponding dimension from down path
            in_dim = self.down_dims[-(i+1)] + skip_dim  # Current + skip connection
            out_dim = skip_dim
            self.up_dims.append((in_dim, out_dim))
            
        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_dim, out_dim = self.up_dims[i]
            self.up_blocks.append(MLPBlock(in_dim, in_dim, out_dim, time_dim))
            
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Log dimensions
        print(f"ConditionalUNet dimensions:")
        print(f"  Down dims: {self.down_dims}")
        print(f"  Up dims: {self.up_dims}")
        
    def forward(self, x, t, condition):
        # Time embedding
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)
        
        # Encode condition
        cond_emb = self.condition_encoder(condition)
        
        # Concatenate input and condition
        x = torch.cat([x, cond_emb], dim=-1)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Down path
        residuals = [x]
        for block in self.down_blocks:
            x = block(x, t_emb)
            residuals.append(x)
            
        # Middle block
        x = self.middle_block(x, t_emb)
        
        # Up path with skip connections
        for block in self.up_blocks:
            residual = residuals.pop()
            x = torch.cat([x, residual], dim=-1)
            x = block(x, t_emb)
            
        # Output projection
        x = self.output_proj(x)
        
        return x


class DiffusionSchedule:
    """
    Diffusion schedule for controlling noise levels during diffusion.
    
    Args:
        num_timesteps: Number of diffusion steps
        schedule_type: Type of schedule ('linear' or 'cosine')
        beta_start: Starting value for beta (linear schedule)
        beta_end: Ending value for beta (linear schedule)
        s: Parameter for cosine schedule
    """
    def __init__(
        self, 
        num_timesteps=1000, 
        schedule_type="linear", 
        beta_start=1e-4, 
        beta_end=0.02,
        s=0.008
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = torch.arange(num_timesteps + 1) / num_timesteps
            alpha_cumprod = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: sample from q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Select appropriate diffusion parameters for timestep t
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Sample from q(x_t | x_0)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from noise model output
        """
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute mean and variance of the posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        
        posterior_mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_t
    
    def p_mean_variance(self, model, x_t, t, condition=None, clip_denoised=True):
        """
        Compute mean and variance of the reverse process p(x_{t-1} | x_t)
        """
        # Predict noise
        if condition is None:
            model_output = model(x_t, t)
        else:
            model_output = model(x_t, t, condition)
        
        # Predict x_0
        x_0 = self.predict_start_from_noise(x_t, t, model_output)
        
        # Clip predicted x_0 for stability
        if clip_denoised:
            x_0 = torch.clamp(x_0, -1., 1.)
            
        # Compute mean and variance of the posterior
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_0, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance, model_output, x_0
    
    def p_sample(self, model, x_t, t, condition=None, clip_denoised=True):
        """
        Sample one step from the reverse process p(x_{t-1} | x_t)
        """
        model_mean, _, posterior_log_variance, _, _ = self.p_mean_variance(
            model, x_t, t, condition, clip_denoised
        )
        
        noise = torch.randn_like(x_t) if any(t > 0) else torch.zeros_like(x_t)
        return model_mean + torch.exp(0.5 * posterior_log_variance) * noise
    
    def _extract(self, a, t, target_shape):
        """
        Extract appropriate elements from 'a' based on timestep t
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu()).to(t.device)
        return out.reshape(batch_size, *((1,) * (len(target_shape) - 1)))

    def q_posterior(self, x_0, x_t, t):
        """
        Compute the posterior mean and log variance for q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: The predicted clean data
            x_t: The noisy data at timestep t
            t: The timestep
            
        Returns:
            posterior_mean: The posterior mean
            posterior_log_variance: The posterior log variance
        """
        # Get the required values
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        # Compute the posterior mean
        posterior_mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        
        return posterior_mean, posterior_log_variance_t


class GFNDiffusion:
    """
    GFlowNet-guided diffusion model.
    
    Args:
        model: UNet or ConditionalUNet model
        dim: Dimension of the data
        schedule: Diffusion schedule
        energy_fn: Energy function for guidance
        device: Device to run on
    """
    def __init__(self, model, dim, schedule, energy_fn=None, device="cpu"):
        self.model = model
        self.dim = dim
        self.schedule = schedule
        self.energy_fn = energy_fn
        self.device = device
        
        # Check if model is conditional
        self.is_conditional = isinstance(model, ConditionalUNet)
    
    def train_step(self, optimizer, x_0, condition=None):
        """
        Train the diffusion model on a batch of data with optional conditioning.
        
        Args:
            optimizer: PyTorch optimizer
            x_0: Initial data
            condition: Conditioning data (for conditional models)
            
        Returns:
            loss: Loss value
        """
        optimizer.zero_grad()
        
        # Check if we need condition
        if self.is_conditional and condition is None:
            raise ValueError("Conditional model requires conditioning data")
        
        # Create random timesteps
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,), device=self.device).long()
        
        # Add noise to data
        noise = torch.randn_like(x_0)
        x_t = self.schedule.q_sample(x_0, t, noise)
        
        # Predict noise with model
        if self.is_conditional:
            noise_pred = self.model(x_t, t, condition)
        else:
            noise_pred = self.model(x_t, t)
        
        # Simple MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def sample(
        self, 
        num_samples, 
        condition=None, 
        use_gfn=False, 
        guidance_scale=1.0, 
        steps=None, 
        clip_denoised=True
    ):
        """
        Sample from the diffusion model with optional GFN guidance.
        
        Args:
            num_samples: Number of samples to generate
            condition: Conditioning data (for conditional models)
            use_gfn: Whether to use GFN guidance
            guidance_scale: Scale factor for GFN guidance
            steps: Number of sampling steps (default: schedule.num_timesteps)
            clip_denoised: Whether to clip the denoised values
            
        Returns:
            samples: Generated samples
        """
        model = self.model
        device = self.device
        
        # Check if we need condition
        if self.is_conditional and condition is None:
            raise ValueError("Conditional model requires conditioning data")
            
        # Check if condition has the right shape
        if self.is_conditional and condition.shape[0] != num_samples:
            if condition.shape[0] == 1:
                # Broadcast condition
                condition = condition.repeat(num_samples, 1)
            else:
                raise ValueError(f"Condition shape {condition.shape} doesn't match num_samples {num_samples}")
        
        # Start with random noise
        x = torch.randn(num_samples, self.dim).to(device)
        
        # Number of sampling steps
        if steps is None:
            steps = self.schedule.num_timesteps
        
        # Select timesteps
        timesteps = torch.linspace(self.schedule.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
        
        # Sampling loop
        for i, t in enumerate(timesteps):
            # Replicate t to match batch size
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                # Predict noise
                if self.is_conditional:
                    noise_pred = model(x, t_batch, condition)
                else:
                    noise_pred = model(x, t_batch)
                
                # Predict x_0
                x_0_pred = self.schedule.predict_start_from_noise(x, t_batch, noise_pred)
                
                # Clip x_0 for stability
                if clip_denoised:
                    x_0_pred = torch.clamp(x_0_pred, -1., 1.)
                
                # Compute posterior mean and variance
                posterior_mean, posterior_log_variance = self.schedule.q_posterior(x_0_pred, x, t_batch)
                
                # Apply GFN guidance if requested
                if use_gfn and self.energy_fn is not None:
                    # Enable gradients just for this part
                    with torch.enable_grad():
                        # Make a copy that requires gradients
                        x_requires_grad = x.detach().clone().requires_grad_(True)
                        
                        # Compute energy
                        if self.is_conditional:
                            energy = self.energy_fn(x_requires_grad, condition)
                        else:
                            energy = self.energy_fn(x_requires_grad)
                        
                        # Check if we need to sum up for batched energy
                        if energy.ndim > 0 and energy.shape[0] > 1:
                            energy_sum = energy.sum()
                        else:
                            energy_sum = energy
                            
                        # Compute gradients
                        energy_grad = torch.autograd.grad(
                            energy_sum, 
                            x_requires_grad,
                            create_graph=False,
                            retain_graph=False
                        )[0]
                    
                    # Apply gradient-based guidance (outside of torch.enable_grad() context)
                    posterior_mean = posterior_mean - guidance_scale * energy_grad
                
                # Sample from posterior
                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                x = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
        
        return x 

    def p_sample_loop(
        self, 
        n=1, 
        device=None, 
        use_gfn=False, 
        condition=None, 
        guidance_scale=None,
        clip_denoised=True
    ):
        """
        Generate samples from the model by running the sampling loop.
        
        Args:
            n: Number of samples to generate
            device: Device to generate samples on
            use_gfn: Whether to use GFN guidance
            condition: Conditioning data (for conditional models)
            guidance_scale: Scale for GFN guidance (if None, use self.guidance_scale)
            clip_denoised: Whether to clip values during denoising
            
        Returns:
            samples: Generated samples
        """
        if device is None:
            device = self.device
            
        if guidance_scale is None:
            guidance_scale = getattr(self, "guidance_scale", 1.0)
            
        # Generate samples
        return self.sample(
            num_samples=n,
            condition=condition,
            use_gfn=use_gfn,
            guidance_scale=guidance_scale,
            clip_denoised=clip_denoised
        ) 