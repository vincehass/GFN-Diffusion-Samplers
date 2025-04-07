"""
Training functions for GFN-guided diffusion models.

This module provides training functions for both unconditional and conditional models
for GFN-guided diffusion. The models are trained to predict noise in the diffusion process
and can be used with energy-based sampling for unconditional generation or
with conditional generation for specific targets.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from pathlib import Path
import sys
from scipy.spatial.distance import cdist
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    print("Warning: sklearn not available, falling back to simple unique position counting")

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import UNet, ConditionalUNet
from models.diffusion import DiffusionSchedule, GFNDiffusion
from models.energy import gmm_energy
from models.embeddings import SinusoidalPositionEmbeddings

# Import metrics and visualization utilities
try:
    from utils.metrics import (
        compute_diversity,
        compute_novelty,
        compute_energy_statistics,
        compute_energy_improvement,
        compute_effective_sample_size,
        compute_entropy,
        compute_coverage_metrics,
        compute_mode_coverage,
        compute_nearest_mode_distribution,
        earth_movers_distance
    )
    from utils.visualization import (
        visualize_energy,
        visualize_samples,
        compare_samples,
        plot_energy_evolution,
        visualize_diffusion_process,
        create_comparative_trajectory_plot,
        create_mode_coverage_plot,
        plot_2d_density_comparison,
        plot_3d_energy_landscape,
        create_metric_comparison_plot,
        log_animated_gfn_process_to_wandb,
        create_comparative_trajectory_plot,
    )
except ImportError as e:
    print(f"Warning: Could not import metrics and visualization utilities: {str(e)}")
    # Define placeholder functions if imports fail
    def compute_diversity(samples, threshold=0.1): 
        # Add safety checks
        if samples is None or (isinstance(samples, np.ndarray) and samples.size == 0) or (isinstance(samples, torch.Tensor) and samples.numel() == 0):
            print("Warning: Empty samples array in compute_diversity. Returning 0.0")
            return 0.0
            
        try:
            if isinstance(samples, torch.Tensor):
                samples = samples.cpu().numpy()
            
            # Make sure samples is 2D
            if len(samples.shape) == 1:
                samples = samples.reshape(-1, 1)
                
            # Safety check for NaN or inf values
            if np.isnan(samples).any() or np.isinf(samples).any():
                print("Warning: NaN or Inf values in samples for compute_diversity. Cleaning data...")
                samples = np.nan_to_num(samples, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # If we have only one sample, return 0 diversity
            if len(samples) <= 1:
                return 0.0
            
            # Compute pairwise distances
            distances = cdist(samples, samples)
            
            # Exclude self-distances (diagonal)
            n = distances.shape[0]
            mask = np.ones((n, n), dtype=bool)
            np.fill_diagonal(mask, 0)
            
            # Mean distance
            diversity = distances[mask].mean()
            
            return diversity
        except Exception as e:
            print(f"Error in compute_diversity: {e}")
            # Return a default value instead of failing
            return 0.0
            
    def compute_novelty(generated_samples, reference_samples=None, threshold=0.1): return 0.0
    def compute_energy_statistics(samples, energy_fn): 
        return {"mean_energy": 0.0, "min_energy": 0.0, "max_energy": 0.0, "std_energy": 0.0, "p50": 0.0}
    def compute_entropy(samples, bins=20, range_val=5.0): return 0.0
    def compute_energy_improvement(standard_samples, gfn_samples, energy_fn): return 0.0, 0.0
    def compute_effective_sample_size(samples, energy_fn, temperature=1.0): return len(samples) * 0.5, 0.5
    def compute_coverage_metrics(samples, reference_centers=None, energy_fn=None, range_val=5.0, n_grid=10):
        return {"grid_coverage": 0.0}
    def compute_mode_coverage(samples, modes, threshold=0.5): return 0.0, []
    def compute_nearest_mode_distribution(samples, modes): return torch.zeros(modes.shape[0])
    def earth_movers_distance(p_samples, q_samples): return 0.0
    def visualize_energy(energy_fn, save_path, title="Energy Function"): pass
    def visualize_samples(samples, save_path, title="Samples"): pass
    def compare_samples(samples1, samples2, save_path, title="Sample Comparison"): pass
    def plot_energy_evolution(energy_values, save_path, title="Energy Evolution"): pass
    def visualize_diffusion_process(trajectory, save_path, title="Diffusion Process", energy_fn=None): pass
    def create_comparative_trajectory_plot(trajectory_standard, trajectory_gfn, energy_fn, timesteps, output_dir, name="comparison"): return ""
    def create_mode_coverage_plot(mode_counts_standard, mode_counts_gfn, save_path, title="Mode Coverage"): pass

import wandb
import random
import torch.nn.functional as F

# Create directory for results
os.makedirs("results", exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a diffusion model for energy-based sampling")
    
    # General training arguments
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr_policy", type=float, default=1e-4, help="Learning rate for policy network")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for networks")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension for diffusion")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--schedule_type", type=str, default="linear", help="Diffusion schedule type (linear, cosine)")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in UNet")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda, cpu)")
    parser.add_argument("--output_dir", type=str, default="outputs/energy_sampling", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval to save model")
    parser.add_argument("--eval_interval", type=int, default=100, help="Interval to evaluate model")
    parser.add_argument("--log_interval", type=int, default=1, help="Log metrics every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--project_name", type=str, default="gfn-diffusion-experiments", help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    
    # Energy function arguments
    parser.add_argument("--energy", type=str, default="25gmm", help="Energy function to use (25gmm, ring, moons, vae)")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Scale for energy guidance")
    
    # VAE-specific arguments
    parser.add_argument("--vae_path", type=str, default=None, help="Path to VAE model")
    parser.add_argument("--vae_type", type=str, default="mnist", help="Type of VAE model (mnist, fashion_mnist)")
    
    # Conditional diffusion arguments
    parser.add_argument("--conditional", action="store_true", help="Use conditional diffusion")
    parser.add_argument("--num_conditions", type=int, default=4, help="Number of conditional diffusion models")
    
    # Loss function arguments
    parser.add_argument("--loss_type", type=str, default="standard", 
                        choices=["standard", "tb_avg"], 
                        help="Type of loss to use (standard, tb_avg)")
    
    return parser.parse_args()


def create_gmm_energy(num_modes=25, std=0.1, device="cpu"):
    """
    Create a Gaussian Mixture Model energy function.
    
    Args:
        num_modes: Number of mixture components (default: 25 in a grid)
        std: Standard deviation of each component
        device: Device to place tensors on
        
    Returns:
        energy_fn: GMM energy function
    """
    side_length = int(np.sqrt(num_modes))
    x_grid = torch.linspace(-4, 4, side_length)
    y_grid = torch.linspace(-4, 4, side_length)
    means = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='ij'), dim=-1).reshape(-1, 2).to(device)
    weights = torch.ones(side_length * side_length).to(device)
    weights = weights / weights.sum()
    
    return lambda x: gmm_energy(x, means, weights, std=std)


def create_multi_modal_energy(device="cpu"):
    """
    Create a multi-modal energy function with 4 well-separated modes.
    
    Args:
        device: Device to place tensors on
        
    Returns:
        energy_fn: Multi-modal energy function
    """
    means = torch.tensor([
        [-3.0, -3.0],
        [3.0, -3.0],
        [-3.0, 3.0],
        [3.0, 3.0]
    ], device=device)
    
    weights = torch.ones(4, device=device) / 4
    
    return lambda x: gmm_energy(x, means, weights, std=0.5)


def complex_mixture_energy(x, device="cpu"):
    """
    A more complex energy function that combines multiple patterns:
    - A 5x5 GMM grid with varying weights (some modes are "deeper" than others)
    - A ring pattern overlaid
    - A diagonal "valley" of low energy
    
    Args:
        x: Input tensor of shape [batch_size, dim]
        device: Device to place tensors on
        
    Returns:
        energy: Energy for each input [batch_size]
    """
    # Ensure input is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    
    # Make sure x has the right shape - should be [batch_size, 2]
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension if missing
    
    if x.shape[1] != 2:
        raise ValueError(f"Expected input with 2 dimensions, got shape {x.shape}")
    
    # Setup for GMM component
    side_length = 5  # 5x5 = 25 modes
    x_grid = torch.linspace(-4, 4, side_length)
    y_grid = torch.linspace(-4, 4, side_length)
    means = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='ij'), dim=-1).reshape(-1, 2).to(device)
    
    # Create non-uniform weights - make some modes "deeper" than others
    weights = torch.ones(side_length * side_length).to(device)
    
    # Make diagonal modes deeper (lower energy)
    for i in range(side_length):
        weights[i * side_length + i] = 3.0  # Diagonal from top-left to bottom-right
        if i < side_length - 1:
            weights[i * side_length + (side_length - i - 1)] = 2.0  # Diagonal from top-right to bottom-left
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # GMM component - manually compute to avoid shape issues
    batch_size = x.shape[0]
    gmm_energies = torch.zeros(batch_size, device=device)
    gmm_std = 0.1
    
    for i in range(batch_size):
        # Calculate distances to all GMM centers
        distances = torch.norm(x[i:i+1].unsqueeze(1) - means.unsqueeze(0), dim=2).squeeze(0)
        # Calculate Gaussian density for each center
        densities = torch.exp(-0.5 * (distances / gmm_std) ** 2) * weights
        # Sum weighted densities
        gmm_energy_i = -torch.log(torch.sum(densities) + 1e-10)
        gmm_energies[i] = gmm_energy_i
    
    # Ring component
    ring_radius = 3.5
    ring_thickness = 0.3
    distances_from_origin = torch.norm(x, dim=1)
    ring_energies = ((distances_from_origin - ring_radius) / ring_thickness) ** 2
    
    # Diagonal valley component - low energy along y=x line
    x_coord = x[:, 0]
    y_coord = x[:, 1]
    diagonal_dist = torch.abs(y_coord - x_coord) / np.sqrt(2)
    valley_width = 0.5
    valley_energies = (diagonal_dist / valley_width) ** 2
    
    # Combine components with different weights
    energy = 0.5 * gmm_energies + 0.3 * ring_energies + 0.2 * valley_energies
    
    # Normalize to 0-100 range
    energy_min = torch.min(energy)
    energy_max = torch.max(energy)
    energy_range = energy_max - energy_min
    if energy_range > 0:
        energy = (energy - energy_min) / energy_range * 100
    
    # Ensure output is the right shape [batch_size]
    assert energy.shape == (batch_size,), f"Expected energy shape ({batch_size},), got {energy.shape}"
    
    return energy


def setup_energy_function(energy_type, device):
    """
    Create an energy function based on the specified type.
    
    Args:
        energy_type: Type of energy function
        device: Device to place tensors on
        
    Returns:
        energy_fn: Energy function
    """
    if energy_type == "25gmm":
        # Create a 5x5 grid of Gaussian means
        side_length = 5  # 5x5 = 25 modes
        x_grid = torch.linspace(-4, 4, side_length)
        y_grid = torch.linspace(-4, 4, side_length)
        means = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='ij'), dim=-1).reshape(-1, 2).to(device)
        weights = torch.ones(side_length * side_length).to(device)
        weights = weights / weights.sum()
        std = 0.1
        
        return lambda x: gmm_energy(x, means, weights, std=std)
    elif energy_type == "ring":
        return lambda x: ring_energy(x)
    elif energy_type == "moons":
        return lambda x: moons_energy(x)
    elif energy_type == "complex_mixture":
        return lambda x: complex_mixture_energy(x, device=device)
    else:
        raise ValueError(f"Unknown energy type: {energy_type}")


def ring_energy(x, radius=3.0, thickness=0.5):
    """
    Ring energy function.
    
    Args:
        x: Input tensor of shape [batch_size, dim]
        radius: Radius of the ring
        thickness: Thickness of the ring
        
    Returns:
        energy: Energy for each input [batch_size]
    """
    # Ensure input is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Make sure x has the right shape - should be [batch_size, dim]
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension if missing
    
    # Ensure we have at least 2 dimensions to work with
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 dimensions, got shape {x.shape}")
    
    # Use only the first 2 dimensions if more are provided
    if x.shape[1] > 2:
        x = x[:, :2]
    
    # Calculate distance from origin
    dist_from_origin = torch.norm(x, dim=-1)
    
    # Calculate distance from ring
    dist_from_ring = torch.abs(dist_from_origin - radius)
    
    # Return energy (squared distance from ring, normalized by thickness)
    energy = (dist_from_ring / thickness) ** 2
    
    # Ensure output shape is correct [batch_size]
    assert energy.shape == (x.shape[0],), f"Expected energy shape ({x.shape[0]}), got {energy.shape}"
    
    return energy


def moons_energy(x, radius=3.0, thickness=0.5, distance=2.0):
    """
    Two moons energy function.
    
    Args:
        x: Input tensor of shape [batch_size, dim]
        radius: Radius of the moons
        thickness: Thickness of the moons
        distance: Distance between the moons
        
    Returns:
        energy: Energy for each input [batch_size]
    """
    # Ensure input is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Make sure x has the right shape - should be [batch_size, dim]
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension if missing
    
    # Ensure we have at least 2 dimensions to work with
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 dimensions, got shape {x.shape}")
    
    # Use only the first 2 dimensions if more are provided
    if x.shape[1] > 2:
        x = x[:, :2]
    
    batch_size = x.shape[0]
    device = x.device
    
    # Create centers for the two moons
    centers = torch.tensor([[-distance/2, 0], [distance/2, 0]], device=device)
    
    # Calculate distances from each point to both centers
    x_expanded = x.unsqueeze(1)  # [batch_size, 1, 2]
    centers_expanded = centers.unsqueeze(0)  # [1, 2, 2]
    
    # Calculate squared distances
    squared_dists = ((x_expanded - centers_expanded) ** 2).sum(dim=-1)  # [batch_size, 2]
    
    # Calculate distance from each point to the nearest moon
    dist_to_nearest = torch.min(squared_dists, dim=-1)[0]
    
    # Return energy (squared distance from nearest moon, normalized by thickness)
    energy = (dist_to_nearest / thickness) ** 2
    
    # Ensure output shape is correct [batch_size]
    assert energy.shape == (batch_size,), f"Expected energy shape ({batch_size},), got {energy.shape}"
    
    return energy


def visualize_energy(energy_fn, filename, title="Energy Function", range_val=4.0, resolution=100):
    """
    Visualize an energy function.
    
    Args:
        energy_fn: Energy function
        filename: File to save the visualization
        title: Title for the plot
        range_val: Range for the plot
        resolution: Resolution of the grid
    """
    try:
        # Create a grid of points
        x = np.linspace(-range_val, range_val, resolution)
        y = np.linspace(-range_val, range_val, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Reshape to [batch_size, 2]
        grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
        
        # Compute energy in batches to avoid memory issues
        batch_size = 1000  # Adjust based on memory constraints
        Z = np.zeros(grid_points.shape[0])
        
        for i in range(0, grid_points.shape[0], batch_size):
            end_idx = min(i + batch_size, grid_points.shape[0])
            batch = torch.tensor(grid_points[i:end_idx], dtype=torch.float32)
            
            try:
                with torch.no_grad():
                    Z[i:end_idx] = energy_fn(batch).cpu().numpy()
            except Exception as e:
                print(f"Error computing energy for batch {i//batch_size}: {str(e)}")
                # Fill with gradient as placeholder for failed batches
                Z[i:end_idx] = np.sin(5 * grid_points[i:end_idx, 0]) * np.cos(5 * grid_points[i:end_idx, 1])
        
        # Reshape back to grid
        Z = Z.reshape(X.shape)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot contour
        contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Energy')
        
        # Add title and labels
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
        
        print(f"Created energy visualization at {filename}")
        
    except Exception as e:
        print(f"Error computing energy for visualization: {str(e)}")
        
        # Create a simple placeholder image as fallback
        plt.figure(figsize=(10, 8))
        
        # Create a simple gradient as placeholder
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(5 * X) * np.cos(5 * Y)
        
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Energy (placeholder)')
        plt.title(f"{title} (placeholder - visualization failed)")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
        
        print(f"Created placeholder energy visualization at {filename}")

def visualize_energy_with_samples(energy_fn, samples, filename, title="Energy with Samples", range_val=4.0, resolution=100):
    """
    Visualize an energy function with samples overlaid.
    
    Args:
        energy_fn: Energy function
        samples: Samples to visualize, tensor of shape [batch_size, 2]
        filename: File to save the visualization
        title: Title for the plot
        range_val: Range for the plot
        resolution: Resolution of the grid
    """
    try:
        # Convert samples to numpy if needed
        if isinstance(samples, torch.Tensor):
            samples_np = samples.cpu().numpy()
        else:
            samples_np = samples
            
        # Create a grid of points
        x = np.linspace(-range_val, range_val, resolution)
        y = np.linspace(-range_val, range_val, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Reshape to [batch_size, 2]
        grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
        
        # Compute energy in batches to avoid memory issues
        batch_size = 1000  # Adjust based on memory constraints
        Z = np.zeros(grid_points.shape[0])
        
        for i in range(0, grid_points.shape[0], batch_size):
            end_idx = min(i + batch_size, grid_points.shape[0])
            batch = torch.tensor(grid_points[i:end_idx], dtype=torch.float32)
            
            try:
                with torch.no_grad():
                    Z[i:end_idx] = energy_fn(batch).cpu().numpy()
            except Exception as e:
                print(f"Error computing energy for batch {i//batch_size}: {str(e)}")
                # Fill with gradient as placeholder for failed batches
                Z[i:end_idx] = np.sin(5 * grid_points[i:end_idx, 0]) * np.cos(5 * grid_points[i:end_idx, 1])
        
        # Reshape back to grid
        Z = Z.reshape(X.shape)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot contour
        contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, label='Energy')
        
        # Plot samples
        plt.scatter(samples_np[:, 0], samples_np[:, 1], color='red', alpha=0.7, s=30, label='Samples')
        
        # Add title and labels
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
        
        print(f"Created energy with samples visualization at {filename}")
        
    except Exception as e:
        print(f"Error computing energy with samples visualization: {str(e)}")
        
        # Create a simple placeholder image as fallback
        plt.figure(figsize=(12, 10))
        
        # Create a simple gradient as placeholder
        x = np.linspace(-range_val, range_val, resolution)
        y = np.linspace(-range_val, range_val, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(5 * X) * np.cos(5 * Y)
        
        plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Energy (placeholder)')
        
        # Add samples if we have them
        if samples_np is not None:
            plt.scatter(samples_np[:, 0], samples_np[:, 1], color='red', alpha=0.7, s=30, label='Samples')
            
        plt.title(f"{title} (placeholder - visualization failed)")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
        
        print(f"Created placeholder energy with samples visualization at {filename}")

def visualize_samples(samples, filename, title="Samples"):
    """
    Visualize samples.
    
    Args:
        samples: Samples to visualize
        filename: File to save the visualization
        title: Title for the plot
    """
    # Convert tensors to NumPy if needed
    if isinstance(samples, torch.Tensor):
        samples_np = samples.cpu().numpy()
    else:
        samples_np = samples
        
    plt.figure(figsize=(10, 8))
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.7, s=20)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_samples(samples_standard, samples_gfn, filename, title="Sample Comparison"):
    """
    Compare samples from standard diffusion and GFN-guided diffusion.
    
    Args:
        samples_standard: Samples from standard diffusion
        samples_gfn: Samples from GFN-guided diffusion
        filename: File to save the visualization
        title: Title for the plot
    """
    # Convert tensors to NumPy if needed
    if isinstance(samples_standard, torch.Tensor):
        samples_standard_np = samples_standard.cpu().numpy()
    else:
        samples_standard_np = samples_standard
        
    if isinstance(samples_gfn, torch.Tensor):
        samples_gfn_np = samples_gfn.cpu().numpy()
    else:
        samples_gfn_np = samples_gfn
    
    plt.figure(figsize=(16, 8))
    
    # Plot standard samples
    plt.subplot(1, 2, 1)
    plt.scatter(samples_standard_np[:, 0], samples_standard_np[:, 1], alpha=0.7, s=20)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title("Standard Diffusion Samples")
    
    # Plot GFN samples
    plt.subplot(1, 2, 2)
    plt.scatter(samples_gfn_np[:, 0], samples_gfn_np[:, 1], alpha=0.7, s=20)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title("GFN-Guided Samples")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def p_sample(diffusion, model, x, t, condition=None):
    """
    Sample from p(x_{t-1} | x_t) for a single timestep.
    
    Args:
        diffusion: Diffusion schedule or GFNDiffusion object
        model: Model to predict noise
        x: Noisy samples at timestep t
        t: Timesteps
        condition: Optional condition for conditional model
        
    Returns:
        x_{t-1}: Samples at timestep t-1
    """
    # Check if diffusion is a GFNDiffusion object
    if hasattr(diffusion, 'diffusion') and diffusion.diffusion is not None:
        # It's a GFNDiffusion object
        diffusion_schedule = diffusion.diffusion
    else:
        # It's a regular DiffusionSchedule
        diffusion_schedule = diffusion
        
    # Get diffusion parameters
    sqrt_alphas_cumprod = diffusion_schedule.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod = diffusion_schedule.sqrt_one_minus_alphas_cumprod
    posterior_mean_coef1 = diffusion_schedule.posterior_mean_coef1
    posterior_mean_coef2 = diffusion_schedule.posterior_mean_coef2
    posterior_variance = diffusion_schedule.posterior_variance
    
    # Predict noise
    with torch.no_grad():
        if condition is not None:
            # Convert condition to long tensor if needed
            if not isinstance(condition, torch.LongTensor) and condition.dtype != torch.int64:
                condition = condition.long()
            pred_noise = model(x, t, condition)
        else:
            pred_noise = model(x, t)
    
    # Get posterior mean and variance
    posterior_mean = posterior_mean_coef1[t].view(-1, 1) * x + posterior_mean_coef2[t].view(-1, 1) * pred_noise
    posterior_var = posterior_variance[t].view(-1, 1)
    
    # Sample
    noise = torch.randn_like(x)
    return posterior_mean + torch.sqrt(posterior_var) * noise


# Create a simple UNet model that avoids dimension issues
class SimpleUNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, time_dim=128, num_layers=1):
        super().__init__()
        self.time_embeddings = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Down path
        down_dims = [hidden_dim]
        for i in range(num_layers):
            down_dims.append(hidden_dim * 2 ** (i + 1))
        
        self.down_layers = nn.ModuleList()
        for i in range(num_layers):
            self.down_layers.append(nn.Sequential(
                nn.Linear(down_dims[i], down_dims[i+1]),
                nn.LayerNorm(down_dims[i+1]),
                nn.SiLU()
            ))
        
        # Middle
        self.middle = nn.Sequential(
            nn.Linear(down_dims[-1], down_dims[-1] * 2),
            nn.LayerNorm(down_dims[-1] * 2),
            nn.SiLU(),
            nn.Linear(down_dims[-1] * 2, down_dims[-1]),
            nn.LayerNorm(down_dims[-1]),
            nn.SiLU()
        )
        
        # Up path
        up_in_dims = []
        up_out_dims = []
        for i in range(num_layers):
            if i == 0:
                up_in_dims.append(down_dims[-1] + down_dims[-2])
            else:
                up_in_dims.append(down_dims[-(i+1)] + down_dims[-(i+2)])
            up_out_dims.append(down_dims[-(i+2)])
        
        self.up_layers = nn.ModuleList()
        for i in range(num_layers):
            self.up_layers.append(nn.Sequential(
                nn.Linear(up_in_dims[i], up_out_dims[i]),
                nn.LayerNorm(up_out_dims[i]),
                nn.SiLU()
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Print dimensions for debugging
        print(f"SimpleUNet dimensions:")
        print(f"  Down dims: {down_dims}")
        print(f"  Up in dims: {up_in_dims}")
        print(f"  Up out dims: {up_out_dims}")
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Down path with skip connections
        residuals = [x]
        for layer in self.down_layers:
            x = layer(x)
            residuals.append(x)
            
        # Middle
        x = self.middle(x)
        
        # Up path with skip connections
        for i, layer in enumerate(self.up_layers):
            residual = residuals[-(i+2)]
            x = torch.cat([x, residual], dim=-1)
            x = layer(x)
            
        # Output projection
        x = self.output_proj(x)
        
        return x


def train_energy_sampling(args):
    """
    Train a diffusion model for energy-based sampling.
    
    Args:
        args: Command line arguments
        
    Returns:
        model: Trained model
    """
    # Import global functions for diversity metrics
    global compute_diversity, compute_novelty, compute_energy_statistics, compute_entropy
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model directory inside output directory
    if not hasattr(args, 'model_dir'):
        model_dir = output_dir / "models"
    else:
        model_dir = Path(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a debug directory for extra logs
    debug_dir = output_dir / "debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Initialize wandb
    if args.wandb:
        # Set default values for wandb attributes if they don't exist
        if not hasattr(args, 'offline'):
            args.offline = False
        if not hasattr(args, 'wandb_entity'):
            args.wandb_entity = None
        if not hasattr(args, 'wandb_project'):
            args.wandb_project = args.project_name if hasattr(args, 'project_name') else "gfn-diffusion-experiments"
            
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=args.__dict__,
            mode="offline" if args.offline else "online"
        )
    
    # Choose model based on conditional flag
    if args.conditional:
        # Create conditional model
        model = ConditionalUNet(
            input_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.latent_dim,
            num_conditions=args.num_conditions,
            num_layers=args.num_layers
        ).to(device)
        
        # Create diffusion schedule
        diffusion = DiffusionSchedule(
            num_diffusion_timesteps=args.num_timesteps,
            schedule_type=args.schedule_type
        )
        
        # Create energy function for each condition
        energy_fns = []
        
        # 1. 25gmm: 25 Gaussian mixture with different parameters
        energy_fns.append(create_gmm_energy(num_modes=25, std=0.1, device=device))
        
        # 2. Ring: Ring-shaped energy function with radius 3.0
        energy_fns.append(lambda x: ring_energy(x, radius=3.0, thickness=0.5))
        
        # 3. 4gmm: 4 Gaussian mixture with larger standard deviation
        energy_fns.append(create_gmm_energy(num_modes=4, std=0.5, device=device))
        
        # 4. Moons: Two moon-shaped energy function
        energy_fns.append(lambda x: moons_energy(x, radius=3.0, thickness=0.5, distance=5.0))
        
        # Default energy function (used if condition is outside range)
        default_energy_fn = energy_fns[0]
        
        # Conditional energy function that selects the appropriate energy function based on condition
        def conditional_energy_fn(x, c=None):
            # Check for NaN input values first
            if torch.isnan(x).any():
                print("Warning: NaN detected in conditional_energy_fn input. Using default high energy.")
                return torch.full((x.shape[0],), 1000.0, device=x.device)  # Default high energy
            
            # Convert condition tensor to int
            if c is None:
                c_int = 0
            elif isinstance(c, torch.Tensor) and c.numel() == 1:
                c_int = c.item()
            elif isinstance(c, torch.Tensor):
                # For batch conditions, use the first element (just for simplicity)
                try:
                    c_int = c[0].item() if c.numel() > 0 else 0
                except:
                    print("Warning: Could not convert condition to int. Using default energy function.")
                    c_int = 0
            else:
                c_int = 0
            
            try:
                # Dispatch to appropriate energy function
                if c_int == 0:  # GMM
                    energy = energy_fns[0](x)
                elif c_int == 1:  # Ring
                    energy = energy_fns[1](x)
                elif c_int == 2:  # Moons
                    energy = energy_fns[2](x)
                elif c_int == 3:  # Multi-modal
                    energy = energy_fns[3](x)
                else:
                    # Default to GMM
                    energy = energy_fns[0](x)
                
                # Check for NaN/Inf values in the output
                if torch.isnan(energy).any() or torch.isinf(energy).any():
                    print(f"Warning: Energy function returned NaN/Inf values for condition {c_int}. Using default high energy.")
                    nan_inf_mask = torch.isnan(energy) | torch.isinf(energy)
                    energy[nan_inf_mask] = 1000.0  # High energy for bad values
                
                return energy
            except Exception as e:
                print(f"Error in conditional_energy_fn: {e}")
                return torch.full((x.shape[0],), 1000.0, device=x.device)  # Default high energy
        
        # Create the GFNDiffusion model
        gfn_diffusion = GFNDiffusion(
            model=model,
            diffusion=diffusion,
            energy_fn=conditional_energy_fn,
            guidance_scale=args.guidance_scale,
            device=device,
            loss_type=args.loss_type
        )
        
        # Visualize different energy functions
        for c in range(args.num_conditions):
            # Create a different energy function for each condition
            if c == 0:
                # Standard 25-mode GMM
                side_length = 5
                x_grid = torch.linspace(-4, 4, side_length)
                y_grid = torch.linspace(-4, 4, side_length)
                means = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='ij'), dim=-1).reshape(-1, 2).to(device)
                weights = torch.ones(side_length * side_length).to(device)
                weights = weights / weights.sum()
                std = 0.1
                energy_fn = lambda x: gmm_energy(x, means, weights, std=std)
            elif c == 1:
                energy_fn = lambda x: ring_energy(x)
            elif c == 2:
                energy_fn = lambda x: moons_energy(x)
            else:
                # Modified GMM with different scale
                side_length = 5
                x_grid = torch.linspace(-4, 4, side_length)
                y_grid = torch.linspace(-4, 4, side_length)
                means = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='ij'), dim=-1).reshape(-1, 2).to(device)
                weights = torch.ones(side_length * side_length).to(args.device)
                weights = weights / weights.sum()
                std = 0.1 * (1 + 0.5 * c)
                energy_fn = lambda x, std=std: gmm_energy(x, means, weights, std=std)
                
            visualize_energy(
                energy_fn, 
                output_dir / f"energy_cond{c}.png",
                title=f"Energy Function (Condition {c})"
            )
    else:
        # Unconditional model
        model = SimpleUNet(
            input_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.latent_dim,
            num_layers=args.num_layers
        ).to(device)
        
        # Create energy function
        energy_fn = setup_energy_function(args.energy, device)
        
        # Create diffusion schedule
        diffusion = DiffusionSchedule(
            num_diffusion_timesteps=args.num_timesteps,
            schedule_type=args.schedule_type
        )
        
        # Create GFN-Diffusion model
        gfn_diffusion = GFNDiffusion(
            model=model,
            diffusion=diffusion,
            energy_fn=energy_fn,
            guidance_scale=args.guidance_scale,
            device=device,
            loss_type=args.loss_type
        )
        
        # Visualize energy function
        visualize_energy(energy_fn, output_dir / f"energy_{args.energy}.png")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_policy)
    
    # Create learning rate scheduler (optional)
    scheduler = None
    if getattr(args, 'use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # Training loop
    progress_bar = tqdm(range(args.epochs))
    for epoch in progress_bar:
        # Generate data
        x = torch.randn(args.batch_size, args.latent_dim, device=device)  # Start from Gaussian noise
        
        # Sample random timesteps for diffusion
        t = torch.randint(0, args.num_timesteps, (args.batch_size,), device=device)
        
        # Train one step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        if args.conditional:
            # For conditional models, randomly sample a condition for training
            condition_indices = torch.randint(0, args.num_conditions, (args.batch_size,), device=device)
            condition_long = condition_indices.long()
            loss = model(x, t, condition_long)
        else:
            # For unconditional models, no condition is provided
            loss = model(x, t)
        
        # Ensure loss is scalar (take mean if needed)
        if not isinstance(loss, torch.Tensor) or loss.numel() > 1:
            loss = loss.mean()
        
        # Safety check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected: {loss.item()}. Skipping backward pass.")
            # Skip this iteration if loss is invalid
            continue
            
        # Backward pass
        loss.backward()
        
        # Use gradient clipping to improve stability
        clip_grad_norm = getattr(args, 'clip_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        
        # Log training metrics to wandb at regular intervals
        log_interval = getattr(args, 'log_interval', 1)  # Changed from 10 to 1 to log every epoch
        
        # Initially set default values for metrics
        diversity_standard = getattr(args, '_last_diversity_standard', 0.0)
        diversity_gfn = getattr(args, '_last_diversity_gfn', 0.0)
        novelty_standard = getattr(args, '_last_novelty_standard', 0.0)
        novelty_gfn = getattr(args, '_last_novelty_gfn', 0.0) 
        unique_positions_standard = getattr(args, '_last_unique_positions_standard', 0)
        unique_positions_gfn = getattr(args, '_last_unique_positions_gfn', 0)
        
        if args.wandb and epoch % log_interval == 0:  # Log every epoch by default
            # Calculate current average reward as negative loss (lower loss is better)
            current_avg_reward = -loss.item()
            
            # Add more detailed metrics for charting - only include metrics updated every iteration
            metrics_dict = {
                "epoch": epoch,
                "train_loss": loss.item(),
                "avg_reward": current_avg_reward,
                "loss": loss.item(),  # Add standard "loss" metric for default charts
                "iteration": epoch,  # Add iteration for proper tracking
            }
            
            # Log with step parameter to ensure proper timeline in charts
            wandb.log(metrics_dict, step=epoch)
            
            # If conditional, also log condition-specific metrics
            if args.conditional:
                for c in range(args.num_conditions):
                    condition_metrics = {
                        f"train_loss_cond{c}": loss.item(),  # We're using the same loss for all conditions during training
                        f"avg_reward_cond{c}": current_avg_reward
                    }
                    # Log with step parameter for proper tracking
                    wandb.log(condition_metrics, step=epoch)
        
        # Learning rate scheduler step if using a scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save model
        if (epoch > 0 and epoch % args.save_interval == 0) or epoch == args.epochs - 1:
            torch.save(model.state_dict(), model_dir / f"gfn_diffusion_epoch{epoch}.pt")
            
        # Evaluation
        if (epoch > 0 and epoch % args.eval_interval == 0) or epoch == args.epochs - 1:
            with torch.no_grad():
                model.eval()
                
                # Increase sample size for evaluation
                eval_samples = 256 if epoch == args.epochs - 1 else 128
                
                # Generate reference dataset for novelty calculation 
                # Use a simple uniform grid as reference points
                grid_size = 10
                x_vals = torch.linspace(-4, 4, grid_size)
                y_vals = torch.linspace(-4, 4, grid_size)
                X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
                reference_grid = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
                
                # Log the reference grid for debugging
                if epoch == args.epochs - 1:
                    np.savetxt(debug_dir / "reference_grid.csv", reference_grid.cpu().numpy(), delimiter=",")
                
                if args.conditional:
                    # Generate samples for each condition
                    for c in range(args.num_conditions):
                        # Create condition
                        condition = torch.full((eval_samples,), c, device=device, dtype=torch.long)
                
                        # ... continue with existing code ...
                        
                # Generate samples using standard diffusion with larger batch size
                samples_standard = gfn_diffusion.p_sample_loop(
                    n=eval_samples,
                    dim=args.latent_dim,
                    use_gfn=False,
                    verbose=False
                )
                
                # Generate samples using GFN-guided diffusion with larger batch size
                samples_gfn = gfn_diffusion.p_sample_loop(
                    n=eval_samples,
                    dim=args.latent_dim,
                    use_gfn=True,
                    verbose=False
                )
                
                # Compute and save raw samples for debugging
                if epoch == args.epochs - 1:
                    np.savetxt(debug_dir / "samples_standard.csv", samples_standard.cpu().numpy(), delimiter=",")
                    np.savetxt(debug_dir / "samples_gfn.csv", samples_gfn.cpu().numpy(), delimiter=",")
                
                # Compute diversity metrics (how diverse the samples are)
                diversity_standard = compute_diversity(samples_standard.cpu().numpy())
                diversity_gfn = compute_diversity(samples_gfn.cpu().numpy())
                
                # Compute novelty metrics with respect to reference grid
                try:
                    novelty_standard = compute_novelty(samples_standard.cpu().numpy(), reference_grid.cpu().numpy())
                    novelty_gfn = compute_novelty(samples_gfn.cpu().numpy(), reference_grid.cpu().numpy())
                    print(f"Novelty metrics - Standard: {novelty_standard:.4f}, GFN: {novelty_gfn:.4f}")
                except Exception as e:
                    print(f"Warning: Could not compute novelty metrics: {e}")
                    novelty_standard = 0.0
                    novelty_gfn = 0.0
                
                # Enhanced unique positions calculation with detailed logging
                def count_unique_with_tolerance(samples, tol=1e-5, debug=False):
                    """Count unique positions with tolerance for floating point."""
                    samples_np = samples.cpu().numpy()
                    if len(samples_np) == 0:
                        return 0
                        
                    # Use a clustering approach to find unique positions
                    from sklearn.cluster import DBSCAN
                    try:
                        clustering = DBSCAN(eps=tol, min_samples=1).fit(samples_np)
                        labels = clustering.labels_
                        unique_count = len(set(labels))
                        
                        if debug and epoch == args.epochs - 1:
                            # Save detailed clustering info for debugging
                            cluster_info = np.column_stack((samples_np, labels))
                            np.savetxt(debug_dir / f"clustering_samples_{tol}.csv", cluster_info, 
                                       delimiter=",", header="x,y,cluster_id")
                            
                            # Log cluster sizes
                            unique_labels = set(labels)
                            cluster_sizes = [sum(labels == l) for l in unique_labels]
                            cluster_size_info = np.column_stack((list(unique_labels), cluster_sizes))
                            np.savetxt(debug_dir / f"cluster_sizes_{tol}.csv", cluster_size_info,
                                      delimiter=",", header="cluster_id,size")
                            
                            print(f"Clustering found {unique_count} unique positions from {len(samples_np)} samples")
                            print(f"Largest cluster size: {max(cluster_sizes)}")
                            print(f"Smallest cluster size: {min(cluster_sizes)}")
                        
                        return unique_count
                    except Exception as e:
                        print(f"Warning: Error in unique position clustering: {e}")
                        # Fall back to simple approach
                        rounded_samples = np.round(samples_np, int(-np.log10(tol)))
                        unique_tuples = set([tuple(s) for s in rounded_samples])
                        
                        if debug and epoch == args.epochs - 1:
                            unique_array = np.array(list(unique_tuples))
                            np.savetxt(debug_dir / f"unique_positions_fallback_{tol}.csv", 
                                      unique_array, delimiter=",")
                            print(f"Fallback method found {len(unique_tuples)} unique positions from {len(samples_np)} samples")
                        
                        return len(unique_tuples)
                
                try:
                    unique_positions_standard = count_unique_with_tolerance(samples_standard, debug=True)
                    unique_positions_gfn = count_unique_with_tolerance(samples_gfn, debug=True)
                    
                    # Try with different tolerance levels to understand the sensitivity
                    if epoch == args.epochs - 1:
                        print("\nUnique positions with different tolerance levels:")
                        for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                            std_count = count_unique_with_tolerance(samples_standard, tol=tol)
                            gfn_count = count_unique_with_tolerance(samples_gfn, tol=tol)
                            print(f"Tolerance {tol}: Standard={std_count}, GFN={gfn_count}")
                
                    # Store the metrics in wandb
                    if args.wandb:
                        wandb.log({
                            "diversity_standard": diversity_standard,
                            "diversity_gfn": diversity_gfn,
                            "novelty_standard": novelty_standard,
                            "novelty_gfn": novelty_gfn,
                            "unique_positions_standard": unique_positions_standard,
                            "unique_positions_gfn": unique_positions_gfn,
                            "unique_positions": unique_positions_gfn  # For backward compatibility
                        }, step=epoch)
                
                except Exception as e:
                    print(f"Warning: Could not compute unique positions: {e}")
                    # Fallback to simple approach if needed
                    unique_positions_standard = len(set([tuple(s.cpu().numpy().tolist()) for s in samples_standard]))
                    unique_positions_gfn = len(set([tuple(s.cpu().numpy().tolist()) for s in samples_gfn]))
                
                model.train()  # Set back to training mode
        else:
            # Use the last computed values or defaults
            diversity_standard = getattr(args, '_last_diversity_standard', 0.0)
            diversity_gfn = getattr(args, '_last_diversity_gfn', 0.0)
            novelty_standard = getattr(args, '_last_novelty_standard', 0.0)
            novelty_gfn = getattr(args, '_last_novelty_gfn', 0.0) 
            unique_positions_standard = getattr(args, '_last_unique_positions_standard', 0)
            unique_positions_gfn = getattr(args, '_last_unique_positions_gfn', 0)

    # Performing final evaluation with enhanced metrics and visualization
    if epoch == args.epochs - 1:
        print("\nPerforming final evaluation with enhanced metrics and visualization...")
        
        # Create output directory for visualizations
        viz_dir = output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualize energy function
        try:
            visualize_energy(energy_fn, viz_dir / f"energy_{args.energy}.png", 
                             title=f"{args.energy} Energy Function", resolution=200)
        except Exception as e:
            print(f"Error visualizing energy function: {e}")
        
        # Visualize energy function with standard diffusion samples
        try:
            visualize_energy_with_samples(
                energy_fn, 
                samples_standard, 
                viz_dir / f"energy_with_standard_samples_{args.energy}.png",
                title=f"{args.energy} Energy with Standard Diffusion Samples", 
                resolution=200
            )
        except Exception as e:
            print(f"Error visualizing energy with standard samples: {e}")
        
        # Visualize energy function with GFN-guided diffusion samples
        try:
            visualize_energy_with_samples(
                energy_fn, 
                samples_gfn, 
                viz_dir / f"energy_with_gfn_samples_{args.energy}.png",
                title=f"{args.energy} Energy with GFN-Guided Samples", 
                resolution=200
            )
        except Exception as e:
            print(f"Error visualizing energy with GFN samples: {e}")
            
        # Visualize sample comparison
        compare_samples(
            samples_standard, 
            samples_gfn, 
            viz_dir / "sample_comparison.png", 
            title=f"Sample Comparison - {args.energy}"
        )
        
    print("Training completed successfully!")
    return model_dir / f"gfn_diffusion_epoch{args.epochs - 1}.pt"


def main():
    """
    Main function to parse arguments and run training.
    """
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run appropriate training function
    model = train_energy_sampling(args)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 