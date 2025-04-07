"""
VAE-based energy functions for GFN-Diffusion.

This module provides energy functions based on VAE models
for use with GFN-guided diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

# Import VAE model
from energies.vae import VAE


def get_vae_energy(vae, temp=1.0, device="cpu"):
    """
    Create an energy function based on a VAE model.
    
    Args:
        vae: Pretrained VAE model
        temp: Temperature parameter for energy scaling
        device: Device to place tensors on
        
    Returns:
        energy_fn: Energy function for sampling
    """
    def energy_fn(z):
        """Energy function based on VAE reconstruction error and KL divergence."""
        # Make sure z requires gradient for Langevin dynamics
        z = z.clone().detach().requires_grad_(True)
        
        # Decode latent
        x_recon = vae.decode(z)
        
        # Since we're not using the KL term in the usual VAE way (we're in latent space already),
        # we'll use a simple Gaussian prior centered at origin
        prior_energy = 0.5 * torch.sum(z ** 2, dim=1)
        
        # We don't have the original x, so we can't compute reconstruction error directly
        # Instead, we compute how "typical" the decoded x_recon is by re-encoding it
        mu, logvar = vae.encode(x_recon)
        
        # Compute reconstruction probability (how likely is z given x_recon?)
        z_mean = mu
        z_var = torch.exp(logvar)
        recon_energy = 0.5 * torch.sum(((z - z_mean) ** 2) / z_var + logvar, dim=1)
        
        # Combined energy (lower is better)
        energy = (prior_energy + recon_energy) / temp
        
        return energy
    
    return energy_fn


class VAEEnergy:
    """
    Class for using a VAE as an energy function for GFN-Diffusion.
    
    Args:
        vae_path: Path to pretrained VAE model
        latent_dim: Dimension of latent space
        input_dim: Dimension of input data
        temp: Temperature parameter for energy scaling
        device: Device to place tensors on
    """
    def __init__(self, vae_path, latent_dim=2, input_dim=784, temp=1.0, device="cpu"):
        self.vae_path = vae_path
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.temp = temp
        self.device = device
        
        # Load VAE model
        try:
            self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=512).to(device)
            self.vae.load_state_dict(torch.load(vae_path, map_location=device))
            self.vae.eval()
            print(f"Successfully loaded VAE from {vae_path}")
        except Exception as e:
            print(f"Warning: Could not load VAE from {vae_path}, error: {str(e)}")
            print("Creating a new VAE model")
            self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=512).to(device)
            self.vae.eval()
    
    def __call__(self, z):
        """
        Compute energy of latent codes.
        
        Args:
            z: Latent codes
            
        Returns:
            energy: Energy values
        """
        # Make sure z requires gradient
        z = z.clone().detach().requires_grad_(True)
        
        # Decode latent
        x_recon = self.vae.decode(z)
        
        # Since we're not using the KL term in the usual VAE way (we're in latent space already),
        # we'll use a simple Gaussian prior centered at origin
        prior_energy = 0.5 * torch.sum(z ** 2, dim=1)
        
        # Encode reconstructed data to compute likelihood of z
        mu, logvar = self.vae.encode(x_recon)
        
        # Compute reconstruction probability (how likely is z given x_recon?)
        z_mean = mu
        z_var = torch.exp(logvar)
        recon_energy = 0.5 * torch.sum(((z - z_mean) ** 2) / z_var + logvar, dim=1)
        
        # Combined energy (lower is better)
        energy = (prior_energy + recon_energy) / self.temp
        
        return energy
    
    def visualize_latent_energy(self, n_points=100, range_val=3, save_path=None):
        """
        Visualize the energy function in latent space.
        
        Args:
            n_points: Number of points per dimension
            range_val: Range of latent space to visualize
            save_path: Path to save the figure
        """
        if self.latent_dim != 2:
            print("Warning: Visualization is only supported for 2D latent space")
            return
        
        # Create grid of points
        x = torch.linspace(-range_val, range_val, n_points)
        y = torch.linspace(-range_val, range_val, n_points)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Compute energy at each point
        z = torch.stack([X.flatten(), Y.flatten()], dim=1).to(self.device)
        
        with torch.no_grad():
            energy = self(z).reshape(n_points, n_points).cpu()
        
        # Normalize for better visualization
        energy = (energy - energy.min()) / (energy.max() - energy.min())
        
        # Plot energy landscape
        plt.figure(figsize=(10, 8))
        plt.contourf(X.cpu(), Y.cpu(), energy, levels=20, cmap='viridis')
        plt.colorbar(label='Normalized Energy')
        plt.title('VAE Latent Space Energy Function')
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_latent_samples(self, samples, save_path=None):
        """
        Visualize samples in the latent space.
        
        Args:
            samples: Latent space samples to visualize
            save_path: Path to save the figure
        """
        if self.latent_dim != 2:
            print("Warning: Visualization is only supported for 2D latent space")
            return
        
        # Compute energy landscape for background
        n_points = 50
        range_val = 3
        x = torch.linspace(-range_val, range_val, n_points)
        y = torch.linspace(-range_val, range_val, n_points)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Compute energy at each point
        z = torch.stack([X.flatten(), Y.flatten()], dim=1).to(self.device)
        
        with torch.no_grad():
            energy = self(z).reshape(n_points, n_points).cpu()
        
        # Normalize for better visualization
        energy = (energy - energy.min()) / (energy.max() - energy.min())
        
        # Plot energy landscape and samples
        plt.figure(figsize=(10, 8))
        plt.contourf(X.cpu(), Y.cpu(), energy, levels=20, cmap='viridis', alpha=0.7)
        plt.scatter(samples[:, 0], samples[:, 1], c='red', alpha=0.5, s=10)
        plt.colorbar(label='Normalized Energy')
        plt.title('Samples in VAE Latent Space')
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.xlim(-range_val, range_val)
        plt.ylim(-range_val, range_val)
        
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def decode_and_visualize(self, samples, grid_shape=(5, 5), save_path=None):
        """
        Decode latent samples and visualize the resulting images.
        
        Args:
            samples: Latent space samples to decode and visualize
            grid_shape: Shape of the visualization grid
            save_path: Path to save the figure
        """
        n_samples = grid_shape[0] * grid_shape[1]
        
        # Ensure we have the right number of samples
        if samples.shape[0] < n_samples:
            print(f"Warning: Not enough samples provided. Expected {n_samples}, got {samples.shape[0]}")
            n_samples = samples.shape[0]
            grid_shape = (1, n_samples)
        
        # Decode samples
        with torch.no_grad():
            decoded = self.vae.decode(samples[:n_samples].to(self.device)).cpu()
        
        # Plot reconstructed images
        plt.figure(figsize=(12, 12))
        for i in range(n_samples):
            plt.subplot(grid_shape[0], grid_shape[1], i + 1)
            
            # Reshape based on expected input dimensions (assuming MNIST-like)
            if self.input_dim == 784:  # 28x28 images
                plt.imshow(decoded[i].reshape(28, 28), cmap='gray')
            else:
                plt.imshow(decoded[i])
            
            plt.axis('off')
        
        plt.suptitle('Decoded Samples from Latent Space')
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def visualize_vae_energy(vae, n_points=100, range_val=3, filename=None):
    """
    Visualize the energy function of a VAE in 2D latent space.
    
    Args:
        vae: VAE model
        n_points: Number of points per dimension
        range_val: Range of latent space to visualize
        filename: Path to save the figure
    """
    if vae.latent_dim != 2:
        print("Warning: Visualization is only supported for 2D latent space")
        return
    
    # Create energy function
    energy_fn = get_vae_energy(vae)
    
    # Create grid of points
    x = torch.linspace(-range_val, range_val, n_points)
    y = torch.linspace(-range_val, range_val, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Compute energy at each point
    z = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        energy = energy_fn(z).reshape(n_points, n_points).cpu()
    
    # Normalize for better visualization
    energy = (energy - energy.min()) / (energy.max() - energy.min())
    
    # Plot energy landscape
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, energy, levels=20, cmap='viridis')
    plt.colorbar(label='Normalized Energy')
    plt.title('VAE Latent Space Energy Function')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create VAE energy function
    vae_energy = VAEEnergy(
        vae_path='models/vae_best.pt',
        device=device
    )
    
    # Visualize the energy function
    vae_energy.visualize_latent_energy(save_path="results/vae_energy.png")
    
    # Generate random samples
    random_samples = torch.randn(100, 2).to(device)
    
    # Visualize random samples
    vae_energy.visualize_latent_samples(random_samples, save_path="results/vae_random_samples.png")
    
    # Decode and visualize
    vae_energy.decode_and_visualize(random_samples[:16], grid_shape=(4, 4), save_path="results/vae_decoded_samples.png")
    
    print("VAE energy visualization completed!") 