"""
GFN-Diffusion VAE module.

This module provides functionality for training and sampling from VAE-based
energy functions using GFlowNet-guided diffusion.
"""

# Make classes and functions available at the package level
from .energies.vae import VAE, load_mnist, train_vae, test_vae
from .energies.vae_energy import VAEEnergy, get_vae_energy 