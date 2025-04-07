#!/usr/bin/env python
"""
Test script to validate VAE model and energy fixes.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try imports with debug info
print("Testing imports...")

try:
    from gfn_diffusion.vae.energies.vae import VAE
    print("✓ Successfully imported VAE")
except Exception as e:
    print(f"✗ Error importing VAE: {str(e)}")

try:
    from gfn_diffusion.vae.energies.vae_energy import VAEEnergy, get_vae_energy
    print("✓ Successfully imported VAE energy functions")
except Exception as e:
    print(f"✗ Error importing VAE energy: {str(e)}")

# Test VAE creation
print("\nTesting VAE creation...")
try:
    device = torch.device("cpu")
    vae = VAE(input_dim=784, latent_dim=2, hidden_dim=512)
    print(f"✓ Successfully created VAE: {vae}")
except Exception as e:
    print(f"✗ Error creating VAE: {str(e)}")

# Test energy function
print("\nTesting VAE energy function...")
try:
    # Create a dummy VAE model for testing
    vae = VAE(input_dim=784, latent_dim=2, hidden_dim=512).to(device)
    
    # Create energy function
    energy_fn = get_vae_energy(vae, device=device)
    
    # Test with random latent vectors
    z = torch.randn(10, 2, device=device)
    energy = energy_fn(z)
    
    print(f"✓ Successfully computed energy: {energy.shape}")
    print(f"  Energy values: {energy}")
except Exception as e:
    print(f"✗ Error computing energy: {str(e)}")

# Test VAE energy class
print("\nTesting VAE energy class...")
try:
    # Create output dir
    os.makedirs("test_results", exist_ok=True)
    
    # We'll create a temporary model file for testing
    temp_model_path = "test_results/temp_vae.pt"
    vae = VAE(input_dim=784, latent_dim=2, hidden_dim=512)
    torch.save(vae.state_dict(), temp_model_path)
    
    # Create energy class
    vae_energy = VAEEnergy(
        vae_path=temp_model_path,
        latent_dim=2,
        input_dim=784,
        device=device
    )
    
    # Test with random latent vectors
    z = torch.randn(10, 2, device=device)
    energy = vae_energy(z)
    
    print(f"✓ Successfully computed energy with class: {energy.shape}")
    
    # Test visualization
    try:
        vae_energy.visualize_latent_energy(
            n_points=20,  # Use a small grid for quick testing
            save_path="test_results/energy_test.png"
        )
        print("✓ Successfully visualized energy")
    except Exception as e:
        print(f"✗ Error visualizing energy: {str(e)}")
    
    # Clean up
    os.remove(temp_model_path)
except Exception as e:
    print(f"✗ Error with VAE energy class: {str(e)}")

print("\nTest completed!") 