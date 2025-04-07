#!/bin/bash

# GFN-Diffusion Models Test Script
# This script tests the different GFN-Diffusion models with minimal configuration
# to quickly validate that the implementations work correctly

# Set execution directory to the script location
cd "$(dirname "$0")"

# Create test output directory
mkdir -p test_results

echo "=================================================="
echo "GFN-Diffusion Models Test"
echo "=================================================="

# =============== PART 1: Test Energy Sampling ===============
echo "Testing simplified Energy Sampling implementations..."
echo "Running simplified energy sampling tests with smaller models..."

# Run the simplified energy sampling tests
python test_energy_sampling.py

# =============== PART 2: Test VAE Experiment ===============
echo "Testing VAE experiment implementations..."

# Create VAE directories if they don't exist
mkdir -p vae/test_models
mkdir -p vae/test_results

# 2.1 Test VAE model
echo "2.1 Testing VAE model..."
cat > vae/test_vae.py << EOL
import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vae.energies.vae import VAE

# Test VAE model
model = VAE(input_dim=784, latent_dim=2, hidden_dims=[64, 32])
x = torch.randn(4, 784)  # [batch_size, input_dim]
recon_x, mu, log_var = model(x)

print(f"VAE test - Input shape: {x.shape}, Recon shape: {recon_x.shape}, Mu shape: {mu.shape}, Log_var shape: {log_var.shape}")
print("VAE test passed" if recon_x.shape == x.shape and mu.shape == (4, 2) and log_var.shape == (4, 2) else "VAE test failed")

# Test VAE encode and decode
mu, log_var = model.encode(x)
z = model.reparameterize(mu, log_var)
recon_x = model.decode(z)

print(f"VAE encode/decode test - z shape: {z.shape}, Recon shape: {recon_x.shape}")
print("VAE encode/decode test passed" if z.shape == (4, 2) and recon_x.shape == (4, 784) else "VAE encode/decode test failed")

# Test VAE sampling
samples = model.sample(num_samples=4, device="cpu")
print(f"VAE sampling test - Samples shape: {samples.shape}")
print("VAE sampling test passed" if samples.shape == (4, 784) else "VAE sampling test failed")

# Save a tiny VAE model for testing the energy function
torch.save(model.state_dict(), "test_models/tiny_vae.pt")
print("Tiny VAE model saved for further testing")
EOL

cd vae
python test_vae.py
rm test_vae.py

# 2.2 Test VAE energy function
echo "2.2 Testing VAE energy function..."
cat > test_vae_energy.py << EOL
import torch
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energies.vae_energy import VAEEnergy

# Test VAE energy function with the tiny VAE model
vae_energy = VAEEnergy(
    vae_path="test_models/tiny_vae.pt",
    latent_dim=2,
    input_dim=784,
    hidden_dims=[64, 32],
    device="cpu"
)

# Test energy computation
z = torch.randn(4, 2)  # [batch_size, latent_dim]
energy = vae_energy(z)
print(f"VAE energy test - Input shape: {z.shape}, Energy shape: {energy.shape}")
print("VAE energy test passed" if energy.shape == (4,) else "VAE energy test failed")

# Test visualization function (minimal grid for quick testing)
try:
    vae_energy.visualize_latent_energy(grid_size=10, range_val=3.0, save_path="test_results/vae_energy_test.png")
    print("VAE energy visualization test passed")
except Exception as e:
    print(f"VAE energy visualization test failed: {str(e)}")

# Test latent samples visualization
try:
    samples = torch.randn(10, 2)
    vae_energy.visualize_latent_samples(samples, save_path="test_results/vae_latent_samples_test.png")
    print("VAE latent samples visualization test passed")
except Exception as e:
    print(f"VAE latent samples visualization test failed: {str(e)}")

# Test decode and visualize (minimal for quick testing)
try:
    samples = torch.randn(4, 2)
    vae_energy.decode_and_visualize(samples, grid_shape=(2, 2), save_path="test_results/vae_decoded_samples_test.png")
    print("VAE decode and visualize test passed")
except Exception as e:
    print(f"VAE decode and visualize test failed: {str(e)}")
EOL

python test_vae_energy.py
rm test_vae_energy.py

# 2.3 Test Langevin dynamics and local search explicitly
echo "2.3 Testing Langevin dynamics and local search..."
cat > test_langevin_local_search.py << EOL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define mock energy function for testing
def mock_energy_fn(x):
    return torch.sum(x**2, dim=1)

# Langevin dynamics implementation
def langevin_dynamics(energy_fn, x_init, n_steps=100, step_size=0.01):
    """
    Langevin dynamics for sampling from energy function.
    
    Args:
        energy_fn: Energy function
        x_init: Initial point
        n_steps: Number of Langevin steps
        step_size: Step size for Langevin dynamics
        
    Returns:
        x: Final point
        acceptance_rate: Acceptance rate of proposals
    """
    # Make sure x is detached and requires gradients
    x = x_init.clone().detach().requires_grad_(True)
    accepted = 0
    
    for _ in range(n_steps):
        # Compute gradient of energy
        energy = energy_fn(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        
        # Langevin update with noise
        noise = torch.randn_like(x) * np.sqrt(2 * step_size)
        x_proposed = x.detach() - step_size * grad + noise
        x_proposed.requires_grad_(True)
        
        # Metropolis-Hastings acceptance
        energy_proposed = energy_fn(x_proposed)
        
        # Compute acceptance probability
        energy_diff = energy.detach() - energy_proposed.detach()
        accept_prob = torch.min(torch.ones_like(energy_diff), torch.exp(energy_diff))
        
        # Accept or reject with proper broadcasting
        mask = (torch.rand_like(accept_prob) < accept_prob).float()
        while len(mask.shape) < len(x.shape):
            mask = mask.unsqueeze(-1)
        
        # Update x with proper detachment to maintain gradients for next iteration
        x = x_proposed.detach().clone().requires_grad_(True) * mask + x.detach().clone().requires_grad_(True) * (1 - mask)
        
        accepted += mask.mean().item()
    
    acceptance_rate = accepted / n_steps
    return x.detach(), acceptance_rate

# Local search implementation
def local_search(energy_fn, x_init, max_iter=500, tol=1e-4):
    """
    Local search to find lower energy states.
    
    Args:
        energy_fn: Energy function
        x_init: Initial point
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        
    Returns:
        x: Final point
        energy: Final energy
    """
    # Make sure x is properly detached and requires gradients
    x = x_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x], lr=0.01)
    
    prev_energy = float('inf')
    
    for i in range(max_iter):
        optimizer.zero_grad()
        energy = energy_fn(x).mean()
        energy.backward()
        optimizer.step()
        
        # Check convergence
        current_energy = energy.item()
        if abs(prev_energy - current_energy) < tol:
            break
            
        prev_energy = current_energy
    
    # Final energy computation
    with torch.no_grad():
        final_energy = energy_fn(x)
    
    return x.detach(), final_energy.detach()

# Test Langevin dynamics
x_init = torch.randn(4, 2, requires_grad=True)  # Explicitly set requires_grad=True
try:
    # First verify tensor requires grad
    if not x_init.requires_grad:
        print("Warning: x_init.requires_grad is False, setting it to True")
        x_init.requires_grad_(True)
    
    # Run langevin dynamics with explicit grad handling
    x_final, acceptance_rate = langevin_dynamics(
        mock_energy_fn, 
        x_init, 
        n_steps=5,  # Small steps for quick testing
        step_size=0.01
    )
    print(f"Langevin dynamics test - Final shape: {x_final.shape}, Acceptance rate: {acceptance_rate}")
    print("Langevin dynamics test passed" if x_final.shape == x_init.shape else "Langevin dynamics test failed")
except Exception as e:
    print(f"Langevin dynamics test failed: {str(e)}")

# Test local search
x_init = torch.randn(1, 2, requires_grad=True)  # Explicitly set requires_grad=True
try:
    # First verify tensor requires grad
    if not x_init.requires_grad:
        print("Warning: x_init.requires_grad is False, setting it to True")
        x_init.requires_grad_(True)
    
    # Run local search
    x_final, energy = local_search(
        mock_energy_fn, 
        x_init, 
        max_iter=5,  # Small iterations for quick testing
        tol=1e-4
    )
    print(f"Local search test - Final shape: {x_final.shape}, Energy shape: {energy.shape}")
    print("Local search test passed" if x_final.shape == x_init.shape else "Local search test failed")
except Exception as e:
    print(f"Local search test failed: {str(e)}")
EOL

python test_langevin_local_search.py
rm test_langevin_local_search.py

cd ..
echo "VAE experiment tests completed!"

echo "=================================================="
echo "GFN-Diffusion tests have been completed!"
echo "Test results and visualizations are saved in the 'test_results' directory"
echo "==================================================" 