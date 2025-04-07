"""
Example script to demonstrate GFN-guided diffusion for both unconditional and conditional sampling.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from energy_sampling.models import UNet, ConditionalUNet, DiffusionSchedule, GFNDiffusion, gmm_energy

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def unconditional_example():
    """
    Example of unconditional GFN-diffusion for energy sampling
    """
    print("\n===== Unconditional GFN-Diffusion Example =====")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup energy function (25 GMM)
    side_length = 5
    x_grid = torch.linspace(-4, 4, side_length)
    y_grid = torch.linspace(-4, 4, side_length)
    means = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='ij'), dim=-1).reshape(-1, 2).to(device)
    weights = torch.ones(side_length * side_length).to(device)
    weights /= weights.sum()
    
    energy_fn = lambda x: gmm_energy(x, means, weights, std=0.1)
    
    # Setup diffusion schedule
    schedule = DiffusionSchedule(num_timesteps=100, schedule_type="linear")
    
    # Setup model
    model = UNet(input_dim=2, hidden_dim=64, output_dim=2)
    
    # Setup diffusion model
    diffusion = GFNDiffusion(
        model=model,
        dim=2,
        schedule=schedule,
        energy_fn=energy_fn,
        device=device
    )
    
    # Display energy function
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        grid = grid.to(device)
        energy = energy_fn(grid).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, energy.reshape(100, 100), levels=20, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.title("Target Energy Function (25 GMM)")
    plt.savefig("energy_contour.png")
    plt.close()
    
    print("Energy function visualization saved as energy_contour.png")
    
    # Generate samples with and without GFN guidance
    print("Sampling with and without GFN guidance...")
    
    with torch.no_grad():
        # Without GFN guidance
        samples_no_gfn = diffusion.sample(
            num_samples=100,
            use_gfn=False,
            num_steps=50
        ).cpu().numpy()
        
        # With GFN guidance
        samples_gfn = diffusion.sample(
            num_samples=100,
            use_gfn=True,
            guidance_scale=1.0,
            num_steps=50
        ).cpu().numpy()
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, energy.reshape(100, 100), levels=20, cmap='viridis', alpha=0.5)
    plt.scatter(samples_no_gfn[:, 0], samples_no_gfn[:, 1], c='r', alpha=0.6, s=10, label="No GFN")
    plt.scatter(samples_gfn[:, 0], samples_gfn[:, 1], c='b', alpha=0.6, s=10, label="With GFN")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.legend()
    plt.title("Comparison: With vs Without GFN Guidance")
    plt.savefig("unconditional_comparison.png")
    plt.close()
    
    print("Comparison plot saved as unconditional_comparison.png")


def conditional_example():
    """
    Example of conditional GFN-diffusion for VAE-like problems
    """
    print("\n===== Conditional GFN-Diffusion Example =====")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup dimensions
    condition_dim = 10
    latent_dim = 2
    
    # Setup diffusion schedule
    schedule = DiffusionSchedule(num_timesteps=100, schedule_type="linear")
    
    # Setup conditional model
    model = ConditionalUNet(
        input_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dim=64,
        output_dim=latent_dim
    )
    
    # Setup diffusion model
    diffusion = GFNDiffusion(
        model=model,
        dim=latent_dim,
        schedule=schedule,
        device=device
    )
    
    # Generate some example condition data (e.g., positions along a circle)
    batch_size = 5
    theta = torch.linspace(0, 2 * np.pi, batch_size).to(device)
    x_circle = torch.cos(theta)
    y_circle = torch.sin(theta)
    
    # Create condition tensor
    condition = torch.zeros(batch_size, condition_dim).to(device)
    condition[:, 0] = x_circle * 3  # Amplify for clarity
    condition[:, 1] = y_circle * 3
    # Add random values to remaining dimensions
    condition[:, 2:] = torch.randn(batch_size, condition_dim - 2).to(device) * 0.1
    
    print(f"Generated {batch_size} condition points along a circle.")
    
    # Sample latents for each condition point
    print("Sampling latents for each condition point...")
    
    samples_per_condition = 10
    all_latents_no_gfn = []
    all_latents_gfn = []
    all_conditions = []
    
    with torch.no_grad():
        for i in range(batch_size):
            single_condition = condition[i:i+1].repeat(samples_per_condition, 1)
            all_conditions.extend([single_condition[0].cpu().numpy()] * samples_per_condition)
            
            # Without GFN guidance
            latents_no_gfn = diffusion.sample(
                num_samples=samples_per_condition,
                condition=single_condition,
                use_gfn=False,
                num_steps=50
            ).cpu().numpy()
            all_latents_no_gfn.append(latents_no_gfn)
            
            # With GFN guidance
            latents_gfn = diffusion.sample(
                num_samples=samples_per_condition,
                condition=single_condition,
                use_gfn=True,
                guidance_scale=1.0,
                num_steps=50
            ).cpu().numpy()
            all_latents_gfn.append(latents_gfn)
    
    all_latents_no_gfn = np.vstack(all_latents_no_gfn)
    all_latents_gfn = np.vstack(all_latents_gfn)
    all_conditions = np.array(all_conditions)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot conditions (first 2 dimensions)
    plt.subplot(1, 3, 1)
    for i in range(batch_size):
        plt.scatter(
            condition[i, 0].cpu().numpy(), 
            condition[i, 1].cpu().numpy(), 
            s=100, 
            label=f"Condition {i+1}" if i == 0 else None
        )
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Condition Points")
    plt.legend()
    
    # Plot latents without GFN
    plt.subplot(1, 3, 2)
    for i in range(batch_size):
        latents = all_latents_no_gfn[i*samples_per_condition:(i+1)*samples_per_condition]
        plt.scatter(latents[:, 0], latents[:, 1], label=f"Condition {i+1}" if i == 0 else None)
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title("Latents without GFN Guidance")
    
    # Plot latents with GFN
    plt.subplot(1, 3, 3)
    for i in range(batch_size):
        latents = all_latents_gfn[i*samples_per_condition:(i+1)*samples_per_condition]
        plt.scatter(latents[:, 0], latents[:, 1], label=f"Condition {i+1}" if i == 0 else None)
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title("Latents with GFN Guidance")
    
    plt.tight_layout()
    plt.savefig("conditional_comparison.png")
    plt.close()
    
    print("Comparison plot saved as conditional_comparison.png")


if __name__ == "__main__":
    # Run examples
    unconditional_example()
    conditional_example() 