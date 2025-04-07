"""
Weights & Biases (wandb) utility functions for GFN-Diffusion.

This file contains utility functions for logging to wandb and
ensuring consistent tracking settings across the project.
"""

import os
import wandb
import numpy as np
import matplotlib.pyplot as plt


def init_wandb(args, project_name=None, tags=None):
    """
    Initialize a wandb run with consistent settings.
    
    Args:
        args: Command-line arguments
        project_name: Project name (overrides args.wandb_project if provided)
        tags: List of tags for the run
        
    Returns:
        wandb_run: The initialized wandb run
    """
    if not args.wandb:
        return None
    
    if project_name is None:
        project_name = args.wandb_project
    
    # Make sure the run name is set
    if args.run_name is None:
        args.run_name = f"gfn-diffusion_{args.energy}"
        if hasattr(args, 'conditional') and args.conditional:
            args.run_name += "_conditional"
        if hasattr(args, 'simple_unet') and args.simple_unet:
            args.run_name += "_simple"
    
    # Initialize wandb
    wandb_run = wandb.init(
        project=project_name,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args),
        tags=tags
    )
    
    return wandb_run


def log_energy_function(energy_fn, device, args, range_val=3.0, resolution=100, log_name="energy_function"):
    """
    Visualize and log an energy function to wandb.
    
    Args:
        energy_fn: Energy function to visualize
        device: Device to run computation on
        args: Command-line arguments
        range_val: Range of values to visualize
        resolution: Resolution of visualization grid
        log_name: Name for the wandb log
    """
    if not args.wandb:
        return
    
    fig = plt.figure(figsize=(8, 8))
    
    # Create grid
    x_range = np.linspace(-range_val, range_val, resolution)
    y_range = np.linspace(-range_val, range_val, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    XY = np.column_stack([X.flatten(), Y.flatten()])
    XY_torch = torch.tensor(XY, dtype=torch.float32, device=device)
    
    # Compute energy
    with torch.no_grad():
        energies = energy_fn(XY_torch)
    
    # Normalize for visualization
    energies = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)
    
    # Plot energy
    Z = energies.cpu().numpy().reshape(X.shape)
    plt.contourf(X, Y, Z, 100, cmap='viridis')
    plt.colorbar(label='Normalized Energy')
    plt.title("Energy Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({log_name: wandb.Image(fig)})
    plt.close(fig)


def log_samples(samples, args, title="Samples", log_name="samples"):
    """
    Log samples to wandb.
    
    Args:
        samples: Samples to log
        args: Command-line arguments
        title: Title for the plot
        log_name: Name for the wandb log
    """
    if not args.wandb:
        return
    
    fig = plt.figure(figsize=(8, 8))
    
    # Plot samples
    plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.7)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({log_name: wandb.Image(fig)})
    plt.close(fig)


def log_comparison(samples_no_gfn, samples_with_gfn, args, title="Samples Comparison", log_name="samples_comparison"):
    """
    Log a comparison of samples with and without GFN guidance to wandb.
    
    Args:
        samples_no_gfn: Samples without GFN guidance
        samples_with_gfn: Samples with GFN guidance
        args: Command-line arguments
        title: Title for the plot
        log_name: Name for the wandb log
    """
    if not args.wandb:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot samples without GFN
    axes[0].scatter(samples_no_gfn[:, 0].cpu(), samples_no_gfn[:, 1].cpu(), alpha=0.7)
    axes[0].set_title("Without GFN Guidance")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    
    # Plot samples with GFN
    axes[1].scatter(samples_with_gfn[:, 0].cpu(), samples_with_gfn[:, 1].cpu(), alpha=0.7)
    axes[1].set_title("With GFN Guidance")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({log_name: wandb.Image(fig)})
    plt.close(fig)


def finish_wandb(args):
    """
    Finish wandb run.
    
    Args:
        args: Command-line arguments
    """
    if args.wandb:
        wandb.finish() 