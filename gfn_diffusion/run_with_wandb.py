#!/usr/bin/env python
"""
Run GFN-Diffusion experiments with wandb logging.

This script runs both energy-based and VAE-based GFN-Diffusion experiments
with wandb logging for tracking results.
"""

import os
import argparse
import subprocess
import torch
from datetime import datetime
import sys


def parse_args():
    # Check if CUDA is actually available
    cuda_available = torch.cuda.is_available()
    default_device = "cuda" if cuda_available else "cpu"
    
    parser = argparse.ArgumentParser(description="Run GFN-Diffusion experiments with wandb logging")
    
    # General parameters
    parser.add_argument("--experiment_type", type=str, default="energy", choices=["energy", "vae", "both"],
                        help="Type of experiment to run")
    parser.add_argument("--device", type=str, default=default_device,
                        help="Device to run experiments on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default="gfn-diffusion",
                        help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="nadhirvincenthassen",
                        help="Wandb entity name")
    parser.add_argument("--run_prefix", type=str, default=None,
                        help="Prefix for run names")
    parser.add_argument("--install_wandb", action="store_true",
                        help="Install wandb if not already installed")
    parser.add_argument("--update_requirements", action="store_true",
                        help="Update requirements.txt file to include wandb")
    
    # Energy experiment parameters
    parser.add_argument("--energy_type", type=str, default="gmm", choices=["gmm", "rings", "moons"],
                        help="Type of energy function for energy-based experiment")
    parser.add_argument("--conditional", action="store_true",
                        help="Whether to run conditional experiments")
    parser.add_argument("--num_conditions", type=int, default=4,
                        help="Number of conditions for conditional experiments")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension of models")
    parser.add_argument("--simple_unet", action="store_true",
                        help="Whether to use simplified UNet implementation")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    
    # GFN parameters
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Scale for GFN guidance")
    
    return parser.parse_args()


def run_energy_experiment(args):
    """
    Run energy-based GFN-Diffusion experiments.
    
    Args:
        args: Command-line arguments
    """
    print("Running energy-based GFN-Diffusion experiment...")
    
    # Create timestamp for run name if prefix not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive run name
    if args.conditional:
        condition_info = f"cond{args.num_conditions}"
    else:
        condition_info = "uncond"
    
    device_info = "gpu" if "cuda" in args.device else "cpu"
    guidance_info = f"gs{args.guidance_scale}"
    
    # Format: [prefix]_energy_[energy-type]_[conditional/unconditional]_gs[scale]_[device]_[timestamp]
    if args.run_prefix:
        run_name = f"{args.run_prefix}_energy_{args.energy_type}_{condition_info}_{guidance_info}_{device_info}_{timestamp}"
    else:
        run_name = f"energy_{args.energy_type}_{condition_info}_{guidance_info}_{device_info}_{timestamp}"
    
    # Map energy_type to what's expected by train.py
    energy_mapping = {
        "gmm": "25gmm",
        "rings": "many_well",
        "moons": "3gmm"
    }
    
    energy_value = energy_mapping.get(args.energy_type, "25gmm")
    
    # Build command
    cmd = [
        "python", "gfn_diffusion/energy_sampling/train.py",
        "--wandb",
        f"--wandb_project={args.wandb_project}",
        f"--device={args.device}",
        f"--seed={args.seed}",
        f"--energy={energy_value}",
        f"--hidden_dim={args.hidden_dim}",
        f"--epochs={args.epochs}",
        f"--lr={args.lr}",
        f"--batch_size={args.batch_size}",
        f"--guidance_scale={args.guidance_scale}",
        f"--run_name={run_name}"
    ]
    
    # Add conditional args if needed
    if args.conditional:
        cmd.append("--conditional")
        cmd.append(f"--num_conditions={args.num_conditions}")
    
    # Add wandb entity if provided
    if args.wandb_entity:
        cmd.append(f"--wandb_entity={args.wandb_entity}")
    
    # Add simple_unet flag if provided
    if args.simple_unet:
        cmd.append("--simple_unet")
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_vae_experiment(args):
    """
    Run VAE-based GFN-Diffusion experiments.
    
    Args:
        args: Command-line arguments
    """
    print("Running VAE-based GFN-Diffusion experiment...")
    
    # Create timestamp for run name if prefix not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive run names
    device_info = "gpu" if "cuda" in args.device else "cpu"
    guidance_info = f"gs{args.guidance_scale}"
    dim_info = f"h{args.hidden_dim}"
    
    # Format: [prefix]_vae_training_[dim-info]_[device]_[timestamp]
    if args.run_prefix:
        vae_run_name = f"{args.run_prefix}_vae_training_{dim_info}_{device_info}_{timestamp}"
        gfn_run_name = f"{args.run_prefix}_vae_gfn_{guidance_info}_{dim_info}_{device_info}_{timestamp}"
    else:
        vae_run_name = f"vae_training_{dim_info}_{device_info}_{timestamp}"
        gfn_run_name = f"vae_gfn_{guidance_info}_{dim_info}_{device_info}_{timestamp}"
    
    # Build command for training VAE
    vae_cmd = [
        "python", "gfn_diffusion/vae/train.py",
        "--train_vae",
        "--wandb",
        f"--wandb_project={args.wandb_project}-vae",
        f"--device={args.device}",
        f"--seed={args.seed}",
        f"--hidden_dim={args.hidden_dim}",
        f"--epochs=1000",  # Fewer epochs for VAE
        f"--batch_size={args.batch_size}",
        f"--run_name={vae_run_name}"
    ]
    
    # Add wandb entity if provided
    if args.wandb_entity:
        vae_cmd.append(f"--wandb_entity={args.wandb_entity}")
    
    # Run VAE training command
    print(f"Running VAE training command: {' '.join(vae_cmd)}")
    subprocess.run(vae_cmd)
    
    # Build command for training GFN-Diffusion with VAE
    gfn_cmd = [
        "python", "gfn_diffusion/vae/train.py",
        "--train_gfn",
        "--wandb",
        f"--wandb_project={args.wandb_project}-vae",
        f"--device={args.device}",
        f"--seed={args.seed}",
        f"--hidden_dim={args.hidden_dim}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--guidance_scale={args.guidance_scale}",
        f"--run_name={gfn_run_name}"
    ]
    
    # Add wandb entity if provided
    if args.wandb_entity:
        gfn_cmd.append(f"--wandb_entity={args.wandb_entity}")
    
    # Add langevin dynamics and local search options
    gfn_cmd.append("--langevin")
    
    # Run GFN-Diffusion training command
    print(f"Running GFN-Diffusion training command: {' '.join(gfn_cmd)}")
    subprocess.run(gfn_cmd)


def main():
    args = parse_args()
    
    # Check CUDA availability if cuda was requested
    if "cuda" in args.device and not torch.cuda.is_available():
        print("WARNING: CUDA was requested but is not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Install wandb if requested
    if args.install_wandb:
        try:
            import wandb
            print("wandb is already installed.")
        except ImportError:
            print("Installing wandb...")
            subprocess.run([sys.executable, "-m", "pip", "install", "wandb"])
            print("wandb installation complete.")
            
            # Import wandb after installation
            import wandb
    
    # Update requirements.txt if requested
    if args.update_requirements:
        req_file = "requirements.txt"
        
        if os.path.exists(req_file):
            with open(req_file, "r") as f:
                requirements = f.read()
                
            if "wandb" not in requirements:
                with open(req_file, "a") as f:
                    f.write("\nwandb>=0.12.0\n")
                print(f"Added wandb to {req_file}")
            else:
                print(f"wandb already in {req_file}")
        else:
            with open(req_file, "w") as f:
                f.write("wandb>=0.12.0\n")
            print(f"Created {req_file} with wandb dependency")
    
    print(f"Running GFN-Diffusion experiments on device: {args.device}")
    print(f"Wandb project: {args.wandb_project}")
    
    # Run selected experiments
    if args.experiment_type == "energy" or args.experiment_type == "both":
        run_energy_experiment(args)
    
    if args.experiment_type == "vae" or args.experiment_type == "both":
        run_vae_experiment(args)
    
    print("All experiments completed!")


if __name__ == "__main__":
    main() 