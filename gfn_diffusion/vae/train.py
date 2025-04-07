"""
Training script for GFN-Diffusion with VAE-based energy functions.

This script implements the VAE experiment from the GFN-Diffusion paper,
training a diffusion model to sample from the latent space of a VAE
with GFlowNet-guided sampling.
"""

import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from pathlib import Path

# Safely import torch first - avoid library version conflicts
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Try to import torchvision with fallback
try:
    from torchvision import datasets, transforms
except ImportError:
    print("Installing torchvision...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "torchvision"])
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("Warning: Unable to import torchvision. Some features may not work.")

# Try to import wandb with a graceful fallback
try:
    import wandb
except ImportError:
    print("Warning: wandb not found. Install with 'pip install wandb' for experiment tracking.")
    wandb = None

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import energy sampling modules
from energy_sampling.models import UNet, ConditionalUNet, DiffusionSchedule, GFNDiffusion
from energies.vae_energy import VAEEnergy, get_vae_energy
from energies.vae import VAE

# Import metrics and visualization utilities
try:
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.metrics import (
        kl_divergence,
        l1_distance, 
        l2_distance,
        earth_movers_distance,
        compute_entropy,
        compute_diversity,
        compute_novelty,
        compute_energy_improvement,
        compute_effective_sample_size,
        compute_coverage_metrics,
        compute_energy_statistics
    )
    from utils.visualization import (
        plot_2d_density_comparison,
        plot_energy_evolution,
        create_comparative_trajectory_plot,
        create_metric_comparison_plot
    )
except ImportError as e:
    print(f"Warning: Could not import metrics and visualization utilities: {str(e)}")
    # Define simple placeholder functions if imports fail
    def compute_diversity(samples, threshold=0.1): return 0.0
    def compute_energy_statistics(samples, energy_fn): 
        return {"mean_energy": 0.0, "min_energy": 0.0, "max_energy": 0.0, "std_energy": 0.0, "p50": 0.0}
    def compute_entropy(samples, bins=20, range_val=5.0): return 0.0
    def compute_energy_improvement(standard_samples, gfn_samples, energy_fn): return 0.0, 0.0
    def compute_effective_sample_size(samples, energy_fn, temperature=1.0): return len(samples) * 0.5, 0.5
    def compute_coverage_metrics(samples, reference_centers=None, energy_fn=None, range_val=5.0, n_grid=10):
        return {"grid_coverage": 0.0}
    def compute_novelty(generated_samples, reference_samples, threshold=0.1): return 0.0
    def kl_divergence(p, q, eps=1e-9): return 0.0
    def l1_distance(p, q): return 0.0
    def l2_distance(p, q): return 0.0
    def earth_movers_distance(p_samples, q_samples): return 0.0

# Create directories for results
os.makedirs("results/vae_experiment", exist_ok=True)
os.makedirs("models/vae_experiment", exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VAE and GFN-Diffusion for VAE latent space sampling")
    
    # General settings
    parser.add_argument("--train_vae", action="store_true", help="Train VAE model")
    parser.add_argument("--train_gfn", action="store_true", help="Train GFN-Diffusion model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="results/vae_experiment", 
                        help="Directory to save results")
    parser.add_argument("--model_dir", type=str, default="models/vae_experiment", 
                        help="Directory to save models")
    
    # VAE parameters
    parser.add_argument("--latent_dim", type=int, default=2, help="Dimension of latent space")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Dimension of hidden layers")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for KL divergence term in loss")
    
    # Data parameters
    parser.add_argument("--input_dim", type=int, default=784, help="Dimension of input data (28x28=784 for MNIST)")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr_policy", type=float, default=1e-3, help="Learning rate for policy network")
    parser.add_argument("--lr_flow", type=float, default=1e-3, help="Learning rate for flow network")
    parser.add_argument("--lr_back", type=float, default=1e-3, help="Learning rate for backbone network")
    
    # Diffusion parameters
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--schedule_type", type=str, default="linear", help="Schedule type for diffusion")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="GFN guidance scale")
    
    # VAE path for GFN-Diffusion training
    parser.add_argument("--vae_path", type=str, default="models/vae_experiment/vae_best.pt", 
                        help="Path to pretrained VAE model")
    
    # Logging and evaluation
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Save interval")
    parser.add_argument("--sample_interval", type=int, default=500, help="Sample visualization interval")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--sampling_steps", type=int, default=100, help="Number of sampling steps")
    parser.add_argument("--visualize_decoded", action="store_true", help="Visualize decoded samples")
    
    # Wandb logging
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--offline", action="store_true", help="Run wandb in offline mode")
    parser.add_argument("--wandb_project", type=str, default="gfn-diffusion-experiments", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="nadhirvincenthassen", help="Wandb entity name")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    
    # Advanced features
    parser.add_argument("--langevin", action="store_true", help="Use Langevin dynamics for comparison")
    parser.add_argument("--ld_step", type=float, default=0.01, help="Step size for Langevin dynamics")
    parser.add_argument("--ld_schedule", action="store_true", help="Use step size schedule for Langevin dynamics")
    parser.add_argument("--target_acceptance_rate", type=float, default=0.574, help="Target acceptance rate for Langevin dynamics")
    
    parser.add_argument("--buffer_size", type=int, default=10000, help="Size of replay buffer")
    parser.add_argument("--prioritized", action="store_true", help="Use prioritized replay buffer")
    parser.add_argument("--rank_weight", type=float, default=0.01, help="Weight for rank-based prioritization")
    
    parser.add_argument("--exploratory", action="store_true", help="Use exploratory objective")
    parser.add_argument("--exploration_factor", type=float, default=0.1, help="Exploration factor for exploratory objective")
    parser.add_argument("--exploration_wd", action="store_true", help="Use weight decay for exploration")
    
    parser.add_argument("--local_search", action="store_true", help="Use local search to find better states")
    parser.add_argument("--max_iter_ls", type=int, default=500, help="Maximum iterations for local search")
    parser.add_argument("--burn_in", type=int, default=200, help="Burn-in period before starting local search")
    
    parser.add_argument("--clipping", action="store_true", help="Use gradient clipping")
    parser.add_argument("--zero_init", action="store_true", help="Initialize policy/flow parameters to zero")
    
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Create visualizations during evaluation")
    
    parser.add_argument("--langevin_steps", type=int, default=10,
                        help="Number of Langevin dynamics steps for exploration")
    
    parser.add_argument("--langevin_step_size", type=float, default=0.1,
                        help="Step size for Langevin dynamics")
    
    parser.add_argument("--langevin_noise_scale", type=float, default=0.01,
                        help="Noise scale for Langevin dynamics")
    
    parser.add_argument("--replay_buffer_size", type=int, default=10000,
                        help="Size of the replay buffer")
    
    parser.add_argument("--replay_buffer_update_freq", type=int, default=10,
                        help="Frequency of replay buffer updates")
    
    parser.add_argument("--prioritized_replay", type=bool, default=True,
                        help="Whether to use prioritized replay")
    
    parser.add_argument("--exploratory_sampling", type=bool, default=True,
                        help="Whether to use exploratory sampling")
    
    parser.add_argument("--adaptive_step_size", type=bool, default=True,
                        help="Whether to use adaptive step size for guidance")
    
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    
    parser.add_argument("--local_search_steps", type=int, default=5,
                        help="Number of local search steps for refinement")
    
    args = parser.parse_args()
    return args


class ReplayBuffer:
    """
    Replay buffer for GFN training with prioritized sampling.
    
    Args:
        buffer_size: Maximum size of the buffer
        prioritization: Type of prioritization ('rank', 'proportional', 'uniform')
        rank_weight: Weight for rank-based prioritization
    """
    def __init__(self, buffer_size, prioritization="rank", rank_weight=0.01):
        self.buffer_size = buffer_size
        self.prioritization = prioritization
        self.rank_weight = rank_weight
        
        self.buffer = []
        self.priorities = []
    
    def add(self, sample, priority=None):
        """
        Add a sample to the buffer.
        
        Args:
            sample: The sample to add
            priority: Priority of the sample (if None, max priority is used)
        """
        if len(self.buffer) >= self.buffer_size:
            # Remove lowest priority sample
            min_idx = np.argmin(self.priorities)
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)
        
        if priority is None and len(self.priorities) > 0:
            priority = max(self.priorities)
        elif priority is None:
            priority = 1.0
            
        self.buffer.append(sample)
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """
        Sample a batch from the buffer based on priorities.
        
        Args:
            batch_size: Number of samples to draw
            
        Returns:
            samples: List of sampled items
        """
        if len(self.buffer) == 0:
            return []
        
        batch_size = min(batch_size, len(self.buffer))
        
        if self.prioritization == "uniform":
            indices = np.random.choice(len(self.buffer), batch_size)
        elif self.prioritization == "proportional":
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        elif self.prioritization == "rank":
            # Sort priorities in descending order
            sorted_indices = np.argsort(self.priorities)[::-1]
            # Compute rank-based probabilities
            ranks = np.arange(1, len(self.buffer) + 1)
            probs = 1.0 / (ranks + self.rank_weight)
            probs = probs / sum(probs)
            # Sample based on rank
            rank_indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            indices = sorted_indices[rank_indices]
        else:
            raise ValueError(f"Unknown prioritization: {self.prioritization}")
        
        return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities of samples.
        
        Args:
            indices: Indices of samples to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


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


def train_vae_gfn_diffusion(args):
    """
    Train a GFN-Diffusion model for VAE latent space sampling.
    
    Args:
        args: Arguments for training
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create run name if not provided
    if args.run_name is None:
        args.run_name = f"vae_gfn_latent{args.latent_dim}_scale{args.guidance_scale}"
        if args.langevin:
            args.run_name += "_langevin"
        if args.local_search:
            args.run_name += "_ls"
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args)
        )
    
    # Setup VAE energy function
    print(f"Setting up VAE energy function from {args.vae_path}")
    vae_energy = VAEEnergy(
        vae_path=args.vae_path,
        latent_dim=args.latent_dim,
        input_dim=args.input_dim,
        device=device
    )
    
    # Visualize the energy function
    vae_energy.visualize_latent_energy(save_path=output_dir / "vae_energy.png")
    
    # Setup diffusion schedule
    schedule = DiffusionSchedule(
        num_diffusion_timesteps=args.num_timesteps, 
        schedule_type=args.schedule_type
    )
    
    # Setup UNet model for diffusion
    model = UNet(
        input_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.latent_dim
    )
    
    # Setup diffusion model
    diffusion = GFNDiffusion(
        model=model,
        diffusion=schedule,
        energy_fn=vae_energy,
        guidance_scale=args.guidance_scale,
        device=device
    )
    
    # Setup optimizer
    # Different learning rates for different parts of the model based on GFN optimization
    policy_params = []
    flow_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'policy' in name:
            policy_params.append(param)
        elif 'flow' in name:
            flow_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.Adam([
        {'params': policy_params, 'lr': args.lr_policy},
        {'params': flow_params, 'lr': args.lr_flow},
        {'params': other_params, 'lr': args.lr_back}
    ])
    
    # Initialize parameters based on zero_init flag
    if args.zero_init:
        for name, param in model.named_parameters():
            if 'policy' in name or 'flow' in name:
                nn.init.zeros_(param)
    
    # Setup replay buffer
    buffer = ReplayBuffer(
        buffer_size=args.buffer_size,
        prioritization=args.prioritized,
        rank_weight=args.rank_weight
    )
    
    # Generate initial samples for the buffer
    print("Generating initial samples for the buffer...")
    
    # Use Langevin dynamics to get good initial samples
    x_init = torch.randn(args.batch_size, args.latent_dim, device=device)
    
    if args.langevin:
        x_samples, _ = langevin_dynamics(
            vae_energy, 
            x_init, 
            n_steps=100, 
            step_size=args.ld_step
        )
    else:
        x_samples = x_init
    
    # Add initial samples to buffer
    for i in range(x_samples.shape[0]):
        buffer.add(x_samples[i])
    
    # Training loop
    num_batches = args.epochs
    losses = []
    
    # Langevin step size schedule
    ld_step = args.ld_step
    
    print("Training diffusion model...")
    pbar = trange(num_batches, desc="Training")
    
    for i in pbar:
        # Sample batch from buffer
        if len(buffer) >= args.batch_size:
            batch_samples = buffer.sample(args.batch_size)
            x_batch = torch.stack(batch_samples).to(device)
        else:
            # If buffer is not filled yet, use random samples
            x_batch = torch.randn(args.batch_size, args.latent_dim, device=device)
        
        # Train diffusion model
        loss = diffusion.train_step(optimizer, x_batch)
        losses.append(loss)
        
        # Apply exploratory objective
        if args.exploratory:
            # Generate new samples
            with torch.no_grad():
                x_new = diffusion.sample(
                    num_samples=args.batch_size,
                    use_gfn=True,
                    guidance_scale=args.guidance_scale,
                    num_steps=args.sampling_steps
                )
            
            # Compute energies
            energies = vae_energy(x_new).detach()
            
            # Add to buffer with priorities based on energy
            for j in range(x_new.shape[0]):
                buffer.add(x_new[j], priority=-energies[j].item())  # Lower energy = higher priority
            
            # Apply local search to some samples
            if args.local_search and i >= args.burn_in:
                # Select samples for local search
                local_search_indices = np.random.choice(args.batch_size, size=min(10, args.batch_size))
                local_search_samples = x_new[local_search_indices]
                
                # Apply local search
                improved_samples = []
                improved_energies = []
                
                for sample in local_search_samples:
                    improved_sample, improved_energy = local_search(
                        vae_energy, 
                        sample.unsqueeze(0), 
                        max_iter=args.max_iter_ls
                    )
                    improved_samples.append(improved_sample.squeeze(0))
                    improved_energies.append(improved_energy.item())
                
                # Add improved samples to buffer
                for sample, energy in zip(improved_samples, improved_energies):
                    buffer.add(sample, priority=-energy)  # Lower energy = higher priority
        
        # Apply Langevin dynamics to some samples
        if args.langevin and i % 10 == 0:
            # Generate new samples
            with torch.no_grad():
                x_langevin = diffusion.sample(
                    num_samples=args.batch_size // 2,
                    use_gfn=True,
                    guidance_scale=args.guidance_scale,
                    num_steps=args.sampling_steps
                )
            
            # Apply Langevin dynamics
            x_improved, acceptance_rate = langevin_dynamics(
                vae_energy, 
                x_langevin, 
                n_steps=100, 
                step_size=ld_step
            )
            
            # Update Langevin step size based on acceptance rate
            if args.ld_schedule:
                if acceptance_rate < args.target_acceptance_rate:
                    ld_step *= 0.9  # Decrease step size
                else:
                    ld_step *= 1.1  # Increase step size
                
                # Clip step size
                ld_step = max(1e-5, min(1e-1, ld_step))
            
            # Compute energies and add to buffer
            energies = vae_energy(x_improved).detach()
            for j in range(x_improved.shape[0]):
                buffer.add(x_improved[j], priority=-energies[j].item())
        
        # Gradient clipping
        if args.clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss:.4f}"})
        
        # Periodically generate and save samples
        if (i + 1) % args.sample_interval == 0 or i == num_batches - 1:
            # Generate samples without GFN guidance
            with torch.no_grad():
                samples_no_gfn = diffusion.sample(
                    num_samples=args.num_samples,
                    use_gfn=False,
                    num_steps=args.sampling_steps
                ).cpu()
            
            # Generate samples with GFN guidance
            with torch.no_grad():
                samples_with_gfn = diffusion.sample(
                    num_samples=args.num_samples,
                    use_gfn=True,
                    guidance_scale=args.guidance_scale,
                    num_steps=args.sampling_steps
                ).cpu()
            
            # Visualize samples in latent space
            vae_energy.visualize_latent_samples(
                samples_no_gfn, 
                save_path=output_dir / f"latent_no_gfn_{i+1}.png"
            )
            
            vae_energy.visualize_latent_samples(
                samples_with_gfn, 
                save_path=output_dir / f"latent_with_gfn_{i+1}.png"
            )
            
            # Decode and visualize samples
            vae_energy.decode_and_visualize(
                samples_no_gfn[:25], 
                grid_shape=(5, 5), 
                save_path=output_dir / f"decoded_no_gfn_{i+1}.png"
            )
            
            vae_energy.decode_and_visualize(
                samples_with_gfn[:25], 
                grid_shape=(5, 5), 
                save_path=output_dir / f"decoded_with_gfn_{i+1}.png"
            )
            
            # Save model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': i,
                'losses': losses
            }, model_dir / f"checkpoint_{i+1}.pt")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, model_dir / "model_final.pt")
    
    print(f"Training completed. Results saved to {output_dir}")
    
    # Final sampling
    try:
        samples_final = diffusion.sample(
            num_samples=args.num_samples,
            use_gfn=True,
            guidance_scale=args.guidance_scale,
            steps=args.sampling_steps
        )
        
        # Decode and visualize final samples
        with torch.no_grad():
            # Visualize latent samples
            plt.figure(figsize=(8, 8))
            plt.scatter(samples_final[:, 0].cpu(), samples_final[:, 1].cpu(), alpha=0.5)
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.title("Final Latent Samples with GFN")
            plt.savefig(f"{output_dir}/{args.run_name}_final_latent_samples.png")
            
            # Log to wandb
            if args.wandb:
                wandb.log({"final_latent_samples": wandb.Image(plt)})
            
            plt.close()
            
            # Decode samples
            if args.input_dim == 784:  # MNIST-like
                decoded_final = vae_energy.vae.decode(samples_final[:64])
                decoded_final = decoded_final.view(-1, 1, 28, 28)
                
                from torchvision.utils import make_grid
                grid_final = make_grid(decoded_final, nrow=8, normalize=True)
                grid_final = grid_final.cpu().numpy().transpose((1, 2, 0))
                
                plt.figure(figsize=(10, 10))
                plt.imshow(grid_final)
                plt.axis('off')
                plt.title("Final Decoded Samples with GFN")
                plt.savefig(f"{output_dir}/{args.run_name}_final_decoded_samples.png")
                
                # Log to wandb
                if args.wandb:
                    wandb.log({"final_decoded_samples": wandb.Image(plt)})
                
                plt.close()
    except Exception as e:
        print(f"Error during final sampling: {str(e)}")
    
    # Close wandb
    if args.wandb:
        wandb.finish()


def train_vae(args):
    """
    Train a VAE model on MNIST dataset.
    
    Args:
        args: Command line arguments
    """
    # Import os at the top level
    import os
    from energies.vae import VAE
    
    print("Training VAE model...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if specified
    if args.wandb and wandb is not None:
        # Set offline mode if requested
        if args.offline:
            os.environ["WANDB_MODE"] = "offline"
            print("Running wandb in offline mode")
            
        # Fix import path to use absolute path
        import sys
        # Add the project root to the path
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(root_dir)
        
        # Define wandb utility functions here if import fails
        def init_wandb(project=None, entity=None, config=None, name=None):
            """Initialize wandb run."""
            wandb.init(
                project=project,
                entity=None if args.offline else entity,
                config=config,
                name=name
            )
            
        def finish_wandb():
            """Finish wandb run."""
            wandb.finish()
        
        # Initialize wandb
        init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name or "vae_training"
        )
        
        if args.offline:
            print(f"Offline run data will be saved to: {os.path.abspath('wandb')}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )
    except Exception as e:
        print(f"Warning: Error loading MNIST dataset: {str(e)}")
        print("Creating dummy datasets for testing")
        # Create dummy data
        train_loader = [(torch.randn(args.batch_size, 1, 28, 28), torch.zeros(args.batch_size)) for _ in range(10)]
        test_loader = [(torch.randn(args.batch_size, 1, 28, 28), torch.zeros(args.batch_size)) for _ in range(2)]
    
    # Create VAE model with corrected parameters
    vae = VAE(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.lr_policy)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        vae.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            # Handle both real MNIST and dummy data
            if isinstance(data, torch.Tensor):
                data = data.view(-1, args.input_dim).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = vae(data)
            
            # Loss calculation
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Adjust loss based on dataset type
        if isinstance(train_loader, list):
            train_loss /= len(train_loader) * args.batch_size
        else:
            train_loss /= len(train_loader.dataset)
        
        # Evaluation
        vae.eval()
        test_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                # Handle both real MNIST and dummy data
                if isinstance(data, torch.Tensor):
                    data = data.view(-1, args.input_dim).to(device)
                
                # Forward pass
                recon_batch, mu, logvar = vae(data)
                
                # Loss calculation
                recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                
                test_loss += loss.item()
        
        # Adjust loss based on dataset type
        if isinstance(test_loader, list):
            test_loss /= len(test_loader) * args.batch_size
        else:
            test_loss /= len(test_loader.dataset)
        
        # Print progress
        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
            
            # Log to wandb
            if args.wandb and wandb is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                })
                
                # Generate and log samples
                with torch.no_grad():
                    # Generate samples
                    z = torch.randn(64, args.latent_dim).to(device)
                    samples = vae.decode(z).cpu()
                    
                    try:
                        # Get some reconstructions too
                        if not isinstance(test_loader, list):
                            test_data = next(iter(test_loader))[0][:8].to(device)
                            test_data_flat = test_data.view(-1, args.input_dim)
                            recon, _, _ = vae(test_data_flat)
                            
                            # Create a grid of original and reconstructed images
                            fig, axs = plt.subplots(4, 4, figsize=(10, 10))
                            
                            # Plot original images (top row)
                            for i in range(4):
                                axs[0, i].imshow(test_data[i].cpu().squeeze(), cmap='gray')
                                axs[0, i].axis('off')
                                axs[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
                                axs[1, i].axis('off')
                            
                            # Plot samples (bottom row)
                            for i in range(4):
                                axs[2, i].imshow(samples[i].view(28, 28), cmap='gray')
                                axs[2, i].axis('off')
                                axs[3, i].imshow(samples[i+4].view(28, 28), cmap='gray')
                                axs[3, i].axis('off')
                            
                            plt.tight_layout()
                            plt.savefig(output_dir / f"vae_samples_epoch_{epoch}.png")
                            plt.close()
                            
                            # Log to wandb
                            wandb.log({f'vae_samples_epoch_{epoch}': wandb.Image(str(output_dir / f"vae_samples_epoch_{epoch}.png"))})
                    except Exception as e:
                        print(f"Warning: Error generating samples: {str(e)}")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(vae.state_dict(), model_dir / "vae_best.pt")
            print(f"Saved best model with test loss: {test_loss:.4f}")
    
    # Save final model
    torch.save(vae.state_dict(), model_dir / "vae_final.pt")
    
    # Finish wandb logging
    if args.wandb and wandb is not None:
        finish_wandb()
    
    print("VAE training completed!")
    return vae


def train_gfn_diffusion(args):
    """
    Train the GFN-guided diffusion model for the VAE latent space.
    
    Args:
        args: Command line arguments
    """
    # Import at the beginning of the function to avoid UnboundLocalError
    import os
    import sys
    from energy_sampling.models import UNet, DiffusionSchedule, GFNDiffusion
    from energies.vae import VAE

    print("Training GFN-Diffusion model with VAE energy function...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if specified
    if args.wandb and wandb is not None:
        try:
            # Set offline mode if requested
            if args.offline:
                os.environ["WANDB_MODE"] = "offline"
                print("Running wandb in offline mode")
                
            # Add the project root to the path
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.append(root_dir)
            
            # Define wandb utility functions here if import fails
            def init_wandb(project=None, entity=None, config=None, name=None):
                """Initialize wandb run."""
                try:
                    wandb.init(
                        project=project,
                        entity=None if args.offline else entity,
                        config=config,
                        name=name
                    )
                    print(f"Wandb initialized: project={project}, entity={entity if not args.offline else None}")
                except Exception as e:
                    print(f"Warning: Failed to initialize wandb: {str(e)}")
                
            def log_energy_function(energy_fn, title="Energy Function", range_val=5):
                """Log energy function visualization to wandb."""
                try:
                    import numpy as np
                    import matplotlib.pyplot as plt
                    x = np.linspace(-range_val, range_val, 100)
                    y = np.linspace(-range_val, range_val, 100)
                    X, Y = np.meshgrid(x, y)
                    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
                    
                    with torch.no_grad():
                        energy = energy_fn(points).cpu().numpy()
                    
                    plt.figure(figsize=(10, 8))
                    plt.contourf(X, Y, energy.reshape(100, 100), levels=20, cmap='viridis')
                    plt.colorbar(label='Energy')
                    plt.title(title)
                    
                    # Save temporarily and log to wandb
                    temp_file = str(output_dir / "temp_energy.png")
                    plt.savefig(temp_file)
                    plt.close()
                    wandb.log({"energy_function": wandb.Image(temp_file)})
                    # Don't remove the file so we can verify it exists
                    print(f"Energy function visualization saved to {temp_file}")
                except Exception as e:
                    print(f"Warning: Failed to log energy function: {str(e)}")
                
            def log_samples(samples, title="Samples", log_name="samples"):
                """Log samples to wandb."""
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 8))
                    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.7, s=10)
                    plt.xlim(-5, 5)
                    plt.ylim(-5, 5)
                    plt.title(title)
                    
                    # Save temporarily and log to wandb
                    temp_file = str(output_dir / f"temp_{log_name}.png")
                    plt.savefig(temp_file)
                    plt.close()
                    wandb.log({log_name: wandb.Image(temp_file)})
                    # Don't remove the file so we can verify it exists
                    print(f"Samples visualization saved to {temp_file}")
                except Exception as e:
                    print(f"Warning: Failed to log samples: {str(e)}")
                
            def log_comparison(samples_standard, samples_gfn, energy_fn=None, title="Comparison"):
                """Log comparison of samples with and without GFN guidance."""
                try:
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Plot standard samples
                    ax1.scatter(samples_standard[:, 0], samples_standard[:, 1], alpha=0.7, s=10)
                    ax1.set_xlim(-5, 5)
                    ax1.set_ylim(-5, 5)
                    ax1.set_title("Standard Diffusion Samples")
                    
                    # Plot GFN samples
                    ax2.scatter(samples_gfn[:, 0], samples_gfn[:, 1], alpha=0.7, s=10)
                    ax2.set_xlim(-5, 5)
                    ax2.set_ylim(-5, 5)
                    ax2.set_title("GFN-Guided Samples")
                    
                    plt.tight_layout()
                    
                    # Save temporarily and log to wandb
                    temp_file = str(output_dir / "temp_comparison.png")
                    plt.savefig(temp_file)
                    plt.close()
                    wandb.log({f"{title}": wandb.Image(temp_file)})
                    # Don't remove the file so we can verify it exists
                    print(f"Comparison visualization saved to {temp_file}")
                except Exception as e:
                    print(f"Warning: Failed to log comparison: {str(e)}")
                
            def finish_wandb():
                """Finish wandb run."""
                try:
                    wandb.finish()
                    print("Wandb run finished")
                except Exception as e:
                    print(f"Warning: Failed to finish wandb run: {str(e)}")
            
            # Initialize wandb
            init_wandb(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=args.run_name or "gfn_diffusion_vae"
            )
            
            if args.offline:
                print(f"Offline run data will be saved to: {os.path.abspath('wandb')}")
        except Exception as e:
            print(f"Warning: Error initializing wandb: {str(e)}")
            print("Training will continue without wandb logging")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load pretrained VAE - use the correct parameters
    vae = VAE(
        input_dim=args.input_dim, 
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim
    ).to(args.device)
    
    try:
        vae.load_state_dict(torch.load(args.vae_path, map_location=args.device))
        vae.eval()
        print(f"Successfully loaded VAE from {args.vae_path}")
    except Exception as e:
        print(f"Warning: Could not load VAE from {args.vae_path}, error: {str(e)}")
        print("Training will continue with randomly initialized VAE")
    
    # Create VAE energy function
    energy_fn = get_vae_energy(vae, device=args.device)
    
    # Create UNet model for score prediction
    unet = UNet(
        input_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.latent_dim
    ).to(args.device)
    
    # Create diffusion schedule
    diffusion = DiffusionSchedule(
        num_diffusion_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type
    )
    
    # Create GFN Diffusion model with specified guidance scale
    gfn_diffusion = GFNDiffusion(
        model=unet,
        diffusion=diffusion,
        energy_fn=energy_fn,
        guidance_scale=args.guidance_scale
    )
    
    # Move models to device and create optimizer
    gfn_diffusion.to(args.device)
    optimizer = torch.optim.Adam(gfn_diffusion.parameters(), lr=args.lr_policy)
    
    # Log energy function
    if args.wandb and wandb is not None:
        log_energy_function(energy_fn, title="VAE Energy Function")
    
    # Training loop
    best_loss = float('inf')
    viz_dir = Path(args.output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    progress_bar = tqdm(range(args.epochs), desc="Training")
    
    for epoch in range(args.epochs):
        gfn_diffusion.train()
        
        # Sample batch of noise
        noise = torch.randn(args.batch_size, args.latent_dim).to(args.device)
        
        # Sample timesteps
        t = torch.randint(0, args.num_timesteps, (args.batch_size,), device=args.device).long()
        
        # Apply diffusion process to get noisy samples (fixing the error here)
        # Instead of calling q_sample directly, use the DiffusionSchedule to add noise
        alpha_cumprod = diffusion.alphas_cumprod[t].view(-1, 1)
        eps = torch.randn_like(noise)
        x_t = torch.sqrt(alpha_cumprod) * noise + torch.sqrt(1 - alpha_cumprod) * eps
        
        # Compute loss - predict the added noise
        loss = F.mse_loss(gfn_diffusion(x_t, t), eps)
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
        
        # Periodic evaluation and logging
        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            gfn_diffusion.eval()
            
            # Log training metrics
            if args.wandb and wandb is not None:
                wandb.log({"train_loss": loss.item(), "epoch": epoch})
            
            # Sample from the model (both standard and GFN-guided)
            with torch.no_grad():
                # Standard diffusion samples
                standard_samples = gfn_diffusion.p_sample_loop(
                    shape=(args.batch_size, args.latent_dim),
                    device=args.device,
                    use_guidance=False
                )
                
                # GFN-guided samples
                gfn_samples = gfn_diffusion.p_sample_loop(
                    shape=(args.batch_size, args.latent_dim),
                    device=args.device,
                    use_guidance=True
                )
                
                # Log samples
                if args.wandb and wandb is not None:
                    log_samples(standard_samples.cpu(), 
                                title="Standard Diffusion Samples", 
                                log_name=f"standard_samples_epoch_{epoch}")
                    log_samples(gfn_samples.cpu(), 
                                title="GFN-Guided Samples", 
                                log_name=f"gfn_samples_epoch_{epoch}")
                    log_comparison(standard_samples.cpu(), gfn_samples.cpu(), 
                                title=f"Sample Comparison (Epoch {epoch})")
                
                # Decode and visualize samples (optional)
                if args.visualize_decoded:
                    vae.eval()
                    with torch.no_grad():
                        decoded_standard = vae.decode(standard_samples[:25])
                        decoded_gfn = vae.decode(gfn_samples[:25])
                        
                        # Save visualizations
                        decoded_dir = output_dir / "decoded_samples"
                        decoded_dir.mkdir(exist_ok=True)
                        
                        # Plot standard samples
                        plt.figure(figsize=(10, 10))
                        for i in range(min(25, decoded_standard.shape[0])):
                            plt.subplot(5, 5, i+1)
                            img = decoded_standard[i].reshape(28, 28).cpu()
                            plt.imshow(img, cmap='gray')
                            plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(decoded_dir / f"standard_decoded_epoch_{epoch}.png")
                        plt.close()
                        
                        # Plot GFN samples
                        plt.figure(figsize=(10, 10))
                        for i in range(min(25, decoded_gfn.shape[0])):
                            plt.subplot(5, 5, i+1)
                            img = decoded_gfn[i].reshape(28, 28).cpu()
                            plt.imshow(img, cmap='gray')
                            plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(decoded_dir / f"gfn_decoded_epoch_{epoch}.png")
                        plt.close()
                        
                        # Log decoded samples to wandb
                        if args.wandb and wandb is not None:
                            wandb.log({
                                f"decoded_standard_epoch_{epoch}": wandb.Image(str(decoded_dir / f"standard_decoded_epoch_{epoch}.png")),
                                f"decoded_gfn_epoch_{epoch}": wandb.Image(str(decoded_dir / f"gfn_decoded_epoch_{epoch}.png"))
                            })
            
            # Save model checkpoint
            if epoch > 0 and epoch % args.save_interval == 0:
                torch.save(gfn_diffusion.state_dict(), model_dir / f"gfn_diffusion_epoch_{epoch}.pt")
        
        # Periodically evaluate the model
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            # Evaluate using the new comprehensive evaluation function
            eval_results, samples_standard, samples_gfn = evaluate_vae_diffusion(
                gfn_diffusion=gfn_diffusion,
                vae=vae,
                energy_fn=energy_fn,
                args=args,
                device=args.device,
                epoch=epoch + 1,
                viz_dir=viz_dir
            )
            
            # Print key metrics
            standard_energy = eval_results["Standard Diffusion"]["mean_energy"]
            gfn_energy = eval_results["GFN-Guided Diffusion"]["mean_energy"]
            standard_diversity = eval_results["Standard Diffusion"]["diversity"]
            gfn_diversity = eval_results["GFN-Guided Diffusion"]["diversity"]
            
            print(f"Epoch {epoch+1} Evaluation:")
            print(f"  Standard Diffusion - Mean Energy: {standard_energy:.4f}, Diversity: {standard_diversity:.4f}")
            print(f"  GFN-Guided Diffusion - Mean Energy: {gfn_energy:.4f}, Diversity: {gfn_diversity:.4f}")
            
            # Save model if it has the best performance
            current_loss = gfn_energy  # Use mean energy as the loss metric
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(gfn_diffusion.state_dict(), os.path.join(args.output_dir, "gfn_diffusion_best.pt"))
                print(f"New best model saved with mean energy: {best_loss:.4f}")
    
    # Final evaluation after training is complete
    print("\nTraining completed. Running final evaluation...")
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, "gfn_diffusion_best.pt")
    try:
        gfn_diffusion.load_state_dict(torch.load(best_model_path, map_location=args.device))
        print(f"Loaded best model from {best_model_path}")
    except Exception as e:
        print(f"Warning: Could not load best model: {str(e)}")
    
    # Run final evaluation with comprehensive metrics
    final_metrics, final_samples_standard, final_samples_gfn = evaluate_vae_diffusion(
        gfn_diffusion=gfn_diffusion,
        vae=vae,
        energy_fn=energy_fn,
        args=args,
        device=args.device,
        viz_dir=viz_dir
    )
    
    # Print summary of results
    print("\nFinal Evaluation Results:")
    print("Standard Diffusion:")
    for k, v in final_metrics["Standard Diffusion"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nGFN-Guided Diffusion:")
    for k, v in final_metrics["GFN-Guided Diffusion"].items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nModel saved to {args.output_dir}/gfn_diffusion_best.pt")
    
    # Return the trained model
    return gfn_diffusion


def evaluate_vae_diffusion(gfn_diffusion, vae, energy_fn, args, device, epoch=None, viz_dir=None):
    """
    Evaluate the VAE diffusion model with comprehensive metrics.
    
    Args:
        gfn_diffusion: The GFNDiffusion model
        vae: The VAE model
        energy_fn: Energy function
        args: Command line arguments
        device: Device to use
        epoch: Current epoch (optional)
        viz_dir: Directory to save visualizations (optional)
    
    Returns:
        metrics: Dictionary of evaluation metrics
        samples_standard: Samples from standard diffusion
        samples_gfn: Samples from GFN-guided diffusion
    """
    # Ensure visualization directory exists
    if viz_dir is None:
        viz_dir = Path(args.output_dir) / "visualizations"
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Switch models to eval mode
    gfn_diffusion.eval()
    vae.eval()
    
    with torch.no_grad():
        # Sample trajectories
        batch_size = 64
        
        # Standard diffusion
        trajectory_standard = []
        x_standard = torch.randn(batch_size, args.latent_dim, device=device)
        energy_values_standard = []
        
        # GFN diffusion
        trajectory_gfn = []
        x_gfn = torch.randn(batch_size, args.latent_dim, device=device)
        energy_values_gfn = []
        
        # Generate trajectories and track energy values
        for t in reversed(range(100)):
            # Record states
            trajectory_standard.append(x_standard.clone())
            trajectory_gfn.append(x_gfn.clone())
            
            # Record energy values
            energy_values_standard.append(energy_fn(x_standard).cpu().numpy())
            energy_values_gfn.append(energy_fn(x_gfn).cpu().numpy())
            
            # Generate next states
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Standard diffusion step
            x_standard = gfn_diffusion.p_sample(x_standard, t_tensor, use_gfn=False)
            
            # GFN diffusion step
            x_gfn = gfn_diffusion.p_sample(x_gfn, t_tensor, use_gfn=True)
        
        # Final samples
        samples_standard = x_standard
        samples_gfn = x_gfn
        
        # Stack trajectory tensors
        trajectory_standard = torch.stack(trajectory_standard)
        trajectory_gfn = torch.stack(trajectory_gfn)
        
        # Compute energy statistics
        stats_standard = compute_energy_statistics(samples_standard, energy_fn)
        stats_gfn = compute_energy_statistics(samples_gfn, energy_fn)
        
        # Compute improvement metrics
        energy_improvement = (stats_standard["mean_energy"] - stats_gfn["mean_energy"]) / abs(stats_standard["mean_energy"]) * 100
        
        # Sort samples by energy
        standard_energies = energy_fn(samples_standard).cpu().numpy().flatten()
        gfn_energies = energy_fn(samples_gfn).cpu().numpy().flatten()
        
        # Top 10% energy improvement
        top_k = max(1, int(0.1 * len(standard_energies)))
        top_standard = np.sort(standard_energies)[:top_k].mean()
        top_gfn = np.sort(gfn_energies)[:top_k].mean()
        top_energy_improvement = (top_standard - top_gfn) / abs(top_standard) * 100
        
        # Compute diversity metrics
        diversity_standard = compute_diversity(samples_standard.cpu())
        diversity_gfn = compute_diversity(samples_gfn.cpu())
        
        # Entropy
        entropy_standard = compute_entropy(samples_standard.cpu())
        entropy_gfn = compute_entropy(samples_gfn.cpu())
        
        # Effective sample size
        standard_ess, standard_ess_ratio = compute_effective_sample_size(samples_standard.cpu(), energy_fn)
        gfn_ess, gfn_ess_ratio = compute_effective_sample_size(samples_gfn.cpu(), energy_fn)
        
        # Compute coverage metrics (how well samples cover the space)
        coverage_metrics_standard = compute_coverage_metrics(samples_standard.cpu(), energy_fn=energy_fn)
        coverage_metrics_gfn = compute_coverage_metrics(samples_gfn.cpu(), energy_fn=energy_fn)
        
        # Calculate metrics comparable to those in motif.py
        # - avg_reward: negative energy (lower energy is better)
        avg_reward_standard = -stats_standard["mean_energy"]
        avg_reward_gfn = -stats_gfn["mean_energy"]
        
        # - novelty: for VAE, we can use latent space coverage or KL divergence as a proxy
        # Here we'll use diversity as a proxy for novelty
        novelty_standard = diversity_standard
        novelty_gfn = diversity_gfn
        
        # - unique_sequences: discretize latent space to count unique positions
        # Define a grid for the latent space
        if args.latent_dim <= 16:  # Only do this for manageable latent dims
            grid_size = max(5, min(20, 100 // args.latent_dim))  # Adjust grid size based on latent dim
            
            def discretize_latent_space(samples, grid_size, latent_dim):
                # Create bins for each dimension
                bins = []
                for d in range(latent_dim):
                    dim_values = samples[:, d].cpu().numpy()
                    dim_min, dim_max = dim_values.min(), dim_values.max()
                    # Add some padding to the range
                    padding = 0.1 * (dim_max - dim_min)
                    bins.append(np.linspace(dim_min - padding, dim_max + padding, grid_size))
                
                # Convert continuous samples to discrete grid cells
                grid_indices = []
                for i in range(samples.shape[0]):
                    sample = samples[i].cpu().numpy()
                    index = []
                    for d in range(latent_dim):
                        idx = np.digitize(sample[d], bins[d])
                        index.append(idx)
                    grid_indices.append(tuple(index))
                
                return set(grid_indices)
            
            unique_positions_standard = discretize_latent_space(samples_standard, grid_size, args.latent_dim)
            unique_positions_gfn = discretize_latent_space(samples_gfn, grid_size, args.latent_dim)
        else:
            # For high dimensional spaces, use a different approach
            # E.g., count samples that are at least epsilon apart
            epsilon = 0.1
            from sklearn.metrics.pairwise import euclidean_distances
            
            def count_distinct_samples(samples, epsilon):
                samples_np = samples.cpu().numpy()
                distances = euclidean_distances(samples_np)
                # Set diagonal to infinity
                np.fill_diagonal(distances, np.inf)
                # Count samples that have at least one neighbor closer than epsilon
                return np.sum(np.min(distances, axis=1) > epsilon)
            
            unique_positions_standard = count_distinct_samples(samples_standard, epsilon)
            unique_positions_gfn = count_distinct_samples(samples_gfn, epsilon)
        
        # Visualize trajectories
        epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
        
        # 1. Energy evolution through diffusion process
        energy_plot_path_standard = str(viz_dir / f"energy_evolution_standard{epoch_suffix}.png")
        from gfn_diffusion.utils.visualization import plot_energy_evolution
        plot_energy_evolution(
            energy_values=energy_values_standard,
            save_path=energy_plot_path_standard,
            title="Energy Evolution (Standard Diffusion)"
        )
        
        energy_plot_path_gfn = str(viz_dir / f"energy_evolution_gfn{epoch_suffix}.png")
        plot_energy_evolution(
            energy_values=energy_values_gfn,
            save_path=energy_plot_path_gfn,
            title="Energy Evolution (GFN-Guided Diffusion)"
        )
        
        # 2. Visualize sample distributions
        if args.latent_dim == 2:
            # 2D latent space - create scatter plots
            standard_samples_path = str(viz_dir / f"standard_samples{epoch_suffix}.png")
            gfn_samples_path = str(viz_dir / f"gfn_samples{epoch_suffix}.png")
            
            # Create scatter plots
            plt.figure(figsize=(8, 8))
            plt.scatter(samples_standard[:, 0].cpu(), samples_standard[:, 1].cpu(), alpha=0.6)
            plt.title("Standard Diffusion Samples")
            plt.xlabel("z1")
            plt.ylabel("z2")
            plt.savefig(standard_samples_path)
            plt.close()
            
            plt.figure(figsize=(8, 8))
            plt.scatter(samples_gfn[:, 0].cpu(), samples_gfn[:, 1].cpu(), alpha=0.6)
            plt.title("GFN-Guided Diffusion Samples")
            plt.xlabel("z1")
            plt.ylabel("z2")
            plt.savefig(gfn_samples_path)
            plt.close()
            
            # Create comparison plot
            comparison_path = str(viz_dir / f"sample_comparison{epoch_suffix}.png")
            plt.figure(figsize=(10, 8))
            plt.scatter(samples_standard[:, 0].cpu(), samples_standard[:, 1].cpu(), alpha=0.6, label="Standard")
            plt.scatter(samples_gfn[:, 0].cpu(), samples_gfn[:, 1].cpu(), alpha=0.6, label="GFN-Guided")
            plt.title("Sample Comparison")
            plt.xlabel("z1")
            plt.ylabel("z2")
            plt.legend()
            plt.savefig(comparison_path)
            plt.close()
        else:
            # Higher dimensional latent space - use t-SNE or PCA for visualization
            from gfn_diffusion.utils.visualization import visualize_high_dim_samples_with_tsne
            tsne_path = str(viz_dir / f"tsne_comparison{epoch_suffix}.png")
            visualize_high_dim_samples_with_tsne(
                samples_standard=samples_standard.cpu(),
                samples_gfn=samples_gfn.cpu(),
                save_path=tsne_path,
                title=f"t-SNE Visualization of Latent Samples" + (f" (Epoch {epoch})" if epoch else "")
            )
        
        # 3. Decode and visualize samples in pixel space
        with torch.no_grad():
            decoded_standard = vae.decode(samples_standard[:25])
            decoded_gfn = vae.decode(samples_gfn[:25])
            
            # Save visualizations
            decoded_standard_path = str(viz_dir / f"decoded_standard{epoch_suffix}.png")
            plt.figure(figsize=(10, 10))
            for i in range(min(25, decoded_standard.shape[0])):
                plt.subplot(5, 5, i+1)
                img = decoded_standard[i].reshape(28, 28).cpu()
                plt.imshow(img, cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(decoded_standard_path)
            plt.close()
            
            decoded_gfn_path = str(viz_dir / f"decoded_gfn{epoch_suffix}.png")
            plt.figure(figsize=(10, 10))
            for i in range(min(25, decoded_gfn.shape[0])):
                plt.subplot(5, 5, i+1)
                img = decoded_gfn[i].reshape(28, 28).cpu()
                plt.imshow(img, cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(decoded_gfn_path)
            plt.close()
        
        # Compile all metrics
        metrics = {
            "Standard Diffusion": {
                "mean_energy": stats_standard["mean_energy"],
                "min_energy": stats_standard["min_energy"],
                "entropy": entropy_standard,
                "diversity": diversity_standard,
                "effective_sample_size": standard_ess_ratio
            },
            "GFN-Guided Diffusion": {
                "mean_energy": stats_gfn["mean_energy"],
                "min_energy": stats_gfn["min_energy"],
                "entropy": entropy_gfn,
                "diversity": diversity_gfn,
                "effective_sample_size": gfn_ess_ratio
            }
        }
        
        # Add grid coverage to metrics
        if 'grid_coverage' in coverage_metrics_standard:
            metrics["Standard Diffusion"]["grid_coverage"] = coverage_metrics_standard["grid_coverage"]
            metrics["GFN-Guided Diffusion"]["grid_coverage"] = coverage_metrics_gfn["grid_coverage"]
        
        # Create metric comparison plot
        from gfn_diffusion.utils.visualization import create_metric_comparison_plot
        metric_plot_path = str(viz_dir / f"metric_comparison{epoch_suffix}.png")
        create_metric_comparison_plot(
            metrics_dict=metrics,
            save_path=metric_plot_path,
            title=f"Performance Metrics Comparison" + (f" (Epoch {epoch})" if epoch else ""),
            higher_is_better=False  # Lower energy is better
        )
        
        # Log everything to wandb
        if args.wandb and wandb is not None:
            # Calculate current iteration
            current_iteration = epoch if epoch is not None else args.epochs
            
            # Log standard metrics format for consistent comparison across experiments
            wandb.log({
                # Consistent metrics across all experiments
                "iteration": current_iteration,
                "loss": stats_standard["mean_energy"],  # Current loss value (using mean energy)
                "avg_reward_standard": avg_reward_standard,
                "avg_reward_gfn": avg_reward_gfn,
                "avg_reward_improvement": (avg_reward_gfn - avg_reward_standard) / (abs(avg_reward_standard) + 1e-8),
                "diversity_standard": diversity_standard,
                "diversity_gfn": diversity_gfn,
                "diversity_improvement": (diversity_gfn - diversity_standard) / (diversity_standard + 1e-8),
                "novelty_standard": novelty_standard,
                "novelty_gfn": novelty_gfn,
                "novelty_improvement": (novelty_gfn - novelty_standard) / (novelty_standard + 1e-8),
                "unique_positions_standard": len(unique_positions_standard) if isinstance(unique_positions_standard, set) else unique_positions_standard,
                "unique_positions_gfn": len(unique_positions_gfn) if isinstance(unique_positions_gfn, set) else unique_positions_gfn,
                "unique_positions_ratio": (len(unique_positions_gfn) if isinstance(unique_positions_gfn, set) else unique_positions_gfn) / 
                                        (len(unique_positions_standard) if isinstance(unique_positions_standard, set) else unique_positions_standard + 1e-8),
                
                # Images
                "viz/standard_samples": wandb.Image(standard_samples_path) if args.latent_dim == 2 else None,
                "viz/gfn_samples": wandb.Image(gfn_samples_path) if args.latent_dim == 2 else None,
                "viz/sample_comparison": wandb.Image(comparison_path) if args.latent_dim == 2 else None,
                "viz/tsne_comparison": wandb.Image(tsne_path) if args.latent_dim > 2 else None,
                "viz/energy_evolution_standard": wandb.Image(energy_plot_path_standard),
                "viz/energy_evolution_gfn": wandb.Image(energy_plot_path_gfn),
                "viz/decoded_standard": wandb.Image(decoded_standard_path),
                "viz/decoded_gfn": wandb.Image(decoded_gfn_path),
                "viz/metric_comparison": wandb.Image(metric_plot_path),
                
                # Energy statistics
                "energy_stats/standard/mean": stats_standard["mean_energy"],
                "energy_stats/standard/min": stats_standard["min_energy"],
                "energy_stats/standard/max": stats_standard["max_energy"],
                "energy_stats/standard/std": stats_standard["std_energy"],
                "energy_stats/standard/p50": stats_standard["p50"],
                
                "energy_stats/gfn/mean": stats_gfn["mean_energy"],
                "energy_stats/gfn/min": stats_gfn["min_energy"],
                "energy_stats/gfn/max": stats_gfn["max_energy"],
                "energy_stats/gfn/std": stats_gfn["std_energy"],
                "energy_stats/gfn/p50": stats_gfn["p50"],
                
                # Energy improvement metrics
                "energy_improvement/mean": energy_improvement,
                "energy_improvement/top_10_percent": top_energy_improvement,
                
                # Distribution metrics
                "distribution/entropy_standard": entropy_standard,
                "distribution/entropy_gfn": entropy_gfn,
                "distribution/diversity_standard": diversity_standard,
                "distribution/diversity_gfn": diversity_gfn,
                "distribution/diversity_ratio": diversity_gfn / (diversity_standard + 1e-8),
                
                # Effective sample size metrics
                "ess/standard": standard_ess,
                "ess/standard_ratio": standard_ess_ratio,
                "ess/gfn": gfn_ess,
                "ess/gfn_ratio": gfn_ess_ratio,
                "ess/improvement": (gfn_ess_ratio - standard_ess_ratio) / (standard_ess_ratio + 1e-8) * 100,
                
                # Coverage metrics
                "coverage/standard_grid": coverage_metrics_standard.get("grid_coverage", 0),
                "coverage/gfn_grid": coverage_metrics_gfn.get("grid_coverage", 0),
                "coverage/grid_ratio": coverage_metrics_gfn.get("grid_coverage", 0) / (coverage_metrics_standard.get("grid_coverage", 1e-8)),
            })
    
    # Return metrics and samples
    return metrics, samples_standard, samples_gfn


def main():
    """
    Main function to parse arguments and run training.
    """
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output and model directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Check if we should train VAE, GFN-Diffusion, or both
    if hasattr(args, 'train_vae') and args.train_vae:
        train_vae(args)
    
    if hasattr(args, 'train_gfn') and args.train_gfn:
        train_gfn_diffusion(args)
    
    # If no specific training is specified, train both by default
    if not (hasattr(args, 'train_vae') or hasattr(args, 'train_gfn')):
        # First train VAE
        vae = train_vae(args)
        
        # Then train GFN-Diffusion
        args.vae_path = os.path.join(args.model_dir, "vae_best.pt")
        train_gfn_diffusion(args)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 