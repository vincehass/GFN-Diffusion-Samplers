from plot_utils import *
import argparse
import torch
import os
import time
import numpy as np
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from utils import set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name, get_batch_metrics
from buffer import ReplayBuffer
from langevin import langevin_dynamics
from models import GFN
from gflownet_losses import *
from energies import *
from evaluations import *

import matplotlib.pyplot as plt
from tqdm import trange
import wandb

parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--buffer_size', type=int, default=300 * 1000 * 2)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='9gmm',
                    choices=('9gmm', '25gmm', 'hard_funnel', 'easy_funnel', 'many_well'))
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
parser.add_argument('--both_ways', action='store_true', default=False)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=200)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=100)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

parser.add_argument('--ld_schedule', action='store_true', default=False)

# target acceptance rate
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)


# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)

# Add new arguments for performance
parser.add_argument('--use_amp', action='store_true', default=False, 
                    help='Use automatic mixed precision for faster training')
parser.add_argument('--num_workers', type=int, default=4, 
                    help='Number of workers for data loading')
parser.add_argument('--log_interval', type=int, default=50,
                    help='How often to log metrics during training')
parser.add_argument('--profile', action='store_true', default=False,
                    help='Enable performance profiling')
parser.add_argument('--prefetch_factor', type=int, default=2,
                    help='Number of batches to prefetch')

args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 2000
final_eval_data_size = 2000
plot_data_size = 2000
final_plot_data_size = 2000

if args.pis_architectures:
    args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True


def get_energy():
    if args.energy == '9gmm':
        energy = NineGaussianMixture(device=device)
    elif args.energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif args.energy == 'hard_funnel':
        energy = HardFunnel(device=device)
    elif args.energy == 'easy_funnel':
        energy = EasyFunnel(device=device)
    elif args.energy == 'many_well':
        energy = ManyWell(device=device)
    return energy


def plot_step(energy, gfn_model, name):
    if args.energy == 'many_well':
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, energy.log_reward)

        vizualizations = viz_many_well(energy, samples)
        fig_samples_x13, ax_samples_x13, fig_kde_x13, ax_kde_x13, fig_contour_x13, ax_contour_x13, fig_samples_x23, ax_samples_x23, fig_kde_x23, ax_kde_x23, fig_contour_x23, ax_contour_x23 = vizualizations

        fig_samples_x13.savefig(f'{name}samplesx13.pdf', bbox_inches='tight')
        fig_samples_x23.savefig(f'{name}samplesx23.pdf', bbox_inches='tight')

        fig_kde_x13.savefig(f'{name}kdex13.pdf', bbox_inches='tight')
        fig_kde_x23.savefig(f'{name}kdex23.pdf', bbox_inches='tight')

        fig_contour_x13.savefig(f'{name}contourx13.pdf', bbox_inches='tight')
        fig_contour_x23.savefig(f'{name}contourx23.pdf', bbox_inches='tight')

        return {"visualization/contourx13": wandb.Image(fig_to_image(fig_contour_x13)),
                "visualization/contourx23": wandb.Image(fig_to_image(fig_contour_x23)),
                "visualization/kdex13": wandb.Image(fig_to_image(fig_kde_x13)),
                "visualization/kdex23": wandb.Image(fig_to_image(fig_kde_x23)),
                "visualization/samplesx13": wandb.Image(fig_to_image(fig_samples_x13)),
                "visualization/samplesx23": wandb.Image(fig_to_image(fig_samples_x23))}

    elif energy.data_ndim != 2:
        return {}

    else:
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, energy.log_reward)
        gt_samples = energy.sample(batch_size)

        fig_contour, ax_contour = get_figure(bounds=(-13., 13.))
        fig_kde, ax_kde = get_figure(bounds=(-13., 13.))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-13., 13.))

        plot_contours(energy.log_reward, ax=ax_contour, bounds=(-13., 13.), n_contour_levels=150, device=device)
        plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-13., 13.))
        plot_kde(samples, ax=ax_kde, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_contour, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_kde_overlay, bounds=(-13., 13.))

        fig_contour.savefig(f'{name}contour.pdf', bbox_inches='tight')
        fig_kde_overlay.savefig(f'{name}kde_overlay.pdf', bbox_inches='tight')
        fig_kde.savefig(f'{name}kde.pdf', bbox_inches='tight')
        # return None
        return {"visualization/contour": wandb.Image(fig_to_image(fig_contour)),
                "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
                "visualization/kde": wandb.Image(fig_to_image(fig_kde))}


def eval_step(eval_data, energy, gfn_model, final_eval=False):
    gfn_model.eval()
    metrics = dict()
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics['final_eval/log_Z'], metrics['final_eval/log_Z_lb'], metrics[
            'final_eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, energy.log_reward)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
            'eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, energy.log_reward)
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    else:
        if final_eval:
            metrics['final_eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                              gfn_model,
                                                                                                              energy.log_reward)
        else:
            metrics['eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                        gfn_model,
                                                                                                        energy.log_reward)
        metrics.update(get_sample_metrics(samples, eval_data, final_eval))
    gfn_model.train()
    return metrics


def setup_wandb(args, name):
    """Configure WandB with enhanced settings"""
    config = args.__dict__
    config["experiment"] = f"{args.energy}_{args.mode_fwd}_{args.mode_bwd}"
    
    # Define more sophisticated run naming
    run_name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Set up tags for better organization
    tags = [args.energy, args.mode_fwd, args.mode_bwd]
    if args.both_ways:
        tags.append("both_ways")
    if args.local_search:
        tags.append("local_search")
    if args.langevin:
        tags.append("langevin")
    if args.pis_architectures:
        tags.append("pis")
    
    # Initialize wandb with enhanced configuration
    wandb.init(
        project="GFN_Energy_Optimized", 
        name=run_name,
        config=config,
        tags=tags,
        settings=wandb.Settings(start_method="thread")
    )
    
    # Create custom panels in wandb
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("eval/*", summary="max")
    wandb.define_metric("perf/*", summary="mean")
    
    # Log system info
    wandb.log({
        "system/gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "system/batch_size": args.batch_size,
        "system/micro_batch_size": min(32, args.batch_size)
    })

def create_dataloader(data, batch_size, num_workers, device):
    """Create an optimized dataloader for tensor data"""
    dataset = TensorDataset(data)
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type=='cuda',
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )

def prefill_buffer(buffer, gfn_model, energy, batch_size, device):
    """Efficiently prefill buffer with initial samples for better stability"""
    print("Prefilling buffer with initial samples...")
    
    # Determine prefill amount (10% of buffer capacity)
    prefill_size = min(buffer.capacity // 10, 10000)
    remaining = prefill_size
    
    # Use batched filling for efficiency
    fill_batch_size = min(batch_size * 4, 1024)
    
    with torch.no_grad():
        while remaining > 0:
            current_batch_size = min(fill_batch_size, remaining)
            
            # Generate samples from both the energy model and the GFN
            energy_samples = energy.sample(current_batch_size // 2).to(device)
            energy_log_rewards = energy.log_reward(energy_samples)
            
            # Generate some samples from the GFN (with higher exploration noise)
            init_states = torch.zeros(current_batch_size // 2, energy.data_ndim, device=device)
            gfn_samples, _, _, _ = gfn_model.get_trajectory_fwd(
                init_states, 
                exploration_std=1.0,  # Higher noise for diversity
                log_r=energy.log_reward
            )
            gfn_samples = gfn_samples[:, -1]  # Get final states
            gfn_log_rewards = energy.log_reward(gfn_samples)
            
            # Combine samples and add to buffer
            combined_samples = torch.cat([energy_samples, gfn_samples], dim=0)
            combined_log_rewards = torch.cat([energy_log_rewards, gfn_log_rewards], dim=0)
            
            buffer.add_batch(combined_samples, combined_log_rewards)
            remaining -= current_batch_size
            
    print(f"Buffer prefilled with {buffer.size} samples")

def train_step_optimized(energy, gfn_model, gfn_optimizer, buffer, buffer_ls, 
                        exploration_std, scaler=None, use_amp=False, 
                        micro_batch_size=32, iteration=0):
    """Optimized training step with micro-batching and gradient accumulation"""
    gfn_optimizer.zero_grad()
    
    # Track total loss for logging
    total_loss = 0.0
    
    # Number of micro-batches for gradient accumulation
    num_micro_batches = args.batch_size // micro_batch_size
    if args.batch_size % micro_batch_size != 0:
        num_micro_batches += 1
    
    # Train forward and backward passes with gradient accumulation
    for i in range(num_micro_batches):
        # Adjust batch size for last micro-batch if needed
        actual_micro_batch = min(micro_batch_size, args.batch_size - i * micro_batch_size)
        if actual_micro_batch <= 0:
            break
        
        # Handle mixed precision training if enabled
        if use_amp:
            with autocast():
                # Forward pass
                if args.both_ways:
                    fwd_loss = fwd_train_step(energy, gfn_model, exploration_std, 
                                             batch_size=actual_micro_batch, return_exp=True)
                    
                    if isinstance(fwd_loss, tuple):
                        loss, states, _, _, log_r = fwd_loss
                        # Add samples to buffer for backward training
                        buffer.add_batch(states[:, -1], log_r)
                    else:
                        loss = fwd_loss
                    
                    # Backward pass using the buffer
                    bwd_loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, 
                                             exploration_std, batch_size=actual_micro_batch, 
                                             it=iteration)
                    
                    # Combine losses
                    loss = args.alpha * loss + (1 - args.alpha) * bwd_loss
                else:
                    # Only forward training
                    loss = fwd_train_step(energy, gfn_model, exploration_std, 
                                         batch_size=actual_micro_batch)
                
                # Scale loss by number of micro-batches to maintain correct gradient magnitude
                loss = loss / num_micro_batches
                
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
        else:
            # Standard training without mixed precision
            if args.both_ways:
                fwd_loss = fwd_train_step(energy, gfn_model, exploration_std, 
                                         batch_size=actual_micro_batch, return_exp=True)
                
                if isinstance(fwd_loss, tuple):
                    loss, states, _, _, log_r = fwd_loss
                    # Add samples to buffer for backward training
                    buffer.add_batch(states[:, -1], log_r)
                else:
                    loss = fwd_loss
                
                # Backward pass using the buffer
                bwd_loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, 
                                         exploration_std, batch_size=actual_micro_batch, 
                                         it=iteration)
                
                # Combine losses
                loss = args.alpha * loss + (1 - args.alpha) * bwd_loss
            else:
                # Only forward training
                loss = fwd_train_step(energy, gfn_model, exploration_std, 
                                     batch_size=actual_micro_batch)
            
            # Scale loss by number of micro-batches
            loss = loss / num_micro_batches
            loss.backward()
        
        # Track total loss for logging
        total_loss += loss.item() * num_micro_batches
    
    # Apply gradients with grad clipping if enabled
    if use_amp:
        # Unscale before clipping
        scaler.unscale_(gfn_optimizer)
        
        # Optional gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(gfn_model.parameters(), args.grad_clip)
            
        scaler.step(gfn_optimizer)
        scaler.update()
    else:
        # Optional gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(gfn_model.parameters(), args.grad_clip)
            
        gfn_optimizer.step()
    
    return total_loss

def fwd_train_step(energy, gfn_model, exploration_std, batch_size=None, return_exp=False):
    """Forward training step with configurable batch size"""
    if batch_size is None:
        batch_size = args.batch_size
        
    init_state = torch.zeros(batch_size, energy.data_ndim, device=device)
    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, 
                                coeff_matrix, exploration_std=exploration_std, return_exp=return_exp)
    return loss

def bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std=None, batch_size=None, it=0):
    """Backward training step with configurable batch size"""
    if batch_size is None:
        batch_size = args.batch_size
        
    # Optimize sampling strategy
    if args.sampling == 'buffer':
        if args.local_search and it % args.ls_cycle < 2:
            # Reduce frequency of local search
            samples, rewards = buffer.sample(batch_size)
            # Optimize Langevin dynamics for speed
            with torch.no_grad():  # No gradients needed during local search
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, 
                                                               device, args)
            buffer_ls.add_batch(local_search_samples, log_r)
        
        samples, rewards = buffer_ls.sample(batch_size) if (args.local_search and buffer_ls.size > 0) else buffer.sample(batch_size)
    elif args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(batch_size, exploration_std)
    else:  # 'energy'
        samples = energy.sample(batch_size)
    
    loss = get_gfn_backward_loss(args.mode_bwd, samples, gfn_model, energy.log_reward,
                                exploration_std=exploration_std)
    return loss

def log_visualizations(energy, gfn_model, name, step):
    """Create and log visualizations to WandB"""
    # Generate plots
    images = plot_step(energy, gfn_model, name)
    
    # Log multiple plots as images
    if isinstance(images, dict):
        for plot_name, img in images.items():
            wandb.log({f"plots/{plot_name}": wandb.Image(img)}, step=step)
    else:
        # If just a single image is returned
        wandb.log({"plots/distribution": wandb.Image(images)}, step=step)
    
    # Close all matplotlib figures to prevent memory leaks
    plt.close('all')

def train():
    name = get_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    # Setup WandB with enhanced configuration
    setup_wandb(args, name)
    
    # Log code to WandB for reproducibility
    if os.path.exists("energy_sampling"):
        wandb.run.log_code("energy_sampling", include_fn=lambda path: path.endswith(".py"))

    energy = get_energy()
    # Only generate evaluation data once at the beginning
    eval_data = energy.sample(eval_data_size).to(device)

    # Initialize model and optimizer
    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, 
                    gfn_clip=args.gfn_clip, langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range, t_scale=args.t_scale, 
                    langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)

    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, 
                                     args.learn_pb, args.conditional_flow_model, 
                                     args.use_weight_decay, args.weight_decay)

    # Initialize buffers with larger capacity but more efficient storage
    buffer = ReplayBuffer(args.buffer_size, device, energy.log_reward, args.batch_size, 
                         data_ndim=energy.data_ndim, beta=args.beta,
                         rank_weight=args.rank_weight, prioritized=args.prioritized)
    buffer_ls = ReplayBuffer(args.buffer_size, device, energy.log_reward, args.batch_size, 
                            data_ndim=energy.data_ndim, beta=args.beta,
                            rank_weight=args.rank_weight, prioritized=args.prioritized)

    # Setup for mixed precision training
    scaler = GradScaler() if args.use_amp else None
    
    # Use gradient accumulation with smaller micro-batches
    micro_batch_size = min(32, args.batch_size)
    
    # Less frequent evaluation
    eval_frequency = 500  # Changed from 100
    save_frequency = 2000  # Changed from 1000
    
    metrics = {}
    gfn_model.train()
    
    # More efficient buffer prefill
    if args.sampling == 'buffer' and args.both_ways:
        prefill_buffer(buffer, gfn_model, energy, args.batch_size, device)
    
    # Create progress bar
    pbar = trange(args.epochs + 1)
    start_time = time.time()
    step_start_time = start_time
    total_samples = 0
    
    # Enable performance profiling if requested
    if args.profile:
        wandb.watch(gfn_model, log="all", log_freq=100)
    
    for i in pbar:
        # Training step with micro-batching and mixed precision
        loss = train_step_optimized(
            energy, gfn_model, gfn_optimizer, buffer, buffer_ls,
            get_exploration_std(i, args.exploratory, args.exploration_factor, args.exploration_wd),
            scaler, args.use_amp, micro_batch_size, i
        )
        
        total_samples += args.batch_size
        
        # Update progress bar with current loss
        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss:.4f}")
        
        # Periodically log training metrics
        if i % args.log_interval == 0:
            # Calculate training speed metrics
            elapsed = time.time() - step_start_time
            examples_per_sec = (args.log_interval * args.batch_size) / max(elapsed, 1e-5)
            
            metrics = {
                'train/loss': loss,
                'train/epoch': i,
                'train/total_samples': total_samples,
                'perf/examples_per_second': examples_per_sec,
                'perf/buffer_size': buffer.size,
                'perf/buffer_ls_size': buffer_ls.size if args.local_search else 0,
                'perf/seconds_per_epoch': elapsed / args.log_interval
            }
            
            wandb.log(metrics, step=i)
            step_start_time = time.time()
        
        # Less frequent evaluation and visualization to reduce overhead
        if i % eval_frequency == 0:
            # Use torch.no_grad for evaluation to save memory
            with torch.no_grad():
                gfn_model.eval()  # Set model to evaluation mode
                eval_metrics = eval_step(eval_data, energy, gfn_model, final_eval=False)
                metrics.update(eval_metrics)
                
                # Visualize and log every 2nd evaluation for efficiency
                if i % (eval_frequency * 2) == 0:
                    log_visualizations(energy, gfn_model, name, i)
                
                gfn_model.train()  # Set back to training mode
            
            # Calculate and log overall training time metrics
            elapsed_total = time.time() - start_time
            metrics.update({
                'perf/total_training_time': elapsed_total,
                'perf/avg_examples_per_second': total_samples / elapsed_total,
            })
            
            wandb.log(metrics, step=i)
            
            # Save model less frequently
            if i % save_frequency == 0 and i > 0:
                model_path = f'{name}model_{i}.pt'
                torch.save(gfn_model.state_dict(), model_path)
                wandb.save(model_path)  # Upload to WandB as well
    
    # Final evaluation
    with torch.no_grad():
        gfn_model.eval()
        eval_results = eval_step(energy.sample(final_eval_data_size).to(device), energy, gfn_model, final_eval=True)
        metrics.update(eval_results)
        
        # Final visualization
        log_visualizations(energy, gfn_model, name, args.epochs)
        
    # Save final model
    final_model_path = f'{name}model_final.pt'
    torch.save(gfn_model.state_dict(), final_model_path)
    wandb.save(final_model_path)
    
    # Log final performance summary
    total_time = time.time() - start_time
    wandb.log({
        'final/training_time_hours': total_time / 3600,
        'final/total_epochs': args.epochs,
        'final/samples_processed': total_samples,
        'final/average_samples_per_second': total_samples / total_time
    })
    
    wandb.finish()
    print(f"Training completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")


def final_eval(energy, gfn_model):
    final_eval_data = energy.sample(final_eval_data_size)
    results = eval_step(final_eval_data, energy, gfn_model, final_eval=True)
    return results


def eval():
    pass


if __name__ == '__main__':
    if args.eval:
        eval()
    else:
        train()
