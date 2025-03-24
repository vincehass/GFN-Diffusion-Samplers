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

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import trange
import wandb

# Timer for performance profiling
class Timer:
    def __init__(self):
        self.times = {}
        self.starts = {}
        self.counts = {}
        
    def start(self, section):
        self.starts[section] = time.time()
        
    def stop(self, section):
        if section in self.starts:
            elapsed = time.time() - self.starts[section]
            if section not in self.times:
                self.times[section] = 0
                self.counts[section] = 0
            self.times[section] += elapsed
            self.counts[section] += 1
            return elapsed
        return 0
    
    def report(self):
        print("\n--- Performance Report ---")
        total = sum(self.times.values())
        for section, time_spent in sorted(self.times.items(), key=lambda x: x[1], reverse=True):
            count = self.counts[section]
            avg = time_spent / max(1, count)
            percentage = 100.0 * time_spent / max(total, 1e-9)
            print(f"{section}: {time_spent:.2f}s total, {avg:.4f}s avg, {count} calls ({percentage:.1f}%)")
        print(f"Total time: {total:.2f}s\n")

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
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight for forward vs backward loss when using both_ways')
parser.add_argument('--grad_clip', type=float, default=0.0,
                    help='Gradient clipping value (0 to disable)')

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
parser.add_argument('--reduced_eval', action='store_true', default=False,
                    help='Perform less frequent and less computation-heavy evaluations')
parser.add_argument('--disable_wandb', action='store_true', default=False,
                    help='Disable WandB logging')
parser.add_argument('--reduced_T', type=int, default=0,
                    help='Use reduced trajectory length for training (0 to use regular T)')
parser.add_argument('--reduced_ls_freq', type=int, default=0,
                    help='Reduce local search frequency (0 to use regular frequency)')

args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

# Performance optimized parameters
eval_data_size = 1000 if args.reduced_eval else 2000  # Reduced evaluation set size
final_eval_data_size = 1000 if args.reduced_eval else 2000
plot_data_size = 1000 if args.reduced_eval else 2000
final_plot_data_size = 1000 if args.reduced_eval else 2000

if args.pis_architectures:
    args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True

# Global timer for profiling
timer = Timer()


def setup_wandb(args, name):
    """Initialize WandB with optimized settings"""
    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        return
    
    # Set offline mode if network issues are detected
    wandb_mode = "online"
    try:
        import socket
        socket.create_connection(("wandb.ai", 443), timeout=1)
    except:
        wandb_mode = "offline"
        print("Network issues detected - using WandB in offline mode")
    
    wandb.init(
        entity="energy-sampling",
        project="energy-sampling",
        name=name,
        config=args,
        mode=wandb_mode,
    )


def get_energy():
    """Get energy model based on arguments"""
    timer.start("energy_init")
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
    timer.stop("energy_init")
    return energy


def plot_step(energy, gfn_model, name):
    """Generate plots for visualization with optimized settings"""
    timer.start("plot_step")
    gfn_model.eval()
    
    with torch.no_grad():
        # Use smaller plot size for speed
        data_size = plot_data_size // 2 if args.reduced_eval else plot_data_size
        
        init_states = torch.zeros(data_size, energy.data_ndim, device=device)
        states, _, _ = gfn_model.get_trajectory_fwd(init_states)
        samples = states[:, -1, :]

        # Generate the visualization
        images = energy.log_and_visualize(samples.cpu().numpy(), name)
    
    gfn_model.train()
    timer.stop("plot_step")
    return images


def prefill_buffer(buffer, gfn_model, energy, batch_size, device):
    """Efficiently prefill buffer with initial samples for better stability"""
    timer.start("prefill_buffer")
    print("Prefilling buffer with initial samples...")
    
    # Determine prefill amount (5% of buffer capacity)
    prefill_size = min(buffer.capacity // 20, 10000)
    prefill_batches = (prefill_size + batch_size - 1) // batch_size
    
    for _ in range(prefill_batches):
        current_batch_size = min(batch_size, prefill_size)
        
        # Generate samples from both the energy model and random initialization
        samples = energy.sample(current_batch_size).to(device)
        log_rewards = energy.log_reward(samples)
        
        # Add to buffer
        buffer.add_batch(samples, log_rewards)
        prefill_size -= current_batch_size
        if prefill_size <= 0:
            break
    
    print(f"Buffer prefilled with {buffer.size} samples")
    timer.stop("prefill_buffer")


def langevin_dynamics_optimized(x, log_reward_fn, device, args, **kwargs):
    """Optimized Langevin dynamics implementation"""
    timer.start("langevin_dynamics")
    max_steps = min(args.max_iter_ls, 50)  # Limit maximum steps
    result = langevin_dynamics(x, log_reward_fn, device, args, max_steps=max_steps, **kwargs)
    timer.stop("langevin_dynamics")
    return result


def fwd_train_step(energy, gfn_model, exploration_std=None, batch_size=None, return_exp=False):
    """Forward training step with optimizations"""
    timer.start("fwd_train_step")
    
    # Use provided batch size or default
    actual_batch_size = batch_size if batch_size is not None else args.batch_size
    
    # Get appropriate loss function based on mode
    loss_fn = get_gfn_forward_loss(args.mode_fwd, coeff_matrix=coeff_matrix)
    
    # Determine trajectory length (use reduced if specified)
    T = args.reduced_T if args.reduced_T > 0 else args.T
    
    # Use mixed precision if enabled
    if args.use_amp:
        with autocast():
            loss, trajectory_info = loss_fn(gfn_model, args.batch_size, energy.log_reward, exploration_std=exploration_std)
    else:
        loss, trajectory_info = loss_fn(gfn_model, actual_batch_size, energy.log_reward, exploration_std=exploration_std)
    
    if return_exp:
        timer.stop("fwd_train_step")
        return loss, *trajectory_info
    
    timer.stop("fwd_train_step")
    return loss


def bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std=None, batch_size=None, it=0):
    """Backward training step with optimizations"""
    timer.start("bwd_train_step")
    
    # Use provided batch size or default
    actual_batch_size = batch_size if batch_size is not None else args.batch_size
    half_batch = actual_batch_size // 2
    
    # Get appropriate loss function
    if args.mode_bwd == "tb":
        loss_fn = get_gfn_backward_loss(args.mode_bwd, coeff_matrix=coeff_matrix)
    else:
        loss_fn = get_gfn_backward_loss(args.mode_bwd)
    
    # Determine if we should run local search
    do_local_search = args.local_search and it >= args.burn_in
    
    # Get samples from buffer
    states, log_rewards = buffer.sample(half_batch)
    
    # Use mixed precision if enabled
    if args.use_amp:
        with autocast():
            loss = loss_fn(gfn_model, states, log_rewards, exploration_std=exploration_std)
    else:
        loss = loss_fn(gfn_model, states, log_rewards, exploration_std=exploration_std)
    
    # Run local search with reduced frequency
    ls_freq = args.reduced_ls_freq if args.reduced_ls_freq > 0 else args.ls_cycle
    if do_local_search and it % ls_freq == 0:
        timer.start("local_search")
        
        # Sample fewer examples for local search to improve speed
        states_ls, log_rewards_ls = buffer.sample(half_batch // 2)
        
        with torch.no_grad():
            improved_states, improved_log_rewards = langevin_dynamics_optimized(
                states_ls, energy.log_reward, device, args
            )
            
            # Add improved samples to buffer
            buffer_ls.add_batch(improved_states, improved_log_rewards)
            
            # Get samples from local search buffer
            if buffer_ls.size >= half_batch:
                states_ls, log_rewards_ls = buffer_ls.sample(half_batch)
                
                # Compute loss using local search samples
                if args.use_amp:
                    with autocast():
                        loss_ls = loss_fn(gfn_model, states_ls, log_rewards_ls, exploration_std=exploration_std)
                else:
                    loss_ls = loss_fn(gfn_model, states_ls, log_rewards_ls, exploration_std=exploration_std)
                
                # Combine losses
                loss = 0.5 * (loss + loss_ls)
        
        timer.stop("local_search")
    
    timer.stop("bwd_train_step")
    return loss


def train_step_optimized(energy, gfn_model, gfn_optimizer, buffer, buffer_ls, 
                        exploration_std, scaler=None, use_amp=False, 
                        micro_batch_size=32, iteration=0):
    """Optimized training step with gradient accumulation and mixed precision support"""
    timer.start("train_step")
    
    # Reset gradients
    gfn_optimizer.zero_grad()
    
    # Calculate how many micro-batches we need
    actual_micro_batch = min(micro_batch_size, args.batch_size)
    num_micro_batches = max(1, args.batch_size // actual_micro_batch)
    total_loss = 0.0
    
    # Gradient accumulation loop
    for _ in range(num_micro_batches):
        if use_amp:
            with autocast():
                if args.both_ways:
                    # Forward training
                    fwd_loss = fwd_train_step(energy, gfn_model, exploration_std, 
                                             batch_size=actual_micro_batch, return_exp=True)
                    
                    if isinstance(fwd_loss, tuple):
                        loss, states, _, _, log_r = fwd_loss
                        # Add samples to buffer for backward training
                        buffer.add_batch(states[:, -1], log_r)
                    else:
                        loss = fwd_loss
                    
                    # Backward training from buffer
                    bwd_loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, 
                                             exploration_std, batch_size=actual_micro_batch, 
                                             it=iteration)
                    
                    # Combine losses with weighting
                    loss = args.alpha * loss + (1 - args.alpha) * bwd_loss
                else:
                    # Only forward training
                    loss = fwd_train_step(energy, gfn_model, exploration_std, 
                                         batch_size=actual_micro_batch)
                
                # Scale loss for gradient accumulation
                loss = loss / num_micro_batches
            
            # Mixed precision backward pass
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
    
    timer.stop("train_step")
    return total_loss


def eval_step(eval_data, energy, gfn_model, final_eval=False):
    """Evaluate model performance"""
    timer.start("eval_step")
    gfn_model.eval()
    
    # Generate samples
    n_samples = eval_data.shape[0]
    init_states = torch.zeros(n_samples, energy.data_ndim, device=device)
    states, logps, pb = gfn_model.get_trajectory_fwd(init_states)
    samples = states[:, -1, :]
    
    # Calculate metrics
    true_log_rewards = energy.log_reward(eval_data)
    pred_log_rewards = energy.log_reward(samples)
    
    # Get relevant metrics
    metrics_dict = get_batch_metrics(samples, pred_log_rewards, eval_data, true_log_rewards)
    
    # Use different prefix for final evaluation vs regular
    prefix = "final" if final_eval else "eval"
    metrics = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
    
    gfn_model.train()
    timer.stop("eval_step")
    return metrics


def log_visualizations(energy, gfn_model, name, step):
    """Create and log visualizations to WandB"""
    if args.disable_wandb:
        return
        
    timer.start("visualizations")
    
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
    
    timer.stop("visualizations")


def train():
    name = get_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    # Setup WandB with enhanced configuration
    if not args.disable_wandb:
        setup_wandb(args, name)
        
        # Log code to WandB for reproducibility
        if os.path.exists("energy_sampling") and not args.disable_wandb:
            wandb.run.log_code("energy_sampling", include_fn=lambda path: path.endswith(".py"))

    # Start timing the initialization
    timer.start("initialization")
    
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
    
    # Prefill buffer for better initial training
    prefill_buffer(buffer, gfn_model, energy, args.batch_size, device)
    
    # Setup evaluation frequency based on total epochs
    eval_frequency = min(args.epochs // 50 + 1, 500)  # At least 50 evals, max every 500 epochs
    if args.reduced_eval:
        eval_frequency *= 2  # Even less frequent for reduced eval mode
    
    # Model saving frequency - save even less frequently
    save_frequency = eval_frequency * 5  
    
    # Determine micro batch size for gradient accumulation
    micro_batch_size = min(args.batch_size, 64)  # Use smaller micro batches for better memory usage
    
    timer.stop("initialization")
    print(f"Initialization completed in {timer.times['initialization']:.2f}s")
    
    # Main training loop
    timer.start("training_loop")
    total_samples = 0
    start_time = time.time()
    
    # Use tqdm for progress tracking
    pbar = trange(args.epochs)
    for i in pbar:
        # Update exploration standard deviation based on iteration
        exploration_std = get_exploration_std(i, args.epochs) if args.exploratory else None
        
        # Track training step time for this batch
        time_step_start = time.time()
        
        # Perform optimized training step
        loss = train_step_optimized(
            energy=energy,
            gfn_model=gfn_model,
            gfn_optimizer=gfn_optimizer,
            buffer=buffer,
            buffer_ls=buffer_ls,
            exploration_std=exploration_std,
            scaler=scaler,
            use_amp=args.use_amp,
            micro_batch_size=micro_batch_size,
            iteration=i
        )
        
        # Track total processed samples
        total_samples += args.batch_size
        
        # Update progress bar description
        step_time = time.time() - time_step_start
        samples_per_sec = args.batch_size / max(step_time, 1e-5)
        pbar.set_description(f"Loss: {loss:.4f}, {samples_per_sec:.1f} samples/s")
        
        # Log metrics at regular intervals
        if i % args.log_interval == 0:
            # Basic metrics to track
            metrics = {
                'train/loss': loss,
                'train/samples_per_second': samples_per_sec,
                'train/iteration': i,
            }
            
            # Log additional detailed metrics for deeper analysis if not reduced
            if not args.reduced_eval and i % (args.log_interval * 5) == 0:
                with torch.no_grad():
                    # Add buffer statistics
                    metrics.update({
                        'buffer/size': buffer.size,
                        'buffer/ls_size': buffer_ls.size if args.local_search else 0,
                    })
            
            # Periodically log detailed evaluation metrics
            if i % eval_frequency == 0:
                # Use torch.no_grad for evaluation to save memory
                with torch.no_grad():
                    gfn_model.eval()  # Set model to evaluation mode
                    eval_metrics = eval_step(eval_data, energy, gfn_model, final_eval=False)
                    metrics.update(eval_metrics)
                    
                    # Visualize and log less frequently for efficiency
                    if not args.reduced_eval and i % (eval_frequency * 2) == 0:
                        log_visualizations(energy, gfn_model, name, i)
                    
                    gfn_model.train()  # Set back to training mode
                
                # Calculate and log overall training time metrics
                elapsed_total = time.time() - start_time
                metrics.update({
                    'perf/total_training_time': elapsed_total,
                    'perf/avg_examples_per_second': total_samples / elapsed_total,
                })
                
                # Report profiling information periodically
                if i % (eval_frequency * 2) == 0:
                    timer.report()
                
                if not args.disable_wandb:
                    wandb.log(metrics, step=i)
                
                # Save model less frequently to reduce I/O overhead
                if i % save_frequency == 0 and i > 0:
                    model_path = f'{name}model_{i}.pt'
                    torch.save(gfn_model.state_dict(), model_path)
                    if not args.disable_wandb:
                        wandb.save(model_path)  # Upload to WandB as well
    
    timer.stop("training_loop")
    
    # Final evaluation
    timer.start("final_evaluation")
    with torch.no_grad():
        gfn_model.eval()
        final_eval_data = energy.sample(final_eval_data_size).to(device)
        eval_results = eval_step(final_eval_data, energy, gfn_model, final_eval=True)
        metrics = eval_results
        
        # Final visualization
        if not args.reduced_eval:
            log_visualizations(energy, gfn_model, name, args.epochs)
    timer.stop("final_evaluation")
    
    # Save final model
    final_model_path = f'{name}model_final.pt'
    torch.save(gfn_model.state_dict(), final_model_path)
    if not args.disable_wandb:
        wandb.save(final_model_path)
    
    # Log final performance summary
    total_time = time.time() - start_time
    if not args.disable_wandb:
        wandb.log({
            'final/training_time_hours': total_time / 3600,
            'final/total_epochs': args.epochs,
            'final/samples_processed': total_samples,
            'final/average_samples_per_second': total_samples / total_time
        })
        wandb.finish()
    
    # Print final report
    print(f"Training completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print(f"Processed {total_samples} samples at {total_samples/total_time:.1f} samples/second")
    timer.report()


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