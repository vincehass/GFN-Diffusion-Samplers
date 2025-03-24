import argparse
import torch
import os
import time
from utils import set_seed
from buffer import ReplayBuffer
from langevin import langevin_dynamics
from models import GFN
from energies.nine_gmm import NineGaussianMixture
from energies.many_well import ManyWell
from tqdm import trange

parser = argparse.ArgumentParser(description='GFN Energy Sampling (Minimal)')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--energy', type=str, default='9gmm', choices=('9gmm', '25gmm', 'many_well'))
parser.add_argument('--local_search', action='store_true', default=False)
parser.add_argument('--max_iter_ls', type=int, default=20)  # Reduced from 200
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--disable_wandb', action='store_true', default=False)
args = parser.parse_args()

# Set seeds for reproducibility
set_seed(args.seed)

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set up energy model
if args.energy == '9gmm':
    energy = NineGaussianMixture(device=device)
elif args.energy == '25gmm':
    energy = NineGaussianMixture(device=device)
elif args.energy == 'many_well':
    energy = ManyWell(device=device)

# Initialize model - using the correct parameter name (T instead of num_steps)
gfn_model = GFN(
    T=args.T,  # Changed back to T as expected by your GFN implementation
    input_dim=energy.data_ndim,
    hidden_dim=args.hidden_dim,
    scale=5.0,
    device=device,
    learn_pb=False
).to(device)

# Simple optimizer
optimizer = torch.optim.Adam(gfn_model.parameters(), lr=args.lr)

# Initialize buffer
buffer = ReplayBuffer(
    capacity=args.buffer_size,
    device=device,
    log_reward_fn=energy.log_reward,
    batch_size=args.batch_size,
    data_ndim=energy.data_ndim
)

# Prefill buffer with initial samples
print("Prefilling buffer...")
with torch.no_grad():
    samples = energy.sample(min(5000, args.buffer_size)).to(device)
    log_rewards = energy.log_reward(samples)
    buffer.add_batch(samples, log_rewards)

# Main training loop
print("Starting training...")
pbar = trange(args.epochs)
start_time = time.time()

for i in pbar:
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass - simple trajectory balance loss
    init_states = torch.zeros(args.batch_size, energy.data_ndim, device=device)
    states, logps, _ = gfn_model.get_trajectory_fwd(init_states)
    final_states = states[:, -1]
    log_rewards = energy.log_reward(final_states)
    
    # Simple TB loss: log p(forward) - log reward
    loss = torch.mean(logps.sum(dim=1) - log_rewards)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Add samples to buffer
    buffer.add_batch(final_states, log_rewards)
    
    # Optional local search with reduced frequency
    if args.local_search and i % 5 == 0:
        with torch.no_grad():
            samples, log_rewards = buffer.sample(args.batch_size // 2)
            improved_samples, improved_log_rewards = langevin_dynamics(
                samples, energy.log_reward, device, args
            )
            buffer.add_batch(improved_samples, improved_log_rewards)
    
    # Update progress bar
    if i % 10 == 0:
        pbar.set_description(f"Loss: {loss.item():.4f}")
    
    # Simple evaluation without visualization
    if i % 200 == 0:
        with torch.no_grad():
            test_samples = torch.zeros(1000, energy.data_ndim, device=device)
            states, _, _ = gfn_model.get_trajectory_fwd(test_samples)
            final_states = states[:, -1]
            test_log_rewards = energy.log_reward(final_states)
            mean_reward = torch.mean(torch.exp(test_log_rewards)).item()
            print(f"Epoch {i}, Mean reward: {mean_reward:.4f}")

print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Save final model
torch.save(gfn_model.state_dict(), f'model_final_{args.energy}.pt') 