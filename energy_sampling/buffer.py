import torch
import numpy as np
from collections import deque

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        super(SampleDataset, self).__init__()
        self.sample_list = sample
      
    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        return sample

    def update(self, sample):
        self.sample_list = torch.cat([self.sample_list, sample], dim=0)

    def deque(self, length):
        self.sample_list = self.sample_list[length:]

    def get_seq(self):
        return self.sample_list

    def __len__(self):
        return len(self.sample_list)

    def collate(data_list):
        return torch.stack(data_list)

class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, rewards):
        super(RewardDataset, self).__init__()
        self.rewards = rewards
        self.raw_tsrs = self.rewards

    def __getitem__(self, idx):
        return self.rewards[idx]
        #return  self.score_list[idx]

    def update(self, rewards):
        new_rewards = rewards
        self.raw_tsrs = torch.cat([self.rewards, new_rewards], dim=0)
        self.rewards = self.raw_tsrs

    def deque(self, length):
        self.raw_tsrs = self.raw_tsrs[length:]
        self.rewards = self.raw_tsrs

    def get_tsrs(self):
        return self.rewards

    def __len__(self):
        return self.rewards.size(0)

    def collate(data_list):
        return torch.stack(data_list)

class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]

    def collate(data_list):
        return [dataset.collate(data_list) for dataset, data_list in zip(self.datasets, zip(*data_list))]

def collate(data_list):
    sample, rewards = zip(*data_list)
    sample_data = SampleDataset.collate(sample)
    reward_data = RewardDataset.collate(rewards)
    return sample_data, reward_data

class ReplayBuffer:
    """
    Optimized Replay Buffer with efficient data storage, batch operations,
    and prioritized experience replay.
    
    Note: This buffer stores log_rewards, not raw rewards, for numerical stability
    in energy-based models.
    """
    def __init__(self, capacity, device, log_reward_fn, batch_size, data_ndim=2, 
                 beta=1.0, rank_weight=1e-2, prioritized='none'):
        self.capacity = capacity
        self.device = device
        self.log_reward_fn = log_reward_fn  # This is the log_reward function
        self.size = 0
        self.beta = beta
        self.rank_weight = rank_weight
        self.prioritized = prioritized
        self.data_ndim = data_ndim
        
        # Preallocate tensors for storage
        self.data = torch.zeros((capacity, data_ndim), device=device)
        self.log_rewards = torch.zeros(capacity, device=device)  # Renamed to log_rewards for clarity
        
        # For faster sampling without replacement
        self._last_sampled = set()
        self._sample_history = deque(maxlen=min(capacity // 10, 1000))
        
        # Statistics tracking for WandB
        self.add_count = 0
        self.sample_count = 0
        
        # Pre-allocate indices for faster sampling
        self._full_indices = torch.arange(capacity, device=device)
        
    def add(self, state, log_reward):
        """Add a single state and log_reward to the buffer"""
        if self.size < self.capacity:
            self.data[self.size] = state
            self.log_rewards[self.size] = log_reward  # Store log_reward
            self.size += 1
        else:
            idx = torch.randint(0, self.capacity, (1,)).item()
            self.data[idx] = state
            self.log_rewards[idx] = log_reward  # Store log_reward
            
        self.add_count += 1
        
    def add_batch(self, states, log_rewards):
        """Add a batch of states and log_rewards to the buffer at once"""
        batch_size = states.shape[0]
        if self.size < self.capacity:
            # Space available, add as many as we can
            end_idx = min(self.size + batch_size, self.capacity)
            add_count = end_idx - self.size
            if add_count > 0:
                self.data[self.size:end_idx] = states[:add_count]
                self.log_rewards[self.size:end_idx] = log_rewards[:add_count]  # Store log_rewards
                self.size = end_idx
                
                # If we have more samples than capacity, replace some existing ones
                if batch_size > add_count:
                    remaining = batch_size - add_count
                    replace_idx = torch.randint(0, self.size - add_count, (remaining,))
                    self.data[replace_idx] = states[add_count:batch_size]
                    self.log_rewards[replace_idx] = log_rewards[add_count:batch_size]  # Store log_rewards
        else:
            # Buffer full, replace random elements
            if batch_size >= self.capacity:
                # If batch is larger than capacity, just use the last portion
                self.data = states[-self.capacity:].clone()
                self.log_rewards = log_rewards[-self.capacity:].clone()  # Store log_rewards
            else:
                # Replace random indices without repetition
                replace_idx = torch.randperm(self.capacity, device=self.device)[:batch_size]
                self.data[replace_idx] = states
                self.log_rewards[replace_idx] = log_rewards  # Store log_rewards
        
        self.add_count += batch_size

    def sample(self, batch_size):
        """Sample a batch from the buffer using prioritization if enabled"""
        if self.size == 0:
            # Handle empty buffer case
            return (
                torch.zeros((batch_size, self.data_ndim), device=self.device),
                torch.zeros(batch_size, device=self.device)
            )
            
        actual_batch_size = min(batch_size, self.size)
        
        if self.prioritized == 'none' or self.size <= batch_size:
            # Simple random sampling without replacement when possible
            if self.size <= batch_size:
                indices = self._full_indices[:self.size]
            else:
                indices = torch.randperm(self.size, device=self.device)[:actual_batch_size]
        
        elif self.prioritized == 'reward':
            # Reward-prioritized sampling - using log_rewards
            if self.beta < 1e-10:  # Almost uniform
                indices = torch.randperm(self.size, device=self.device)[:actual_batch_size]
            else:
                # Need to exponentiate log_rewards for prioritized sampling
                # Since we want to prioritize higher rewards, not higher log_rewards
                exp_rewards = torch.exp(self.beta * self.log_rewards[:self.size])
                # Handle potential numerical overflow by normalizing
                exp_rewards = exp_rewards / (exp_rewards.sum() + 1e-10)
                
                # Sample using the calculated probabilities
                indices = torch.multinomial(exp_rewards, actual_batch_size, replacement=False)
                
        elif self.prioritized == 'rank':
            # Rank-based prioritized sampling
            # Sort log_rewards and assign probabilities by rank
            _, sorted_indices = torch.sort(self.log_rewards[:self.size], descending=True)
            ranks = torch.arange(1, self.size + 1, dtype=torch.float32, device=self.device)
            probs = (1.0 / (ranks + self.rank_weight)) 
            probs = probs / probs.sum()
            
            # Sample ranks based on probability
            selected_ranks = torch.multinomial(probs, actual_batch_size, replacement=False)
            indices = sorted_indices[selected_ranks]
            
        # Get the selected data
        batch_data = self.data[indices]
        batch_log_rewards = self.log_rewards[indices]  # Return log_rewards
        
        # Track sampling for statistics
        self.sample_count += actual_batch_size
        self._sample_history.extend(indices.tolist())
        
        return batch_data, batch_log_rewards
    
    def get_stats(self):
        """Get buffer statistics for monitoring"""
        if self.size == 0:
            return {
                "buffer/size": 0,
                "buffer/capacity": self.capacity,
                "buffer/fullness": 0.0,
                "buffer/log_reward_mean": 0.0,
                "buffer/log_reward_std": 0.0,
                "buffer/add_count": self.add_count,
                "buffer/sample_count": self.sample_count,
            }
            
        valid_log_rewards = self.log_rewards[:self.size]
        
        # Calculate stats in both log and original space
        with torch.no_grad():
            # For log rewards
            log_r_mean = valid_log_rewards.mean().item()
            log_r_std = valid_log_rewards.std().item() if self.size > 1 else 0.0
            log_r_min = valid_log_rewards.min().item()
            log_r_max = valid_log_rewards.max().item()
            
            # For actual rewards (exponentiate with care)
            # Shift log rewards to avoid overflow
            shifted_log_r = valid_log_rewards - valid_log_rewards.max()
            exp_r = torch.exp(shifted_log_r)
            mean_r = exp_r.mean().item()
            
        return {
            "buffer/size": self.size,
            "buffer/capacity": self.capacity,
            "buffer/fullness": self.size / self.capacity,
            "buffer/log_reward_mean": log_r_mean,
            "buffer/log_reward_std": log_r_std,
            "buffer/log_reward_min": log_r_min,
            "buffer/log_reward_max": log_r_max,
            "buffer/reward_mean": mean_r,  # This is approximate
            "buffer/add_count": self.add_count,
            "buffer/sample_count": self.sample_count,
            "buffer/unique_ratio": len(set(self._sample_history)) / len(self._sample_history) if self._sample_history else 1.0,
        }