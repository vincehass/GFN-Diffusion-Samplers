import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class DiffusionSchedule:
    """
    Diffusion scheduling for forward and reverse processes.
    """
    def __init__(self, num_diffusion_timesteps=1000, schedule_type='linear', beta_start=1e-4, beta_end=0.02):
        """
        Initialize the diffusion schedule.
        
        Args:
            num_diffusion_timesteps: Number of diffusion timesteps
            schedule_type: Type of schedule ('linear' or 'cosine')
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'linear':
            # Linear schedule
            self.betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
        elif schedule_type == 'cosine':
            # Cosine schedule as per improved DDPM paper
            steps = num_diffusion_timesteps + 1
            x = torch.linspace(0, num_diffusion_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0, 0.999)
        else:
            raise ValueError(f"Unknown schedule type {schedule_type}")
        
        # Calculate alphas and alpha cumprod quantities
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_variance = torch.cat([self.posterior_variance[1:], torch.tensor([0.0])])
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:], torch.tensor([1e-20])]))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - the forward diffusion process.
        
        Args:
            x_0: Initial samples [batch_size, dims]
            t: Timesteps [batch_size]
            noise: Optional noise to use
            
        Returns:
            x_t: Noisy samples at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get alpha values for each timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        # Sample from q(x_t | x_0)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, x_0, t, noise=None, condition=None):
        """
        Compute the loss for denoising diffusion.
        
        Args:
            model: The model to predict noise
            x_0: Initial samples [batch_size, dims]
            t: Timesteps [batch_size]
            noise: Optional noise to use
            condition: Optional condition for conditional diffusion
            
        Returns:
            loss: MSE loss between predicted and true noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get noisy samples
        x_t = self.q_sample(x_0, t, noise=noise)
        
        # Predict noise using the model
        if condition is not None:
            predicted_noise = model(x_t, t, condition)
        else:
            predicted_noise = model(x_t, t)
        
        # Compute MSE loss
        return F.mse_loss(predicted_noise, noise)
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Given the noisy sample x_t and the predicted noise, compute an estimate of the clean sample x_0.
        
        Args:
            x_t: Noisy input at timestep t
            t: Timestep
            noise: Predicted noise
            
        Returns:
            x_0_pred: Prediction of the clean sample
        """
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior(self, x_0, x_t, t):
        mean = self.posterior_mean_coef1[t].reshape(-1, 1) * x_0 + self.posterior_mean_coef2[t].reshape(-1, 1) * x_t
        var = self.posterior_variance[t].reshape(-1, 1)
        log_var = self.posterior_log_variance_clipped[t].reshape(-1, 1)
        return mean, var, log_var
    
    def p_mean_variance(self, model, x_t, t, model_output=None, condition=None):
        """
        Calculate the mean and variance of the diffusion posterior p(x_{t-1} | x_t)
        
        Args:
            model: The model to predict noise
            x_t: Noisy samples at timestep t
            t: Timesteps
            model_output: Optional precomputed model output
            condition: Optional condition for conditional diffusion
            
        Returns:
            mean, variance, log_variance
        """
        if model_output is None:
            # If model_output not provided, compute it with the model
            if condition is not None:
                predicted_noise = model(x_t, t, condition)
            else:
                predicted_noise = model(x_t, t)
        else:
            # Use provided model_output
            predicted_noise = model_output
        
        # Extract alphas from scheduler for timestep t
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t]
        
        # Calculate mean
        x_0_pred = self.predict_start_from_noise(x_t, t, predicted_noise)
        model_mean = self.q_posterior_mean(x_0_pred, x_t, t)
        
        # Calculate variance and log variance
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance = self.posterior_log_variance_clipped[t]
        
        model_variance = posterior_variance.view(-1, 1)
        model_log_variance = posterior_log_variance.view(-1, 1)
        
        return model_mean, model_variance, model_log_variance
    
    def p_sample(self, model, x_t, t, condition=None):
        """
        Sample from p(x_{t-1} | x_t), optionally with GFN guidance.
        
        Args:
            model: The model to predict noise
            x_t: Noisy samples at timestep t
            t: Timesteps
            condition: Optional condition for conditional models
            
        Returns:
            x_{t-1}: Samples at timestep t-1
        """
        # For conditional models with GFN guidance
        if hasattr(self, 'energy_fn') and self.energy_fn is not None and condition is not None:
            if not isinstance(condition, torch.Tensor):
                condition = torch.tensor(condition, device=self.device)
            if condition.dtype != torch.long:
                condition = condition.long()
            return self.p_sample_with_guidance(model, x_t, t, condition)
            
        # For standard diffusion, get model mean and variance
        if condition is not None and hasattr(model, 'condition_embedding'):
            # Conditional model but no GFN guidance
            if not isinstance(condition, torch.Tensor):
                condition = torch.tensor(condition, device=self.device)
            if condition.dtype != torch.long:
                condition = condition.long()
            model_output = model(x_t, t, condition)
            model_mean, model_variance, model_log_variance = self.p_mean_variance(model, x_t, t, model_output, condition)
        else:
            # Unconditional model
            model_mean, model_variance, model_log_variance = self.p_mean_variance(model, x_t, t)
        
        # For t=0, don't add noise
        if torch.all(t == 0):
            return model_mean
        
        # Sample with Gaussian noise
        noise = torch.randn_like(x_t)
        
        # If using GFN guidance, generate multiple candidates and select by energy
        if hasattr(self, 'energy_fn') and self.energy_fn is not None:
            # Sample multiple candidates
            num_candidates = 10
            batch_noise = torch.randn(num_candidates, *x_t.shape, device=self.device)
            
            # Generate candidates
            candidates = model_mean.unsqueeze(0) + torch.exp(0.5 * model_log_variance).unsqueeze(0) * batch_noise
            
            # Reshape for energy computation
            batch_size = x_t.shape[0]
            candidates_flat = candidates.reshape(num_candidates * batch_size, -1)
            
            # Compute energy for each candidate
            with torch.no_grad():
                energies = self.energy_fn(candidates_flat)
                
                # Check the shape of energies
                total_elements = energies.numel()
                expected_elements = num_candidates * batch_size
                
                # If the energy function returns a single value per input
                if total_elements == expected_elements:
                    energies = energies.reshape(num_candidates, batch_size)
                else:
                    # If the energy function returns multiple values per input
                    print(f"Warning: Energy function returned unexpected shape. Using mean energy per candidate.")
                    
                    # Force energies to be a 1D tensor of the expected size
                    if total_elements > expected_elements:
                        # If we have more elements than expected, take the first expected_elements
                        energies = energies.view(-1)[:expected_elements]
                    elif total_elements < expected_elements:
                        # If we have fewer elements than expected, pad with high energy values
                        padded = torch.full((expected_elements,), 100.0, device=energies.device)
                        padded[:total_elements] = energies.view(-1)
                        energies = padded
                    
                    # Reshape to [num_candidates, batch_size]
                    energies = energies.reshape(num_candidates, batch_size)
            
            # Use softmax to get sampling probabilities based on energy
            probs = F.softmax(-energies * self.guidance_scale, dim=0)
            
            # Add detailed diagnostic logging for probability values
            prob_min = probs.min().item()
            prob_max = probs.max().item()
            prob_sum = probs.sum(dim=0).mean().item()
            prob_has_nan = torch.isnan(probs).any().item()
            prob_has_inf = torch.isinf(probs).any().item()
            prob_has_neg = (probs < 0).any().item()
            
            # Check for inf, nan, or negative values in probs
            if prob_has_nan or prob_has_inf or prob_has_neg:
                # Log detailed diagnostic information
                print(f"Warning: Invalid probability values detected. min={prob_min}, max={prob_max}, sum={prob_sum}")
                print(f"NaN: {prob_has_nan}, Inf: {prob_has_inf}, Negative: {prob_has_neg}")
                print(f"Energy min: {energies.min().item()}, max: {energies.max().item()}, guidance_scale: {self.guidance_scale}")
                
                # Before falling back, try to fix the issue by clamping energies
                if prob_has_inf or abs(energies.max().item()) > 30:
                    print("Attempting to fix by clamping extreme energy values...")
                    # Clamp energy values to avoid overflow in softmax
                    energies = torch.clamp(energies, min=-20.0, max=20.0)
                    # Try again with clamped values
                    probs = F.softmax(-energies * self.guidance_scale, dim=0)
                    
                    # Check if fixed
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                        print("Clamping didn't fix the issue. Falling back to uniform sampling.")
                        probs = torch.ones_like(probs) / num_candidates
                    else:
                        print("Successfully fixed probability values with clamping.")
                else:
                    # Fall back to uniform distribution if there's a problem
                    print("Falling back to uniform sampling.")
                    probs = torch.ones_like(probs) / num_candidates
            
            # Sample indices based on probabilities
            indices = torch.multinomial(probs.T, 1).squeeze(-1)
            
            # Select the sampled candidates
            selected_candidates = torch.zeros_like(x_t)
            for i in range(batch_size):
                selected_candidates[i] = candidates[indices[i], i]
            
            return selected_candidates
        else:
            # Standard diffusion sampling
            return model_mean + torch.exp(0.5 * model_log_variance) * noise
    
    def q_posterior_mean(self, x_0, x_t, t):
        """
        Compute the mean of the posterior q(x_{t-1} | x_t, x_0)
        
        Args:
            x_0: Clean sample
            x_t: Noisy sample at timestep t
            t: Timestep
            
        Returns:
            mean: Mean of the posterior
        """
        posterior_mean_coef1 = self.posterior_mean_coef1[t].reshape(-1, 1)
        posterior_mean_coef2 = self.posterior_mean_coef2[t].reshape(-1, 1)
        
        # Compute posterior mean
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        
        return posterior_mean
    
    def extract(self, a, t, x_shape):
        """
        Extract values from a tensor at indices t and reshape to match x_shape.
        
        Args:
            a: Source tensor
            t: Timestep indices
            x_shape: Target shape
            
        Returns:
            out: Extracted values reshaped to match x_shape
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        
        # Reshape to match x_shape
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class GFNDiffusion:
    """
    GFN-guided diffusion model.
    """
    def __init__(self, model, diffusion, energy_fn=None, guidance_scale=1.0, device='cpu', loss_type='standard'):
        """
        Initialize the GFN-guided diffusion model.
        
        Args:
            model: The model to predict noise
            diffusion: The diffusion schedule
            energy_fn: Energy function for guidance
            guidance_scale: Scale for energy guidance
            device: Device to run on
            loss_type: Type of loss to use ('standard', 'tb_avg')
        """
        self.model = model
        self.diffusion = diffusion
        self.energy_fn = energy_fn
        self.guidance_scale = guidance_scale
        self.device = device
        self.loss_type = loss_type
        
    def to(self, device):
        """Move the model to the specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def parameters(self):
        """Return the model parameters."""
        return self.model.parameters()
    
    def train(self):
        """Set the model to training mode."""
        self.model.train()
        
    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
    
    def get_trajectory_fwd(self, x_0, exploration_std=None, log_reward_fn=None):
        """
        Get forward trajectory for GFlowNet training.
        
        Args:
            x_0: Initial samples (batch_size, dim)
            exploration_std: Exploration noise standard deviation
            log_reward_fn: Function to compute log rewards
            
        Returns:
            states: States along trajectory (batch_size, num_steps, dim)
            log_pfs: Log forward probabilities (batch_size, num_steps)
            log_pbs: Log backward probabilities (batch_size, num_steps)
            log_fs: Log flow values (batch_size, num_steps+1)
        """
        batch_size = x_0.shape[0]
        num_timesteps = self.diffusion.num_diffusion_timesteps
        dim = x_0.shape[1]
        
        # Initialize arrays to store trajectory information
        states = torch.zeros(batch_size, num_timesteps + 1, dim, device=self.device)
        log_pfs = torch.zeros(batch_size, num_timesteps, device=self.device)
        log_pbs = torch.zeros(batch_size, num_timesteps, device=self.device)
        log_fs = torch.zeros(batch_size, num_timesteps + 1, device=self.device)
        
        # Start with Gaussian noise
        states[:, 0] = torch.randn_like(x_0)
        
        # Set initial flow value based on standard normal density
        log_fs[:, 0] = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * torch.sum(states[:, 0] ** 2, dim=1)
        
        # Forward diffusion process
        for t in range(num_timesteps):
            # Current timestep in diffusion indexing
            diffusion_t = num_timesteps - 1 - t
            diffusion_t_tensor = torch.full((batch_size,), diffusion_t, device=self.device, dtype=torch.long)
            
            # Generate next state using the model
            with torch.no_grad():
                x_next = self.p_sample(states[:, t], diffusion_t_tensor, use_gfn=False)
                states[:, t+1] = x_next
            
            # Compute forward probability (p(x_{t+1} | x_t))
            alpha_t = self.diffusion.alphas[diffusion_t]
            beta_t = self.diffusion.betas[diffusion_t]
            
            # p(x_{t+1} | x_t) is Gaussian with:
            # mean = sqrt(alpha_t) * x_t
            # variance = beta_t
            noise = states[:, t+1] - torch.sqrt(alpha_t) * states[:, t]
            log_prob_fwd = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi * beta_t, device=self.device))
            log_prob_fwd = log_prob_fwd - 0.5 * torch.sum(noise ** 2, dim=1) / beta_t
            log_pfs[:, t] = log_prob_fwd
            
            # Compute backward probability (p(x_t | x_{t+1}))
            # For DDPM, this is also a Gaussian
            alpha_bar_t = self.diffusion.alphas_cumprod[diffusion_t]
            posterior_mean_coef1 = self.diffusion.posterior_mean_coef1[diffusion_t]
            posterior_mean_coef2 = self.diffusion.posterior_mean_coef2[diffusion_t]
            posterior_variance = self.diffusion.posterior_variance[diffusion_t]
            
            # Estimate x_0 from x_{t+1}
            if diffusion_t > 0:
                pred_noise = self.model(states[:, t+1], diffusion_t_tensor)
                pred_x0 = self.diffusion.predict_start_from_noise(states[:, t+1], diffusion_t_tensor, pred_noise)
                posterior_mean = posterior_mean_coef1 * states[:, t+1] + posterior_mean_coef2 * pred_x0
            else:
                # At t=0, x_0 is directly observed
                posterior_mean = states[:, t+1]
            
            # Compute log probability of x_t given x_{t+1}
            backward_noise = states[:, t] - posterior_mean
            log_prob_bwd = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi * posterior_variance, device=self.device))
            log_prob_bwd = log_prob_bwd - 0.5 * torch.sum(backward_noise ** 2, dim=1) / posterior_variance
            log_pbs[:, t] = log_prob_bwd
            
            # Compute flow value at the next state
            # For intermediate states, flow is computed recursively
            if t < num_timesteps - 1:
                log_fs[:, t+1] = log_fs[:, t] + log_pfs[:, t] - log_pbs[:, t]
            else:
                # For terminal state, compute log reward
                if log_reward_fn is not None:
                    log_fs[:, t+1] = log_reward_fn(states[:, t+1])
        
        return states, log_pfs, log_pbs, log_fs
    
    def get_trajectory_bwd(self, x_T, exploration_std=None, log_reward_fn=None):
        """
        Get backward trajectory for GFlowNet training.
        
        Args:
            x_T: Terminal samples (batch_size, dim)
            exploration_std: Exploration noise standard deviation
            log_reward_fn: Function to compute log rewards
            
        Returns:
            states: States along trajectory (batch_size, num_steps, dim)
            log_pfs: Log forward probabilities (batch_size, num_steps)
            log_pbs: Log backward probabilities (batch_size, num_steps)
            log_fs: Log flow values (batch_size, num_steps+1)
        """
        batch_size = x_T.shape[0]
        num_timesteps = self.diffusion.num_diffusion_timesteps
        dim = x_T.shape[1]
        
        # Initialize arrays to store trajectory information
        states = torch.zeros(batch_size, num_timesteps + 1, dim, device=self.device)
        log_pfs = torch.zeros(batch_size, num_timesteps, device=self.device)
        log_pbs = torch.zeros(batch_size, num_timesteps, device=self.device)
        log_fs = torch.zeros(batch_size, num_timesteps + 1, device=self.device)
        
        # Start with terminal state
        states[:, -1] = x_T
        
        # Set terminal flow value based on log reward
        if log_reward_fn is not None:
            log_fs[:, -1] = log_reward_fn(x_T)
        
        # Backward diffusion process
        for t in range(num_timesteps):
            # Current timestep in diffusion indexing
            diffusion_t = t
            diffusion_t_tensor = torch.full((batch_size,), diffusion_t, device=self.device, dtype=torch.long)
            
            # Generate previous state using the model
            if t < num_timesteps - 1:
                with torch.no_grad():
                    # We're going backwards, so we're predicting x_{t-1} from x_t
                    x_prev = self.p_sample(states[:, -(t+1)], diffusion_t_tensor, use_gfn=False)
                    states[:, -(t+2)] = x_prev
                
                # Compute forward probability (p(x_{t+1} | x_t)) - in backward direction
                alpha_t = self.diffusion.alphas[diffusion_t]
                beta_t = self.diffusion.betas[diffusion_t]
                
                # p(x_{t+1} | x_t) is Gaussian with:
                # mean = sqrt(alpha_t) * x_t
                # variance = beta_t
                noise = states[:, -(t+1)] - torch.sqrt(alpha_t) * states[:, -(t+2)]
                log_prob_fwd = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi * beta_t, device=self.device))
                log_prob_fwd = log_prob_fwd - 0.5 * torch.sum(noise ** 2, dim=1) / beta_t
                log_pfs[:, -(t+1)] = log_prob_fwd
                
                # Compute backward probability (p(x_t | x_{t+1})) - in backward direction
                # For DDPM, this is also a Gaussian
                alpha_bar_t = self.diffusion.alphas_cumprod[diffusion_t]
                posterior_mean_coef1 = self.diffusion.posterior_mean_coef1[diffusion_t]
                posterior_mean_coef2 = self.diffusion.posterior_mean_coef2[diffusion_t]
                posterior_variance = self.diffusion.posterior_variance[diffusion_t]
                
                # Estimate x_0 from x_{t+1}
                pred_noise = self.model(states[:, -(t+1)], diffusion_t_tensor)
                pred_x0 = self.diffusion.predict_start_from_noise(states[:, -(t+1)], diffusion_t_tensor, pred_noise)
                posterior_mean = posterior_mean_coef1 * states[:, -(t+1)] + posterior_mean_coef2 * pred_x0
                
                # Compute log probability of x_t given x_{t+1}
                backward_noise = states[:, -(t+2)] - posterior_mean
                log_prob_bwd = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi * posterior_variance, device=self.device))
                log_prob_bwd = log_prob_bwd - 0.5 * torch.sum(backward_noise ** 2, dim=1) / posterior_variance
                log_pbs[:, -(t+1)] = log_prob_bwd
                
                # Compute flow value at the next state (going backwards)
                log_fs[:, -(t+2)] = log_fs[:, -(t+1)] - log_pfs[:, -(t+1)] + log_pbs[:, -(t+1)]
            else:
                # For t = num_timesteps - 1, we reach the initial state
                states[:, 0] = torch.randn_like(x_T)
                
                # Initial flow value is based on standard normal density
                log_fs[:, 0] = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * torch.sum(states[:, 0] ** 2, dim=1)
                
                # Compute last step probabilities
                alpha_t = self.diffusion.alphas[diffusion_t]
                beta_t = self.diffusion.betas[diffusion_t]
                
                noise = states[:, 1] - torch.sqrt(alpha_t) * states[:, 0]
                log_prob_fwd = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi * beta_t, device=self.device))
                log_prob_fwd = log_prob_fwd - 0.5 * torch.sum(noise ** 2, dim=1) / beta_t
                log_pfs[:, 0] = log_prob_fwd
                
                # Backward probability from initial state
                alpha_bar_t = self.diffusion.alphas_cumprod[diffusion_t]
                posterior_mean_coef1 = self.diffusion.posterior_mean_coef1[diffusion_t]
                posterior_mean_coef2 = self.diffusion.posterior_mean_coef2[diffusion_t]
                posterior_variance = self.diffusion.posterior_variance[diffusion_t]
                
                pred_noise = self.model(states[:, 1], diffusion_t_tensor)
                pred_x0 = self.diffusion.predict_start_from_noise(states[:, 1], diffusion_t_tensor, pred_noise)
                posterior_mean = posterior_mean_coef1 * states[:, 1] + posterior_mean_coef2 * pred_x0
                
                backward_noise = states[:, 0] - posterior_mean
                log_prob_bwd = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi * posterior_variance, device=self.device))
                log_prob_bwd = log_prob_bwd - 0.5 * torch.sum(backward_noise ** 2, dim=1) / posterior_variance
                log_pbs[:, 0] = log_prob_bwd
        
        return states, log_pfs, log_pbs, log_fs
    
    def train_step(self, optimizer, x_0, condition=None):
        """
        Perform a training step.
        
        Args:
            optimizer: Optimizer for training
            x_0: Initial samples
            condition: Optional condition for conditional diffusion
            
        Returns:
            loss: Training loss
        """
        optimizer.zero_grad()
        
        # Sample t randomly for each sample in the batch
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.diffusion.num_diffusion_timesteps, (batch_size,), device=self.device).long()
        
        # Compute loss
        if self.loss_type == 'tb_avg' and self.energy_fn is not None:
            # Use TB-Average loss
            from energy_sampling.gflownet_losses import fwd_tb_avg
            
            # Convert energy function to log reward function
            log_reward_fn = lambda x: -self.energy_fn(x)
            
            # Compute TB-Average loss
            loss = fwd_tb_avg(x_0, self, log_reward_fn)
        else:
            # Use standard diffusion loss
            loss = self.diffusion.p_losses(self.model, x_0, t, condition=condition)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def p_sample(self, x_t, t, use_gfn=False, condition=None):
        """
        Sample from p(x_{t-1} | x_t), optionally with GFN guidance.
        
        Args:
            x_t: Noisy samples at timestep t
            t: Timesteps
            use_gfn: Whether to use GFN guidance
            condition: Optional condition for conditional models
            
        Returns:
            x_{t-1}: Samples at timestep t-1
        """
        if not use_gfn or self.energy_fn is None:
            # Standard diffusion sampling
            return self.diffusion.p_sample(self.model, x_t, t, condition=condition)
        
        # Get model mean and variance
        if condition is not None and hasattr(self.model, 'condition_embedding'):
            # Conditional model
            if not isinstance(condition, torch.Tensor):
                condition = torch.tensor(condition, device=self.device)
            if condition.dtype != torch.long:
                condition = condition.long()
            model_output = self.model(x_t, t, condition)
            model_mean, model_variance, model_log_variance = self.diffusion.p_mean_variance(self.model, x_t, t, model_output, condition)
        else:
            # Unconditional model
            model_mean, model_variance, model_log_variance = self.diffusion.p_mean_variance(self.model, x_t, t)
        
        # For t=0, don't add noise
        if torch.all(t == 0):
            return model_mean
        
        # Sample with Gaussian noise
        noise = torch.randn_like(x_t)
        
        # If using GFN guidance, generate multiple candidates and select by energy
        if hasattr(self, 'energy_fn') and self.energy_fn is not None:
            # Sample multiple candidates
            num_candidates = 10
            batch_noise = torch.randn(num_candidates, *x_t.shape, device=self.device)
            
            # Generate candidates
            candidates = model_mean.unsqueeze(0) + torch.exp(0.5 * model_log_variance).unsqueeze(0) * batch_noise
            
            # Reshape for energy computation
            batch_size = x_t.shape[0]
            candidates_flat = candidates.reshape(num_candidates * batch_size, -1)
            
            # Compute energy for each candidate
            with torch.no_grad():
                energies = self.energy_fn(candidates_flat)
                
                # Check the shape of energies
                total_elements = energies.numel()
                expected_elements = num_candidates * batch_size
                
                # If the energy function returns a single value per input
                if total_elements == expected_elements:
                    energies = energies.reshape(num_candidates, batch_size)
                else:
                    # If the energy function returns multiple values per input
                    print(f"Warning: Energy function returned unexpected shape. Using mean energy per candidate.")
                    
                    # Force energies to be a 1D tensor of the expected size
                    if total_elements > expected_elements:
                        # If we have more elements than expected, take the first expected_elements
                        energies = energies.view(-1)[:expected_elements]
                    elif total_elements < expected_elements:
                        # If we have fewer elements than expected, pad with high energy values
                        padded = torch.full((expected_elements,), 100.0, device=energies.device)
                        padded[:total_elements] = energies.view(-1)
                        energies = padded
                    
                    # Reshape to [num_candidates, batch_size]
                    energies = energies.reshape(num_candidates, batch_size)
            
            # Use softmax to get sampling probabilities based on energy
            probs = F.softmax(-energies * self.guidance_scale, dim=0)
            
            # Add detailed diagnostic logging for probability values
            prob_min = probs.min().item()
            prob_max = probs.max().item()
            prob_sum = probs.sum(dim=0).mean().item()
            prob_has_nan = torch.isnan(probs).any().item()
            prob_has_inf = torch.isinf(probs).any().item()
            prob_has_neg = (probs < 0).any().item()
            
            # Check for inf, nan, or negative values in probs
            if prob_has_nan or prob_has_inf or prob_has_neg:
                # Log detailed diagnostic information
                print(f"Warning: Invalid probability values detected. min={prob_min}, max={prob_max}, sum={prob_sum}")
                print(f"NaN: {prob_has_nan}, Inf: {prob_has_inf}, Negative: {prob_has_neg}")
                print(f"Energy min: {energies.min().item()}, max: {energies.max().item()}, guidance_scale: {self.guidance_scale}")
                
                # Before falling back, try to fix the issue by clamping energies
                if prob_has_inf or abs(energies.max().item()) > 30:
                    print("Attempting to fix by clamping extreme energy values...")
                    # Clamp energy values to avoid overflow in softmax
                    energies = torch.clamp(energies, min=-20.0, max=20.0)
                    # Try again with clamped values
                    probs = F.softmax(-energies * self.guidance_scale, dim=0)
                    
                    # Check if fixed
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                        print("Clamping didn't fix the issue. Falling back to uniform sampling.")
                        probs = torch.ones_like(probs) / num_candidates
                    else:
                        print("Successfully fixed probability values with clamping.")
                else:
                    # Fall back to uniform distribution if there's a problem
                    print("Falling back to uniform sampling.")
                    probs = torch.ones_like(probs) / num_candidates
            
            # Sample indices based on probabilities
            indices = torch.multinomial(probs.T, 1).squeeze(-1)
            
            # Select the sampled candidates
            selected_candidates = torch.zeros_like(x_t)
            for i in range(batch_size):
                selected_candidates[i] = candidates[indices[i], i]
            
            return selected_candidates
        else:
            # Standard diffusion sampling
            return model_mean + torch.exp(0.5 * model_log_variance) * noise
    
    def p_sample_with_guidance(self, model, x_t, t, condition=None):
        """
        Sample from p(x_{t-1} | x_t) with guidance for conditional models.
        
        Args:
            model: The diffusion model
            x_t: Noisy samples at timestep t
            t: Timesteps
            condition: Condition for conditional models
            
        Returns:
            x_{t-1}: Samples at timestep t-1
        """
        # Get model mean and variance, passing condition to the model
        if condition is not None:
            # Ensure condition is a tensor on the right device
            if not isinstance(condition, torch.Tensor):
                condition = torch.tensor(condition, device=self.device)
            
            # Ensure condition is a long tensor for embedding layers
            if condition.dtype != torch.long:
                condition = condition.long()
            
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).repeat(x_t.shape[0], 1)
                
            # Get model prediction with condition
            model_output = model(x_t, t, condition)
            model_mean, model_variance, model_log_variance = self.diffusion.p_mean_variance(model, x_t, t, model_output)
        else:
            model_mean, model_variance, model_log_variance = self.diffusion.p_mean_variance(model, x_t, t)
        
        # For t=0, don't add noise
        if torch.all(t == 0):
            return model_mean
        
        # Sample multiple candidates
        num_candidates = 10
        noise = torch.randn(num_candidates, *x_t.shape, device=self.device)
        
        # Generate candidates
        candidates = model_mean.unsqueeze(0) + torch.exp(0.5 * model_log_variance).unsqueeze(0) * noise
        
        # Reshape for energy computation
        batch_size = x_t.shape[0]
        candidates_flat = candidates.reshape(num_candidates * batch_size, -1)
        
        # Compute energy for each candidate
        with torch.no_grad():
            # Prepare condition for energy computation if needed
            if condition is not None:
                # Reshape condition to match candidates
                condition_expanded = condition.unsqueeze(0).expand(num_candidates, -1, -1)
                condition_flat = condition_expanded.reshape(num_candidates * batch_size, -1)
                energies = self.energy_fn(candidates_flat, condition_flat)
            else:
                energies = self.energy_fn(candidates_flat)
            
            # Check the shape of energies
            total_elements = energies.numel()
            expected_elements = num_candidates * batch_size
            
            # If the energy function returns a single value per input
            if total_elements == expected_elements:
                energies = energies.reshape(num_candidates, batch_size)
            else:
                # If the energy function returns multiple values per input
                print(f"Warning: Energy function returned unexpected shape. Using mean energy per candidate.")
                
                # Force energies to be a 1D tensor of the expected size
                if total_elements > expected_elements:
                    # If we have more elements than expected, take the first expected_elements
                    energies = energies.view(-1)[:expected_elements]
                elif total_elements < expected_elements:
                    # If we have fewer elements than expected, pad with high energy values
                    padded = torch.full((expected_elements,), 100.0, device=energies.device)
                    padded[:total_elements] = energies.view(-1)
                    energies = padded
                
                # Reshape to [num_candidates, batch_size]
                energies = energies.reshape(num_candidates, batch_size)
        
        # Use softmax to get sampling probabilities based on energy
        probs = F.softmax(-energies * self.guidance_scale, dim=0)
        
        # Add detailed diagnostic logging for probability values
        prob_min = probs.min().item()
        prob_max = probs.max().item()
        prob_sum = probs.sum(dim=0).mean().item()
        prob_has_nan = torch.isnan(probs).any().item()
        prob_has_inf = torch.isinf(probs).any().item()
        prob_has_neg = (probs < 0).any().item()
        
        # Check for inf, nan, or negative values in probs
        if prob_has_nan or prob_has_inf or prob_has_neg:
            # Log detailed diagnostic information
            print(f"Warning: Invalid probability values detected. min={prob_min}, max={prob_max}, sum={prob_sum}")
            print(f"NaN: {prob_has_nan}, Inf: {prob_has_inf}, Negative: {prob_has_neg}")
            print(f"Energy min: {energies.min().item()}, max: {energies.max().item()}, guidance_scale: {self.guidance_scale}")
            
            # Before falling back, try to fix the issue by clamping energies
            if prob_has_inf or abs(energies.max().item()) > 30:
                print("Attempting to fix by clamping extreme energy values...")
                # Clamp energy values to avoid overflow in softmax
                energies = torch.clamp(energies, min=-20.0, max=20.0)
                # Try again with clamped values
                probs = F.softmax(-energies * self.guidance_scale, dim=0)
                
                # Check if fixed
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print("Clamping didn't fix the issue. Falling back to uniform sampling.")
                    probs = torch.ones_like(probs) / num_candidates
                else:
                    print("Successfully fixed probability values with clamping.")
            else:
                # Fall back to uniform distribution if there's a problem
                print("Falling back to uniform sampling.")
                probs = torch.ones_like(probs) / num_candidates
        
        # Sample indices based on probabilities
        indices = torch.multinomial(probs.T, 1).squeeze(-1)
        
        # Select the sampled candidates
        selected_candidates = torch.zeros_like(x_t)
        for i in range(batch_size):
            selected_candidates[i] = candidates[indices[i], i]
        
        return selected_candidates
    
    def p_sample_loop(self, n=1, dim=2, use_gfn=False, verbose=True, energy_fn=None, guidance_scale=None, schedule=None, condition=None):
        """
        Generate samples from the model, optionally with GFN guidance.
        
        Args:
            n: Number of samples to generate
            dim: Dimension of each sample
            use_gfn: Whether to use GFN guidance
            verbose: Show progress bar
            energy_fn: Optional energy function to override the model's energy_fn
            guidance_scale: Optional guidance scale to override the model's guidance_scale
            schedule: Optional diffusion schedule to override the model's diffusion
            condition: Optional condition for conditional models
            
        Returns:
            x_0: Generated samples
        """
        # Use provided values or fall back to model's values
        energy_fn = energy_fn if energy_fn is not None else self.energy_fn
        guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale
        diffusion = schedule if schedule is not None else self.diffusion
        
        # Create temporary GFNDiffusion if overriding parameters
        if energy_fn is not self.energy_fn or guidance_scale is not self.guidance_scale or diffusion is not self.diffusion:
            temp_gfn = GFNDiffusion(self.model, diffusion, energy_fn, guidance_scale, self.device)
            return temp_gfn.p_sample_loop(n, dim, use_gfn, verbose, condition=condition)
        
        # Start from pure noise
        shape = (n, dim)
        x_t = torch.randn(shape, device=self.device)
        
        # Iteratively denoise
        iterator = tqdm(reversed(range(self.diffusion.num_diffusion_timesteps)), desc='Sampling') if verbose else reversed(range(self.diffusion.num_diffusion_timesteps))
        
        for t in iterator:
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, use_gfn=use_gfn, condition=condition)
        
        return x_t
    
    def state_dict(self):
        """Return the model state_dict."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state_dict."""
        self.model.load_state_dict(state_dict)
