import torch
import torch.nn.functional as F

def gmm_energy(x, means, weights, std=0.1):
    """
    Gaussian Mixture Model energy function.
    
    Args:
        x: Input tensor of shape [batch_size, dim]
        means: Means of Gaussian components [num_components, dim]
        weights: Weights of Gaussian components [num_components]
        std: Standard deviation of Gaussian components
        
    Returns:
        energy: Energy for each input [batch_size]
    """
    # Default energy value - lower to create better gradient signals
    DEFAULT_ENERGY = 100.0  # Changed from 1000.0 to 100.0
    
    # Initial tensor shape validation
    if not isinstance(x, torch.Tensor):
        print("Warning: x is not a tensor. Converting to tensor.")
        x = torch.tensor(x, dtype=torch.float32)
    
    if not isinstance(means, torch.Tensor):
        print("Warning: means is not a tensor. Converting to tensor.")
        means = torch.tensor(means, dtype=torch.float32)
    
    if not isinstance(weights, torch.Tensor):
        print("Warning: weights is not a tensor. Converting to tensor.")
        weights = torch.tensor(weights, dtype=torch.float32)
    
    # Ensure devices match
    device = x.device
    means = means.to(device)
    weights = weights.to(device)

    # Ensure x is 2D: [batch_size, dim]
    original_shape = x.shape
    if len(original_shape) == 1:  # If 1D, add batch dimension
        x = x.unsqueeze(0)
    elif len(original_shape) > 2:  # If more than 2D, flatten to 2D
        x = x.reshape(-1, original_shape[-1])
    
    batch_size = x.shape[0]
    
    # Check for NaN/Inf input values & clean up the data
    if torch.isnan(x).any() or torch.isinf(x).any():
        # Replace NaN/Inf with zeros instead of returning default energy immediately
        # This can help when only a few values are problematic
        nan_inf_mask = torch.isnan(x) | torch.isinf(x)
        if nan_inf_mask.all():
            # If all values are NaN/Inf, return default energy
            print("Warning: All values in input are NaN/Inf. Using default energy.")
            return torch.full((batch_size,), DEFAULT_ENERGY, device=device)
        else:
            # Count how many entries are problematic
            problem_count = nan_inf_mask.sum().item()
            total_count = nan_inf_mask.numel()
            if problem_count / total_count > 0.5:  # If more than 50% are problematic
                print(f"Warning: {problem_count}/{total_count} values in input are NaN/Inf. Using default energy.")
                return torch.full((batch_size,), DEFAULT_ENERGY, device=device)
            
            # Otherwise, try to clean the data
            print(f"Warning: Replacing {problem_count} NaN/Inf values with zeros.")
            x = x.clone()  # Create a copy to avoid modifying the original
            x[nan_inf_mask] = 0.0
    
    if torch.isnan(means).any() or torch.isinf(means).any():
        print("Warning: NaN/Inf detected in means. Using default energy.")
        return torch.full((batch_size,), DEFAULT_ENERGY, device=device)
    
    if torch.isnan(weights).any() or torch.isinf(weights).any():
        print("Warning: NaN/Inf detected in weights. Using uniform weights.")
        weights = torch.ones(weights.shape, device=device) / weights.shape[0]
    
    # Ensure positive std value to prevent NaN
    std = max(std, 1e-5)
    
    # Normalize weights
    if weights.sum() <= 0 or torch.isnan(weights.sum()) or torch.isinf(weights.sum()):
        weights = torch.ones_like(weights) / weights.size(0)
    else:
        weights = weights / weights.sum()  # Ensure weights sum to 1
    
    # Clip input values to reasonable range to avoid numerical issues
    x_clipped = torch.clamp(x, min=-100.0, max=100.0)
    
    try:
        # Expand dimensions for broadcasting
        # x: [batch_size, 1, dim]
        # means: [1, num_components, dim]
        x_expanded = x_clipped.unsqueeze(1)
        means_expanded = means.unsqueeze(0)
        
        # Calculate squared distances between points and means
        # Use a more stable calculation method
        squared_dists = torch.sum((x_expanded - means_expanded) ** 2, dim=-1)
        
        # Apply numerical stability trick: scale distances to avoid extreme values
        # Note: We're working in log space, which helps with numerical stability
        log_coefficient = -0.5 * torch.log(2 * torch.tensor(3.14159, device=device) * (std ** 2))
        log_exponent = -0.5 * squared_dists / (std ** 2)
        
        # Add log of weights (with small epsilon to avoid log(0))
        log_weights = torch.log(weights + 1e-10)
        
        # Compute log probabilities for each component
        log_probs = log_coefficient + log_exponent + log_weights
        
        # LogSumExp trick for numerical stability
        max_log_prob = torch.max(log_probs, dim=1, keepdim=True)[0]
        log_sum_exp = max_log_prob + torch.log(
            torch.sum(torch.exp(log_probs - max_log_prob), dim=1)
        )
        
        # Energy is negative log probability
        energy = -log_sum_exp.squeeze(-1)
        
        # Final safety check: Replace any remaining NaN/Inf with default energy
        if torch.isnan(energy).any() or torch.isinf(energy).any():
            nan_inf_mask = torch.isnan(energy) | torch.isinf(energy)
            problem_count = nan_inf_mask.sum().item()
            print(f"Warning: {problem_count} NaN/Inf values in final energy. Replacing with default energy.")
            energy[nan_inf_mask] = DEFAULT_ENERGY
        
        # Clip energy to reasonable range as final safeguard (using our new DEFAULT_ENERGY value)
        energy = torch.clamp(energy, min=-DEFAULT_ENERGY, max=DEFAULT_ENERGY)
        
        # Count and log how many samples are using the default energy
        default_energy_count = (torch.abs(energy - DEFAULT_ENERGY) < 1e-5).sum().item()
        if default_energy_count > 0:
            print(f"Info: {default_energy_count}/{energy.shape[0]} samples ({default_energy_count/energy.shape[0]*100:.2f}%) using default energy value.")
        
        # Ensure the output has the correct batch dimension
        if len(original_shape) == 1 and energy.shape[0] == 1:
            energy = energy.squeeze(0)  # Return a scalar for single inputs
        
        return energy
        
    except Exception as e:
        print(f"Error in gmm_energy: {e}")
        return torch.full((batch_size,), DEFAULT_ENERGY, device=device)  # Default energy value
