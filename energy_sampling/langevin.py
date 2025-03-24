import torch
import time

def adjust_ld_step(current_ld_step, current_acceptance_rate, target_acceptance_rate=0.574, adjustment_factor=0.01):
    """
    Adjust the Langevin dynamics step size based on the current acceptance rate.
    
    :param current_ld_step: Current Langevin dynamics step size.
    :param current_acceptance_rate: Current observed acceptance rate.
    :param target_acceptance_rate: Target acceptance rate, default is 0.574.
    :param adjustment_factor: Factor to adjust the ld_step.
    :return: Adjusted Langevin dynamics step size.
    """
    if current_acceptance_rate > target_acceptance_rate:
        return current_ld_step + adjustment_factor * current_ld_step
    else:
        return current_ld_step - adjustment_factor * current_ld_step

def langevin_dynamics(samples, log_reward_fn, device, args):
    """
    Highly optimized Langevin dynamics implementation that leverages
    batched operations and adaptive step sizes.
    
    Parameters:
    -----------
    samples : torch.Tensor
        Initial samples to run dynamics on
    log_reward_fn : callable
        Function that computes log reward/density
    device : torch.device
        Device to run computations on
    args : argparse.Namespace
        Arguments including ld_step and max_iter_ls
        
    Returns:
    --------
    local_samples : torch.Tensor
        Samples after Langevin dynamics
    final_log_r : torch.Tensor
        Log rewards of final samples
    """
    batch_size, dim = samples.shape
    
    # Make a copy to avoid modifying the original samples
    local_samples = samples.clone()
    
    # Track step sizes per sample for adaptive adjustment
    step_sizes = torch.ones(batch_size, device=device) * args.ld_step
    
    # Tracking acceptance rates for step size adaptation
    accepts = torch.zeros(batch_size, device=device)
    accepted_steps = 0
    
    # Pre-allocate tensors for efficiency
    noise = torch.zeros_like(local_samples)
    proposal = torch.zeros_like(local_samples)
    
    # For performance monitoring
    start_time = time.time()
    
    with torch.enable_grad():
        for i in range(args.max_iter_ls):
            # Enable gradients only for the current samples
            local_samples.requires_grad_(True)
            log_r = log_reward_fn(local_samples)
            grad_log_r = torch.autograd.grad(log_r.sum(), local_samples)[0]
            local_samples.requires_grad_(False)
            
            # Handle NaN/Inf gradients
            grad_log_r = torch.nan_to_num(grad_log_r, nan=0.0, posinf=1e8, neginf=-1e8)
            
            # Optional gradient clipping
            if hasattr(args, 'grad_clip') and args.grad_clip > 0:
                grad_norm = torch.norm(grad_log_r, dim=1, keepdim=True) 
                scale = torch.clamp(args.grad_clip / (grad_norm + 1e-6), max=1.0)
                grad_log_r = grad_log_r * scale
            
            # Generate proposal with vectorized operations
            noise.normal_()
            # Step in the gradient direction and add noise
            proposal = local_samples + \
                      step_sizes.view(-1, 1) * grad_log_r + \
                      torch.sqrt(2 * step_sizes.view(-1, 1)) * noise
            
            # Evaluate log probabilities
            with torch.no_grad():
                log_r_proposal = log_reward_fn(proposal)
                
            # Metropolis-Hastings acceptance criterion (vectorized)
            log_accept_prob = log_r_proposal - log_r
            u = torch.rand_like(log_accept_prob, device=device)
            accept_mask = u < torch.exp(log_accept_prob.clamp(max=0))  # clamp for numerical stability
            
            # Update samples where accepted
            local_samples = torch.where(accept_mask.unsqueeze(1), proposal, local_samples)
            
            # Track acceptance rate for adaptation
            accepts += accept_mask.float()
            accepted_steps += accept_mask.sum().item()
            
            # Adapt step sizes (optional) - increase if accepting too often, decrease if rarely
            if args.ld_schedule and i >= args.burn_in and (i % 10 == 0):
                # Current acceptance rate
                accept_rate = accepts / (i + 1)
                
                # Adjust step size based on acceptance rate vs target
                step_sizes = torch.where(
                    accept_rate > args.target_acceptance_rate,
                    step_sizes * 1.01,  # Increase step size if accepting too often
                    step_sizes * 0.99   # Decrease step size if rarely accepting
                )
                
                # Avoid extreme step sizes
                step_sizes.clamp_(min=1e-6, max=0.1)
    
    # Calculate final rewards for prioritization
    with torch.no_grad():
        final_log_r = log_reward_fn(local_samples)
    
    # Compute stats for monitoring
    elapsed = time.time() - start_time
    avg_acceptance = accepted_steps / (batch_size * args.max_iter_ls)
    steps_per_sec = args.max_iter_ls * batch_size / elapsed
    
    # Log Langevin dynamics stats to wandb if it's available
    try:
        import wandb
        if wandb.run:
            wandb.log({
                "langevin/acceptance_rate": avg_acceptance,
                "langevin/steps_per_second": steps_per_sec,
                "langevin/avg_step_size": step_sizes.mean().item(),
                "langevin/time_seconds": elapsed
            })
    except (ImportError, NameError):
        pass
    
    return local_samples, final_log_r