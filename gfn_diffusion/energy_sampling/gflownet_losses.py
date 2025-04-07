import torch
import torch.nn.functional as F

def fwd_tb(initial_state, gfn, log_reward_fn, exploration_std=None, return_exp=False):
    """
    Forward trajectory balance loss for GFlowNet.
    
    Args:
        initial_state: Initial state to start trajectories from
        gfn: GFlowNet model
        log_reward_fn: Function that returns log rewards for terminal states
        exploration_std: Standard deviation for exploration
        return_exp: Whether to return additional information
        
    Returns:
        Loss or (loss, additional info)
    """
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
        loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r) ** 2)
    
    if return_exp:
        return loss.mean(), states, log_pfs, log_pbs, log_r
    else:
        return loss.mean()

def fwd_tb_avg(initial_state, gfn, log_reward_fn, exploration_std=None, return_exp=False):
    """
    Forward trajectory balance with average loss (more stable version).
    
    Args:
        initial_state: Initial state to start trajectories from
        gfn: GFlowNet model
        log_reward_fn: Function that returns log rewards for terminal states
        exploration_std: Standard deviation for exploration
        return_exp: Whether to return additional information
        
    Returns:
        Loss or (loss, additional info)
    """
    states, log_pfs, log_pbs, _ = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
        # Estimate log partition function (normalization constant)
        log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
        loss = log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1))
    
    if return_exp:
        return 0.5 * (loss ** 2).mean(), states, log_pfs, log_pbs, log_r
    else:
        return 0.5 * (loss ** 2).mean()

def bwd_tb(initial_state, gfn, log_reward_fn, exploration_std=None):
    """
    Backward trajectory balance loss for GFlowNet.
    
    Args:
        initial_state: Initial state to start trajectories from
        gfn: GFlowNet model
        log_reward_fn: Function that returns log rewards for terminal states
        exploration_std: Standard deviation for exploration
        
    Returns:
        Loss
    """
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(initial_state, exploration_std, log_reward_fn)
    
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
        loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r) ** 2)
    
    return loss.mean()

def bwd_tb_avg(initial_state, gfn, log_reward_fn, exploration_std=None):
    """
    Backward trajectory balance with average loss (more stable version).
    
    Args:
        initial_state: Initial state to start trajectories from
        gfn: GFlowNet model
        log_reward_fn: Function that returns log rewards for terminal states
        exploration_std: Standard deviation for exploration
        
    Returns:
        Loss
    """
    states, log_pfs, log_pbs, _ = gfn.get_trajectory_bwd(initial_state, exploration_std, log_reward_fn)
    
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
        # Estimate log partition function (normalization constant)
        log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
        loss = log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1))
    
    return 0.5 * (loss ** 2).mean()

def db(initial_state, gfn, log_reward_fn, exploration_std=None, return_exp=False):
    """
    Detailed balance loss for GFlowNet.
    
    Args:
        initial_state: Initial state to start trajectories from
        gfn: GFlowNet model
        log_reward_fn: Function that returns log rewards for terminal states
        exploration_std: Standard deviation for exploration
        return_exp: Whether to return additional information
        
    Returns:
        Loss or (loss, additional info)
    """
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1]).detach()
        loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)
    
    if return_exp:
        return loss.mean(), states, log_pfs, log_pbs, log_fs[:, -1]
    else:
        return loss.mean() 