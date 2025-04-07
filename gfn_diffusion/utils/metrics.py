import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os
from scipy.spatial.distance import cdist

def kl_divergence(p, q, eps=1e-9):
    """
    Compute KL divergence between two distributions p and q.
    
    Args:
        p: First distribution (torch.Tensor)
        q: Second distribution (torch.Tensor)
        eps: Small constant to avoid numerical issues
        
    Returns:
        KL(p||q): The KL divergence
    """
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p).float()
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q).float()
    
    # Normalize distributions if they don't sum to 1
    if abs(p.sum() - 1.0) > 1e-5:
        p = p / p.sum()
    if abs(q.sum() - 1.0) > 1e-5:
        q = q / q.sum()
    
    # Add small epsilon to avoid log(0)
    p_safe = p + eps
    q_safe = q + eps
    
    return (p_safe * torch.log(p_safe / q_safe)).sum().item()

def reverse_kl_divergence(p, q, eps=1e-9):
    """
    Compute reverse KL divergence between two distributions.
    
    Args:
        p: First distribution (torch.Tensor)
        q: Second distribution (torch.Tensor)
        eps: Small constant to avoid numerical issues
        
    Returns:
        KL(q||p): The reverse KL divergence
    """
    return kl_divergence(q, p, eps)

def jensen_shannon_divergence(p, q, eps=1e-9):
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Args:
        p: First distribution (torch.Tensor)
        q: Second distribution (torch.Tensor)
        eps: Small constant to avoid numerical issues
        
    Returns:
        JSD(p||q): The Jensen-Shannon divergence
    """
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, eps) + kl_divergence(q, m, eps))

def l1_distance(p, q):
    """
    Compute L1 distance between two distributions.
    
    Args:
        p: First distribution (torch.Tensor)
        q: Second distribution (torch.Tensor)
        
    Returns:
        |p-q|_1: The L1 distance
    """
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p).float()
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q).float()
    
    return (p - q).abs().mean().item()

def l2_distance(p, q):
    """
    Compute L2 distance between two distributions.
    
    Args:
        p: First distribution (torch.Tensor)
        q: Second distribution (torch.Tensor)
        
    Returns:
        |p-q|_2: The L2 distance
    """
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p).float()
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q).float()
    
    return ((p - q) ** 2).sum().sqrt().item()

def earth_movers_distance(p_samples, q_samples):
    """
    Compute the Earth Mover's Distance (EMD) between two sets of samples.
    
    Args:
        p_samples: First set of samples
        q_samples: Second set of samples
        
    Returns:
        emd: The Earth Mover's Distance
    """
    # Convert to numpy if torch tensors
    if isinstance(p_samples, torch.Tensor):
        p_samples = p_samples.cpu().numpy()
    if isinstance(q_samples, torch.Tensor):
        q_samples = q_samples.cpu().numpy()
    
    # Check for empty arrays
    if p_samples.size == 0 or q_samples.size == 0:
        print("Warning: Empty array in EMD calculation. Returning 0.")
        return 0.0
    
    # Reshape if needed - ensure we have 2D arrays
    if len(p_samples.shape) == 1:
        p_samples = p_samples.reshape(-1, 1)
    if len(q_samples.shape) == 1:
        q_samples = q_samples.reshape(-1, 1)
    
    # Verify dimensions
    if len(p_samples.shape) < 2 or len(q_samples.shape) < 2:
        print(f"Warning: Invalid shapes for EMD: p_samples {p_samples.shape}, q_samples {q_samples.shape}. Returning 0.")
        return 0.0
        
    # For 1D data, compute histograms and use the 1D Wasserstein distance
    if p_samples.shape[1] == 1:
        return wasserstein_distance(p_samples.flatten(), q_samples.flatten())
    
    # For multi-dimensional data, we average the 1D EMDs for each dimension
    emd = 0
    for i in range(p_samples.shape[1]):
        emd += wasserstein_distance(p_samples[:, i], q_samples[:, i])
    
    return emd / p_samples.shape[1]

def compute_nearest_mode_distribution(samples, modes, std=0.1):
    """
    Compute the distribution of samples' nearest modes.
    
    Args:
        samples: Sample points (torch.Tensor or np.ndarray)
        modes: Mode centers (torch.Tensor or np.ndarray)
        std: Standard deviation threshold for mode assignment
        
    Returns:
        mode_dist: Mode distribution
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    if isinstance(modes, torch.Tensor):
        modes = modes.cpu().numpy()
    
    # Compute distances from each sample to each mode
    distances = cdist(samples, modes)
    
    # Find the nearest mode for each sample
    nearest_modes = np.argmin(distances, axis=1)
    
    # Count occurrences of each mode
    mode_counts = np.bincount(nearest_modes, minlength=len(modes))
    
    # Convert to distribution
    mode_dist = mode_counts / len(samples)
    
    return mode_dist

def compute_mode_coverage(samples, modes, threshold=0.1):
    """
    Compute the proportion of modes covered by the samples.
    
    Args:
        samples: Sample points (torch.Tensor or np.ndarray)
        modes: Mode centers (torch.Tensor or np.ndarray)
        threshold: Distance threshold for mode coverage
        
    Returns:
        coverage: Proportion of modes covered
        covered_modes: List of covered mode indices
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    if isinstance(modes, torch.Tensor):
        modes = modes.cpu().numpy()
    
    # For each mode, check if there's at least one sample within threshold distance
    covered_modes = []
    for i, mode in enumerate(modes):
        distances = np.sqrt(np.sum((samples - mode)**2, axis=1))
        if np.min(distances) < threshold:
            covered_modes.append(i)
    
    coverage = len(covered_modes) / len(modes)
    
    return coverage, covered_modes

def compute_reverse_mode_coverage(samples, modes, threshold=0.1):
    """
    Compute the proportion of samples that are within threshold of at least one mode.
    
    Args:
        samples: Sample points (torch.Tensor or np.ndarray)
        modes: Mode centers (torch.Tensor or np.ndarray)
        threshold: Distance threshold for mode coverage
        
    Returns:
        reverse_coverage: Proportion of samples covering modes
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    if isinstance(modes, torch.Tensor):
        modes = modes.cpu().numpy()
    
    # For each sample, check if it's within threshold distance of at least one mode
    samples_near_modes = 0
    for sample in samples:
        distances = np.sqrt(np.sum((modes - sample)**2, axis=1))
        if np.min(distances) < threshold:
            samples_near_modes += 1
    
    reverse_coverage = samples_near_modes / len(samples)
    
    return reverse_coverage

def plot_histograms(samples, title, save_dir, filename):
    """
    Plot histograms for each dimension of the samples.
    
    Args:
        samples: Samples to visualize
        title: Title for the plot
        save_dir: Directory to save the visualization
        filename: Name of the file to save
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    
    n_dims = samples.shape[1]
    
    plt.figure(figsize=(12, 3 * n_dims))
    for i in range(n_dims):
        plt.subplot(n_dims, 1, i + 1)
        plt.hist(samples[:, i], bins=50, alpha=0.7, density=True)
        plt.xlabel(f'Dimension {i+1}')
        plt.ylabel('Density')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_energy_landscape_with_samples(energy_fn, samples, range_val=5.0, resolution=100, 
                                       title="Energy Landscape with Samples", 
                                       save_dir="./results", filename="energy_with_samples.png"):
    """
    Plot the energy landscape with samples overlaid.
    
    Args:
        energy_fn: Energy function
        samples: Sample points (torch.Tensor)
        range_val: Range for the plot
        resolution: Resolution of the energy landscape grid
        title: Title for the plot
        save_dir: Directory to save the visualization
        filename: Name of the file to save
    """
    # Create grid for energy landscape
    x = torch.linspace(-range_val, range_val, resolution)
    y = torch.linspace(-range_val, range_val, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten and stack
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Compute energy
    with torch.no_grad():
        energy = energy_fn(points).reshape(resolution, resolution)
    
    # Normalize for better visualization
    energy_min = energy.min().item()
    energy_max = energy.max().item()
    energy_norm = (energy - energy_min) / (energy_max - energy_min)
    
    # Convert tensors to NumPy arrays for matplotlib
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    energy_norm_np = energy_norm.cpu().numpy()
    
    if isinstance(samples, torch.Tensor):
        samples_np = samples.cpu().numpy()
    else:
        samples_np = samples
    
    # Create colormap with alpha gradient
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    newcolors[:25, :] = white
    newcmp = ListedColormap(newcolors)
    
    plt.figure(figsize=(12, 10))
    plt.contourf(X_np, Y_np, energy_norm_np, levels=50, cmap=newcmp)
    plt.colorbar(label='Normalized Energy')
    
    # Plot samples with contour heat map
    plt.scatter(samples_np[:, 0], samples_np[:, 1], c='red', alpha=0.7, s=15, edgecolor='black', linewidth=0.5)
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-range_val, range_val)
    plt.ylim(-range_val, range_val)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_trajectory_animation(trajectories, energy_fn=None, range_val=5.0, resolution=100,
                             title="Sample Trajectories", save_dir="./results", filename="trajectory_animation.gif"):
    """
    Create an animation of sample trajectories with the energy landscape as background.
    
    Args:
        trajectories: List of trajectory tensors [T, batch_size, dim]
        energy_fn: Energy function (optional)
        range_val: Range for the plot
        resolution: Resolution of the energy landscape grid
        title: Title for the animation
        save_dir: Directory to save the visualization
        filename: Name of the file to save
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        print("matplotlib.animation is required for creating animations. Please install it.")
        return
    
    if isinstance(trajectories[0], torch.Tensor):
        trajectories_np = [t.cpu().numpy() for t in trajectories]
    else:
        trajectories_np = trajectories
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # If energy function is provided, plot the energy landscape
    if energy_fn is not None:
        # Create grid for energy landscape
        x = torch.linspace(-range_val, range_val, resolution)
        y = torch.linspace(-range_val, range_val, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Flatten and stack
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Compute energy
        with torch.no_grad():
            energy = energy_fn(points).reshape(resolution, resolution)
        
        # Normalize for better visualization
        energy_min = energy.min().item()
        energy_max = energy.max().item()
        energy_norm = (energy - energy_min) / (energy_max - energy_min)
        
        # Convert tensors to NumPy arrays for matplotlib
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        energy_norm_np = energy_norm.cpu().numpy()
        
        # Plot energy landscape
        contour = ax.contourf(X_np, Y_np, energy_norm_np, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Normalized Energy')
    
    # Function to update animation at each frame
    def update(frame):
        ax.clear()
        
        # If energy function is provided, redraw the energy landscape
        if energy_fn is not None:
            contour = ax.contourf(X_np, Y_np, energy_norm_np, levels=50, cmap='viridis')
        
        # Plot trajectories up to the current frame
        for traj in trajectories_np:
            ax.scatter(traj[frame, :, 0], traj[frame, :, 1], c='red', alpha=0.7, s=15)
            # Add trajectory lines
            for i in range(traj.shape[1]):
                ax.plot(traj[:frame+1, i, 0], traj[:frame+1, i, 1], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f"{title} (Step {frame+1}/{len(trajectories_np[0])})")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-range_val, range_val)
        ax.set_ylim(-range_val, range_val)
        
        return ax,
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(trajectories_np[0]), interval=200, blit=False)
    
    # Save animation
    os.makedirs(save_dir, exist_ok=True)
    ani.save(os.path.join(save_dir, filename), writer='pillow', fps=5)
    plt.close()

def compute_energy_statistics(samples, energy_fn):
    """
    Compute statistics of energy values for a set of samples.
    
    Args:
        samples: Sample points (torch.Tensor)
        energy_fn: Energy function
        
    Returns:
        stats: Dictionary of energy statistics
    """
    with torch.no_grad():
        energies = energy_fn(samples)
    
    mean_energy = energies.mean().item()
    min_energy = energies.min().item()
    max_energy = energies.max().item()
    std_energy = energies.std().item()
    
    # Compute percentiles
    percentiles = {}
    for p in [5, 25, 50, 75, 95]:
        percentiles[f'p{p}'] = torch.quantile(energies, p/100).item()
    
    stats = {
        'mean_energy': mean_energy,
        'min_energy': min_energy,
        'max_energy': max_energy,
        'std_energy': std_energy,
        **percentiles
    }
    
    return stats

def compute_entropy(samples, bins=20, range_val=5.0):
    """
    Compute the entropy of a sample distribution.
    
    Args:
        samples: Sample points (torch.Tensor)
        bins: Number of bins per dimension for histogram
        range_val: Range for histogram
        
    Returns:
        entropy: Estimated entropy of the distribution
    """
    if isinstance(samples, torch.Tensor):
        samples_np = samples.cpu().numpy()
    else:
        samples_np = samples
    
    # For 2D data, compute 2D histogram
    if samples_np.shape[1] == 2:
        hist, _ = np.histogramdd(samples_np, bins=bins, range=[[-range_val, range_val], [-range_val, range_val]])
        
        # Normalize histogram to get probability distribution
        hist_sum = hist.sum()
        if hist_sum > 0:  # Avoid division by zero
            hist = hist / hist_sum
        else:
            # If histogram is empty, return default entropy
            print("Warning: Empty histogram in entropy calculation")
            return 0.0
        
        # Compute entropy, avoiding log(0)
        non_zero_mask = hist > 0
        if np.any(non_zero_mask):
            entropy = -np.sum(hist[non_zero_mask] * np.log(hist[non_zero_mask]))
        else:
            entropy = 0.0
        
        return entropy
    else:
        # For higher dimensions, use KDE approximation
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(samples_np)
        
        # Evaluate log density at sample points
        log_density = kde.score_samples(samples_np)
        
        # Estimate entropy as negative average log density
        entropy = -np.mean(log_density)
        
        return entropy

def compute_diversity(samples, threshold=0.1):
    """
    Compute diversity of samples as mean pairwise distances.
    
    Args:
        samples: Sample points (torch.Tensor or np.ndarray)
        threshold: Threshold for considering samples as diverse
        
    Returns:
        diversity: Average pairwise diversity
    """
    # Add safety checks
    if samples is None or (isinstance(samples, np.ndarray) and samples.size == 0) or (isinstance(samples, torch.Tensor) and samples.numel() == 0):
        print("Warning: Empty samples array in compute_diversity. Returning 0.0")
        return 0.0
        
    try:
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        
        # Make sure samples is 2D
        if len(samples.shape) == 1:
            samples = samples.reshape(-1, 1)
            
        # Safety check for NaN or inf values
        if np.isnan(samples).any() or np.isinf(samples).any():
            print("Warning: NaN or Inf values in samples for compute_diversity. Cleaning data...")
            samples = np.nan_to_num(samples, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # If we have only one sample, return 0 diversity
        if len(samples) <= 1:
            return 0.0
        
        # Compute pairwise distances
        distances = cdist(samples, samples)
        
        # Exclude self-distances (diagonal)
        n = distances.shape[0]
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, 0)
        
        # Mean distance
        diversity = distances[mask].mean()
        
        return diversity
    except Exception as e:
        print(f"Error in compute_diversity: {e}")
        # Return a default value instead of failing
        return 0.0

def compute_novelty(generated_samples, reference_samples=None, threshold=0.1):
    """
    Compute novelty of generated samples with respect to reference samples.
    
    Args:
        generated_samples: Generated sample points (torch.Tensor or np.ndarray)
        reference_samples: Reference sample points (torch.Tensor or np.ndarray)
        threshold: Threshold for considering samples as novel
        
    Returns:
        novelty: Average novelty score
    """
    # Add safety checks
    if generated_samples is None or (isinstance(generated_samples, np.ndarray) and generated_samples.size == 0) or \
       (isinstance(generated_samples, torch.Tensor) and generated_samples.numel() == 0):
        print("Warning: Empty generated_samples array in compute_novelty. Returning 0.0")
        return 0.0
        
    try:
        if isinstance(generated_samples, torch.Tensor):
            generated_samples = generated_samples.cpu().numpy()
            
        # Make sure generated_samples is 2D
        if len(generated_samples.shape) == 1:
            generated_samples = generated_samples.reshape(-1, 1)
        
        # Safety check for NaN or inf values
        if np.isnan(generated_samples).any() or np.isinf(generated_samples).any():
            print("Warning: NaN or Inf values in generated_samples for compute_novelty. Cleaning data...")
            generated_samples = np.nan_to_num(generated_samples, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # If reference samples are not provided, compute diversity instead
        if reference_samples is None:
            return compute_diversity(generated_samples, threshold)
            
        if isinstance(reference_samples, torch.Tensor):
            reference_samples = reference_samples.cpu().numpy()
            
        # Make sure reference_samples is 2D
        if len(reference_samples.shape) == 1:
            reference_samples = reference_samples.reshape(-1, 1)
            
        # Safety check for NaN or inf values in reference
        if np.isnan(reference_samples).any() or np.isinf(reference_samples).any():
            print("Warning: NaN or Inf values in reference_samples for compute_novelty. Cleaning data...")
            reference_samples = np.nan_to_num(reference_samples, nan=0.0, posinf=1e6, neginf=-1e6)
            
        # If shapes don't match in dimension, try to adapt
        if generated_samples.shape[1] != reference_samples.shape[1]:
            min_dim = min(generated_samples.shape[1], reference_samples.shape[1])
            generated_samples = generated_samples[:, :min_dim]
            reference_samples = reference_samples[:, :min_dim]
            print(f"Warning: Dimension mismatch in compute_novelty. Using first {min_dim} dimensions.")
        
        # Compute pairwise distances to nearest reference
        distances = cdist(generated_samples, reference_samples)
        
        # Find minimum distance to reference for each generated sample
        min_distances = distances.min(axis=1)
        
        # Mean minimum distance
        novelty = min_distances.mean()
        
        return novelty
    except Exception as e:
        print(f"Error in compute_novelty: {e}")
        # Return a default value instead of failing
        return 0.0

def compute_energy_improvement(standard_samples, gfn_samples, energy_fn):
    """
    Compute the relative improvement in energy values between standard samples
    and GFN-guided samples.
    
    Args:
        standard_samples: Samples from standard method (torch.Tensor)
        gfn_samples: Samples from GFN-guided method (torch.Tensor)
        energy_fn: Energy function
        
    Returns:
        improvement: Percentage improvement in mean energy
        top_improvement: Percentage improvement in top-10% samples
    """
    with torch.no_grad():
        standard_energies = energy_fn(standard_samples)
        gfn_energies = energy_fn(gfn_samples)
    
    # Compute mean improvement
    mean_standard = standard_energies.mean().item()
    mean_gfn = gfn_energies.mean().item()
    improvement = (mean_standard - mean_gfn) / mean_standard * 100
    
    # Compute top-10% improvement
    k = max(1, int(0.1 * len(standard_energies)))
    top_standard = torch.topk(standard_energies, k, largest=False).values.mean().item()
    top_gfn = torch.topk(gfn_energies, k, largest=False).values.mean().item()
    top_improvement = (top_standard - top_gfn) / top_standard * 100
    
    return improvement, top_improvement

def compute_effective_sample_size(samples, energy_fn, temperature=1.0):
    """
    Compute the effective sample size (ESS) given energy values and temperature.
    ESS measures how many independent samples would be equivalent to the weighted samples.
    
    Args:
        samples: Sample points (torch.Tensor)
        energy_fn: Energy function
        temperature: Temperature parameter for energy-to-weight conversion
        
    Returns:
        ess: Effective sample size
        ess_ratio: ESS divided by actual number of samples
    """
    with torch.no_grad():
        energies = energy_fn(samples)
    
    # Convert energies to weights (unnormalized)
    # Lower energy = higher weight
    weights = torch.exp(-energies / temperature)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Compute ESS = 1 / sum(w_i^2)
    ess = 1.0 / (weights**2).sum().item()
    
    # Compute ratio (0-1, higher is better)
    ess_ratio = ess / len(samples)
    
    return ess, ess_ratio

def compute_kl_from_target(samples, target_fn, bins=50, range_val=5.0):
    """
    Compute the KL divergence between the empirical distribution of samples
    and a target distribution.
    
    Args:
        samples: Sample points (torch.Tensor)
        target_fn: Target probability function (not energy function)
        bins: Number of bins per dimension for histogram
        range_val: Range for histogram
        
    Returns:
        kl: KL divergence from empirical to target
    """
    if isinstance(samples, torch.Tensor):
        samples_np = samples.cpu().numpy()
    else:
        samples_np = samples
    
    # For 2D data, compute 2D histogram of samples
    if samples_np.shape[1] == 2:
        hist_samples, edges_x, edges_y = np.histogram2d(
            samples_np[:, 0], samples_np[:, 1], 
            bins=bins, 
            range=[[-range_val, range_val], [-range_val, range_val]]
        )
        
        # Normalize histogram to get probability distribution
        hist_samples = hist_samples / hist_samples.sum()
        
        # Compute target distribution on the same grid
        x_centers = (edges_x[:-1] + edges_x[1:]) / 2
        y_centers = (edges_y[:-1] + edges_y[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
        
        # Get target probabilities
        with torch.no_grad():
            target_probs = target_fn(grid_points).reshape(bins, bins).cpu().numpy()
        
        # Normalize target probabilities
        target_probs = target_probs / target_probs.sum()
        
        # Compute KL divergence
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        kl = np.sum(target_probs * np.log((target_probs + eps) / (hist_samples.T + eps)))
        
        return kl
    else:
        # For higher dimensions, not implemented
        raise NotImplementedError("KL divergence estimation only implemented for 2D data")

def compute_coverage_metrics(samples, reference_centers=None, energy_fn=None, range_val=5.0, n_grid=10):
    """
    Compute various coverage metrics for a set of samples.
    
    Args:
        samples: Samples to evaluate (torch.Tensor or np.ndarray)
        reference_centers: Optional reference centers for mode coverage (torch.Tensor or np.ndarray)
        energy_fn: Optional energy function for energy-based coverage
        range_val: Range of the domain for grid coverage
        n_grid: Number of grid cells per dimension
        
    Returns:
        metrics: Dictionary of coverage metrics
    """
    try:
        # Convert samples to numpy if they are torch tensors
        if isinstance(samples, torch.Tensor):
            samples_np = samples.cpu().numpy()
        else:
            samples_np = samples
            
        # Make sure samples is 2D
        if len(samples_np.shape) == 1:
            samples_np = samples_np.reshape(-1, 1)
            
        # Basic checks for NaN/Inf values
        if np.isnan(samples_np).any() or np.isinf(samples_np).any():
            print("Warning: NaN or Inf values in samples for compute_coverage_metrics. Cleaning data...")
            samples_np = np.nan_to_num(samples_np, nan=0.0, posinf=1e6, neginf=-1e6)
            
        # Initialize metrics dictionary
        metrics = {}
        
        # Compute grid coverage
        dim = samples_np.shape[1]
        
        if dim <= 2:  # Only compute grid coverage for 1D and 2D data
            # Create grid
            if dim == 1:
                grid_points = np.linspace(-range_val, range_val, n_grid).reshape(-1, 1)
            else:  # dim == 2
                x = np.linspace(-range_val, range_val, n_grid)
                y = np.linspace(-range_val, range_val, n_grid)
                X, Y = np.meshgrid(x, y)
                grid_points = np.column_stack([X.flatten(), Y.flatten()])
            
            # Compute distances from each grid point to nearest sample
            distances = cdist(grid_points, samples_np)
            min_distances = distances.min(axis=1)
            
            # Count grid points that are "covered" (have a sample within some threshold)
            threshold = range_val / (n_grid * 2)  # Half the grid cell size
            covered_points = (min_distances < threshold).sum()
            
            # Compute grid coverage metric
            grid_coverage = covered_points / len(grid_points)
            metrics['grid_coverage'] = grid_coverage
        
        # If reference centers are provided, compute mode coverage
        if reference_centers is not None:
            if isinstance(reference_centers, torch.Tensor):
                reference_centers_np = reference_centers.cpu().numpy()
            else:
                reference_centers_np = reference_centers
                
            # Make sure reference_centers is 2D
            if len(reference_centers_np.shape) == 1:
                reference_centers_np = reference_centers_np.reshape(-1, 1)
                
            # Basic checks for NaN/Inf values
            if np.isnan(reference_centers_np).any() or np.isinf(reference_centers_np).any():
                print("Warning: NaN or Inf values in reference_centers for compute_coverage_metrics. Cleaning data...")
                reference_centers_np = np.nan_to_num(reference_centers_np, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Compute mode coverage
            coverage, covered_modes = compute_mode_coverage(samples_np, reference_centers_np)
            metrics['mode_coverage'] = coverage
            metrics['covered_modes'] = covered_modes
        
        # If energy function is provided, compute energy-based coverage
        if energy_fn is not None:
            try:
                # Use energy function to identify low-energy regions
                if dim <= 2:  # Only for 1D and 2D data
                    # Create a fine grid for energy evaluation
                    n_fine = 100
                    if dim == 1:
                        grid_x = np.linspace(-range_val, range_val, n_fine)
                        grid_points = grid_x.reshape(-1, 1)
                    else:  # dim == 2
                        grid_x = np.linspace(-range_val, range_val, n_fine)
                        grid_y = np.linspace(-range_val, range_val, n_fine)
                        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
                        grid_points = np.column_stack([grid_X.flatten(), grid_Y.flatten()])
                    
                    # Convert to torch tensor for energy computation
                    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
                    
                    # Compute energy at each grid point
                    with torch.no_grad():
                        energies = energy_fn(grid_tensor).cpu().numpy()
                    
                    # Identify low energy points (e.g., below median energy)
                    energy_threshold = np.median(energies)
                    low_energy_mask = energies < energy_threshold
                    
                    # Handle the case when the mask shape doesn't match
                    if low_energy_mask.shape != grid_points.shape:
                        # Reshape the mask to match grid_points shape for indexing
                        if len(low_energy_mask.shape) == 1 and low_energy_mask.shape[0] == grid_points.shape[0]:
                            # 1D mask for 2D points - this is the expected case, no reshape needed
                            pass
                        else:
                            print(f"Warning: Mask shape {low_energy_mask.shape} doesn't match grid_points shape {grid_points.shape}. Adjusting mask.")
                            
                            # For safety, flatten the mask if it's multi-dimensional
                            if len(low_energy_mask.shape) > 1:
                                low_energy_mask = low_energy_mask.flatten()
                            
                            # Ensure the mask length matches the number of grid points
                            if len(low_energy_mask) > len(grid_points):
                                # Truncate if too long
                                low_energy_mask = low_energy_mask[:len(grid_points)]
                            elif len(low_energy_mask) < len(grid_points):
                                # Extend with False if too short
                                print("Warning: Mask is too short. Extending with False values.")
                                extended_mask = np.zeros(len(grid_points), dtype=bool)
                                extended_mask[:len(low_energy_mask)] = low_energy_mask
                                low_energy_mask = extended_mask
                    
                    # Apply mask to get low energy points
                    low_energy_points = grid_points[low_energy_mask]
                    
                    # Skip if no low energy points found
                    if len(low_energy_points) == 0:
                        print("Warning: Unable to select low energy points. Using all grid points.")
                        low_energy_points = grid_points
                    
                    # For each low energy point, compute distance to nearest sample
                    energy_distances = []
                    for point in low_energy_points:
                        # Convert point to proper numpy array with the right shape
                        if isinstance(point, torch.Tensor):
                            point = point.cpu().numpy()
                            
                        # Make sure point is properly shaped for subtraction (1D array)
                        if len(point.shape) == 0:  # scalar
                            point = np.array([point])
                        elif len(point.shape) > 1:  # multi-dimensional
                            point = point.flatten()
                        
                        # Compute distances to all samples
                        try:
                            if len(samples_np.shape) == 1:
                                # Handle 1D case
                                distances_to_point = np.abs(samples_np - point)
                            else:
                                # Handle multi-dimensional case, ensuring dimensions match
                                if len(point) != samples_np.shape[1]:
                                    # Truncate or pad point to match samples_np dimensions
                                    if len(point) > samples_np.shape[1]:
                                        point = point[:samples_np.shape[1]]
                                    else:
                                        padded_point = np.zeros(samples_np.shape[1])
                                        padded_point[:len(point)] = point
                                        point = padded_point
                                        
                                distances_to_point = np.sqrt(np.sum((samples_np - point) ** 2, axis=1))
                        except Exception as e:
                            print(f"Error computing distances: {e}")
                            print(f"samples_np shape: {samples_np.shape}, point shape: {np.array(point).shape}")
                            print(f"samples_np type: {type(samples_np)}, point type: {type(point)}")
                            # Use a fallback method
                            point_reshaped = np.array(point).reshape(1, -1)
                            distances_to_point = cdist(point_reshaped, samples_np)[0]
                            
                        # Add minimum distance to list
                        min_distance = np.min(distances_to_point)
                        energy_distances.append(min_distance)
                    
                    # Compute average minimum distance to low energy points
                    avg_energy_distance = np.mean(energy_distances)
                    metrics['energy_coverage_distance'] = avg_energy_distance
                    
                    # Compute fraction of low energy points that are "covered"
                    energy_threshold = range_val / (n_fine * 2)
                    covered_energy_points = np.sum(np.array(energy_distances) < energy_threshold)
                    energy_coverage = covered_energy_points / len(low_energy_points)
                    metrics['energy_coverage'] = energy_coverage
            except Exception as e:
                print(f"Error in energy-based coverage computation: {e}")
                metrics['energy_coverage_distance'] = float('nan')
                metrics['energy_coverage'] = float('nan')
        
        return metrics
    except Exception as e:
        print(f"Error in compute_coverage_metrics: {e}")
        # Return a default dictionary with NaN values
        return {'grid_coverage': float('nan'), 'mode_coverage': float('nan')}

def compute_all_metrics(standard_samples, gfn_samples, energy_fn, reference_samples=None, reference_centers=None):
    """
    Compute comprehensive metrics comparing standard and GFN-guided samples.
    
    Args:
        standard_samples: Samples from standard diffusion (torch.Tensor)
        gfn_samples: Samples from GFN-guided diffusion (torch.Tensor)
        energy_fn: Energy function
        reference_samples: Optional reference samples for novelty computation
        reference_centers: Optional reference centers for coverage computation
        
    Returns:
        metrics: Dictionary of all metrics
    """
    metrics = {}
    
    # Basic energy statistics
    standard_stats = compute_energy_statistics(standard_samples, energy_fn)
    gfn_stats = compute_energy_statistics(gfn_samples, energy_fn)
    
    # Add prefixes for clarity
    for k, v in standard_stats.items():
        metrics[f'standard_{k}'] = v
    for k, v in gfn_stats.items():
        metrics[f'gfn_{k}'] = v
    
    # Energy improvement metrics
    improvement, top_improvement = compute_energy_improvement(standard_samples, gfn_samples, energy_fn)
    metrics['energy_improvement'] = improvement
    metrics['top_energy_improvement'] = top_improvement
    
    # Diversity metrics
    metrics['standard_diversity'] = compute_diversity(standard_samples)
    metrics['gfn_diversity'] = compute_diversity(gfn_samples)
    metrics['diversity_ratio'] = metrics['gfn_diversity'] / max(metrics['standard_diversity'], 1e-10)
    
    # Novelty metrics (if reference samples provided)
    if reference_samples is not None:
        metrics['standard_novelty'] = compute_novelty(standard_samples, reference_samples)
        metrics['gfn_novelty'] = compute_novelty(gfn_samples, reference_samples)
        metrics['novelty_ratio'] = metrics['gfn_novelty'] / max(metrics['standard_novelty'], 1e-10)
    
    # Entropy metrics
    metrics['standard_entropy'] = compute_entropy(standard_samples)
    metrics['gfn_entropy'] = compute_entropy(gfn_samples)
    metrics['entropy_ratio'] = metrics['gfn_entropy'] / max(metrics['standard_entropy'], 1e-10)
    
    # Effective sample size
    standard_ess, standard_ess_ratio = compute_effective_sample_size(standard_samples, energy_fn)
    gfn_ess, gfn_ess_ratio = compute_effective_sample_size(gfn_samples, energy_fn)
    metrics['standard_ess'] = standard_ess
    metrics['standard_ess_ratio'] = standard_ess_ratio
    metrics['gfn_ess'] = gfn_ess
    metrics['gfn_ess_ratio'] = gfn_ess_ratio
    metrics['ess_improvement'] = (gfn_ess_ratio - standard_ess_ratio) / max(standard_ess_ratio, 1e-10) * 100
    
    # Coverage metrics
    standard_coverage = compute_coverage_metrics(standard_samples, reference_centers, energy_fn)
    gfn_coverage = compute_coverage_metrics(gfn_samples, reference_centers, energy_fn)
    
    for k, v in standard_coverage.items():
        metrics[f'standard_{k}'] = v
    for k, v in gfn_coverage.items():
        metrics[f'gfn_{k}'] = v
        if f'standard_{k}' in metrics:
            metrics[f'{k}_ratio'] = metrics[f'gfn_{k}'] / max(metrics[f'standard_{k}'], 1e-10)
    
    return metrics 