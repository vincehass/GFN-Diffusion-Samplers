import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import io
from PIL import Image
import wandb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.gridspec as gridspec

def plot_2d_density_comparison(samples_standard, samples_gfn, save_path, title="Density Comparison", 
                             range_val=5.0, bins=100):
    """
    Create a side-by-side comparison of density plots for standard and GFN samples.
    
    Args:
        samples_standard: Samples from standard diffusion
        samples_gfn: Samples from GFN-guided diffusion
        save_path: Path to save the figure
        title: Title for the overall plot
        range_val: Range for the plot
        bins: Number of bins for the histogram
    """
    if isinstance(samples_standard, torch.Tensor):
        samples_standard = samples_standard.cpu().numpy()
    if isinstance(samples_gfn, torch.Tensor):
        samples_gfn = samples_gfn.cpu().numpy()
    
    plt.figure(figsize=(16, 6))
    
    # Standard samples density
    plt.subplot(1, 2, 1)
    sns.kdeplot(
        x=samples_standard[:, 0], 
        y=samples_standard[:, 1],
        fill=True, 
        cmap="Blues",
        thresh=0.05
    )
    plt.xlim(-range_val, range_val)
    plt.ylim(-range_val, range_val)
    plt.title("Standard Diffusion Density")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # GFN samples density
    plt.subplot(1, 2, 2)
    sns.kdeplot(
        x=samples_gfn[:, 0], 
        y=samples_gfn[:, 1],
        fill=True, 
        cmap="Reds",
        thresh=0.05
    )
    plt.xlim(-range_val, range_val)
    plt.ylim(-range_val, range_val)
    plt.title("GFN-Guided Diffusion Density")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_3d_energy_landscape(energy_fn, save_path, title="3D Energy Landscape", 
                           range_val=4.0, resolution=100):
    """
    Create a 3D visualization of the energy landscape.
    
    Args:
        energy_fn: Energy function
        save_path: Path to save the figure
        title: Title for the plot
        range_val: Range for the plot
        resolution: Number of points per dimension
    """
    try:
        # Try to compute the actual energy landscape
        x = torch.linspace(-range_val, range_val, resolution)
        y = torch.linspace(-range_val, range_val, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Flatten and stack
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Use batching to compute energy in smaller chunks to avoid memory issues
        batch_size = 1000
        num_batches = (points.shape[0] + batch_size - 1) // batch_size  # Ceiling division
        energy_list = []
        
        # Compute energy in batches
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, points.shape[0])
                batch_points = points[start_idx:end_idx]
                batch_energy = energy_fn(batch_points)
                
                # Check if energy_fn returned a single value per point
                if batch_energy.shape[0] == batch_points.shape[0]:
                    energy_list.append(batch_energy)
                else:
                    # If not, take the mean along all dimensions except the first
                    batch_energy = batch_energy.view(batch_points.shape[0], -1).mean(dim=1)
                    energy_list.append(batch_energy)
            
            # Concatenate all batches
            energy = torch.cat(energy_list, dim=0)
            
            # Check if we can reshape to the expected dimensions
            if energy.numel() == resolution * resolution:
                energy = energy.reshape(resolution, resolution)
            else:
                print(f"Warning: Energy output size {energy.numel()} doesn't match expected size {resolution * resolution}")
                # Create a placeholder energy landscape
                energy = torch.zeros(resolution, resolution)
                # Add some simple pattern
                for i in range(resolution):
                    for j in range(resolution):
                        # Simple radial pattern
                        distance = ((i - resolution//2)**2 + (j - resolution//2)**2) ** 0.5
                        energy[i, j] = np.sin(distance / 5) * np.exp(-distance / 30)
        
        # Convert tensors to NumPy
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        energy_np = energy.cpu().numpy()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a surface plot
        surf = ax.plot_surface(
            X_np, Y_np, energy_np,
            rstride=3, cstride=3,
            cmap='viridis',
            linewidth=0,
            antialiased=True
        )
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Energy')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Energy')
        ax.set_title(title)
        
        # Set the viewing angle
        ax.view_init(elev=30, azim=45)
        
    except Exception as e:
        print(f"Error generating 3D energy landscape: {e}")
        print("Creating placeholder visualization instead")
        
        # Create a placeholder visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create simple synthetic data
        x = np.linspace(-range_val, range_val, resolution)
        y = np.linspace(-range_val, range_val, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        # Create a surface plot
        surf = ax.plot_surface(
            X, Y, Z,
            rstride=3, cstride=3,
            cmap='viridis',
            linewidth=0,
            antialiased=True
        )
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Energy (Placeholder)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Energy (Placeholder)')
        ax.set_title(f"{title} (Placeholder)")
        
        # Set the viewing angle
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_energy_evolution(energy_values, save_path, title="Energy Evolution"):
    """
    Plot the evolution of energy values over diffusion timesteps.
    
    Args:
        energy_values: Array-like of energy values (can be 1D, 2D, or higher)
        save_path: Path to save the figure
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Ensure energy_values is properly shaped and convert to numpy if tensor
    if isinstance(energy_values, torch.Tensor):
        energy_values = energy_values.cpu().numpy()
    
    try:
        # Handle object arrays with multiple shapes
        if isinstance(energy_values, np.ndarray) and energy_values.dtype == np.dtype('O'):
            print(f"Processing object array of energy values with length {len(energy_values)}")
            # Convert to a more uniform representation by extracting mean values
            processed_values = []
            for ev in energy_values:
                if isinstance(ev, np.ndarray):
                    # For array entries, compute mean
                    if ev.size > 0:
                        processed_values.append(float(np.mean(ev)))
                    else:
                        processed_values.append(0.0)
                elif isinstance(ev, (int, float)):
                    # For scalar entries, use directly
                    processed_values.append(float(ev))
                else:
                    # For other types, try conversion or use default
                    try:
                        processed_values.append(float(ev))
                    except (TypeError, ValueError):
                        processed_values.append(0.0)
            
            # Create a simple 1D plot of the processed values
            x = np.arange(len(processed_values))
            plt.plot(x, processed_values, color='blue', label='Mean Energy', marker='o', markersize=3)
            
        else:
            # Try normal array handling
            energy_values = np.array(energy_values)
            
            # Handle different shapes of energy_values
            if len(energy_values.shape) == 1:
                # Simple 1D array, just plot directly
                x = np.arange(len(energy_values))
                plt.plot(x, energy_values, color='blue', label='Energy')
            
            elif len(energy_values.shape) == 2:
                # 2D array: timesteps × batch_size or timesteps × energy_dimensions
                x = np.arange(energy_values.shape[0])
                
                # If second dimension is large, assume it's batch elements
                if energy_values.shape[1] > 5:  # Arbitrary threshold
                    mean_energy = np.mean(energy_values, axis=1)
                    std_energy = np.std(energy_values, axis=1)
                    
                    # Plot mean energy
                    plt.plot(x, mean_energy, color='blue', label='Mean Energy')
                    
                    # Plot confidence interval
                    plt.fill_between(
                        x, 
                        mean_energy - std_energy, 
                        mean_energy + std_energy, 
                        color='blue', 
                        alpha=0.2, 
                        label='Standard Deviation'
                    )
                else:
                    # If second dimension is small, plot each dimension separately
                    for i in range(energy_values.shape[1]):
                        plt.plot(x, energy_values[:, i], label=f'Energy Dim {i}')
            
            elif len(energy_values.shape) >= 3:
                # Higher dimensional array, flatten all but the first dimension
                # and compute statistics across the flattened dimensions
                x = np.arange(energy_values.shape[0])
                flattened = energy_values.reshape(energy_values.shape[0], -1)
                mean_energy = np.mean(flattened, axis=1)
                std_energy = np.std(flattened, axis=1)
                
                plt.plot(x, mean_energy, color='blue', label='Mean Energy')
                plt.fill_between(
                    x, 
                    mean_energy - std_energy, 
                    mean_energy + std_energy, 
                    color='blue', 
                    alpha=0.2, 
                    label='Standard Deviation'
                )
    except Exception as e:
        print(f"Error processing energy values for plotting: {e}")
        # Create a simple placeholder plot
        plt.text(0.5, 0.5, f"Error plotting energy values: {e}", 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.xlabel('Diffusion Timestep')
    plt.ylabel('Energy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def visualize_diffusion_process(trajectory, save_path, title="Diffusion Process", 
                              energy_fn=None, num_frames=10):
    """
    Create an animation showing the diffusion process through time.
    
    Args:
        trajectory: Tensor of shape [timesteps, batch_size, dimensions]
        save_path: Path to save the animation
        title: Title for the animation
        energy_fn: Optional energy function for background
        num_frames: Number of frames to show (will sample from trajectory)
    """
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    
    # Sample frames if there are too many
    if trajectory.shape[0] > num_frames:
        indices = np.linspace(0, trajectory.shape[0]-1, num_frames, dtype=int)
        traj_sampled = trajectory[indices]
    else:
        traj_sampled = trajectory
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the plot
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # If energy function provided, create background contour
    if energy_fn is not None:
        try:
            resolution = 100
            x = torch.linspace(-5, 5, resolution)
            y = torch.linspace(-5, 5, resolution)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            points = torch.stack([X.flatten(), Y.flatten()], dim=1)
            
            with torch.no_grad():
                # Compute energy in batches to avoid memory issues
                batch_size = 1000
                num_batches = (points.shape[0] + batch_size - 1) // batch_size
                energy_list = []
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, points.shape[0])
                    batch_points = points[start_idx:end_idx]
                    
                    batch_energy = energy_fn(batch_points)
                    # If energy returns more than one value per input, take the mean
                    if batch_energy.numel() != batch_points.shape[0]:
                        batch_energy = batch_energy.view(batch_points.shape[0], -1).mean(dim=1)
                    
                    energy_list.append(batch_energy)
                
                # Concatenate batches
                energy = torch.cat(energy_list, dim=0)
                
                # Check if the resulting energy can be reshaped
                if energy.numel() == resolution * resolution:
                    energy = energy.reshape(resolution, resolution)
                else:
                    print(f"Warning: Energy output size {energy.numel()} doesn't match expected size {resolution * resolution}")
                    # Create a placeholder energy landscape
                    energy = torch.zeros(resolution, resolution)
                    # Add some simple pattern
                    for i in range(resolution):
                        for j in range(resolution):
                            # Simple radial pattern
                            distance = ((i - resolution//2)**2 + (j - resolution//2)**2) ** 0.5
                            energy[i, j] = np.sin(distance / 5) * np.exp(-distance / 30)
            
            X_np = X.cpu().numpy()
            Y_np = Y.cpu().numpy()
            energy_np = energy.cpu().numpy()
            
            # Create a normalized colormap
            contour_filled = plt.contourf(X_np, Y_np, energy_np, levels=50, cmap='viridis', alpha=0.5)
            plt.colorbar(contour_filled, label='Energy')
            
        except Exception as e:
            print(f"Error creating energy visualization: {e}")
            print("Continuing without energy background...")
    
    # Empty scatter plot for data points that will be updated
    scatter = ax.scatter([], [], c='red', alpha=0.7, s=10)
    
    # Function to update the scatter plot for each frame
    def update(frame):
        points = traj_sampled[frame]
        scatter.set_offsets(points)
        ax.set_title(f"{title} (Step {frame+1}/{len(traj_sampled)})")
        return scatter,
    
    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(traj_sampled), 
        interval=500, blit=True
    )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=2)
    plt.close()


def visualize_high_dim_samples_with_tsne(samples_standard, samples_gfn, save_path, 
                                       title="t-SNE Visualization of Samples",
                                       perplexity=30, n_iter=1000):
    """
    Visualize high-dimensional samples using t-SNE dimensionality reduction.
    
    Args:
        samples_standard: Samples from standard diffusion
        samples_gfn: Samples from GFN-guided diffusion
        save_path: Path to save the figure
        title: Title for the plot
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
    """
    if isinstance(samples_standard, torch.Tensor):
        samples_standard = samples_standard.cpu().numpy()
    if isinstance(samples_gfn, torch.Tensor):
        samples_gfn = samples_gfn.cpu().numpy()
    
    # Combine samples for joint embedding
    all_samples = np.vstack([samples_standard, samples_gfn])
    labels = np.concatenate([
        np.zeros(len(samples_standard)), 
        np.ones(len(samples_gfn))
    ])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    samples_tsne = tsne.fit_transform(all_samples)
    
    # Split back to standard and GFN samples
    standard_tsne = samples_tsne[labels == 0]
    gfn_tsne = samples_tsne[labels == 1]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    plt.scatter(
        standard_tsne[:, 0], standard_tsne[:, 1],
        c='blue', alpha=0.5, label='Standard Diffusion'
    )
    plt.scatter(
        gfn_tsne[:, 0], gfn_tsne[:, 1],
        c='red', alpha=0.5, label='GFN-Guided Diffusion'
    )
    
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def create_mode_coverage_plot(mode_counts_standard, mode_counts_gfn, 
                            save_path, modes=None, title="Mode Coverage Comparison"):
    """
    Create a bar chart showing the coverage of different modes by standard and GFN samples.
    
    Args:
        mode_counts_standard: Array of sample counts for each mode from standard diffusion
        mode_counts_gfn: Array of sample counts for each mode from GFN-guided diffusion
        save_path: Path to save the figure
        modes: Optional list of mode names/indices
        title: Title for the plot
    """
    if isinstance(mode_counts_standard, torch.Tensor):
        mode_counts_standard = mode_counts_standard.cpu().numpy()
    if isinstance(mode_counts_gfn, torch.Tensor):
        mode_counts_gfn = mode_counts_gfn.cpu().numpy()
    
    # Create mode labels if not provided
    if modes is None:
        modes = [f"Mode {i+1}" for i in range(len(mode_counts_standard))]
    
    # Convert counts to percentages
    total_standard = mode_counts_standard.sum()
    total_gfn = mode_counts_gfn.sum()
    
    percent_standard = (mode_counts_standard / total_standard) * 100
    percent_gfn = (mode_counts_gfn / total_gfn) * 100
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(modes))
    width = 0.35
    
    plt.bar(x - width/2, percent_standard, width, label='Standard Diffusion', color='blue', alpha=0.7)
    plt.bar(x + width/2, percent_gfn, width, label='GFN-Guided Diffusion', color='red', alpha=0.7)
    
    plt.xlabel('Modes')
    plt.ylabel('Percentage of Samples (%)')
    plt.title(title)
    plt.xticks(x, modes, rotation=45 if len(modes) > 10 else 0)
    plt.legend()
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def create_metric_comparison_plot(metrics_dict, save_path, 
                                title="Performance Metrics Comparison",
                                higher_is_better=True):
    """
    Create a radar/spider chart to compare multiple metrics between methods.
    
    Args:
        metrics_dict: Dictionary with model names as keys and metric dictionaries as values
        save_path: Path to save the figure
        title: Title for the plot
        higher_is_better: Whether higher values are better for the metrics
    """
    # Extract method names and all unique metrics
    methods = list(metrics_dict.keys())
    all_metrics = set()
    for method_metrics in metrics_dict.values():
        all_metrics.update(method_metrics.keys())
    
    # Sort metrics for consistent order
    all_metrics = sorted(list(all_metrics))
    
    # Number of metrics (variables)
    N = len(all_metrics)
    
    # Create angles for the radar chart
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the polygon by repeating the first angle
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Add variable labels
    plt.xticks(angles[:-1], all_metrics, size=12)
    
    # Add method data
    for i, method in enumerate(methods):
        # Get values for this method (use 0 for missing metrics)
        values = []
        for metric in all_metrics:
            if metric in metrics_dict[method]:
                values.append(metrics_dict[method][metric])
            else:
                values.append(0)
        
        # If lower is better, invert the values
        if not higher_is_better:
            max_vals = []
            for j, metric in enumerate(all_metrics):
                max_val = max([m.get(metric, 0) for m in metrics_dict.values()])
                max_vals.append(max_val if max_val > 0 else 1)
            
            values = [max_vals[j] - v for j, v in enumerate(values)]
        
        # Close the polygon by repeating the first value
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title(title, size=15, y=1.1)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def log_animated_gfn_process_to_wandb(trajectories, energy_fn, output_dir, name="gfn_process"):
    """
    Create and log an animated GIF of the GFN process to wandb.
    
    Args:
        trajectories: List of trajectory tensors [T, batch_size, dim]
        energy_fn: Energy function
        output_dir: Directory to save the temporary images
        name: Name for the wandb log entry
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create temporary directory for frames
    frames_dir = os.path.join(output_dir, f"{name}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Convert trajectories to numpy if they're torch tensors
    if isinstance(trajectories[0], torch.Tensor):
        trajectories_np = [t.cpu().numpy() for t in trajectories]
    else:
        trajectories_np = trajectories
    
    # Create background energy landscape
    resolution = 100
    range_val = 5.0
    x = torch.linspace(-range_val, range_val, resolution)
    y = torch.linspace(-range_val, range_val, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        try:
            # Compute energy in batches
            batch_size = 1000
            num_batches = (points.shape[0] + batch_size - 1) // batch_size
            energy_list = []
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, points.shape[0])
                batch_points = points[start_idx:end_idx]
                
                batch_energy = energy_fn(batch_points)
                # If energy returns more than one value per input, take the mean
                if batch_energy.numel() != batch_points.shape[0]:
                    batch_energy = batch_energy.view(batch_points.shape[0], -1).mean(dim=1)
                
                energy_list.append(batch_energy)
            
            # Concatenate batches
            energy = torch.cat(energy_list, dim=0)
            
            # Check if the resulting energy can be reshaped
            if energy.numel() == resolution * resolution:
                energy = energy.reshape(resolution, resolution)
            else:
                print(f"Warning: Energy output size {energy.numel()} doesn't match expected size {resolution * resolution}")
                # Create a placeholder energy landscape
                energy = torch.zeros(resolution, resolution)
                # Add some simple pattern
                for i in range(resolution):
                    for j in range(resolution):
                        # Simple radial pattern
                        distance = ((i - resolution//2)**2 + (j - resolution//2)**2) ** 0.5
                        energy[i, j] = np.sin(distance / 5) * np.exp(-distance / 30)
        except Exception as e:
            print(f"Error computing energy landscape: {e}")
            # Create a placeholder energy landscape
            energy = torch.zeros(resolution, resolution)
            # Add some simple pattern
            for i in range(resolution):
                for j in range(resolution):
                    # Simple radial pattern
                    distance = ((i - resolution//2)**2 + (j - resolution//2)**2) ** 0.5
                    energy[i, j] = np.sin(distance / 5) * np.exp(-distance / 30)
    
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    energy_np = energy.cpu().numpy()
    
    # Create frames
    frames = []
    for t in range(len(trajectories_np[0])):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot energy landscape
        contour = ax.contourf(X_np, Y_np, energy_np, levels=50, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='Energy')
        
        # Plot all trajectories at this timestep
        for traj in trajectories_np:
            ax.scatter(traj[t, :, 0], traj[t, :, 1], c='red', alpha=0.7, s=15)
            
            # Add trajectory lines
            for i in range(traj.shape[1]):
                ax.plot(traj[:t+1, i, 0], traj[:t+1, i, 1], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f"GFN Sampling Process (Step {t+1}/{len(trajectories_np[0])})")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-range_val, range_val)
        ax.set_ylim(-range_val, range_val)
        
        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{t:03d}.png")
        plt.savefig(frame_path)
        frames.append(frame_path)
        plt.close()
    
    # Create GIF
    images = [Image.open(frame) for frame in frames]
    gif_path = os.path.join(output_dir, f"{name}.gif")
    
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=200,
        loop=0
    )
    
    # Log to wandb
    wandb.log({name: wandb.Image(gif_path)})
    
    return gif_path


def create_comparative_trajectory_plot(trajectory_standard, trajectory_gfn, 
                                     energy_fn, timesteps, output_dir, name="trajectory_comparison"):
    """
    Create a plot comparing trajectories from standard and GFN-guided diffusion.
    
    Args:
        trajectory_standard: Trajectory from standard diffusion
        trajectory_gfn: Trajectory from GFN-guided diffusion
        energy_fn: Energy function for visualization
        timesteps: Specific timesteps to visualize
        output_dir: Directory to save plot
        name: Filename for saving
        
    Returns:
        Path to saved plot
    """
    range_val = 5.0
    
    # Make sure the requested timesteps are valid
    num_steps = min(trajectory_standard.shape[0], trajectory_gfn.shape[0])
    valid_timesteps = []
    for t in timesteps:
        if t < num_steps:
            valid_timesteps.append(t)
        else:
            print(f"Warning: Timestep {t} exceeds available steps ({num_steps}). Skipping.")
    
    # If no valid timesteps, select some evenly spaced ones
    if not valid_timesteps:
        num_to_show = min(5, num_steps)
        indices = np.linspace(0, num_steps-1, num_to_show, dtype=int)
        valid_timesteps = [int(i) for i in indices]
        print(f"Using timesteps {valid_timesteps} instead")
    
    # Create a visualization of the energy landscape
    try:
        # Create meshgrid for visualization
        resolution = 100
        x = torch.linspace(-range_val, range_val, resolution)
        y = torch.linspace(-range_val, range_val, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Reshape for energy computation
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Compute energy values
        energy = torch.zeros(resolution*resolution)
        try:
            with torch.no_grad():
                energy = energy_fn(points)
                energy = energy.reshape(resolution, resolution)
        except Exception as e:
            print(f"Error computing energy for visualization: {e}")
            # Create a placeholder energy function
            energy = torch.zeros(resolution, resolution)
            for i in range(resolution):
                for j in range(resolution):
                    # Simple radial pattern
                    distance = ((i - resolution//2)**2 + (j - resolution//2)**2) ** 0.5
                    energy[i, j] = np.sin(distance / 5) * np.exp(-distance / 30)
    
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        energy_np = energy.cpu().numpy()
        
        # Create plot
        n_steps = len(valid_timesteps)
        fig = plt.figure(figsize=(15, 10))
        
        # Create a common colormap for all subplots
        vmin = energy_np.min()
        vmax = energy_np.max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        for i, t in enumerate(valid_timesteps):
            # Standard diffusion subplot
            ax1 = fig.add_subplot(2, n_steps, i+1)
            cf1 = ax1.contourf(X_np, Y_np, energy_np, levels=50, cmap='viridis', alpha=0.7, norm=norm)
            ax1.scatter(trajectory_standard[t, :, 0], trajectory_standard[t, :, 1], c='blue', alpha=0.7, s=15)
            ax1.set_title(f"Standard (t={t})")
            ax1.set_xlim(-range_val, range_val)
            ax1.set_ylim(-range_val, range_val)
            
            # GFN diffusion subplot
            ax2 = fig.add_subplot(2, n_steps, i+n_steps+1)
            cf2 = ax2.contourf(X_np, Y_np, energy_np, levels=50, cmap='viridis', alpha=0.7, norm=norm)
            ax2.scatter(trajectory_gfn[t, :, 0], trajectory_gfn[t, :, 1], c='red', alpha=0.7, s=15)
            ax2.set_title(f"GFN-Guided (t={t})")
            ax2.set_xlim(-range_val, range_val)
            ax2.set_ylim(-range_val, range_val)
        
        # Add a common colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(cf2, cax=cbar_ax, label='Energy')
        
        # Set title and adjust layout
        plt.suptitle("Comparison of Diffusion Trajectories", fontsize=16)
        try:
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        except Warning:
            # Layout might not be perfect but figure will still be created
            pass
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{name}.png"))
        plt.close()
        
        return os.path.join(output_dir, f"{name}.png")
    except Exception as e:
        print(f"Error creating comparative trajectory plot: {e}")
        # Create a simple fallback plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                horizontalalignment='center', verticalalignment='center')
        plt.title("Error in Visualization")
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{name}_error.png"))
        plt.close()
        return os.path.join(output_dir, f"{name}_error.png")


def plot_sample_trajectories(trajectories, energies, energy_fn=None, range_val=4.0, save_path=None, title=None):
    """
    Plot a set of sample trajectories with energy values.
    
    Args:
        trajectories: List of trajectories [T, n, d]
        energies: List of energy values for each trajectory [T, n]
        energy_fn: Optional energy function to visualize the energy landscape
        range_val: Range of the plot
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create subplot grid with room for colorbar
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    cax1 = plt.subplot(gs[0, 1])
    cax2 = plt.subplot(gs[1, 1])
    
    # Plot energy landscape if energy function is provided
    if energy_fn is not None:
        # Create a grid of points
        x = torch.linspace(-range_val, range_val, 100)
        y = torch.linspace(-range_val, range_val, 100)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Compute energy values
        with torch.no_grad():
            grid_energies = energy_fn(grid_points).reshape(100, 100).cpu().numpy()
        
        # Plot energy landscape
        im1 = ax1.imshow(grid_energies, extent=[-range_val, range_val, -range_val, range_val], 
                         origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(im1, cax=cax1, label='Energy')
    
    # Plot the trajectories
    trajectory_data = trajectories[0].cpu().numpy()
    energy_data = energies[0].cpu().numpy()
    
    # Get colormap for time steps
    cmap = cm.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, len(trajectory_data))]
    
    # Plot each trajectory
    for i in range(trajectory_data.shape[1]):
        # Extract trajectory for this sample
        traj = trajectory_data[:, i, :]
        # Plot with color gradient based on time
        for t in range(len(traj) - 1):
            ax1.plot(traj[t:t+2, 0], traj[t:t+2, 1], '-', color=colors[t], alpha=0.5, linewidth=1)
            ax2.plot([t, t+1], [energy_data[t, i], energy_data[t+1, i]], '-', color=colors[t], linewidth=1)
    
    # Add starting and ending points
    ax1.scatter(trajectory_data[0, :, 0], trajectory_data[0, :, 1], c='blue', s=10, label='Start')
    ax1.scatter(trajectory_data[-1, :, 0], trajectory_data[-1, :, 1], c='red', s=20, label='End')
    
    # Set plot labels and title
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_xlim(-range_val, range_val)
    ax1.set_ylim(-range_val, range_val)
    ax1.legend()
    
    # Set second plot labels
    ax2.set_xlabel('Diffusion Time Step')
    ax2.set_ylabel('Energy Value')
    
    # Create a colorbar for the time steps
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(trajectory_data) - 1))
    sm.set_array([])
    plt.colorbar(sm, cax=cax2, label='Time Step')
    
    if title:
        plt.suptitle(title)
    
    # Adjust layout
    try:
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    except Warning:
        # Layout might not be perfect but figure will still be created
        pass
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 