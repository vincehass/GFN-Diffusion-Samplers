"""
VAE model implementation for GFN-Diffusion experiments.

This script provides:
1. A variational autoencoder (VAE) implementation
2. Training functionality for pretraining the VAE
3. Utility functions for evaluating and visualizing the VAE
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from pathlib import Path
import argparse

# Create directories for models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for MNIST.
    
    Args:
        input_dim: Dimension of input data (e.g., 784 for MNIST)
        latent_dim: Dimension of latent space
        hidden_dim: Dimension of hidden layers (default: 512)
    """
    def __init__(self, input_dim=784, latent_dim=2, hidden_dim=512):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log variance for latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        Encode input data to latent space parameters.
        
        Args:
            x: Input data
            
        Returns:
            mu, logvar: Mean and log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to reconstructed input.
        
        Args:
            z: Latent vector
            
        Returns:
            x_recon: Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input data
            
        Returns:
            x_recon, mu, logvar: Reconstructed input, mean and log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def load_mnist(batch_size=128, flatten=True):
    """
    Load MNIST dataset.
    """
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1) if flatten else x)
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def train_vae(model, train_loader, optimizer, device, beta=1.0):
    """
    Train the VAE for one epoch.
    
    Args:
        model: VAE model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        beta: Weight for KL divergence term in loss
        
    Returns:
        avg_loss: Average loss for the epoch
        avg_recon_loss: Average reconstruction loss
        avg_kl_loss: Average KL divergence loss
    """
    model.train()
    train_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy_with_logits(recon_batch, data, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
    
    # Calculate average losses
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon_loss = recon_loss_sum / len(train_loader.dataset)
    avg_kl_loss = kl_loss_sum / len(train_loader.dataset)
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def test_vae(model, test_loader, device, beta=1.0):
    """
    Test the VAE on the test set.
    
    Args:
        model: VAE model
        test_loader: DataLoader for test data
        device: Device to test on
        beta: Weight for KL divergence term in loss
        
    Returns:
        avg_loss: Average loss for the test set
        avg_recon_loss: Average reconstruction loss
        avg_kl_loss: Average KL divergence loss
    """
    model.eval()
    test_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            recon_batch, mu, log_var = model(data)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy_with_logits(recon_batch, data, reduction='sum')
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = recon_loss + beta * kl_loss
            
            test_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
    
    # Calculate average losses
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon_loss = recon_loss_sum / len(test_loader.dataset)
    avg_kl_loss = kl_loss_sum / len(test_loader.dataset)
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def visualize_vae_results(model, test_loader, device, num_samples=10):
    """
    Visualize VAE reconstructions and samples.
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader for test data
        device: Device to run on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get a batch of test images
    test_data = next(iter(test_loader))[0][:num_samples].to(device)
    
    # Reconstruct the images
    with torch.no_grad():
        reconstructions = torch.sigmoid(model.reconstruct(test_data))
        
    # Generate random samples
    with torch.no_grad():
        samples = torch.sigmoid(model.sample(num_samples, device))
    
    # Plot reconstructions
    plt.figure(figsize=(15, 5))
    
    # Original images
    for i in range(num_samples):
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(test_data[i].cpu().reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')
    
    # Reconstructed images
    for i in range(num_samples):
        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(reconstructions[i].cpu().reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    # Generated samples
    for i in range(num_samples):
        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(samples[i].cpu().reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Generated')
    
    plt.tight_layout()
    plt.savefig('results/vae_visualization.png')
    plt.close()
    
    # Plot latent space (only works for 2D latent space)
    if model.fc_mu.out_features == 2:
        visualize_latent_space(model, test_loader, device)


def visualize_latent_space(model, test_loader, device, n_samples=1000):
    """
    Visualize the latent space of the VAE (for 2D latent space).
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader for test data
        device: Device to run on
        n_samples: Number of test samples to visualize
    """
    model.eval()
    
    # Get samples from test set
    data_samples = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            data_samples.append(data)
            labels.append(label)
            
            if len(torch.cat(data_samples)) >= n_samples:
                break
    
    data_samples = torch.cat(data_samples)[:n_samples]
    labels = torch.cat(labels)[:n_samples]
    
    # Get latent representations
    mu, _ = model.encode(data_samples)
    mu = mu.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Plot latent space
    plt.figure(figsize=(10, 8))
    
    # Scatter plot colored by digit label
    scatter = plt.scatter(mu[:, 0], mu[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space (2D)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/vae_latent_space.png')
    plt.close()


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist(batch_size=args.batch_size)
    
    # Create model
    model = VAE(
        input_dim=784,  # 28x28 images flattened
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    train_losses = []
    train_recon_losses = []
    train_kl_losses = []
    
    test_losses = []
    test_recon_losses = []
    test_kl_losses = []
    
    best_loss = float('inf')
    
    print("Starting VAE training...")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_recon_loss, train_kl_loss = train_vae(
            model, train_loader, optimizer, device, beta=args.beta
        )
        
        # Test
        test_loss, test_recon_loss, test_kl_loss = test_vae(
            model, test_loader, device, beta=args.beta
        )
        
        # Store losses
        train_losses.append(train_loss)
        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_kl_loss)
        
        test_losses.append(test_loss)
        test_recon_losses.append(test_recon_loss)
        test_kl_losses.append(test_kl_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}) | "
              f"Test Loss: {test_loss:.4f} (Recon: {test_recon_loss:.4f}, KL: {test_kl_loss:.4f})")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "models/vae_best.pt")
            print(f"Saved best model with test loss: {best_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "models/vae_final.pt")
    print("Saved final model.")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Reconstruction loss
    plt.subplot(1, 3, 2)
    plt.plot(train_recon_losses, label='Train')
    plt.plot(test_recon_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # KL loss
    plt.subplot(1, 3, 3)
    plt.plot(train_kl_losses, label='Train')
    plt.plot(test_kl_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/vae_training_curves.png')
    plt.close()
    
    # Load best model for visualization
    model.load_state_dict(torch.load("models/vae_best.pt"))
    
    # Visualize results
    visualize_vae_results(model, test_loader, device)
    
    print("Training completed. Results saved to 'results/' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE model")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # VAE parameters
    parser.add_argument("--latent_dim", type=int, default=2, help="Dimension of latent space")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Dimension of hidden layers")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for KL divergence term in loss")
    
    args = parser.parse_args()
    main(args) 