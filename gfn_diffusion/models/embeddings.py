import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding.
    
    This module generates sinusoidal embeddings for diffusion timesteps.
    """
    def __init__(self, dim):
        """
        Initialize the sinusoidal position embeddings.
        
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Generate embeddings for timesteps.
        
        Args:
            time: Timestep tensor [batch_size]
            
        Returns:
            embeddings: Timestep embeddings [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
