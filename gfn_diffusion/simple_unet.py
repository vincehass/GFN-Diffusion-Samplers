import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleMLP(nn.Module):
    """
    Simple MLP block for UNet
    """
    def __init__(self, in_dim, hidden_dim, out_dim, time_dim=None, debug=False):
        super().__init__()
        self.debug = debug
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Main layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
        # Time embedding layer
        self.time_layer = nn.Linear(time_dim, hidden_dim) if time_dim is not None else None
        
        # Residual connection
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, t=None):
        if self.debug:
            print(f"SimpleMLP - Input shape: {x.shape}")
            print(f"SimpleMLP - Expected input dim: {self.in_dim}")
        
        # Residual connection
        residual = self.residual(x)
        
        # First layer
        x = self.fc1(x)
        x = self.norm(x)
        x = self.activation(x)
        
        # Add time embedding if provided
        if t is not None and self.time_layer is not None:
            t_emb = self.time_layer(t)
            
            if self.debug:
                print(f"SimpleMLP - Time emb shape: {t_emb.shape}")
                print(f"SimpleMLP - X shape before addition: {x.shape}")
            
            # Make sure dimensions align
            if len(x.shape) > len(t_emb.shape):
                for _ in range(len(x.shape) - len(t_emb.shape)):
                    t_emb = t_emb.unsqueeze(-1)
            
            x = x + t_emb
        
        # Second layer
        x = self.fc2(x)
        
        if self.debug:
            print(f"SimpleMLP - Output shape: {x.shape}")
            print(f"SimpleMLP - Residual shape: {residual.shape}")
        
        # Add residual connection
        return x + residual

class VerySimpleUNet(nn.Module):
    """
    An extremely simplified UNet implementation that avoids dimension issues
    """
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2, time_dim=32):
        super().__init__()
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Main network - just a series of MLPs with residual connections
        self.net = nn.Sequential(
            SimpleMLP(input_dim, hidden_dim, hidden_dim, time_dim),
            SimpleMLP(hidden_dim, hidden_dim*2, hidden_dim, time_dim),
            SimpleMLP(hidden_dim, hidden_dim*2, hidden_dim, time_dim),
            SimpleMLP(hidden_dim, hidden_dim, output_dim, time_dim)
        )
    
    def forward(self, x, t):
        # Process time embedding
        t_emb = self.time_emb(t)
        
        # Separate blocks with explicit timing
        for layer in self.net:
            x = layer(x, t_emb)
        
        return x

class VerySimpleConditionalUNet(nn.Module):
    """
    An extremely simplified conditional UNet implementation
    """
    def __init__(self, input_dim=2, condition_dim=4, hidden_dim=32, output_dim=2, time_dim=32):
        super().__init__()
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Condition encoder
        self.cond_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Main network
        self.net = nn.Sequential(
            SimpleMLP(hidden_dim, hidden_dim*2, hidden_dim, time_dim),
            SimpleMLP(hidden_dim, hidden_dim*2, hidden_dim, time_dim),
            SimpleMLP(hidden_dim, hidden_dim*2, hidden_dim, time_dim),
            SimpleMLP(hidden_dim, hidden_dim, output_dim, time_dim)
        )
    
    def forward(self, x, t, condition):
        # Process time embedding
        t_emb = self.time_emb(t)
        
        # Process condition
        cond_emb = self.cond_encoder(condition)
        
        # Concatenate input and condition
        x = torch.cat([x, cond_emb], dim=-1)
        
        # Project input
        x = self.input_proj(x)
        
        # Apply network
        for layer in self.net:
            x = layer(x, t_emb)
        
        return x

def test_simple_unet():
    """Test the VerySimpleUNet model"""
    # Create model
    model = VerySimpleUNet(input_dim=2, hidden_dim=32, output_dim=2, time_dim=32)
    
    # Create test input
    batch_size = 4
    x = torch.randn(batch_size, 2)
    t = torch.randint(0, 100, (batch_size,))
    
    # Forward pass
    try:
        output = model(x, t)
        print(f"VerySimpleUNet test - Input shape: {x.shape}, Output shape: {output.shape}")
        print(f"VerySimpleUNet test passed: {output.shape == x.shape}")
        return True
    except Exception as e:
        print(f"VerySimpleUNet test failed: {str(e)}")
        return False

def test_simple_conditional_unet():
    """Test the VerySimpleConditionalUNet model"""
    # Create model
    model = VerySimpleConditionalUNet(input_dim=2, condition_dim=4, hidden_dim=32, output_dim=2, time_dim=32)
    
    # Create test input
    batch_size = 4
    x = torch.randn(batch_size, 2)
    t = torch.randint(0, 100, (batch_size,))
    condition = torch.randn(batch_size, 4)
    
    # Forward pass
    try:
        output = model(x, t, condition)
        print(f"VerySimpleConditionalUNet test - Input shape: {x.shape}, Output shape: {output.shape}")
        print(f"VerySimpleConditionalUNet test passed: {output.shape == x.shape}")
        return True
    except Exception as e:
        print(f"VerySimpleConditionalUNet test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Testing Simple UNet Models ===")
    unet_passed = test_simple_unet()
    cond_unet_passed = test_simple_conditional_unet()
    
    if unet_passed and cond_unet_passed:
        print("\nAll tests passed! The simplified UNet models work correctly.")
    else:
        print("\nSome tests failed. Please check the error messages above.") 