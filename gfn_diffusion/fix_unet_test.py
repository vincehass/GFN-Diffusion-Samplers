import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from energy_sampling.models import UNet, ConditionalUNet, DiffusionSchedule

def test_unet_dimensions():
    """
    Test UNet dimensions to diagnose and fix dimension mismatch issues.
    """
    print("Testing UNet dimensions...")
    
    # Create a simple UNet with minimal dimensions
    input_dim = 2
    hidden_dim = 16
    output_dim = 2
    time_dim = 16
    num_layers = 2
    
    model = UNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                time_dim=time_dim, num_layers=num_layers)
    
    # Create input data
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    t = torch.randint(0, 100, (batch_size,))
    
    # Enable debug printing of dimensions
    print(f"Input x shape: {x.shape}")
    print(f"Timestep t shape: {t.shape}")
    
    # Track the dimensions through the network
    # Time embedding
    t_emb = model.time_embeddings(t)
    t_emb = model.time_mlp(t_emb)
    print(f"Time embedding shape: {t_emb.shape}")
    
    # Initial projection
    x_proj = model.input_proj(x)
    print(f"Initial projection shape: {x_proj.shape}")
    
    # Down path
    residuals = [x_proj]
    x_down = x_proj
    
    print("\n--- Down path ---")
    for i, block in enumerate(model.down_blocks):
        x_down = block(x_down, t_emb)
        print(f"Down block {i} output shape: {x_down.shape}")
        residuals.append(x_down)
        
    # Middle block
    print("\n--- Middle block ---")
    x_middle = model.middle_block(x_down, t_emb)
    print(f"Middle block output shape: {x_middle.shape}")
    
    # Up path
    print("\n--- Up path ---")
    x_up = x_middle
    
    for i, block in enumerate(model.up_blocks):
        residual = residuals.pop()
        print(f"Up block {i} - Current x shape: {x_up.shape}, Residual shape: {residual.shape}")
        x_cat = torch.cat([x_up, residual], dim=-1)
        print(f"After concatenation shape: {x_cat.shape}")
        x_up = block(x_cat, t_emb)
        print(f"Up block {i} output shape: {x_up.shape}")
        
    # Output projection
    output = model.output_proj(x_up)
    print(f"\nFinal output shape: {output.shape}")
    
    # Check if output shape matches input shape
    print(f"Output shape matches input shape: {output.shape == x.shape}")
    
    # Test forward pass
    try:
        output_direct = model(x, t)
        print(f"Direct forward pass output shape: {output_direct.shape}")
        print(f"Direct forward pass successful: {output_direct.shape == x.shape}")
    except Exception as e:
        print(f"Direct forward pass failed: {str(e)}")
    
    return output.shape == x.shape

def test_conditional_unet_dimensions():
    """
    Test ConditionalUNet dimensions to diagnose and fix dimension mismatch issues.
    """
    print("\nTesting ConditionalUNet dimensions...")
    
    # Create a simple ConditionalUNet with minimal dimensions
    input_dim = 2
    condition_dim = 4
    hidden_dim = 16
    output_dim = 2
    time_dim = 16
    num_layers = 2
    
    model = ConditionalUNet(input_dim=input_dim, condition_dim=condition_dim, 
                           hidden_dim=hidden_dim, output_dim=output_dim, 
                           time_dim=time_dim, num_layers=num_layers)
    
    # Create input data
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    t = torch.randint(0, 100, (batch_size,))
    condition = torch.randn(batch_size, condition_dim)
    
    print(f"Input x shape: {x.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Timestep t shape: {t.shape}")
    
    # Test forward pass
    try:
        output = model(x, t, condition)
        print(f"Forward pass output shape: {output.shape}")
        print(f"Forward pass successful: {output.shape == x.shape}")
        return output.shape == x.shape
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")
        return False

if __name__ == "__main__":
    unet_success = test_unet_dimensions()
    conditional_success = test_conditional_unet_dimensions()
    
    if unet_success and conditional_success:
        print("\nAll UNet dimension tests passed!")
    else:
        print("\nUNet dimension tests failed. Please check the dimensional analysis above for issues.") 