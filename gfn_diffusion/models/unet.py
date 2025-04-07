import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import SinusoidalPositionEmbeddings

class Block(nn.Module):
    """
    Basic block for UNet with time embeddings.
    """
    def __init__(self, dim, dim_out, time_dim=None, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.use_time = time_dim is not None
        
        if self.use_time:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, dim_out)
            )
    
    def forward(self, x, time=None):
        h = self.proj(x)
        h = self.norm(h)
        
        if self.use_time and time is not None:
            time_emb = self.time_mlp(time)
            h = h + time_emb.reshape(time_emb.shape[0], -1, 1, 1)
            
        return self.act(h)

class SimpleUNet(nn.Module):
    """
    A simplified UNet model for diffusion models.
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, time_dim=128, num_layers=1):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial conv to get to hidden dimension
        self.init_conv = nn.Conv2d(input_dim, hidden_dim, 1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        curr_dim = hidden_dim
        
        for _ in range(num_layers):
            self.down_blocks.append(nn.ModuleList([
                Block(curr_dim, curr_dim, time_dim),
                Block(curr_dim, curr_dim * 2, time_dim),
                nn.MaxPool2d(2)
            ]))
            curr_dim *= 2
        
        # Middle blocks
        self.mid_block1 = Block(curr_dim, curr_dim, time_dim)
        self.mid_block2 = Block(curr_dim, curr_dim, time_dim)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for _ in range(num_layers):
            self.up_blocks.append(nn.ModuleList([
                Block(curr_dim * 2, curr_dim, time_dim),
                Block(curr_dim, curr_dim // 2, time_dim),
                nn.Upsample(scale_factor=2, mode='nearest')
            ]))
            curr_dim //= 2
        
        # Final conv to get back to output dimension
        self.final_conv = nn.Sequential(
            Block(hidden_dim * 2, hidden_dim, time_dim),
            nn.Conv2d(hidden_dim, output_dim, 1)
        )
    
    def forward(self, x, t):
        # For 2D data like (batch_size, 2), reshape to (batch_size, 2, 1, 1)
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1, 1)
            
        # Get time embedding
        t_emb = self.time_mlp(t)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Store residual connections
        residuals = [x]
        
        # Downsampling
        for down_block in self.down_blocks:
            x = down_block[0](x, t_emb)
            residuals.append(x)
            x = down_block[1](x, t_emb)
            residuals.append(x)
            x = down_block[2](x)
            
        # Middle blocks
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)
        
        # Upsampling with skip connections
        for up_block in self.up_blocks:
            x = torch.cat([x, residuals.pop()], dim=1)
            x = up_block[0](x, t_emb)
            x = torch.cat([x, residuals.pop()], dim=1)
            x = up_block[1](x, t_emb)
            x = up_block[2](x)
            
        # Final conv
        x = torch.cat([x, residuals.pop()], dim=1)
        x = self.final_conv(x)
        
        # Reshape back to original dimensions
        return x.reshape(x.shape[0], x.shape[1])

class UNet(nn.Module):
    """
    Simple UNet for diffusion models with 1D data.
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, time_dim=128, num_layers=1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4)
        )
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        # Down layers
        current_dim = hidden_dim
        for _ in range(num_layers):
            self.down_layers.append(nn.ModuleList([
                nn.Linear(current_dim, current_dim * 2),
                nn.Linear(hidden_dim * 4, current_dim * 2),
                nn.GELU()
            ]))
            current_dim *= 2
        
        # Middle layer
        self.middle_layer = nn.Sequential(
            nn.Linear(current_dim, current_dim * 2),
            nn.GELU(),
            nn.Linear(current_dim * 2, current_dim)
        )
        
        # Up layers
        for _ in range(num_layers):
            self.up_layers.append(nn.ModuleList([
                nn.Linear(current_dim * 2, current_dim // 2),
                nn.Linear(hidden_dim * 4, current_dim // 2),
                nn.GELU()
            ]))
            current_dim //= 2
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, t):
        # Time embedding
        time_emb = self.time_mlp(t)
        
        # Initial layer
        h = self.input_layer(x)
        
        # Store residuals
        residuals = [h]
        
        # Down path
        for linear, time_linear, act in self.down_layers:
            h = linear(h)
            h = h + time_linear(time_emb)
            h = act(h)
            residuals.append(h)
        
        # Middle
        h = self.middle_layer(h)
        
        # Up path with skip connections
        for linear, time_linear, act in self.up_layers:
            h = torch.cat([h, residuals.pop()], dim=-1)
            h = linear(h)
            h = h + time_linear(time_emb)
            h = act(h)
        
        # Final layer
        return self.output_layer(h)

class ConditionalUNet(nn.Module):
    """
    Conditional UNet for diffusion models with condition labels.
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, time_dim=128, num_conditions=4, num_layers=1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4)
        )
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(num_conditions, hidden_dim * 4)
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        # Down layers
        current_dim = hidden_dim
        for _ in range(num_layers):
            self.down_layers.append(nn.ModuleList([
                nn.Linear(current_dim, current_dim * 2),
                nn.Linear(hidden_dim * 4, current_dim * 2),  # Time embedding
                nn.Linear(hidden_dim * 4, current_dim * 2),  # Condition embedding
                nn.GELU()
            ]))
            current_dim *= 2
        
        # Middle layer
        self.middle_layer = nn.Sequential(
            nn.Linear(current_dim, current_dim * 2),
            nn.GELU(),
            nn.Linear(current_dim * 2, current_dim)
        )
        
        # Up layers
        for _ in range(num_layers):
            self.up_layers.append(nn.ModuleList([
                nn.Linear(current_dim * 2, current_dim // 2),
                nn.Linear(hidden_dim * 4, current_dim // 2),  # Time embedding
                nn.Linear(hidden_dim * 4, current_dim // 2),  # Condition embedding
                nn.GELU()
            ]))
            current_dim //= 2
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, t, c=None):
        # Time embedding
        time_emb = self.time_mlp(t)
        
        # Condition embedding
        if c is not None:
            cond_emb = self.condition_embedding(c)
        else:
            batch_size = x.shape[0]
            cond_emb = torch.zeros(batch_size, self.condition_embedding.embedding_dim, device=x.device)
        
        # Initial layer
        h = self.input_layer(x)
        
        # Store residuals
        residuals = [h]
        
        # Down path
        for linear, time_linear, cond_linear, act in self.down_layers:
            h = linear(h)
            h = h + time_linear(time_emb)
            h = h + cond_linear(cond_emb)
            h = act(h)
            residuals.append(h)
        
        # Middle
        h = self.middle_layer(h)
        
        # Up path with skip connections
        for linear, time_linear, cond_linear, act in self.up_layers:
            h = torch.cat([h, residuals.pop()], dim=-1)
            h = linear(h)
            h = h + time_linear(time_emb)
            h = h + cond_linear(cond_emb)
            h = act(h)
        
        # Final layer
        return self.output_layer(h)
