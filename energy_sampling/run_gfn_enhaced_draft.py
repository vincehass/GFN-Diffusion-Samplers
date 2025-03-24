class GFNDiffusion(nn.Module):
    def __init__(self, input_dim, num_steps=100, hidden_dim=256, 
                harmonics_dim=64, t_dim=64, s_emb_dim=64, device='cuda'):
        super().__init__()
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.device = device
        
        # Noise schedule
        self.betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0)
        
        # GFN components
        self.gfn = GFN(
            dim=input_dim,
            s_emb_dim=s_emb_dim,
            hidden_dim=hidden_dim,
            harmonics_dim=harmonics_dim,
            t_dim=t_dim,
            trajectory_length=num_steps,
            langevin_scaling_per_dimension=True,
            device=device
        )
        
        # Learnable partition function
        self.log_z = nn.Parameter(torch.zeros(1).to(device))

    def forward_process(self, x0, t):
        alpha_bar = self.alpha_bars[t].view(-1, 1)
        noise = torch.randn_like(x0)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    def get_loss(self, x0, energy_fn):
        batch_size = x0.size(0)
        t = torch.randint(0, self.num_steps, (batch_size,)).to(self.device)
        
        # Forward diffusion
        xt = self.forward_process(x0, t)
        
        # GFN trajectory
        states, log_pfs, log_pbs, _ = self.gfn.get_trajectory_fwd(
            xt, None, energy_fn
        )
        
        # Trajectory balance loss
        log_r = -energy_fn(states[:, -1])
        loss = (self.log_z + log_pfs.sum(-1) - log_pbs.sum(-1) - log_r).pow(2).mean()
        
        return loss

    def sample(self, num_samples, energy_fn):
        return self.gfn.sample(num_samples, energy_fn)