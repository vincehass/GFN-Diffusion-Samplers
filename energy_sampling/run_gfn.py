import argparse
import torch
import os
from datetime import datetime
from plot_utils import viz_many_well, plot_contours, plot_samples
from energies import ManyWell, NineGaussianMixture, TwentyFiveGaussianMixture
import torch
import torch.nn as nn
from models.gfn import GFN  # Assuming you have the GFN class in gfn.py
from buffer import ReplayBuffer
from matplotlib import pyplot as plt


class GFNDiffusionConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='GFN Diffusion Training')
        self.parser.add_argument('--energy', type=str, default='many_well',
                                choices=('many_well', '9gmm', '25gmm'))
        self.parser.add_argument('--input_dim', type=int, default=2)
        self.parser.add_argument('--num_steps', type=int, default=100)
        self.parser.add_argument('--hidden_dim', type=int, default=256)
        self.parser.add_argument('--harmonics_dim', type=int, default=64)
        self.parser.add_argument('--t_dim', type=int, default=64)
        self.parser.add_argument('--s_emb_dim', type=int, default=64)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--lr_logz', type=float, default=1e-2)
        self.parser.add_argument('--buffer_size', type=int, default=int(1e5))
        self.parser.add_argument('--epochs', type=int, default=10000)
        self.parser.add_argument('--log_interval', type=int, default=100)
        self.parser.add_argument('--eval_interval', type=int, default=500)
        self.parser.add_argument('--visualize', action='store_true')
        self.parser.add_argument('--use_wandb', action='store_true')
        self.parser.add_argument('--save_dir', type=str, default='experiments')
        self.parser.add_argument('--device', type=str, 
                               default='cuda' if torch.cuda.is_available() else 'cpu')

    def parse(self):
        args = self.parser.parse_args()
        args.save_dir = os.path.join(args.save_dir, 
                                   f"{args.energy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        return args


class GFNDiffusion(nn.Module):
    def __init__(self, input_dim, num_steps=100, hidden_dim=256,
                 harmonics_dim=64, t_dim=64, s_emb_dim=64, device='cuda'):
        super().__init__()
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.device = device
        
        # Configure noise schedule
        self.betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0)
        
        # Initialize GFN components
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
        """Forward diffusion process q(x_t|x_0)"""
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


class GFNDiffusionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize energy model
        self.energy = self._create_energy_model(config.energy)
        
        # Initialize GFN diffusion model
        self.model = GFNDiffusion(
            input_dim=config.input_dim,
            num_steps=config.num_steps,
            hidden_dim=config.hidden_dim,
            harmonics_dim=config.harmonics_dim,
            t_dim=config.t_dim,
            s_emb_dim=config.s_emb_dim,
            device=self.device
        ).to(self.device)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            config.buffer_size, 
            self.device,
            log_reward=lambda x: -self.energy.energy(x),
            batch_size=config.batch_size
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': [self.model.log_z], 'lr': config.lr_logz}
        ], lr=config.lr)
        
        # Setup logging
        if config.use_wandb:
            import wandb
            wandb.init(project="gfn-diffusion", config=vars(config))
            self.wandb = wandb
        else:
            self.wandb = None
            
        os.makedirs(config.save_dir, exist_ok=True)

    def _create_energy_model(self, energy_name):
        energy_map = {
            'many_well': ManyWell,
            '9gmm': NineGaussianMixture,
            '25gmm': TwentyFiveGaussianMixture
        }
        return energy_map[energy_name](device=self.device)

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(self.device)
            
            # Train step
            loss = self._train_step(data)
            total_loss += loss.item()

            # Logging
            if batch_idx % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Train Epoch: {self.epoch} [{batch_idx}/{len(data_loader)}] "
                      f"Loss: {avg_loss:.4f}")
                
                if self.wandb:
                    self.wandb.log({"train/loss": avg_loss}, step=self.global_step)
                
            self.global_step += 1

        return total_loss / len(data_loader)

    def _train_step(self, data_batch):
        # Sample from buffer
        if len(self.buffer) > 0:
            buffer_samples, _ = self.buffer.sample()
            inputs = torch.cat([data_batch, buffer_samples])
        else:
            inputs = data_batch
            
        # Forward pass and loss calculation
        loss = self.model.get_loss(inputs, self.energy.energy)
        
        # Update buffer with new samples
        with torch.no_grad():
            samples = self.model.gfn.sample(data_batch.size(0), self.energy.energy)
            log_r = -self.energy.energy(samples)
            self.buffer.add(samples, log_r)
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            # Generate samples
            samples = self.model.gfn.sample(1000, self.energy.energy)
            
            # Calculate metrics
            metrics = {
                'energy_mean': self.energy.energy(samples).mean().item(),
                'energy_std': self.energy.energy(samples).std().item()
            }
            
            # Visualization
            if self.config.visualize:
                fig, ax = plot_contours(self.energy.energy)
                plot_samples(samples, ax=ax)
                
                if self.wandb:
                    self.wandb.log({"samples": self.wandb.Image(fig)})
                
                plt.savefig(os.path.join(self.config.save_dir, f'samples_{self.epoch}.png'))
                plt.close()
            
            return metrics

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.config.save_dir, f'checkpoint_{self.epoch}.pt')
        torch.save({
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'buffer': self.buffer.state_dict(),
        }, ckpt_path)

def main():
    config = GFNDiffusionConfig().parse()
    trainer = GFNDiffusionTrainer(config)
    
    # Create dummy dataset (replace with real data)
    dataset = torch.utils.data.TensorDataset(torch.randn(1000, config.input_dim))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    for epoch in range(config.epochs):
        trainer.epoch = epoch
        trainer.global_step = epoch * len(data_loader)
        
        # Training
        train_loss = trainer.train_epoch(data_loader)
        
        # Evaluation
        if epoch % config.eval_interval == 0:
            eval_metrics = trainer.evaluate()
            print(f"Evaluation [{epoch}/{config.epochs}]: {eval_metrics}")
            
            if trainer.wandb:
                trainer.wandb.log({"eval": eval_metrics}, step=trainer.global_step)
            
            trainer.save_checkpoint()

if __name__ == '__main__':
    main()