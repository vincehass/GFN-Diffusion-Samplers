Looking at the wandb logs, we can analyze the trends you've noticed:

1. **Increasing Loss with High Guidance Scale**:

   - In the high guidance experiment, we observed that loss tends to increase over time. This is actually expected with high guidance scales (5.0, 8.0) because the model is being strongly pushed toward low-energy regions, potentially at the expense of natural diffusion dynamics.
   - The energy function acts as a bias, and when the guidance scale is high, it may dominate the natural diffusion process, causing the standard loss to increase.

2. **Flat Diversity Metrics**:

   - The diversity metrics appear flat in the wandb plots because they're only computed at certain evaluation intervals, not continuously during training.
   - When comparing final results, we see significant differences in diversity across experiments:
     - Ring energy: 3.84299 (GFN) vs 2.55776 (Standard)
     - 25GMM energy: 2.62127 (GFN) vs 2.55776 (Standard)

3. **Distribution/Coverage Metrics as Points**:

   - These metrics (entropy, grid coverage) also appear as points because they're only computed during the final evaluation phase.
   - The important comparison is between GFN-guided and standard diffusion:
     - For Ring energy: grid coverage of 0.12 (GFN) vs 0.18 (Standard)
     - For 25GMM: grid coverage of 0.20 (GFN) vs 0.18 (Standard)

4. **Energy Improvement**:
   - Despite the increasing loss, we see substantial energy improvements:
     - Ring energy: 88.66% mean improvement
     - 25GMM: 3.35% mean improvement

The key insights from these experiments:

1. GFN-guided diffusion with TB-Avg loss effectively targets low-energy regions, especially with the Ring energy function.

2. Higher guidance scales (5.0, 8.0) make the energy bias stronger, which explains the increasing loss but also leads to better energy values.

3. The diversity metrics show that GFN guidance doesn't necessarily reduce sample diversity - in fact, for the Ring energy, GFN samples are significantly more diverse (3.84 vs 2.56).

These results suggest that the GFlowNet-guided diffusion approach is working as intended, especially with the Ring energy function where we see both improved energy values and increased diversity.

'''bash
python -m gfn_diffusion.energy_sampling.train --epochs 20 --batch_size 32 --hidden_dim 64 --latent_dim 2 --num_timesteps 100 --device cpu --schedule_type linear --energy 25gmm --guidance_scale 2.0 --loss_type tb_avg --output_dir results/speed_debug --run_name "tb_avg_speed_debug" --wandb --eval_interval 1
'''
