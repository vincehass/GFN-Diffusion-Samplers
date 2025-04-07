import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

print("Starting wandb test script...")

# Set up wandb
try:
    wandb.init(
        project="gfn-diffusion-experiments",
        entity="nadhirvincenthassen",
        name="test_run",
        config={"test": True}
    )
    print(f"wandb initialized with run ID: {wandb.run.id}")
    print(f"wandb run URL: {wandb.run.get_url()}")
except Exception as e:
    print(f"Error initializing wandb: {e}")
    exit(1)

# Create and log a dummy metric
for i in range(10):
    wandb.log({"test_metric": i * 0.1})
    print(f"Logged test_metric: {i * 0.1}")

# Create and log a plot
try:
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title("Test Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("test_plot.png")
    wandb.log({"test_plot": wandb.Image("test_plot.png")})
    print("Logged test plot")
except Exception as e:
    print(f"Error creating/logging plot: {e}")

# Finish the run
try:
    wandb.finish()
    print("wandb run finished")
except Exception as e:
    print(f"Error finishing wandb run: {e}")

print("Test script completed!") 