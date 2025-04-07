#!/usr/bin/env python
"""
Script to verify wandb configuration and create a test run.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Test Weights & Biases integration")
    parser.add_argument("--project", type=str, default="gfn-diffusion-experiments", 
                        help="Wandb project name")
    parser.add_argument("--entity", type=str, default="nadhirvincenthassen", 
                        help="Wandb entity name")
    parser.add_argument("--offline", action="store_true",
                        help="Run in offline mode (no authentication required)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Testing Weights & Biases integration...")
    print(f"- Project: {args.project}")
    print(f"- Entity: {args.entity}")
    print(f"- Mode: {'offline' if args.offline else 'online'}")
    
    # Set offline mode if requested
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        print("- Running in offline mode (no authentication required)")
    
    # Import wandb after setting environment variables
    try:
        import wandb
        print(f"- wandb version: {wandb.__version__}")
    except ImportError:
        print("Error: wandb not installed")
        print("Please install wandb: pip install wandb")
        return False
    
    # Check login only if in online mode
    if not args.offline:
        try:
            # Try anonymous login if not authenticated
            status = wandb.login(anonymous="allow")
            if status:
                print("- Successfully logged into Weights & Biases")
            else:
                print("- Warning: Not logged into Weights & Biases")
                print("  Continuing anyway, but runs may be anonymous")
        except Exception as e:
            print(f"- Warning: Login check failed: {str(e)}")
            print("  Continuing anyway, but runs may be anonymous")
    
    # Create a simple test run
    try:
        run = wandb.init(
            project=args.project,
            entity=None if args.offline else args.entity,
            name="test_run",
            config={"test": True}
        )
        
        print("- Successfully created a test run")
        
        # Create and log a simple plot
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title("Test Plot")
        plt.xlabel("x")
        plt.ylabel("sin(x)")
        
        # Log the figure
        wandb.log({"test_plot": wandb.Image(plt)})
        plt.close()
        
        # Log some metrics
        for i in range(10):
            wandb.log({
                "test_metric": i * 0.1,
                "test_loss": 1.0 / (i + 1)
            })
        
        print("- Successfully logged test data")
        
        # Finish the run
        wandb.finish()
        print("- Successfully finished the test run")
        
        if not args.offline:
            print("\nTest successful! You should be able to see the test run at:")
            print(f"https://wandb.ai/{args.entity}/{args.project}/runs/{run.id}")
        else:
            print("\nTest successful in offline mode!")
            print(f"Offline run data saved to: {os.path.abspath('wandb')}")
        
        return True
    
    except Exception as e:
        print(f"Error during test run: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nWandb integration test passed!")
    else:
        print("\nWandb integration test failed. Please fix the issues above.")
        sys.exit(1) 