"""
Simple test: Verify robot actually moves in Isaac Lab

This sends absolute joint position commands to move the robot.
If this works, the problem is with VLA integration.
If this doesn't work, the problem is with Isaac Lab environment setup.
"""

import argparse
from pathlib import Path
import sys
import torch
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.headless = False
args.num_envs = 1
args.enable_cameras = True  # REQUIRED for camera sensors!

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv

print("\n" + "="*70)
print("ROBOT MOVEMENT TEST")
print("="*70)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"\n1. Creating environment...")
env = FrankaPickPlaceEnv(
    simulation_app=simulation_app,
    num_envs=1,
    device=device,
    render=True,
    max_episode_length=200
)
print("✅ Environment created")

print(f"\n2. Resetting environment...")
obs = env.reset()
print(f"✅ Environment reset")
print(f"   Robot state shape: {obs['observation.state'].shape}")
print(f"   Robot state: {obs['observation.state'][0, :7].cpu().numpy()}")

# Get home position
home_position = obs['observation.state'][0, :7].clone()
print(f"\n3. Home position (7 DOF): {home_position.cpu().numpy()}")

print(f"\n4. Moving robot through 5 waypoints...")
print(f"   Watch the Isaac Lab viewer - robot should move!")

waypoints = [
    # Joint positions [j0, j1, j2, j3, j4, j5, j6]
    torch.tensor([[ 0.0,  0.0,  0.0, -1.5,  0.0,  1.5,  0.0]], device=device, dtype=torch.float32),  # Home
    torch.tensor([[ 0.5, -0.5,  0.0, -1.2,  0.0,  1.8,  0.5]], device=device, dtype=torch.float32),  # Right
    torch.tensor([[-0.5, -0.5,  0.0, -1.2,  0.0,  1.8, -0.5]], device=device, dtype=torch.float32),  # Left
    torch.tensor([[ 0.0, -0.8,  0.0, -0.8,  0.0,  2.0,  0.0]], device=device, dtype=torch.float32),  # Up
    torch.tensor([[ 0.0,  0.0,  0.0, -1.5,  0.0,  1.5,  0.0]], device=device, dtype=torch.float32),  # Home
]

for i, waypoint in enumerate(waypoints):
    print(f"\n   Waypoint {i+1}/5: {waypoint[0].cpu().numpy()}")

    # Hold position for 30 steps (1 second at 30Hz)
    for step in range(30):
        obs, done, info = env.step(waypoint)
        time.sleep(0.03)  # Slow down for visualization

        if step % 10 == 0:
            current_pos = obs['observation.state'][0, :7]
            error = torch.norm(current_pos - waypoint[0]).item()
            print(f"      Step {step}: error = {error:.4f}")

    print(f"   ✅ Reached waypoint {i+1}")
    time.sleep(0.5)

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")
print("\nDid you see the robot move between 5 different poses?")
print("   ✅ YES → Environment works, problem is VLA integration")
print("   ❌ NO  → Environment not working, Isaac Lab setup issue")
print("="*70 + "\n")

env.close()
simulation_app.close()
