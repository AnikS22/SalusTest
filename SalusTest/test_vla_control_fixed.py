"""
Test VLA control with fixes:
1. 6D→7D conversion handled properly
2. Lighting added for cameras
3. Direct VLA control (no scripts)

This will show if VLA can actually control the robot successfully.
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
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
from salus.core.vla.wrapper import SmolVLAEnsemble

print("\n" + "="*70)
print("VLA CONTROL TEST (FIXED)")
print("="*70)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading VLA...")
model_path = Path.home() / "models" / "smolvla" / "smolvla_base"

if not model_path.exists():
    print(f"❌ VLA not found at {model_path}")
    sys.exit(1)

vla = SmolVLAEnsemble(
    model_path=str(model_path),
    ensemble_size=1,
    device=device
)
print(f"✅ VLA loaded")

print(f"\n2. Creating environment with fixed lighting...")
env = FrankaPickPlaceEnv(
    simulation_app=simulation_app,
    num_envs=1,
    device=device,
    render=True,
    max_episode_length=200
)
print(f"✅ Environment created")

print(f"\n3. Running episode with VLA control...")
obs = env.reset()
print(f"✅ Reset complete")

print(f"\n📹 Watch the Isaac Lab viewer!")
print(f"🤖 VLA is now controlling the robot directly")
print(f"🎯 Task: Pick up red cube\n")

episode_reward = 0
success = False

for t in range(200):
    # Prepare VLA observation
    obs_vla = {
        'observation.images.camera1': obs['observation.images.camera1'],
        'observation.images.camera2': obs['observation.images.camera2'],
        'observation.images.camera3': obs['observation.images.camera3'],
        'observation.state': obs['observation.state'][:, :6],  # VLA expects 6D state
        'task': ['Pick up the red cube and place it in the blue zone']
    }

    # VLA inference
    with torch.no_grad():
        vla_output = vla(obs_vla, return_internals=True)

    # Get 6D action
    action_6d = vla_output['action']  # (1, 6)

    # Environment will automatically convert 6D → 7D
    obs, done, info = env.step(action_6d)

    episode_reward += info.get('reward', torch.tensor([0.0]))[0].item()

    if t % 20 == 0:
        robot_pos = obs['observation.state'][0, :7].cpu().numpy()
        action_mag = torch.norm(action_6d[0]).item()
        epistemic_unc = vla_output['epistemic_uncertainty'][0].item()

        print(f"Step {t:3d}: "
              f"action_mag={action_mag:.4f}, "
              f"epistemic_unc={epistemic_unc:.4f}, "
              f"reward={episode_reward:.2f}")

    if done[0]:
        success = info.get('success', [False])[0]
        break

    time.sleep(0.03)  # 30 Hz

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\nEpisode length: {t+1} steps")
print(f"Total reward: {episode_reward:.2f}")
print(f"Success: {'✅ YES' if success else '❌ NO'}")

if success:
    print(f"\n🎉 VLA SUCCESSFULLY PICKED UP THE CUBE!")
elif episode_reward > 0:
    print(f"\n⚠️  VLA made progress but didn't complete task")
else:
    print(f"\n❌ VLA didn't make progress")

print(f"\n{'='*70}\n")

env.close()
simulation_app.close()
