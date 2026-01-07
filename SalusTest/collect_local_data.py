"""
Collect local training data with VLA control for SALUS.

Collects episodes with:
- VLA controlling robot
- 18D signals extracted every step
- Success/failure labels
- Stored in Zarr format

Fast local version (50 episodes, ~30 minutes).
"""

import argparse
from pathlib import Path
import sys
import torch
import numpy as np
from datetime import datetime
import zarr

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_episodes", type=int, default=50, help="Episodes to collect")
parser.add_argument("--max_steps", type=int, default=100, help="Max steps per episode")
parser.add_argument("--output_dir", type=str, default="local_data", help="Output directory")
args = parser.parse_args()

# Headless mode for speed
args.headless = True
args.enable_cameras = True
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
from salus.core.vla.wrapper import SmolVLAEnsemble, EnhancedSignalExtractor

print("\n" + "="*70)
print("LOCAL DATA COLLECTION FOR SALUS")
print("="*70)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_path = output_dir / f"salus_data_{timestamp}.zarr"

print(f"\nOutput: {data_path}")
print(f"Episodes: {args.num_episodes}")
print(f"Max steps: {args.max_steps}")

# Initialize Zarr storage
print(f"\nInitializing Zarr storage...")
root = zarr.open(str(data_path), mode='w')

# Create datasets
max_total_steps = args.num_episodes * args.max_steps
signals_ds = root.create_dataset('signals', shape=(0, 18), chunks=(1000, 18), dtype='f4')
actions_ds = root.create_dataset('actions', shape=(0, 6), chunks=(1000, 6), dtype='f4')
robot_state_ds = root.create_dataset('robot_state', shape=(0, 7), chunks=(1000, 7), dtype='f4')
episode_ids_ds = root.create_dataset('episode_id', shape=(0,), chunks=(1000,), dtype='i4')
timesteps_ds = root.create_dataset('timestep', shape=(0,), chunks=(1000,), dtype='i4')
success_ds = root.create_dataset('success', shape=(0,), chunks=(1000,), dtype='bool')
done_ds = root.create_dataset('done', shape=(0,), chunks=(1000,), dtype='bool')

print(f"✅ Zarr storage ready")

# Load VLA
print(f"\nLoading VLA ensemble...")
vla = SmolVLAEnsemble(
    str(Path.home() / "models/smolvla/smolvla_base"),
    ensemble_size=3,
    device=device
)
print(f"✅ VLA loaded")

# Signal extractor
print(f"\nInitializing signal extractor...")
signal_extractor = EnhancedSignalExtractor(device)
print(f"✅ Signal extractor ready")

# Environment
print(f"\nCreating environment...")
env = FrankaPickPlaceEnv(
    simulation_app=simulation_app,
    num_envs=1,
    device=device,
    render=False,
    max_episode_length=args.max_steps
)
print(f"✅ Environment ready")

print(f"\n{'='*70}")
print("STARTING DATA COLLECTION")
print(f"{'='*70}\n")

total_steps = 0
successes = 0
failures = 0

for episode in range(args.num_episodes):
    print(f"\n{'─'*70}", flush=True)
    print(f"Episode {episode+1}/{args.num_episodes}", flush=True)
    print(f"{'─'*70}", flush=True)

    # Reset
    obs = env.reset()
    signal_extractor.reset()

    episode_data = {
        'signals': [],
        'actions': [],
        'robot_state': [],
        'timesteps': [],
        'success': False,
        'done': False
    }

    # Run episode
    for t in range(args.max_steps):
        # Prepare VLA observation
        obs_vla = {
            'observation.images.camera1': obs['observation.images.camera1'],
            'observation.images.camera2': obs['observation.images.camera2'],
            'observation.images.camera3': obs['observation.images.camera3'],
            'observation.state': obs['observation.state'][:, :6],
            'task': ['Pick up the red cube and place it in the blue zone']
        }

        # VLA inference
        with torch.no_grad():
            vla_output = vla(obs_vla, return_internals=True)

        # Extract signals
        robot_state = obs['observation.state'].to(device)
        signals = signal_extractor.extract(vla_output, robot_state=robot_state)

        # Get action
        action_6d = vla_output['action']

        # Store data
        episode_data['signals'].append(signals[0].cpu().numpy())
        episode_data['actions'].append(action_6d[0].cpu().numpy())
        episode_data['robot_state'].append(robot_state[0, :7].cpu().numpy())
        episode_data['timesteps'].append(t)

        # Apply action
        obs, done, info = env.step(action_6d)

        # Check done
        if done[0]:
            episode_data['success'] = info.get('success', [False])[0]
            episode_data['done'] = True
            break

    # Save episode data to Zarr
    episode_len = len(episode_data['signals'])

    signals_array = np.array(episode_data['signals'], dtype=np.float32)
    actions_array = np.array(episode_data['actions'], dtype=np.float32)
    robot_state_array = np.array(episode_data['robot_state'], dtype=np.float32)
    timesteps_array = np.array(episode_data['timesteps'], dtype=np.int32)
    episode_ids_array = np.full(episode_len, episode, dtype=np.int32)
    success_array = np.full(episode_len, episode_data['success'], dtype=bool)
    done_array = np.zeros(episode_len, dtype=bool)
    done_array[-1] = True  # Mark last step as done

    # Append to Zarr
    signals_ds.append(signals_array)
    actions_ds.append(actions_array)
    robot_state_ds.append(robot_state_array)
    episode_ids_ds.append(episode_ids_array)
    timesteps_ds.append(timesteps_array)
    success_ds.append(success_array)
    done_ds.append(done_array)

    total_steps += episode_len

    if episode_data['success']:
        successes += 1
        status = "✅ SUCCESS"
    else:
        failures += 1
        status = "❌ FAILURE"

    print(f"   Steps: {episode_len}", flush=True)
    print(f"   Status: {status}", flush=True)
    print(f"   Total: {total_steps} steps, {successes} successes, {failures} failures", flush=True)

    # Progress update every 10 episodes
    if (episode + 1) % 10 == 0:
        success_rate = successes / (episode + 1) * 100
        print(f"\n   📊 Progress: {episode+1}/{args.num_episodes} ({success_rate:.1f}% success rate)")

print(f"\n{'='*70}")
print("DATA COLLECTION COMPLETE")
print(f"{'='*70}")
print(f"\nStatistics:")
print(f"   Total episodes: {args.num_episodes}")
print(f"   Total steps: {total_steps}")
print(f"   Successes: {successes} ({successes/args.num_episodes*100:.1f}%)")
print(f"   Failures: {failures} ({failures/args.num_episodes*100:.1f}%)")
print(f"\nData saved to: {data_path}")
print(f"   Signals shape: {signals_ds.shape}")
print(f"   Actions shape: {actions_ds.shape}")

# Save metadata
root.attrs['num_episodes'] = args.num_episodes
root.attrs['total_steps'] = total_steps
root.attrs['successes'] = successes
root.attrs['failures'] = failures
root.attrs['signal_dim'] = 18
root.attrs['action_dim'] = 6
root.attrs['collection_date'] = timestamp

print(f"\n✅ Ready for SALUS training!")
print(f"{'='*70}\n")

env.close()
simulation_app.close()
