"""
A100-Optimized Parallel Data Collection for SALUS
Runs multiple Isaac Sim environments in parallel to maximize GPU utilization.
"""

# CRITICAL: Only minimal imports before AppLauncher
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Isaac Lab AppLauncher (MUST be before other imports)
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Parallel data collection for A100")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_episodes", type=int, default=500, help="Total episodes to collect")
parser.add_argument("--num_envs", type=int, default=8, help="Parallel environments (recommend 8 for A100)")
parser.add_argument("--save_dir", type=str, default="a100_data/training_500eps", help="Save directory")
parser.add_argument("--config", type=str, default="configs/a100_config.yaml", help="Config file")
parser.add_argument("--batch_collection", action="store_true", help="Collect in batches of num_envs")
args = parser.parse_args()

# Create AppLauncher
print(f"Creating AppLauncher with {args.num_envs} parallel environments...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"✅ AppLauncher created successfully!")

# NOW safe to import everything else
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

from salus.core.vla.wrapper import SmolVLAEnsemble, EnhancedSignalExtractor
from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
from salus.data.recorder import ScalableDataRecorder
from salus.utils.config import load_config

print("✅ All modules imported")


def collect_episode_batch(
    env, vla, signal_extractor,
    start_episode_id: int,
    num_episodes_batch: int,
    max_episode_length: int,
    use_vla: bool = True
):
    """
    Collect a batch of episodes in parallel environments.

    Args:
        env: FrankaPickPlaceEnv with multiple parallel environments
        vla: VLA ensemble
        signal_extractor: Signal extractor
        start_episode_id: Starting episode ID
        num_episodes_batch: Number of episodes in this batch
        max_episode_length: Max timesteps per episode
        use_vla: Use VLA or random actions

    Returns:
        batch_data: List of episode data dicts
    """
    num_envs = env.num_envs
    batch_data = []

    # Reset all environments
    obs = env.reset()
    signal_extractor.reset()

    # Initialize storage for each environment
    env_actions = [[] for _ in range(num_envs)]
    env_states = [[] for _ in range(num_envs)]
    env_images = [[] for _ in range(num_envs)]
    env_signals = [[] for _ in range(num_envs)]
    env_lengths = [0] * num_envs
    env_done = [False] * num_envs

    # Collect episodes
    for step in range(max_episode_length):
        # Get actions from VLA for all environments
        if use_vla:
            vla_device = vla.device
            obs_vla = {
                'observation.images.camera1': obs['observation.images.camera1'].to(vla_device).float() / 255.0,
                'observation.images.camera2': obs['observation.images.camera2'].to(vla_device).float() / 255.0,
                'observation.images.camera3': obs['observation.images.camera3'].to(vla_device).float() / 255.0,
                'observation.state': obs['observation.state'].to(vla_device),
                'task': obs['task']
            }

            with torch.no_grad():
                action_dict = vla(obs_vla, return_internals=True)

            action = action_dict['action'].to(obs['observation.state'].device)

            # Extract 18D signals from VLA output + robot state
            robot_state = obs['observation.state'].to(vla_device)
            signals = signal_extractor.extract(action_dict, robot_state=robot_state)
        else:
            action = torch.randn(num_envs, 7, device=env.device) * 0.1
            signals = torch.zeros(num_envs, 18, device=env.device)  # 18D signals now

        # Store data for active environments
        for env_idx in range(num_envs):
            if not env_done[env_idx]:
                env_actions[env_idx].append(action[env_idx].cpu().numpy())
                env_states[env_idx].append(obs['observation.state'][env_idx].cpu().numpy())

                # Store images (3 cameras)
                images = np.stack([
                    obs['observation.images.camera1'][env_idx].cpu().numpy(),
                    obs['observation.images.camera2'][env_idx].cpu().numpy(),
                    obs['observation.images.camera3'][env_idx].cpu().numpy()
                ], axis=0)
                env_images[env_idx].append(images)

                env_signals[env_idx].append(signals[env_idx].cpu().numpy())
                env_lengths[env_idx] += 1

        # Step all environments
        obs, reward, done, info = env.step(action)

        # Check which environments finished
        for env_idx in range(num_envs):
            if done[env_idx].item() and not env_done[env_idx]:
                env_done[env_idx] = True

        # If all environments done, break
        if all(env_done):
            break

    # Process collected data for each environment
    for env_idx in range(min(num_envs, num_episodes_batch)):
        episode_id = start_episode_id + env_idx

        # Convert to numpy arrays
        actions = np.stack(env_actions[env_idx], axis=0) if env_actions[env_idx] else np.zeros((0, 7))
        states = np.stack(env_states[env_idx], axis=0) if env_states[env_idx] else np.zeros((0, 7))
        images = np.stack(env_images[env_idx], axis=0) if env_images[env_idx] else np.zeros((0, 3, 3, 256, 256))
        signals = np.stack(env_signals[env_idx], axis=0) if env_signals[env_idx] else np.zeros((0, 12))

        # Pad to max_episode_length
        actual_length = env_lengths[env_idx]
        pad_length = max_episode_length - actual_length

        if pad_length > 0:
            actions = np.pad(actions, ((0, pad_length), (0, 0)))
            states = np.pad(states, ((0, pad_length), (0, 0)))
            images = np.pad(images, ((0, pad_length), (0, 0), (0, 0), (0, 0), (0, 0)))
            signals = np.pad(signals, ((0, pad_length), (0, 0)))

        # Get episode outcome
        success = info['success'][env_idx].item() if 'success' in info else False
        failure_type = info['failure_type'][env_idx].item() if 'failure_type' in info else 3

        episode_data = {
            'episode_id': episode_id,
            'actions': actions[np.newaxis, ...],  # Add batch dimension
            'states': states[np.newaxis, ...],
            'images': images.transpose(1, 0, 2, 3, 4)[np.newaxis, ...],  # (T, 3, C, H, W) -> (1, T, 3, C, H, W)
            'signals': signals[np.newaxis, ...],
            'success': success,
            'failure_type': failure_type,
            'episode_length': actual_length
        }

        batch_data.append(episode_data)

    return batch_data


def main():
    """Main parallel data collection loop."""

    print("=" * 70)
    print(f"SALUS A100 Parallel Data Collection")
    print("=" * 70)

    # Configuration
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"⚠️  Config not found: {config_path}, using base_config.yaml")
        config_path = project_root / "configs" / "base_config.yaml"

    config = load_config(config_path)
    max_episode_length = config['data_collection.max_episode_length']

    # Setup save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / args.save_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📋 Configuration:")
    print(f"   Total episodes: {args.num_episodes}")
    print(f"   Parallel environments: {args.num_envs}")
    print(f"   Max episode length: {max_episode_length}")
    print(f"   Batch collection: {args.batch_collection}")
    print(f"   Save directory: {save_dir}")

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    # Load VLA ensemble
    print(f"\n🤖 Loading VLA Ensemble...")
    vla = SmolVLAEnsemble(
        model_path=config['vla.model_path'],
        ensemble_size=config['vla.ensemble_size'],
        device=config['vla.device']
    )
    signal_extractor = EnhancedSignalExtractor(device=config['vla.device'])
    print(f"   ✅ VLA ensemble loaded ({config['vla.ensemble_size']} models)")

    # Initialize environment with multiple parallel envs
    print(f"\n🏗️  Initializing Franka Environment ({args.num_envs} parallel)...")
    env_device = args.device if hasattr(args, 'device') else "cuda:0"
    env = FrankaPickPlaceEnv(
        simulation_app=simulation_app,
        num_envs=args.num_envs,
        device=env_device,
        render=args.enable_cameras if args.headless else True,
        max_episode_length=max_episode_length
    )
    print(f"   ✅ Environment initialized")

    # Initialize data recorder
    print(f"\n💾 Initializing Data Recorder...")
    recorder = ScalableDataRecorder(
        save_dir=save_dir,
        max_episodes=args.num_episodes,
        max_episode_length=max_episode_length,
        image_shape=(3, 256, 256),
        state_dim=7,
        action_dim=7,
        signal_dim=12,
        num_cameras=3,
        chunk_size=1000,
        compression="zstd"
    )
    print(f"   ✅ Data recorder ready")

    # Save configuration
    config.save(save_dir / "config.yaml")
    print(f"   ✅ Config saved")

    print(f"\n🚀 Starting Parallel Data Collection...")
    print("=" * 70)

    # Collection statistics
    success_count = 0
    failure_count = 0

    try:
        # Collect in batches
        num_batches = (args.num_episodes + args.num_envs - 1) // args.num_envs
        episode_id = 0

        for batch_idx in tqdm(range(num_batches), desc="Collecting batches"):
            # Number of episodes in this batch
            num_episodes_batch = min(args.num_envs, args.num_episodes - episode_id)

            # Collect batch
            batch_data = collect_episode_batch(
                env, vla, signal_extractor,
                episode_id, num_episodes_batch,
                max_episode_length, use_vla=True
            )

            # Store episodes
            for ep_data in batch_data:
                # Create horizon labels (placeholder)
                horizon_labels = np.zeros((1, max_episode_length, 4, 4), dtype=np.float32)

                recorder.store_episode(
                    episode_idx=ep_data['episode_id'],
                    actions=ep_data['actions'],
                    states=ep_data['states'],
                    images=ep_data['images'],
                    signals=ep_data['signals'],
                    horizon_labels=horizon_labels,
                    success=ep_data['success'],
                    failure_type=ep_data['failure_type'],
                    episode_length=ep_data['episode_length']
                )

                # Update statistics
                if ep_data['success']:
                    success_count += 1
                else:
                    failure_count += 1

            episode_id += num_episodes_batch

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"\n   Progress: {episode_id}/{args.num_episodes} episodes")
                print(f"   Success: {success_count}, Failure: {failure_count}")
                print(f"   Success rate: {success_count/episode_id*100:.1f}%")

                # Checkpoint
                recorder.save_checkpoint()

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final checkpoint
        recorder.save_checkpoint()

        # Final statistics
        total_episodes = success_count + failure_count
        print(f"\n{'='*70}")
        print("Collection Complete!")
        print(f"{'='*70}")
        print(f"\n📊 Final Statistics:")
        print(f"   Total episodes: {total_episodes}")
        print(f"   Success: {success_count} ({success_count/total_episodes*100:.1f}%)")
        print(f"   Failure: {failure_count} ({failure_count/total_episodes*100:.1f}%)")
        print(f"   Saved to: {save_dir}")

        # Close
        recorder.close()
        env.close()

        print(f"\n✅ Parallel data collection finished!")


if __name__ == "__main__":
    main()
