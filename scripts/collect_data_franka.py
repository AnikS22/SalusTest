"""
SALUS Data Collection with Real Franka Environment
Collects data using IsaacSim with Franka robot and real physics

Usage:
    # Headless with rendering (recommended for servers)
    python scripts/collect_data_franka.py --headless --enable_cameras --num_episodes 10

    # With GUI (requires display)
    python scripts/collect_data_franka.py --num_episodes 10
"""

# CRITICAL: Only import argparse before AppLauncher!
# All other imports must come AFTER AppLauncher creation
import argparse
from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# NOTE: AppLauncher must be created BEFORE importing other modules
from isaaclab.app import AppLauncher

# Setup argument parser
parser = argparse.ArgumentParser(description='SALUS Data Collection with Franka')
parser.add_argument('--num_episodes', type=int, default=10,
                    help='Number of episodes to collect')
parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                    help='Path to config file')
parser.add_argument('--save_dir', type=str, default=None,
                    help='Data save directory')

# Add AppLauncher arguments (--headless, --enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)
print("Parsing arguments...", flush=True)
args = parser.parse_args()
print(f"Arguments parsed: {args}", flush=True)

# Create AppLauncher early (this initializes IsaacSim)
print("Creating AppLauncher...", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app  # CRITICAL: Access the simulation_app property
print(f"AppLauncher created successfully! simulation_app = {simulation_app}")

# NOW we can import other modules (AFTER AppLauncher creation!)
print("Importing Python stdlib and third-party modules...", flush=True)
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
print("  ✓ Python stdlib and third-party modules imported", flush=True)

print("Importing custom SALUS modules...", flush=True)
print("  - Importing wrapper.py...", flush=True)
from salus.core.vla.wrapper import SmolVLAEnsemble, SignalExtractor
print("  ✓ wrapper.py imported", flush=True)

print("  - Importing franka_pick_place_env.py...", flush=True)
from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
print("  ✓ franka_pick_place_env.py imported", flush=True)

print("  - Importing recorder.py...", flush=True)
from salus.data.recorder import ScalableDataRecorder
print("  ✓ recorder.py imported", flush=True)

print("  - Importing config.py...", flush=True)
from salus.utils.config import load_config, get_data_path
print("  ✓ config.py imported", flush=True)

print("✅ All modules imported successfully!", flush=True)


def compute_horizon_labels(labels_list, horizons=[6, 10, 13, 16]):
    """Compute multi-horizon failure prediction labels"""
    T = len(labels_list)
    horizon_labels = np.zeros((T, len(horizons), 4), dtype=np.float32)

    # Find failure timestep
    failure_time = None
    failure_type = None

    for t, label in enumerate(labels_list):
        if not label['success']:
            failure_time = t
            failure_type = int(label['failure_type']) if isinstance(label['failure_type'], (int, torch.Tensor)) else 3
            break

    if failure_time is not None:
        # Label timesteps before failure
        for t in range(failure_time):
            steps_until_failure = failure_time - t

            for h_idx, horizon in enumerate(horizons):
                if steps_until_failure <= horizon:
                    horizon_labels[t, h_idx, failure_type] = 1.0

    return horizon_labels


def collect_episode(env, vla, signal_extractor, episode_id, max_steps, use_vla=True):
    """Collect a single episode"""
    # Reset environment and signal extractor
    obs = env.reset()
    signal_extractor.reset()

    # Storage for episode
    episode_data = {
        'images': [],
        'states': [],
        'actions': [],
        'signals': [],
        'labels': []
    }

    step = 0
    done = torch.zeros(1, dtype=torch.bool)

    while not done.any() and step < max_steps:
        if use_vla and vla is not None:
            # Use VLA forward pass
            with torch.no_grad():
                # Move observations to VLA device and convert uint8 images to float32 [0, 1]
                vla_device = vla.device

                # Debug: print devices
                if step == 0:
                    print(f"   DEBUG: VLA device = {vla_device}")
                    print(f"   DEBUG: obs camera1 device before = {obs['observation.images.camera1'].device}")

                cam1 = obs['observation.images.camera1'].to(vla_device).float() / 255.0
                cam2 = obs['observation.images.camera2'].to(vla_device).float() / 255.0
                cam3 = obs['observation.images.camera3'].to(vla_device).float() / 255.0

                if step == 0:
                    print(f"   DEBUG: cam1 device after = {cam1.device}")

                obs_vla = {
                    'observation.images.camera1': cam1,
                    'observation.images.camera2': cam2,
                    'observation.images.camera3': cam3,
                    'observation.state': obs['observation.state'].to(vla_device),
                    'task': obs['task']
                }

                # Get action from VLA
                action_dict = vla(obs_vla)
                action = action_dict['action'].to(obs['observation.state'].device)  # Move back to env device

                # Pad action to 7 dimensions if needed
                if action.shape[-1] == 6:
                    action = torch.cat([action, torch.zeros(action.shape[0], 1, device=action.device)], dim=-1)
                
                # Scale action - SmolVLA outputs normalized actions, scale to reasonable joint space
                # Franka joint limits are roughly [-2.9, 2.9] radians, so scale by ~1.0-2.0
                action = action * 0.5  # Scale down for smoother control
                
                if step == 0:
                    print(f"   DEBUG: Action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]")

                # Extract signals
                signals = signal_extractor.extract(action_dict)
        else:
            # Use random actions for testing
            action = torch.randn(1, 7, device="cuda:0") * 0.1
            signals = torch.randn(1, 12, device="cuda:0")

        # Step environment
        next_obs, done, info = env.step(action)

        # Store data
        # Images from env are (C, H, W), need to convert to (H, W, C) for zarr
        images_stacked = torch.stack([
            obs['observation.images.camera1'][0].permute(1, 2, 0),  # (C,H,W) -> (H,W,C)
            obs['observation.images.camera2'][0].permute(1, 2, 0),
            obs['observation.images.camera3'][0].permute(1, 2, 0)
        ], dim=0)  # Result: (3, H, W, C) = (num_cameras, 256, 256, 3)

        episode_data['images'].append(images_stacked.cpu().numpy().astype(np.uint8))
        episode_data['states'].append(obs['observation.state'][0].cpu().numpy().astype(np.float32))
        episode_data['actions'].append(action[0].cpu().numpy().astype(np.float32))
        episode_data['signals'].append(signals[0].cpu().numpy().astype(np.float32))

        episode_data['labels'].append({
            'success': bool(info['success'][0].item()),
            'failure_type': int(info['failure_type'][0].item())
        })

        obs = next_obs
        step += 1

    # Convert lists to numpy arrays
    episode_arrays = {
        'images': np.stack(episode_data['images'], axis=0),
        'states': np.stack(episode_data['states'], axis=0),
        'actions': np.stack(episode_data['actions'], axis=0),
        'signals': np.stack(episode_data['signals'], axis=0),
        'horizon_labels': compute_horizon_labels(episode_data['labels'])
    }

    # Episode metadata
    final_label = episode_data['labels'][-1]
    episode_info = {
        'episode_id': episode_id,
        'success': final_label['success'],
        'failure_type': final_label['failure_type'] if not final_label['success'] else None,
        'episode_length': step,
        'timestamp': datetime.now().isoformat()
    }

    return episode_arrays, episode_info


def main():
    print("="*70)
    print("SALUS Data Collection - Real Franka Environment")
    print("="*70)

    # Load configuration
    config = load_config(args.config)
    num_episodes = args.num_episodes
    max_episode_length = config['data_collection.max_episode_length']

    print(f"\n📋 Configuration:")
    print(f"   Episodes to collect: {num_episodes}")
    print(f"   Max episode length: {max_episode_length}")
    print(f"   Headless: {args.headless}")
    print(f"   Enable cameras: {args.enable_cameras}")

    # Setup data directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = get_data_path(config, 'raw_episodes')

    save_dir = save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Save directory: {save_dir}")

    # Initialize VLA
    print("\n🤖 Loading VLA Ensemble...")
    try:
        vla = SmolVLAEnsemble(
            model_path=config['vla.model_path'],
            ensemble_size=config['vla.ensemble_size'],
            device="cuda:0"
        )
        signal_extractor = SignalExtractor()
        use_vla = True
        print(f"   ✅ VLA ensemble loaded ({config['vla.ensemble_size']} models)")
    except Exception as e:
        print(f"   ⚠️  Failed to load VLA: {e}")
        print("   Using random actions instead")
        vla = None
        signal_extractor = SignalExtractor()
        use_vla = False

    # Initialize Franka environment
    print(f"\n🏗️  Initializing Franka Environment...")
    # Use the device from args (set by AppLauncher)
    env_device = args.device if hasattr(args, 'device') else "cuda:0"
    env = FrankaPickPlaceEnv(
        simulation_app=simulation_app,  # Pass the simulation_app from AppLauncher
        num_envs=1,
        device=env_device,
        render=args.enable_cameras if args.headless else True,
        max_episode_length=max_episode_length
    )

    # Initialize data recorder
    print(f"\n💾 Initializing Data Recorder...")
    recorder = ScalableDataRecorder(
        save_dir=save_dir,
        max_episodes=num_episodes,
        max_episode_length=max_episode_length,
        chunk_size=config.get('data_collection.chunk_size', 50)
    )

    # Save config
    config.save(save_dir / "config.yaml")
    print(f"   ✅ Config saved")

    # Data collection loop
    print(f"\n🚀 Starting Data Collection...")
    print(f"{'='*70}\n")

    success_count = 0
    failure_count = 0

    try:
        for episode_id in tqdm(range(num_episodes), desc="Collecting episodes"):
            # Collect episode
            episode_data, episode_info = collect_episode(
                env, vla, signal_extractor,
                episode_id, max_episode_length,
                use_vla=use_vla
            )

            # Record episode
            recorder.record_episode(episode_data, episode_info)

            # Update statistics
            if episode_info['success']:
                success_count += 1
            else:
                failure_count += 1

            # Print progress every 10 episodes
            if (episode_id + 1) % 10 == 0:
                stats = recorder.get_statistics()
                print(f"\n   Progress: {episode_id + 1}/{num_episodes} episodes")
                print(f"   Success: {success_count}, Failure: {failure_count}")
                print(f"   Storage: {stats['storage_size_gb']:.2f} GB")

            # Checkpoint every 50 episodes
            if (episode_id + 1) % 50 == 0:
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
        print(f"\n{'='*70}")
        print("Collection Complete!")
        print(f"{'='*70}")

        stats = recorder.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Total episodes: {stats['num_episodes']}")
        print(f"   Success: {success_count}")
        print(f"   Failure: {failure_count}")
        print(f"   Total timesteps: {stats['total_timesteps']}")
        print(f"   Storage size: {stats['storage_size_gb']:.2f} GB")
        print(f"   Saved to: {save_dir}")

        # Close recorder and environment
        recorder.close()
        env.close()

        print(f"\n✅ Data collection finished!")


if __name__ == "__main__":
    main()
