"""
SALUS Data Collection Script
Scalable, production-ready data collection for VLA safety research

Usage:
    # Test with dummy environment (fast)
    python scripts/collect_data.py --num_episodes 10 --use_dummy

    # Production with real IsaacSim (requires sim running)
    python scripts/collect_data.py --num_episodes 500

    # Custom config
    python scripts/collect_data.py --config configs/custom_config.yaml
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import json

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from salus.core.vla.wrapper import SmolVLAEnsemble, SignalExtractor
from salus.simulation.isaaclab_env import SimplePickPlaceEnv
from salus.data.recorder import ScalableDataRecorder
from salus.utils.config import load_config, get_data_path, get_log_path


def compute_horizon_labels(labels_list, horizons=[6, 10, 13, 16]):
    """
    Compute multi-horizon failure prediction labels

    Args:
        labels_list: List of dicts with 'success' and 'failure_type'
        horizons: Lookahead steps [200ms, 300ms, 400ms, 500ms] at 30Hz

    Returns:
        horizon_labels: (T, 4, 4) array - T timesteps, 4 horizons, 4 failure types
    """
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
                    # Will fail within this horizon
                    horizon_labels[t, h_idx, failure_type] = 1.0

    return horizon_labels


def collect_episode(
    env,
    vla,
    signal_extractor,
    episode_id: int,
    max_steps: int,
    use_vla: bool = True
):
    """
    Collect a single episode

    Returns:
        episode_data: Dict with all collected data
        episode_info: Dict with metadata
    """
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
                # Convert uint8 images to float32 [0, 1] for VLA
                obs_vla = {
                    'observation.images.camera1': obs['observation.images.camera1'].float() / 255.0,
                    'observation.images.camera2': obs['observation.images.camera2'].float() / 255.0,
                    'observation.images.camera3': obs['observation.images.camera3'].float() / 255.0,
                    'observation.state': obs['observation.state'],
                    'task': obs['task']
                }

                # Get action from VLA
                action_dict = vla(obs_vla)
                action = action_dict['action']  # (1, action_dim)

                # Pad action to 7 dimensions if needed (VLA outputs 6, Franka needs 7)
                if action.shape[-1] == 6:
                    # Pad with zeros for the 7th joint (or duplicate last value)
                    action = torch.cat([action, torch.zeros(action.shape[0], 1, device=action.device)], dim=-1)

                # Extract signals from ensemble
                signals = signal_extractor.extract(action_dict)
        else:
            # Use random actions for testing
            action = torch.randn(1, 7, device="cuda:0") * 0.1
            signals = torch.randn(1, 12, device="cuda:0")

        # Step environment
        next_obs, done, info = env.step(action)

        # Store data
        # Convert images from (1, 3, 256, 256) to (3, 3, 256, 256) for 3 cameras
        images_stacked = torch.stack([
            obs['observation.images.camera1'][0],
            obs['observation.images.camera2'][0],
            obs['observation.images.camera3'][0]
        ], dim=0)  # (3, 3, 256, 256)

        episode_data['images'].append(images_stacked.cpu().numpy().astype(np.uint8))
        episode_data['states'].append(obs['observation.state'][0].cpu().numpy().astype(np.float32))
        episode_data['actions'].append(action[0].cpu().numpy().astype(np.float32))

        # Store extracted signals
        episode_data['signals'].append(signals[0].cpu().numpy().astype(np.float32))

        # Store labels
        episode_data['labels'].append({
            'success': bool(info['success'][0].item()),
            'failure_type': int(info['failure_type'][0].item())
        })

        obs = next_obs
        step += 1

    # Debug: Check shapes before stacking
    if len(episode_data['states']) > 0:
        print(f"Debug: First state shape: {episode_data['states'][0].shape}")
        print(f"Debug: First action shape: {episode_data['actions'][0].shape}")
        print(f"Debug: First signal shape: {episode_data['signals'][0].shape}")

        # Check for inconsistent shapes
        state_shapes = [s.shape for s in episode_data['states']]
        if len(set(state_shapes)) > 1:
            print(f"Warning: Inconsistent state shapes: {set(state_shapes)}")

    # Convert lists to numpy arrays
    episode_arrays = {
        'images': np.stack(episode_data['images'], axis=0),  # (T, 3, 3, 256, 256)
        'states': np.stack(episode_data['states'], axis=0),  # (T, 7)
        'actions': np.stack(episode_data['actions'], axis=0),  # (T, 7)
        'signals': np.stack(episode_data['signals'], axis=0),  # (T, 12)
        'horizon_labels': compute_horizon_labels(episode_data['labels'])  # (T, 4, 4)
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
    parser = argparse.ArgumentParser(description='SALUS Data Collection')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Number of episodes to collect (overrides config)')
    parser.add_argument('--use_dummy', action='store_true',
                        help='Use dummy environment (no IsaacSim required)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Data save directory (overrides config)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--render', action='store_true',
                        help='Enable GUI rendering (requires X11 display)')

    args = parser.parse_args()

    # Load configuration
    print("="*70)
    print("SALUS Data Collection")
    print("="*70)
    print(f"\n📋 Loading configuration from: {args.config}")

    config = load_config(args.config)

    # Override config with command line args
    if args.num_episodes is not None:
        config.data['data_collection']['num_episodes'] = args.num_episodes

    num_episodes = config['data_collection.num_episodes']
    max_episode_length = config['data_collection.max_episode_length']
    num_envs = 1  # Start with 1 for simplicity

    print(f"   Episodes to collect: {num_episodes}")
    print(f"   Max episode length: {max_episode_length}")
    print(f"   Using dummy environment: {args.use_dummy}")
    print(f"   GUI rendering: {args.render}")

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
            device=args.device
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

    # Initialize environment
    print(f"\n🏗️  Initializing Environment...")
    if args.use_dummy:
        from salus.simulation.isaaclab_env import SimplePickPlaceEnv
        env = SimplePickPlaceEnv(num_envs=num_envs, device=args.device, render=args.render)
    else:
        print("   ✅ Using real IsaacLab environment with Franka robot")
        from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
        env = FrankaPickPlaceEnv(num_envs=num_envs, device=args.device, render=args.render, max_episode_length=max_episode_length)

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
    print(f"   ✅ Config saved to {save_dir / 'config.yaml'}")

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
                print(f"   Success rate: {stats['success_rate']*100:.1f}%")
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
        if stats['num_episodes'] > 0:
            print(f"   Success: {success_count} ({success_count/stats['num_episodes']*100:.1f}%)")
            print(f"   Failure: {failure_count} ({failure_count/stats['num_episodes']*100:.1f}%)")
        else:
            print(f"   Success: {success_count}")
            print(f"   Failure: {failure_count}")
        print(f"   Total timesteps: {stats['total_timesteps']}")
        print(f"   Storage size: {stats['storage_size_gb']:.2f} GB")
        print(f"   Saved to: {save_dir}")

        # Close recorder
        recorder.close()

        # Close environment
        env.close()

        print(f"\n✅ Data collection finished successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Verify data: python scripts/verify_data.py {save_dir}")
        print(f"   2. Train predictor: python scripts/train_predictor.py --data {save_dir}")


if __name__ == "__main__":
    main()
