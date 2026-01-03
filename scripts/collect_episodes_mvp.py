"""
SALUS MVP Data Collection
Simple control loop that:
  1. Runs TinyVLA ensemble
  2. Extracts 6D signals
  3. Records episodes with failure labels
  4. Saves to Zarr for training

Usage:
    python scripts/collect_episodes_mvp.py --num_episodes 500
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from salus.core.vla.tinyvla_wrapper import TinyVLAEnsemble, SimpleSignalExtractor
from salus.simulation.isaaclab_env import SimplePickPlaceEnv
from salus.data.recorder import ScalableDataRecorder


def collect_episode(env, vla, signal_extractor, episode_id, max_steps):
    """
    Collect single episode with VLA + signal extraction

    Returns:
        episode_data: Arrays with all data
        episode_info: Metadata including success/failure labels
    """
    # Reset
    obs = env.reset()
    signal_extractor.reset()

    # Storage
    episode_data = {
        'images': [],
        'states': [],
        'actions': [],
        'signals': [],  # 6D signals
        'labels': []
    }

    step = 0
    done = torch.zeros(1, dtype=torch.bool, device='cuda:0')

    # Episode loop
    while not done.any() and step < max_steps:
        # VLA forward pass
        with torch.no_grad():
            # Prepare observation for VLA
            vla_obs = {
                'image': obs['observation.images.camera1'],  # Use first camera
                'state': obs['observation.state'],
                'instruction': obs['task']
            }

            # Get action from ensemble
            vla_output = vla(vla_obs)
            action = vla_output['action']  # (1, 7)

            # Extract signals (6D)
            signals = signal_extractor.extract(vla_output)  # (1, 6)

        # Step environment
        next_obs, done, info = env.step(action)

        # Store data
        # Use first camera only for simplicity
        image = obs['observation.images.camera1'][0]  # (3, 256, 256)

        episode_data['images'].append(image.cpu().numpy().astype(np.uint8))
        episode_data['states'].append(obs['observation.state'][0].cpu().numpy().astype(np.float32))
        episode_data['actions'].append(action[0].cpu().numpy().astype(np.float32))
        episode_data['signals'].append(signals[0].cpu().numpy().astype(np.float32))

        # Store labels
        episode_data['labels'].append({
            'success': bool(info['success'][0].item()),
            'failure_type': int(info['failure_type'][0].item())
        })

        obs = next_obs
        step += 1

    # Convert to arrays
    images = np.stack(episode_data['images'], axis=0)  # (T, 3, 256, 256)
    images = np.expand_dims(images, axis=1)  # (T, 1, 3, 256, 256) - add camera dim

    episode_arrays = {
        'images': images,  # (T, 1, 3, 256, 256)
        'states': np.stack(episode_data['states'], axis=0),    # (T, 7)
        'actions': np.stack(episode_data['actions'], axis=0),  # (T, 7)
        'signals': np.stack(episode_data['signals'], axis=0),  # (T, 6) - MVP signals
        'horizon_labels': np.zeros((len(episode_data['states']), 4, 4), dtype=np.float32)  # Dummy for MVP
    }

    # Label entire episode
    final_label = episode_data['labels'][-1]
    episode_info = {
        'episode_id': episode_id,
        'success': final_label['success'],
        'failure_type': final_label['failure_type'] if not final_label['success'] else -1,
        'episode_length': step,
        'timestamp': datetime.now().isoformat()
    }

    return episode_arrays, episode_info


def main():
    parser = argparse.ArgumentParser(description='SALUS MVP Data Collection')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to collect')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--save_dir', type=str, default='data/mvp_episodes',
                        help='Where to save data')
    parser.add_argument('--use_real_vla', action='store_true',
                        help='Use real TinyVLA (requires installation)')

    args = parser.parse_args()

    print("="*70)
    print("SALUS MVP Data Collection")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Episodes: {args.num_episodes}")
    print(f"   Device: {args.device}")
    print(f"   Save dir: {args.save_dir}")
    print(f"   Use real VLA: {args.use_real_vla}")

    # Setup save directory
    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Saving to: {save_dir}")

    # Initialize VLA
    print(f"\n🤖 Loading VLA...")
    if args.use_real_vla:
        try:
            vla = TinyVLAEnsemble(
                model_path="~/models/tinyvla/tinyvla-1b",
                ensemble_size=3,  # 3 models for MVP
                device=args.device
            )
            signal_extractor = SimpleSignalExtractor()
            print(f"   ✅ TinyVLA ensemble loaded")
        except Exception as e:
            print(f"   ❌ Failed to load TinyVLA: {e}")
            print(f"   💡 Install with:")
            print(f"      cd ~/")
            print(f"      git clone https://github.com/OpenDriveLab/TinyVLA.git")
            print(f"      cd TinyVLA && pip install -e .")
            return
    else:
        print(f"   ⚠️  Using dummy VLA (random actions + signals)")
        vla = None
        signal_extractor = SimpleSignalExtractor()

    # Initialize environment
    print(f"\n🏗️  Initializing Environment...")
    env = SimplePickPlaceEnv(num_envs=1, device=args.device, render=False)

    # Initialize recorder
    print(f"\n💾 Initializing Recorder...")
    recorder = ScalableDataRecorder(
        save_dir=save_dir,
        max_episodes=args.num_episodes,
        max_episode_length=200,
        signal_dim=6,  # MVP uses 6D signals
        num_cameras=1,  # MVP uses 1 camera
        chunk_size=50
    )

    # Collect episodes
    print(f"\n🚀 Starting Collection...")
    print("="*70 + "\n")

    success_count = 0
    failure_count = 0

    try:
        for episode_id in tqdm(range(args.num_episodes), desc="Collecting episodes"):
            if vla is not None:
                # Use real VLA
                episode_data, episode_info = collect_episode(
                    env, vla, signal_extractor, episode_id, max_steps=200
                )
            else:
                # Use random actions (for testing pipeline)
                obs = env.reset()
                signal_extractor.reset()

                images, states, actions, signals, labels = [], [], [], [], []

                for step in range(200):
                    # Random action
                    action = torch.randn(1, 7, device=args.device) * 0.1

                    # Dummy VLA output for signal extraction
                    dummy_output = {
                        'action': action,
                        'action_var': torch.rand(1, 7, device=args.device) * 0.1,
                        'epistemic_uncertainty': torch.rand(1, device=args.device) * 0.5
                    }

                    sig = signal_extractor.extract(dummy_output)

                    obs, done, info = env.step(action)

                    # Store
                    images.append(obs['observation.images.camera1'][0].cpu().numpy().astype(np.uint8))
                    states.append(obs['observation.state'][0].cpu().numpy().astype(np.float32))
                    actions.append(action[0].cpu().numpy().astype(np.float32))
                    signals.append(sig[0].cpu().numpy().astype(np.float32))
                    labels.append({
                        'success': bool(info['success'][0].item()),
                        'failure_type': int(info['failure_type'][0].item())
                    })

                    if done.any():
                        break

                # Convert arrays and add camera dimension
                imgs = np.stack(images)  # (T, 3, 256, 256)
                imgs = np.expand_dims(imgs, axis=1)  # (T, 1, 3, 256, 256)

                episode_data = {
                    'images': imgs,
                    'states': np.stack(states),
                    'actions': np.stack(actions),
                    'signals': np.stack(signals),
                    'horizon_labels': np.zeros((len(images), 4, 4), dtype=np.float32)  # Dummy for MVP
                }

                final_label = labels[-1]
                episode_info = {
                    'episode_id': episode_id,
                    'success': final_label['success'],
                    'failure_type': final_label['failure_type'] if not final_label['success'] else -1,
                    'episode_length': len(images),
                    'timestamp': datetime.now().isoformat()
                }

            # Record episode
            recorder.record_episode(episode_data, episode_info)

            # Update stats
            if episode_info['success']:
                success_count += 1
            else:
                failure_count += 1

            # Print progress every 50 episodes
            if (episode_id + 1) % 50 == 0:
                stats = recorder.get_statistics()
                print(f"\n   Progress: {episode_id + 1}/{args.num_episodes}")
                print(f"   Success: {success_count}, Failure: {failure_count}")
                print(f"   Storage: {stats['storage_size_gb']:.2f} GB")

            # Checkpoint every 100 episodes
            if (episode_id + 1) % 100 == 0:
                recorder.save_checkpoint()

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final checkpoint
        recorder.save_checkpoint()

        # Print statistics
        print(f"\n{'='*70}")
        print("Collection Complete!")
        print(f"{'='*70}")

        stats = recorder.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Total episodes: {stats['num_episodes']}")
        print(f"   Success: {success_count} ({success_count/max(stats['num_episodes'], 1)*100:.1f}%)")
        print(f"   Failure: {failure_count} ({failure_count/max(stats['num_episodes'], 1)*100:.1f}%)")
        print(f"   Total timesteps: {stats['total_timesteps']}")
        print(f"   Storage: {stats['storage_size_gb']:.2f} GB")
        print(f"   Saved to: {save_dir}")

        # Close
        recorder.close()
        env.close()

        print(f"\n✅ Collection finished!")
        print(f"\n📋 Next steps:")
        print(f"   1. Train predictor: python scripts/train_predictor_mvp.py --data {save_dir}")
        print(f"   2. Evaluate: python scripts/evaluate_mvp.py --checkpoint checkpoints/best.pth")


if __name__ == "__main__":
    main()
