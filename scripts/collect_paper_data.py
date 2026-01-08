"""
SALUS Paper Data Collection Script
Collects training/validation/test datasets with comprehensive logging.
"""

# CRITICAL: Only minimal imports before AppLauncher
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Isaac Lab AppLauncher (MUST be before other imports)
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Collect SALUS paper dataset")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to collect")
parser.add_argument("--dataset_type", type=str, default="training", choices=["training", "validation", "test"], help="Dataset type")
parser.add_argument("--paper_data_dir", type=str, default="paper_data", help="Root paper data directory")
parser.add_argument("--device_explicit", action="store_true", default=False, help="Device was explicitly set")
args = parser.parse_args()

# Create AppLauncher
print("Creating AppLauncher...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"AppLauncher created successfully! simulation_app = {simulation_app}")

# NOW safe to import everything else
print("Importing Python stdlib and third-party modules...")
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
print("  ✓ Python stdlib and third-party modules imported")

print("Importing custom SALUS modules...")
print("  - Importing wrapper.py...")
from salus.core.vla.wrapper import SmolVLAEnsemble, SignalExtractor
print("  ✓ wrapper.py imported")

print("  - Importing franka_pick_place_env.py...")
from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
print("  ✓ franka_pick_place_env.py imported")

print("  - Importing recorder.py...")
from salus.data.recorder import ScalableDataRecorder
print("  ✓ recorder.py imported")

print("  - Importing config.py...")
from salus.utils.config import load_config, get_data_path
print("  ✓ config.py imported")

print("  - Importing paper_data_logger.py...")
from scripts.paper_data_logger import PaperDataLogger
print("  ✓ paper_data_logger.py imported")

print("✅ All modules imported successfully!")


def main():
    """Main data collection loop with paper logging."""

    # Configuration
    config_path = project_root / "configs" / "base_config.yaml"
    config = load_config(config_path)

    # Update config for paper data collection
    max_episode_length = config['data_collection.max_episode_length']

    # Setup save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_data_root = project_root / args.paper_data_dir
    dataset_dir = paper_data_root / args.dataset_type / timestamp
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SALUS Paper Data Collection - {args.dataset_type.upper()} Dataset")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Dataset type: {args.dataset_type}")
    print(f"   Episodes to collect: {args.num_episodes}")
    print(f"   Max episode length: {max_episode_length}")
    print(f"   Headless: {args.headless}")
    print(f"   Save directory: {dataset_dir}")

    # Initialize paper data logger
    log_dir = paper_data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    paper_logger = PaperDataLogger(log_dir, dataset_type=args.dataset_type)

    # Load VLA ensemble
    print(f"\n🤖 Loading VLA Ensemble...")
    vla = SmolVLAEnsemble(
        model_path=config['vla.model_path'],
        ensemble_size=config['vla.ensemble_size'],
        device=config['vla.device']
    )
    print(f"   ✅ VLA ensemble loaded ({config['vla.ensemble_size']} models)")

    # Initialize environment
    print(f"\n🏗️  Initializing Franka Environment...")
    env_device = args.device if hasattr(args, 'device') else "cuda:0"
    env = FrankaPickPlaceEnv(
        simulation_app=simulation_app,
        num_envs=1,
        device=env_device,
        render=args.enable_cameras if args.headless else True,
        max_episode_length=max_episode_length
    )
    print(f"   ✅ Environment initialized")

    # Initialize data recorder
    print(f"\n💾 Initializing Data Recorder...")
    recorder = ScalableDataRecorder(
        save_dir=dataset_dir,
        max_episodes=args.num_episodes,
        max_episode_length=max_episode_length,
        image_shape=(3, 224, 224),
        state_dim=7,
        action_dim=7,
        signal_dim=12,
        num_cameras=3,
        chunk_size=1000,
        compression="zstd"
    )
    print(f"   ✅ Data recorder ready")

    # Save configuration
    config_save_path = dataset_dir / "config.yaml"
    config.save(config_save_path)
    print(f"   ✅ Config saved to {config_save_path}")

    print(f"\n🚀 Starting Data Collection...")
    print("=" * 70)

    # Data collection loop
    for episode_idx in tqdm(range(args.num_episodes), desc="Collecting episodes"):
        paper_logger.start_episode(episode_idx)

        # Reset environment
        obs = env.reset()

        # Episode data storage
        episode_actions = []
        episode_states = []
        episode_images = []
        episode_signals = []

        episode_length = 0
        done = False

        # Episode loop
        while not done and episode_length < max_episode_length:
            # Move observations to VLA device
            vla_device = vla.device
            obs_vla = {
                'observation.images.camera1': obs['observation.images.camera1'].to(vla_device).float() / 255.0,
                'observation.images.camera2': obs['observation.images.camera2'].to(vla_device).float() / 255.0,
                'observation.images.camera3': obs['observation.images.camera3'].to(vla_device).float() / 255.0,
                'observation.state': obs['observation.state'].to(vla_device),
                'task': obs['task']
            }

            # Get action from VLA
            with torch.no_grad():
                action_dict = vla(obs_vla)

            # Move action back to environment device
            action = action_dict['action'].to(obs['observation.state'].device)

            # Store data
            episode_actions.append(action.cpu().numpy())
            episode_states.append(obs['observation.state'].cpu().numpy())

            # Store images (3 cameras)
            images = np.stack([
                obs['observation.images.camera1'].cpu().numpy(),
                obs['observation.images.camera2'].cpu().numpy(),
                obs['observation.images.camera3'].cpu().numpy()
            ], axis=0)  # Shape: (3, 1, 3, 224, 224)
            episode_images.append(images)

            # Extract and store signals
            signals = action_dict.get('signals', torch.zeros(1, 12, device=vla_device))
            episode_signals.append(signals.cpu().numpy())

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_length += 1

        # Get episode outcome
        success = info.get('success', torch.tensor([False], device=vla_device))[0].item()
        failure_type_int = info.get('failure_type', torch.tensor([3], device=vla_device))[0].item()

        # Map failure type integer to string
        failure_type_map = {0: "none", 1: "drop", 2: "timeout", 3: "other"}
        failure_type = failure_type_map.get(failure_type_int, "other")

        # Compute final cube distance
        if 'final_cube_distance' in info:
            final_cube_distance = info['final_cube_distance']
        else:
            final_cube_distance = 0.0

        # Convert to numpy arrays
        episode_actions = np.concatenate(episode_actions, axis=0)
        episode_states = np.concatenate(episode_states, axis=0)
        episode_images = np.concatenate(episode_images, axis=1)  # (3, T, 3, 224, 224)
        episode_signals = np.concatenate(episode_signals, axis=0)

        # Pad to max_episode_length
        pad_length = max_episode_length - episode_length
        if pad_length > 0:
            episode_actions = np.pad(episode_actions, ((0, pad_length), (0, 0)))
            episode_states = np.pad(episode_states, ((0, pad_length), (0, 0)))
            episode_images = np.pad(episode_images, ((0, 0), (0, pad_length), (0, 0), (0, 0), (0, 0)))
            episode_signals = np.pad(episode_signals, ((0, pad_length), (0, 0)))

        # Add batch dimension
        episode_actions = episode_actions[np.newaxis, ...]  # (1, T, 7)
        episode_states = episode_states[np.newaxis, ...]  # (1, T, 7)
        episode_images = episode_images[np.newaxis, ...]  # (1, 3, T, 3, 224, 224)
        episode_signals = episode_signals[np.newaxis, ...]  # (1, T, 12)

        # Create labels (placeholder for now - will be computed during training)
        horizon_labels = np.zeros((1, max_episode_length, 4, 4), dtype=np.float32)

        # Store in recorder
        recorder.store_episode(
            episode_idx=episode_idx,
            actions=episode_actions,
            states=episode_states,
            images=episode_images,
            signals=episode_signals,
            horizon_labels=horizon_labels,
            success=success,
            failure_type=failure_type,
            episode_length=episode_length
        )

        # Log to paper logger
        data_file = dataset_dir / "data.zarr"
        paper_logger.log_episode(
            episode_id=episode_idx,
            success=success,
            failure_type=failure_type,
            episode_length=episode_length,
            final_cube_distance=final_cube_distance,
            data_file=str(data_file)
        )

        # Print progress every 10 episodes
        if (episode_idx + 1) % 10 == 0:
            paper_logger.print_progress(episode_idx + 1, args.num_episodes)

    # Save all data
    print("\n💾 Saving collected data...")
    recorder.finalize()
    print(f"   ✅ Data saved to {dataset_dir / 'data.zarr'}")

    # Save paper logger summary
    paper_logger.save_summary()

    print("\n" + "=" * 70)
    print("✅ Data Collection Complete!")
    print("=" * 70)

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
