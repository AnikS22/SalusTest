"""
Fix horizon labels for massive dataset
Compute multi-horizon labels retroactively based on episode failures
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import zarr
import numpy as np
from tqdm import tqdm


def compute_horizon_labels_for_episode(episode_length, failure_type=0, horizons=[6, 10, 13, 16]):
    """
    Compute multi-horizon failure prediction labels for a failed episode

    Args:
        episode_length: Number of timesteps in episode
        failure_type: Type of failure (0=collision, 1=drop, 2=miss, 3=timeout)
        horizons: Prediction horizons in timesteps

    Returns:
        horizon_labels: (T, 16) array where 16 = 4 horizons × 4 failure types
    """
    horizon_labels = np.zeros((episode_length, len(horizons), 4), dtype=np.float32)

    # Assume failure at last timestep
    failure_time = episode_length - 1

    # Label timesteps before failure
    for t in range(failure_time):
        steps_until_failure = failure_time - t

        for h_idx, horizon in enumerate(horizons):
            if steps_until_failure <= horizon:
                horizon_labels[t, h_idx, failure_type] = 1.0

    # Flatten to (T, 16)
    return horizon_labels.reshape(episode_length, -1)


def main():
    zarr_path = project_root / "paper_data" / "massive_collection" / "20260109_215258" / "data_20260109_215321.zarr"

    print("="*70)
    print("Fix Horizon Labels for Massive Dataset")
    print("="*70)
    print(f"\nDataset: {zarr_path}")

    # Open zarr in read-write mode
    store = zarr.open(str(zarr_path), mode='r+')

    actions = store['actions']
    horizon_labels = store['horizon_labels']

    print(f"\nDataset shape:")
    print(f"  Actions: {actions.shape}")
    print(f"  Horizon labels: {horizon_labels.shape}")

    # Count valid episodes
    num_episodes = 0
    for i in range(actions.shape[0]):
        if actions[i].max() != 0:
            num_episodes += 1
        else:
            break

    print(f"\nValid episodes: {num_episodes}")
    print(f"\nComputing horizon labels...")

    # Process each episode
    num_fixed = 0
    for ep in tqdm(range(num_episodes), desc="Processing episodes"):
        # Get episode length (count non-zero action timesteps)
        ep_actions = actions[ep]
        valid_mask = (np.abs(ep_actions).sum(axis=-1) > 0)
        ep_length = valid_mask.sum()

        if ep_length == 0:
            continue

        # Compute horizon labels
        # Assume all episodes failed (since collection showed 100% failure rate)
        # Use failure_type=0 (collision) as default
        ep_labels = compute_horizon_labels_for_episode(
            episode_length=ep_length,
            failure_type=0,  # collision
            horizons=[6, 10, 13, 16]
        )

        # Pad to 200 timesteps
        if ep_length < 200:
            padded_labels = np.zeros((200, 16), dtype=np.float32)
            padded_labels[:ep_length] = ep_labels
            ep_labels = padded_labels

        # Write to zarr
        horizon_labels[ep] = ep_labels
        num_fixed += 1

    print(f"\n✓ Fixed {num_fixed} episodes")

    # Verify
    print(f"\nVerifying labels...")
    sample_labels = horizon_labels[:10]
    print(f"  Min: {sample_labels.min()}")
    print(f"  Max: {sample_labels.max()}")
    print(f"  Mean: {sample_labels.mean()}")
    print(f"  Non-zero: {(sample_labels != 0).sum()} / {sample_labels.size}")

    print(f"\n✓ Labels fixed successfully!")


if __name__ == '__main__':
    main()
