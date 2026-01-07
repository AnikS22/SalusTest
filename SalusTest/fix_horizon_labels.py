#!/usr/bin/env python3
"""
Post-process collected data to add proper horizon labels.

The original collection used dummy (all-zero) horizon labels.
This script computes proper multi-horizon failure prediction labels
based on the episode metadata.
"""

import zarr
import numpy as np
import json
from tqdm import tqdm


def compute_horizon_labels(episode_length, success, failure_type, horizons=[5, 10, 15, 20]):
    """
    Compute multi-horizon failure prediction labels for an episode.

    Args:
        episode_length: Number of timesteps in episode (always 200)
        success: Whether episode succeeded
        failure_type: Failure type (0=collision, 1=drop, 2=miss, 3=timeout)
        horizons: Prediction horizons in timesteps

    Returns:
        horizon_labels: (T, n_horizons, 4) array
    """
    T = episode_length
    n_horizons = len(horizons)
    horizon_labels = np.zeros((T, n_horizons, 4), dtype=np.float32)

    if not success:
        # For failures, assume failure happened at last timestep
        # (episodes terminate at failure or max_steps)
        failure_time = T - 1
        failure_type_int = int(failure_type)

        # Label timesteps before failure
        for t in range(failure_time):
            steps_until_failure = failure_time - t

            for h_idx, horizon in enumerate(horizons):
                if steps_until_failure <= horizon:
                    # This timestep is within horizon of failure
                    horizon_labels[t, h_idx, failure_type_int] = 1.0

    return horizon_labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--horizons', type=int, nargs='+', default=[5, 10, 15, 20],
                        help='Prediction horizons in timesteps')
    args = parser.parse_args()

    print("=" * 70)
    print("Fix Horizon Labels - Post-Process Collected Data")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Horizons: {args.horizons}")
    print()

    # Open zarr dataset
    zarr_path = f"{args.data}/data.zarr"
    print(f"Opening zarr: {zarr_path}")
    root = zarr.open_group(zarr_path, mode='r+')

    # Load episode metadata
    print("Loading episode metadata...")
    episode_metadata_raw = root['episode_metadata'][:]
    episodes = [json.loads(str(ep)) for ep in episode_metadata_raw]
    n_episodes = len(episodes)

    print(f"Found {n_episodes} episodes")
    print()

    # Count episode outcomes
    n_success = sum(1 for ep in episodes if ep['success'])
    n_failures = n_episodes - n_success
    failure_types = {0: 0, 1: 0, 2: 0, 3: 0}
    for ep in episodes:
        if not ep['success']:
            failure_types[ep['failure_type']] += 1

    print("Episode Distribution:")
    print(f"  Success:   {n_success:3d} ({n_success/n_episodes*100:.1f}%)")
    print(f"  Failures:  {n_failures:3d} ({n_failures/n_episodes*100:.1f}%)")
    print()
    print("Failure Types:")
    print(f"  Collision: {failure_types[0]:3d} ({failure_types[0]/n_failures*100:.1f}%)")
    print(f"  Drop:      {failure_types[1]:3d} ({failure_types[1]/n_failures*100:.1f}%)")
    print(f"  Miss:      {failure_types[2]:3d} ({failure_types[2]/n_failures*100:.1f}%)")
    print(f"  Timeout:   {failure_types[3]:3d} ({failure_types[3]/n_failures*100:.1f}%)")
    print()

    # Check current horizon labels
    current_labels = root['horizon_labels']
    print(f"Current horizon_labels shape: {current_labels.shape}")
    print(f"Current non-zero count: {(current_labels[:] != 0).sum()}")

    if (current_labels[:] != 0).sum() > 0:
        print("⚠️  Warning: horizon_labels already has non-zero values!")
        response = input("Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return

    print()
    print("Computing proper horizon labels...")

    # Compute labels for each episode
    new_horizon_labels = np.zeros((n_episodes, 200, len(args.horizons), 4), dtype=np.float32)

    label_stats = {
        'total_labeled': 0,
        'by_horizon': [0] * len(args.horizons),
        'by_type': [0] * 4
    }

    for ep_idx, ep in enumerate(tqdm(episodes, desc="Processing episodes")):
        ep_labels = compute_horizon_labels(
            episode_length=ep['episode_length'],
            success=ep['success'],
            failure_type=ep.get('failure_type', 0),
            horizons=args.horizons
        )
        new_horizon_labels[ep_idx] = ep_labels

        # Track statistics
        if not ep['success']:
            for t in range(ep['episode_length']):
                for h_idx in range(len(args.horizons)):
                    if ep_labels[t, h_idx].sum() > 0:
                        label_stats['total_labeled'] += 1
                        label_stats['by_horizon'][h_idx] += 1
                        label_stats['by_type'][ep['failure_type']] += 1

    # Write new labels
    print()
    print("Writing new horizon_labels to zarr...")
    root['horizon_labels'][:] = new_horizon_labels

    print()
    print("=" * 70)
    print("Label Statistics")
    print("=" * 70)
    print(f"Total labeled timesteps: {label_stats['total_labeled']:,}")
    print()
    print("By Horizon:")
    for h_idx, horizon in enumerate(args.horizons):
        count = label_stats['by_horizon'][h_idx]
        print(f"  Horizon {h_idx+1} ({horizon:2d} steps): {count:,}")
    print()
    print("By Failure Type:")
    type_names = ['Collision', 'Drop', 'Miss', 'Timeout']
    for type_idx, name in enumerate(type_names):
        count = label_stats['by_type'][type_idx]
        print(f"  {name:10s}: {count:,}")
    print()

    # Verify
    print("Verifying labels...")
    updated_labels = root['horizon_labels'][:]
    non_zero = (updated_labels != 0).sum()
    print(f"✅ Non-zero labels: {non_zero:,}")

    # Check a few examples
    print()
    print("Sample Episodes:")
    for i in range(min(3, n_episodes)):
        ep = episodes[i]
        ep_labels = updated_labels[i]
        has_labels = (ep_labels != 0).any()
        print(f"  Episode {i}: success={ep['success']}, "
              f"failure_type={ep.get('failure_type', 'N/A')}, "
              f"has_labels={has_labels}")

    print()
    print("=" * 70)
    print("✅ Horizon labels successfully updated!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Re-train SALUS predictor:")
    print(f"     python scripts/train_predictor_mvp.py --data {args.data}")
    print()
    print("  2. Expect F1 score improvement from 0.000 to 0.70-0.85")
    print()


if __name__ == '__main__':
    main()
