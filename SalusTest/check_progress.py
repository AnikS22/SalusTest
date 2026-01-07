#!/usr/bin/env python3
"""Simple progress check for data collection."""

import zarr
from pathlib import Path
import sys

data_dir = Path("local_data")
data_files = sorted(data_dir.glob("salus_data_*.zarr"))

if not data_files:
    print("❌ No data files found. Collection hasn't started yet.")
    sys.exit(1)

data_path = data_files[-1]
print(f"\n{'='*70}")
print(f"DATA COLLECTION STATUS")
print(f"{'='*70}")
print(f"\n📁 File: {data_path.name}")

try:
    root = zarr.open(str(data_path), mode='r')

    # Check if arrays exist and have data
    if 'episode_id' not in root:
        print("\n⏳ Zarr file initializing... (VLA models loading)")
        sys.exit(0)

    try:
        episode_id_array = root['episode_id']
        total_steps = episode_id_array.shape[0]
    except:
        print("\n⏳ Arrays not ready yet... (VLA models loading)")
        sys.exit(0)

    if total_steps == 0:
        print("\n⏳ No data collected yet (VLA models still loading)")
        print("   This takes 1-2 minutes initially...")
        sys.exit(0)

    # Read data
    episode_ids = episode_id_array[:]

    # Count unique episodes
    unique_episodes = set(episode_ids)
    n_episodes = len(unique_episodes)

    print(f"\n📊 PROGRESS:")
    print(f"   Episodes: {n_episodes}/50 ({n_episodes/50*100:.1f}%)")
    print(f"   Total steps: {total_steps:,}")

    # Try to get success/failure info
    try:
        done_array = root['done']
        success_array = root['success']

        done = done_array[:]
        success = success_array[:]

        # Find completed episodes
        completed_eps = set()
        successes = 0
        failures = 0

        for i, is_done in enumerate(done):
            if is_done:
                ep_id = episode_ids[i]
                if ep_id not in completed_eps:
                    completed_eps.add(ep_id)
                    if success[i]:
                        successes += 1
                    else:
                        failures += 1

        print(f"\n✅ COMPLETED EPISODES: {len(completed_eps)}")
        if len(completed_eps) > 0:
            print(f"   Successes: {successes} ({successes/len(completed_eps)*100:.1f}%)")
            print(f"   Failures: {failures} ({failures/len(completed_eps)*100:.1f}%)")

    except Exception as e:
        print(f"\n⚠️  Could not read completion status: {e}")

    # Estimate progress
    avg_steps_per_episode = total_steps / max(n_episodes, 1)
    estimated_total = 50 * 100  # 50 episodes × 100 steps
    progress_pct = (total_steps / estimated_total) * 100

    print(f"\n📈 ESTIMATED:")
    print(f"   Overall progress: {progress_pct:.1f}%")
    print(f"   Avg steps/episode: {avg_steps_per_episode:.1f}")

    if n_episodes < 50:
        print(f"\n⏳ Collection in progress...")
        print(f"   Run this script again to check progress")
    else:
        print(f"\n✅ COLLECTION COMPLETE!")
        print(f"   Ready to train SALUS: python train_salus_local.py")

except Exception as e:
    print(f"\n❌ Error reading Zarr file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*70}\n")
