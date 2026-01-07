#!/usr/bin/env python3
"""
Monitor data collection progress in real-time.
Shows episodes collected, success rate, and estimated time remaining.
"""

import time
import zarr
from pathlib import Path
import sys

def monitor_collection(refresh_interval=5):
    """Monitor collection progress."""

    data_dir = Path("local_data")

    print("\n" + "="*70)
    print("DATA COLLECTION MONITOR")
    print("="*70)
    print("\nPress Ctrl+C to exit\n")

    last_total_steps = 0
    start_time = time.time()

    try:
        while True:
            # Find most recent Zarr file
            data_files = sorted(data_dir.glob("salus_data_*.zarr"))

            if not data_files:
                print("⏳ Waiting for data collection to start...")
                time.sleep(refresh_interval)
                continue

            data_path = data_files[-1]

            try:
                # Open Zarr file
                root = zarr.open(str(data_path), mode='r')

                # Get current stats
                total_steps = len(root['episode_id'])

                if total_steps == 0:
                    print("⏳ Data collection initializing...")
                    time.sleep(refresh_interval)
                    continue

                episode_ids = root['episode_id'][:]
                success = root['success'][:]
                done = root['done'][:]

                # Count episodes
                unique_episodes = len(set(episode_ids))

                # Count successes/failures (only for completed episodes)
                completed_episodes = set()
                for i, is_done in enumerate(done):
                    if is_done:
                        completed_episodes.add(episode_ids[i])

                n_completed = len(completed_episodes)

                # Count successes
                successes = 0
                failures = 0
                for ep_id in completed_episodes:
                    ep_mask = episode_ids == ep_id
                    ep_success = success[ep_mask]
                    if ep_success[0]:  # All same for episode
                        successes += 1
                    else:
                        failures += 1

                # Calculate rate
                elapsed_time = time.time() - start_time
                steps_per_sec = (total_steps - last_total_steps) / refresh_interval if last_total_steps > 0 else 0
                last_total_steps = total_steps

                # Estimate remaining time (assume 50 episodes, ~100 steps each = 5000 total)
                estimated_total_steps = 5000
                remaining_steps = max(0, estimated_total_steps - total_steps)
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                # Clear screen and print stats
                print("\033[H\033[J", end='')  # Clear screen

                print("="*70)
                print("DATA COLLECTION PROGRESS")
                print("="*70)

                print(f"\n📁 File: {data_path.name}")
                print(f"⏱️  Running for: {elapsed_time/60:.1f} minutes")

                print(f"\n📊 PROGRESS:")
                print(f"   Episodes completed: {n_completed}/50 ({n_completed/50*100:.1f}%)")
                print(f"   Current episode: {unique_episodes}")
                print(f"   Total steps: {total_steps:,}")

                # Progress bar
                progress = n_completed / 50
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"   [{bar}] {progress*100:.1f}%")

                print(f"\n✅ SUCCESS/FAILURE:")
                if n_completed > 0:
                    success_rate = successes / n_completed * 100
                    print(f"   Successes: {successes} ({success_rate:.1f}%)")
                    print(f"   Failures:  {failures} ({100-success_rate:.1f}%)")
                else:
                    print(f"   Successes: {successes}")
                    print(f"   Failures:  {failures}")

                print(f"\n⚡ RATE:")
                print(f"   Steps/sec: {steps_per_sec:.1f}")
                if steps_per_sec > 0:
                    episodes_per_min = (steps_per_sec * 60) / 100  # Assume 100 steps per episode
                    print(f"   Episodes/min: {episodes_per_min:.1f}")

                print(f"\n⏳ ESTIMATED TIME:")
                if eta_minutes > 0 and n_completed < 50:
                    print(f"   Remaining: {eta_minutes:.1f} minutes")
                    print(f"   ETA: {time.strftime('%H:%M:%S', time.localtime(time.time() + eta_seconds))}")
                elif n_completed >= 50:
                    print(f"   ✅ COLLECTION COMPLETE!")
                else:
                    print(f"   Calculating...")

                print(f"\n{'='*70}")
                print(f"Refreshing every {refresh_interval}s... (Ctrl+C to exit)")

                # Check if done
                if n_completed >= 50:
                    print(f"\n🎉 Data collection complete!")
                    print(f"\n📊 Final Statistics:")
                    print(f"   Total episodes: {n_completed}")
                    print(f"   Total steps: {total_steps:,}")
                    print(f"   Success rate: {successes/n_completed*100:.1f}%")
                    print(f"   Time taken: {elapsed_time/60:.1f} minutes")
                    print(f"\n✅ Ready for training! Run:")
                    print(f"   python train_salus_local.py")
                    break

            except Exception as e:
                print(f"⚠️  Error reading Zarr file: {e}")
                print(f"   File may be being written to...")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print(f"\n\n👋 Monitoring stopped by user")
        if last_total_steps > 0:
            print(f"\nCurrent progress: {n_completed}/50 episodes, {total_steps:,} steps")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    args = parser.parse_args()

    monitor_collection(refresh_interval=args.refresh)
