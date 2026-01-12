"""
Live Dashboard for SALUS Data Collection

Monitors zarr dataset growth in real-time and displays collection statistics.

Usage:
    python scripts/data_collection_dashboard.py --output_dir paper_data/massive_collection
"""

import sys
from pathlib import Path
import argparse
import time
import zarr
import numpy as np
from datetime import datetime, timedelta
import os

# Clear screen command
def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.2f}h"


def get_dataset_stats(zarr_path):
    """Extract statistics from zarr dataset."""
    if not Path(zarr_path).exists():
        return None

    try:
        store = zarr.open(str(zarr_path), mode='r')

        signals = store['signals']
        labels = store['horizon_labels']
        actions = store['actions']

        # Count valid episodes (non-zero actions)
        num_episodes = 0
        total_timesteps = 0
        failure_timesteps = 0

        for i in range(actions.shape[0]):
            if actions[i].max() != 0:
                num_episodes += 1
                # Count valid timesteps in this episode
                valid_mask = (np.abs(actions[i]).sum(axis=-1) > 0)
                ep_timesteps = valid_mask.sum()
                total_timesteps += ep_timesteps

                # Count failures
                ep_labels = labels[i, :ep_timesteps]
                failure_timesteps += (ep_labels.sum(axis=1) > 0).sum()
            else:
                break

        failure_rate = failure_timesteps / total_timesteps if total_timesteps > 0 else 0

        return {
            'num_episodes': num_episodes,
            'total_timesteps': total_timesteps,
            'failure_timesteps': failure_timesteps,
            'failure_rate': failure_rate,
            'avg_ep_length': total_timesteps / num_episodes if num_episodes > 0 else 0
        }
    except Exception as e:
        return None


def display_dashboard(stats, start_time, target_episodes, refresh_interval):
    """Display the live dashboard."""
    clear_screen()

    elapsed = time.time() - start_time

    print("=" * 80)
    print(" " * 20 + "SALUS DATA COLLECTION DASHBOARD")
    print("=" * 80)
    print()

    if stats is None:
        print("⏳ Waiting for data collection to start...")
        print(f"\n   Dataset will appear when collection begins.")
        print(f"   Refreshing every {refresh_interval}s...")
        return

    # Progress metrics
    progress = (stats['num_episodes'] / target_episodes * 100) if target_episodes > 0 else 0
    episodes_per_sec = stats['num_episodes'] / elapsed if elapsed > 0 else 0

    # ETA calculation
    if episodes_per_sec > 0 and target_episodes > stats['num_episodes']:
        remaining_episodes = target_episodes - stats['num_episodes']
        eta_seconds = remaining_episodes / episodes_per_sec
        eta_str = format_time(eta_seconds)
    else:
        eta_str = "N/A"

    # Display progress bar
    bar_width = 50
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"📊 PROGRESS")
    print(f"   [{bar}] {progress:.1f}%")
    print(f"   Episodes: {stats['num_episodes']:,} / {target_episodes:,}")
    print()

    print(f"⏱️  TIMING")
    print(f"   Elapsed: {format_time(elapsed)}")
    print(f"   ETA: {eta_str}")
    print(f"   Rate: {episodes_per_sec:.2f} episodes/sec ({episodes_per_sec * 3600:.0f} episodes/hour)")
    print()

    print(f"📈 DATASET STATISTICS")
    print(f"   Total Timesteps: {stats['total_timesteps']:,}")
    print(f"   Avg Episode Length: {stats['avg_ep_length']:.1f} steps")
    print(f"   Failure Rate: {stats['failure_rate']:.2%}")
    print(f"   Failure Timesteps: {stats['failure_timesteps']:,}")
    print()

    # Data quality metrics
    print(f"✓ DATA QUALITY")
    if stats['failure_rate'] > 0.05:
        quality_status = "🟢 Excellent (good failure rate)"
    elif stats['failure_rate'] > 0.02:
        quality_status = "🟡 Good (acceptable failure rate)"
    else:
        quality_status = "🔴 Low (few failures captured)"
    print(f"   Status: {quality_status}")
    print()

    # Storage estimate
    bytes_per_timestep = 12 * 4 + 16 * 4 + 7 * 4 + (256*256*3*3) * 1  # signals, labels, actions, images
    estimated_size_mb = (stats['total_timesteps'] * bytes_per_timestep) / (1024 * 1024)
    print(f"💾 STORAGE")
    print(f"   Estimated Size: {estimated_size_mb:.1f} MB")
    if target_episodes > stats['num_episodes']:
        final_size_mb = estimated_size_mb * (target_episodes / stats['num_episodes'])
        print(f"   Final Size (est): {final_size_mb:.1f} MB ({final_size_mb/1024:.2f} GB)")
    print()

    print("=" * 80)
    print(f"🔄 Auto-refreshing every {refresh_interval}s... (Ctrl+C to exit)")
    print(f"   Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def find_latest_dataset(output_dir):
    """Find the most recently modified zarr dataset in output_dir."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    zarr_dirs = list(output_path.glob("**/data.zarr"))
    if not zarr_dirs:
        return None

    # Return the most recently modified
    latest = max(zarr_dirs, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    parser = argparse.ArgumentParser(description='Data Collection Dashboard')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory where data is being collected')
    parser.add_argument('--target_episodes', type=int, default=5000,
                       help='Target number of episodes to collect')
    parser.add_argument('--refresh_interval', type=int, default=5,
                       help='Dashboard refresh interval in seconds')
    parser.add_argument('--zarr_path', type=str, default=None,
                       help='Specific zarr file to monitor (auto-detects if not provided)')
    args = parser.parse_args()

    start_time = time.time()

    print("Starting SALUS Data Collection Dashboard...")
    print(f"Monitoring: {args.output_dir}")
    print(f"Target: {args.target_episodes:,} episodes")
    print()
    time.sleep(2)

    try:
        while True:
            # Find dataset
            if args.zarr_path:
                zarr_path = Path(args.zarr_path)
            else:
                zarr_path = find_latest_dataset(args.output_dir)

            # Get stats
            stats = None
            if zarr_path and zarr_path.exists():
                stats = get_dataset_stats(zarr_path)

            # Display
            display_dashboard(stats, start_time, args.target_episodes, args.refresh_interval)

            # Check if target reached
            if stats and stats['num_episodes'] >= args.target_episodes:
                print()
                print("=" * 80)
                print("🎉 TARGET REACHED! Data collection complete.")
                print("=" * 80)
                break

            # Wait before next refresh
            time.sleep(args.refresh_interval)

    except KeyboardInterrupt:
        print()
        print()
        print("=" * 80)
        print("Dashboard stopped by user.")
        print("=" * 80)
        if stats:
            print(f"\nFinal Statistics:")
            print(f"  Episodes collected: {stats['num_episodes']:,}")
            print(f"  Total timesteps: {stats['total_timesteps']:,}")
            print(f"  Failure rate: {stats['failure_rate']:.2%}")


if __name__ == '__main__':
    main()
