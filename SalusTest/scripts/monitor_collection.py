"""
Monitor SALUS data collection progress
"""

import argparse
import time
from pathlib import Path
import zarr
import subprocess


def get_gpu_memory():
    """Get GPU memory usage for GPU 0"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip())
    except:
        return -1


def monitor_collection(save_dir: str, log_file: str = None):
    """
    Monitor data collection progress

    Args:
        save_dir: Directory where data is being saved
        log_file: Optional log file to tail
    """
    save_path = Path(save_dir)

    print("=" * 70)
    print("SALUS Data Collection Monitor")
    print("=" * 70)
    print(f"Save directory: {save_path}")
    if log_file:
        print(f"Log file: {log_file}")
    print("=" * 70)

    last_episode_count = 0
    start_time = time.time()

    while True:
        try:
            # Find most recent data directory
            data_dirs = list(save_path.glob("*/data.zarr"))
            if not data_dirs:
                print("\n⏳ Waiting for data collection to start...")
                time.sleep(10)
                continue

            # Use most recent directory
            zarr_path = data_dirs[-1]
            store = zarr.open(str(zarr_path), mode='r')

            # Get episode count
            if 'actions' in store:
                episodes_collected = store['actions'].shape[0]
            else:
                episodes_collected = 0

            # Check if new episodes completed
            if episodes_collected > last_episode_count:
                elapsed_time = time.time() - start_time
                avg_time_per_episode = elapsed_time / episodes_collected if episodes_collected > 0 else 0

                # GPU memory
                gpu_mem = get_gpu_memory()

                # Print progress
                print(f"\n📊 Progress Update [{time.strftime('%H:%M:%S')}]")
                print(f"   Episodes collected: {episodes_collected}")
                print(f"   New episodes: {episodes_collected - last_episode_count}")
                print(f"   Elapsed time: {elapsed_time/3600:.2f} hours")
                print(f"   Avg time/episode: {avg_time_per_episode/60:.2f} minutes")
                if gpu_mem > 0:
                    print(f"   GPU 0 memory: {gpu_mem} MB / 11264 MB ({gpu_mem/112.64:.1f}%)")
                print(f"   Data location: {zarr_path}")

                # Data size
                data_size_mb = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / 1024 / 1024
                print(f"   Data size: {data_size_mb:.1f} MB")

                last_episode_count = episodes_collected

            # Check if complete
            if log_file:
                log_path = Path(log_file)
                if log_path.exists():
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                        if "✅ Data Collection Complete!" in log_content:
                            print("\n" + "=" * 70)
                            print("✅ DATA COLLECTION COMPLETE!")
                            print("=" * 70)
                            break

            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor SALUS data collection")
    parser.add_argument("--save_dir", type=str, required=True, help="Data save directory")
    parser.add_argument("--log_file", type=str, help="Log file to monitor")
    args = parser.parse_args()

    monitor_collection(args.save_dir, args.log_file)
