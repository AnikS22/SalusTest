#!/usr/bin/env python3
"""
Comprehensive CLI Dashboard for SALUS Data Collection
Shows detailed real-time information about all stages of the pipeline.
"""

import time
import zarr
from pathlib import Path
import sys
import psutil
import numpy as np
from datetime import datetime, timedelta

def format_bytes(bytes_val):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"

def format_time(seconds):
    """Format seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"

def get_process_info():
    """Get info about collect_local_data.py process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'collect_local_data.py' in ' '.join(cmdline):
                return {
                    'pid': proc.info['pid'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory': proc.info['memory_info'].rss,
                    'runtime': time.time() - proc.info['create_time']
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def analyze_signals(signals):
    """Analyze signal statistics."""
    if len(signals) == 0:
        return None

    # Get last 10 timesteps
    recent = signals[-min(10, len(signals)):]

    stats = {
        'epistemic_unc': recent[:, 0].mean() if recent.shape[1] > 0 else 0,
        'action_mag': recent[:, 1].mean() if recent.shape[1] > 1 else 0,
        'latent_drift': recent[:, 12].mean() if recent.shape[1] > 12 else 0,
        'ood_distance': recent[:, 13].mean() if recent.shape[1] > 13 else 0,
        'sensitivity': recent[:, 15].mean() if recent.shape[1] > 15 else 0,
        'constraint_margin': recent[:, 17].mean() if recent.shape[1] > 17 else 0,
    }

    # Check if signals are all zeros
    all_zero = np.allclose(signals, 0)

    return {'stats': stats, 'all_zero': all_zero}

def render_dashboard(proc_info, zarr_info, start_time, last_update):
    """Render the complete dashboard."""

    # Clear screen
    print("\033[H\033[J", end='')

    # Header
    print("=" * 80)
    print("SALUS DATA COLLECTION - COMPREHENSIVE DASHBOARD".center(80))
    print("=" * 80)
    print()

    # System Time
    now = datetime.now()
    print(f"⏰ Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Dashboard Runtime: {format_time(time.time() - start_time)}")
    print(f"🔄 Last Update: {last_update.strftime('%H:%M:%S')}")
    print()

    # Process Status
    print("─" * 80)
    print("🖥️  PROCESS STATUS")
    print("─" * 80)

    if proc_info:
        print(f"✅ Collection Process RUNNING")
        print(f"   PID:        {proc_info['pid']}")
        print(f"   Runtime:    {format_time(proc_info['runtime'])}")
        print(f"   CPU Usage:  {proc_info['cpu_percent']:.1f}%")
        print(f"   Memory:     {format_bytes(proc_info['memory'])}")
    else:
        print(f"❌ Collection Process NOT RUNNING")
        print(f"   Start with: python collect_local_data.py")
    print()

    # Data File Status
    print("─" * 80)
    print("📁 DATA FILE STATUS")
    print("─" * 80)

    if zarr_info:
        print(f"✅ Data File: {zarr_info['filename']}")
        print(f"   File Size:  {format_bytes(zarr_info['file_size'])}")
        print(f"   Created:    {zarr_info['created'].strftime('%H:%M:%S')}")
        print(f"   Age:        {format_time(zarr_info['age'])}")
    else:
        print(f"⏳ Waiting for data file...")
    print()

    # Collection Stage
    print("─" * 80)
    print("🔬 COLLECTION STAGE")
    print("─" * 80)

    if not zarr_info:
        print("⏳ Stage: Initializing...")
        print("   - Creating Zarr file structure")
        print("   - Preparing datasets")
    elif zarr_info['stage'] == 'vla_loading':
        print("⏳ Stage: VLA Models Loading")
        print("   - Loading SmolVLA ensemble (3× 865MB models)")
        print("   - Initializing signal extractor")
        print("   - This takes 1-2 minutes initially...")
        print()
        print("   Expected: ~2 minutes")
        if proc_info:
            print(f"   Elapsed:  {format_time(proc_info['runtime'])}")
    elif zarr_info['stage'] == 'env_setup':
        print("⏳ Stage: Environment Setup")
        print("   - Initializing Isaac Lab environment")
        print("   - Loading Franka Panda robot model")
        print("   - Setting up cameras")
    elif zarr_info['stage'] == 'collecting':
        print("✅ Stage: Collecting Episodes")
        print(f"   - Episode {zarr_info['episodes_completed']}/{zarr_info['total_episodes']}")
        print(f"   - Current episode step: {zarr_info['current_episode_steps']}")
    elif zarr_info['stage'] == 'complete':
        print("🎉 Stage: COMPLETE!")
        print(f"   - All {zarr_info['total_episodes']} episodes collected")
    print()

    # Progress
    if zarr_info and zarr_info['stage'] == 'collecting':
        print("─" * 80)
        print("📊 COLLECTION PROGRESS")
        print("─" * 80)

        print(f"Episodes: {zarr_info['episodes_completed']}/{zarr_info['total_episodes']} " +
              f"({zarr_info['episodes_completed']/zarr_info['total_episodes']*100:.1f}%)")

        # Progress bar
        progress = zarr_info['episodes_completed'] / zarr_info['total_episodes']
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"[{bar}]")
        print()

        print(f"Total Steps: {zarr_info['total_steps']:,}")
        print(f"Current Episode: {zarr_info['current_episode']}")
        print(f"Steps in Current: {zarr_info['current_episode_steps']}")
        print()

        # Success/Failure
        if zarr_info['episodes_completed'] > 0:
            success_rate = zarr_info['successes'] / zarr_info['episodes_completed'] * 100
            print(f"✅ Successes: {zarr_info['successes']} ({success_rate:.1f}%)")
            print(f"❌ Failures:  {zarr_info['failures']} ({100-success_rate:.1f}%)")
        print()

    # Signal Analysis
    if zarr_info and zarr_info.get('signal_analysis'):
        print("─" * 80)
        print("📡 SIGNAL ANALYSIS (Last 10 Steps)")
        print("─" * 80)

        sig = zarr_info['signal_analysis']

        if sig['all_zero']:
            print("⚠️  WARNING: All signals are ZERO!")
            print("   This indicates VLA signal extraction may not be working.")
        else:
            print("✅ Signals are NON-ZERO (VLA extraction working)")
            print()
            stats = sig['stats']
            print(f"   Epistemic Uncertainty:  {stats['epistemic_unc']:.4f}")
            print(f"   Action Magnitude:       {stats['action_mag']:.4f}")
            print(f"   Latent Drift (VLA):     {stats['latent_drift']:.4f}")
            print(f"   OOD Distance:           {stats['ood_distance']:.4f}")
            print(f"   Perturbation Sensitivity: {stats['sensitivity']:.4f}")
            print(f"   Constraint Margin:      {stats['constraint_margin']:.4f}")
        print()

    # Performance Metrics
    if zarr_info and zarr_info['stage'] == 'collecting' and zarr_info.get('rate_info'):
        print("─" * 80)
        print("⚡ PERFORMANCE METRICS")
        print("─" * 80)

        rate = zarr_info['rate_info']
        print(f"Steps/Second:    {rate['steps_per_sec']:.2f}")
        print(f"Episodes/Minute: {rate['episodes_per_min']:.2f}")
        print()

        if rate['eta_seconds'] > 0:
            print(f"⏳ Estimated Time Remaining: {format_time(rate['eta_seconds'])}")
            eta_time = datetime.now() + timedelta(seconds=rate['eta_seconds'])
            print(f"   Estimated Completion: {eta_time.strftime('%H:%M:%S')}")
        print()

    # Footer
    print("─" * 80)
    print("Press Ctrl+C to exit dashboard (collection will continue in background)")
    print("=" * 80)

def monitor(refresh_interval=2):
    """Main monitoring loop."""

    data_dir = Path("local_data")
    start_time = time.time()

    last_total_steps = 0
    last_check_time = time.time()

    print("\n🚀 Starting SALUS Dashboard...\n")
    time.sleep(1)

    try:
        while True:
            # Get process info
            proc_info = get_process_info()

            # Find most recent Zarr file
            data_files = sorted(data_dir.glob("salus_data_*.zarr"))

            zarr_info = None
            if data_files:
                data_path = data_files[-1]

                try:
                    # Get file stats
                    file_size = sum(f.stat().st_size for f in data_path.rglob('*') if f.is_file())
                    created_time = datetime.fromtimestamp(data_path.stat().st_ctime)
                    age = time.time() - data_path.stat().st_ctime

                    # Open Zarr
                    root = zarr.open(str(data_path), mode='r')

                    # Determine stage
                    stage = 'vla_loading'
                    episodes_completed = 0
                    total_steps = 0
                    successes = 0
                    failures = 0
                    current_episode = 0
                    current_episode_steps = 0

                    if 'episode_id' in root and len(root['episode_id']) > 0:
                        stage = 'collecting'

                        episode_ids = root['episode_id'][:]
                        total_steps = len(episode_ids)

                        done = root['done'][:]
                        success = root['success'][:]

                        # Count completed episodes
                        completed_eps = set()
                        for i, is_done in enumerate(done):
                            if is_done:
                                ep_id = episode_ids[i]
                                if ep_id not in completed_eps:
                                    completed_eps.add(ep_id)
                                    if success[i]:
                                        successes += 1
                                    else:
                                        failures += 1

                        episodes_completed = len(completed_eps)

                        # Current episode
                        current_episode = int(episode_ids[-1])
                        current_ep_mask = episode_ids == current_episode
                        current_episode_steps = int(current_ep_mask.sum())

                        # Check if complete
                        if episodes_completed >= 50:
                            stage = 'complete'

                    zarr_info = {
                        'filename': data_path.name,
                        'file_size': file_size,
                        'created': created_time,
                        'age': age,
                        'stage': stage,
                        'episodes_completed': episodes_completed,
                        'total_episodes': 50,
                        'total_steps': total_steps,
                        'successes': successes,
                        'failures': failures,
                        'current_episode': current_episode,
                        'current_episode_steps': current_episode_steps,
                    }

                    # Signal analysis
                    if 'signals' in root and len(root['signals']) > 0:
                        signals = root['signals'][:]
                        zarr_info['signal_analysis'] = analyze_signals(signals)

                    # Rate calculation
                    if stage == 'collecting':
                        current_time = time.time()
                        time_delta = current_time - last_check_time

                        if time_delta > 0 and last_total_steps > 0:
                            steps_delta = total_steps - last_total_steps
                            steps_per_sec = steps_delta / time_delta

                            if steps_per_sec > 0:
                                # Estimate remaining
                                remaining_episodes = 50 - episodes_completed
                                avg_steps_per_episode = 80  # Rough estimate
                                remaining_steps = remaining_episodes * avg_steps_per_episode
                                eta_seconds = remaining_steps / steps_per_sec

                                episodes_per_min = (steps_per_sec * 60) / avg_steps_per_episode

                                zarr_info['rate_info'] = {
                                    'steps_per_sec': steps_per_sec,
                                    'episodes_per_min': episodes_per_min,
                                    'eta_seconds': eta_seconds
                                }

                        last_total_steps = total_steps
                        last_check_time = current_time

                except Exception as e:
                    # File may be being written
                    pass

            # Render dashboard
            render_dashboard(proc_info, zarr_info, start_time, datetime.now())

            # Check if complete
            if zarr_info and zarr_info['stage'] == 'complete':
                print("\n\n🎉 COLLECTION COMPLETE!")
                print(f"\n📊 Final Statistics:")
                print(f"   Total episodes: {zarr_info['episodes_completed']}")
                print(f"   Total steps: {zarr_info['total_steps']:,}")
                if zarr_info['episodes_completed'] > 0:
                    success_rate = zarr_info['successes'] / zarr_info['episodes_completed'] * 100
                    print(f"   Success rate: {success_rate:.1f}%")
                print(f"\n✅ Ready to train SALUS!")
                print(f"   Run: python train_salus_local.py")
                break

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
        if proc_info:
            print(f"\n⚠️  Collection process (PID {proc_info['pid']}) is still running in background")
            print(f"   Run dashboard again to monitor: python dashboard.py")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive SALUS data collection dashboard")
    parser.add_argument("--refresh", type=int, default=2, help="Refresh interval in seconds")
    args = parser.parse_args()

    monitor(refresh_interval=args.refresh)
