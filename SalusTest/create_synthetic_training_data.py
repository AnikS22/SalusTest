"""
Create synthetic 12D training data for SALUS testing.

Generates realistic failure patterns to validate the training pipeline.
"""

import torch
import numpy as np
import zarr
from pathlib import Path
from datetime import datetime

print("\n" + "="*70)
print("CREATING SYNTHETIC 12D TRAINING DATA")
print("="*70)

# Configuration
num_episodes = 50
max_steps = 100
output_dir = Path("local_data")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_path = output_dir / f"salus_data_{timestamp}.zarr"

print(f"\nConfiguration:")
print(f"  Episodes: {num_episodes}")
print(f"  Steps per episode: {max_steps}")
print(f"  Output: {data_path}")

# Initialize Zarr storage
root = zarr.open(str(data_path), mode='w')

# Create datasets
signals_ds = root.create_dataset('signals', shape=(0, 12), chunks=(1000, 12), dtype='f4')
actions_ds = root.create_dataset('actions', shape=(0, 6), chunks=(1000, 6), dtype='f4')
robot_state_ds = root.create_dataset('robot_state', shape=(0, 7), chunks=(1000, 7), dtype='f4')
episode_ids_ds = root.create_dataset('episode_id', shape=(0,), chunks=(1000,), dtype='i4')
timesteps_ds = root.create_dataset('timestep', shape=(0,), chunks=(1000,), dtype='i4')
success_ds = root.create_dataset('success', shape=(0,), chunks=(1000,), dtype='bool')
done_ds = root.create_dataset('done', shape=(0,), chunks=(1000,), dtype='bool')

print(f"\n✓ Zarr datasets created")

# Generate synthetic data with realistic failure patterns
all_signals = []
all_actions = []
all_robot_state = []
all_episode_ids = []
all_timesteps = []
all_success = []
all_done = []

successes = 0
failures = 0

print(f"\nGenerating episodes...")

for ep in range(num_episodes):
    # 50% success rate
    is_success = (ep % 2 == 0)

    if is_success:
        successes += 1
    else:
        failures += 1

    # Failure episodes have higher signal values (more uncertainty)
    # Success episodes have lower signal values (more stable)

    for t in range(max_steps):
        # Simulate temporal progression
        # Failures: signals increase over time
        # Success: signals stay low

        progress = t / max_steps  # 0 to 1

        if is_success:
            # Success: low, stable signals
            base_level = 0.1 + 0.05 * np.random.randn()
            temporal_trend = 0.0  # No increase
        else:
            # Failure: signals increase over time
            base_level = 0.3 + 0.1 * np.random.randn()
            temporal_trend = progress * 0.5  # Increase toward end

        # Generate 12D signals with realistic patterns
        signals_t = np.zeros(12, dtype=np.float32)

        # Signals 1-4: Temporal Action Dynamics
        signals_t[0] = max(0, base_level + temporal_trend + 0.1 * np.random.randn())  # Volatility
        signals_t[1] = max(0, 0.2 + 0.1 * np.random.randn())  # Magnitude
        signals_t[2] = max(0, base_level + temporal_trend * 0.5 + 0.05 * np.random.randn())  # Acceleration
        signals_t[3] = max(0, base_level + temporal_trend + 0.08 * np.random.randn())  # Divergence

        # Signals 5-7: VLA Internal Stability
        signals_t[4] = max(0, base_level + temporal_trend + 0.1 * np.random.randn())  # Latent drift
        signals_t[5] = max(0, 1.0 + temporal_trend * 0.3 + 0.15 * np.random.randn())  # Norm spike
        signals_t[6] = max(0, base_level * 2 + temporal_trend * 2 + 0.2 * np.random.randn())  # OOD distance

        # Signals 8-9: Model Uncertainty
        signals_t[7] = max(0, base_level * 1.5 + temporal_trend * 1.2 + 0.12 * np.random.randn())  # Softmax entropy
        signals_t[8] = min(1.0, max(0, 0.7 - temporal_trend * 0.3 + 0.1 * np.random.randn()))  # Max prob

        # Signals 10-11: Physics Checks
        signals_t[9] = max(0, base_level + temporal_trend + 0.05 * np.random.randn())  # Execution mismatch
        signals_t[10] = max(0, base_level + temporal_trend * 0.4 + 0.06 * np.random.randn())  # Constraint margin

        # Signal 12: Temporal Consistency
        signals_t[11] = max(0, base_level + temporal_trend * 0.6 + 0.08 * np.random.randn())  # Volatility std

        # Random action and robot state
        action_t = np.random.randn(6).astype(np.float32) * 0.1
        robot_state_t = np.random.randn(7).astype(np.float32) * 0.5

        all_signals.append(signals_t)
        all_actions.append(action_t)
        all_robot_state.append(robot_state_t)
        all_episode_ids.append(ep)
        all_timesteps.append(t)
        all_success.append(is_success)
        all_done.append(t == max_steps - 1)

    if (ep + 1) % 10 == 0:
        print(f"  Generated {ep + 1}/{num_episodes} episodes...")

# Append all data
signals_ds.append(np.array(all_signals))
actions_ds.append(np.array(all_actions))
robot_state_ds.append(np.array(all_robot_state))
episode_ids_ds.append(np.array(all_episode_ids))
timesteps_ds.append(np.array(all_timesteps))
success_ds.append(np.array(all_success))
done_ds.append(np.array(all_done))

# Save metadata
total_steps = num_episodes * max_steps
root.attrs['num_episodes'] = num_episodes
root.attrs['total_steps'] = total_steps
root.attrs['successes'] = successes
root.attrs['failures'] = failures
root.attrs['signal_dim'] = 12
root.attrs['action_dim'] = 6
root.attrs['collection_date'] = timestamp

print(f"\n" + "="*70)
print("DATA GENERATION COMPLETE")
print("="*70)

print(f"\nStatistics:")
print(f"  Total episodes: {num_episodes}")
print(f"  Total steps: {total_steps}")
print(f"  Successes: {successes} ({successes/num_episodes*100:.1f}%)")
print(f"  Failures: {failures} ({failures/num_episodes*100:.1f}%)")

print(f"\nData saved to: {data_path}")
print(f"  Signals shape: {signals_ds.shape}")
print(f"  Signal dimension: 12D (single-model)")

print(f"\n✅ Ready for SALUS training!")
print("="*70 + "\n")

print("Next step:")
print(f"  python train_salus_local.py")
