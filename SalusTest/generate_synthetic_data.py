"""
Generate synthetic training data for SALUS testing.
FAST - no VLA, no Isaac Lab, just synthetic 18D signals with realistic patterns.

This lets us test if SALUS can learn WITHOUT waiting for slow data collection.
"""

import numpy as np
import zarr
from pathlib import Path
from datetime import datetime

print("\n" + "="*70)
print("GENERATE SYNTHETIC DATA FOR SALUS")
print("="*70)

# Config
num_episodes = 100
steps_per_episode = 80
signal_dim = 18

output_dir = Path("local_data")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_path = output_dir / f"synthetic_data_{timestamp}.zarr"

print(f"\nGenerating {num_episodes} episodes with {steps_per_episode} steps each")
print(f"Output: {data_path}")

# Initialize Zarr
root = zarr.open(str(data_path), mode='w')

all_signals = []
all_actions = []
all_robot_states = []
all_episode_ids = []
all_timesteps = []
all_success = []
all_done = []

print(f"\nGenerating episodes...")

successes = 0
failures = 0

for ep in range(num_episodes):
    # Decide if this episode will succeed (60% success rate)
    will_succeed = np.random.random() < 0.6

    # Generate base signals
    episode_signals = []

    for t in range(steps_per_episode):
        # Create 18D signal with realistic patterns
        signals = np.zeros(18, dtype=np.float32)

        if will_succeed:
            # SUCCESS PATTERN: Low uncertainty, decreasing over time
            progress = t / steps_per_episode

            signals[0] = 0.02 + np.random.randn() * 0.01  # Low epistemic
            signals[1] = 0.5 + np.random.randn() * 0.1    # Moderate action mag
            signals[2] = 0.03 + np.random.randn() * 0.01  # Low action var
            signals[3] = 0.02 * (1 - progress)            # Decreasing smoothness
            signals[4] = 0.01 + np.random.randn() * 0.005 # Low divergence
            signals[5:8] = np.random.randn(3) * 0.02      # Low joint var
            signals[8:12] = np.random.randn(4) * 0.02     # Low unc stats
            signals[12] = 0.05 * (1 - progress)           # Decreasing latent drift
            signals[13] = 0.1 + np.random.randn() * 0.05  # Low OOD
            signals[14] = 0.02 + np.random.randn() * 0.01 # High stability
            signals[15] = 0.1 + np.random.randn() * 0.05  # Low sensitivity
            signals[16] = 0.05 * (1 - progress)           # Decreasing mismatch
            signals[17] = 0.2 + np.random.randn() * 0.1   # Safe constraints

        else:
            # FAILURE PATTERN: High uncertainty, INCREASING over time
            progress = t / steps_per_episode

            # Failure point around 70% through episode
            failure_point = 0.7
            approaching_failure = max(0, (progress - failure_point + 0.3) / 0.3)

            signals[0] = 0.08 + approaching_failure * 0.3 + np.random.randn() * 0.02  # HIGH epistemic
            signals[1] = 0.8 + approaching_failure * 0.5 + np.random.randn() * 0.2   # High action mag
            signals[2] = 0.1 + approaching_failure * 0.2 + np.random.randn() * 0.03  # High action var
            signals[3] = 0.1 + approaching_failure * 0.3                              # Increasing roughness
            signals[4] = 0.05 + approaching_failure * 0.2 + np.random.randn() * 0.02 # Increasing divergence
            signals[5:8] = np.random.randn(3) * 0.05 + approaching_failure * 0.1     # High joint var
            signals[8:12] = np.random.randn(4) * 0.05 + approaching_failure * 0.1    # High unc stats
            signals[12] = 0.15 + approaching_failure * 0.5                            # INCREASING latent drift
            signals[13] = 0.3 + approaching_failure * 0.8 + np.random.randn() * 0.1  # HIGH OOD
            signals[14] = 0.08 + approaching_failure * 0.2                            # Low stability
            signals[15] = 0.3 + approaching_failure * 0.5                             # HIGH sensitivity
            signals[16] = 0.15 + approaching_failure * 0.4                            # INCREASING mismatch
            signals[17] = 0.6 + approaching_failure * 0.4                             # UNSAFE constraints

        episode_signals.append(signals)

    # Convert to arrays
    episode_signals = np.array(episode_signals, dtype=np.float32)
    episode_actions = np.random.randn(steps_per_episode, 6).astype(np.float32) * 0.5
    episode_robot_states = np.random.randn(steps_per_episode, 7).astype(np.float32) * 0.5
    episode_ids = np.full(steps_per_episode, ep, dtype=np.int32)
    timesteps = np.arange(steps_per_episode, dtype=np.int32)
    success_labels = np.full(steps_per_episode, will_succeed, dtype=bool)
    done_flags = np.zeros(steps_per_episode, dtype=bool)
    done_flags[-1] = True

    # Append
    all_signals.append(episode_signals)
    all_actions.append(episode_actions)
    all_robot_states.append(episode_robot_states)
    all_episode_ids.append(episode_ids)
    all_timesteps.append(timesteps)
    all_success.append(success_labels)
    all_done.append(done_flags)

    if will_succeed:
        successes += 1
    else:
        failures += 1

    if (ep + 1) % 20 == 0:
        print(f"   Generated {ep+1}/{num_episodes} episodes")

# Concatenate all
all_signals = np.concatenate(all_signals, axis=0)
all_actions = np.concatenate(all_actions, axis=0)
all_robot_states = np.concatenate(all_robot_states, axis=0)
all_episode_ids = np.concatenate(all_episode_ids, axis=0)
all_timesteps = np.concatenate(all_timesteps, axis=0)
all_success = np.concatenate(all_success, axis=0)
all_done = np.concatenate(all_done, axis=0)

print(f"\n✅ Generated {len(all_signals):,} timesteps")

# Save to Zarr
print(f"\nSaving to Zarr...")
root['signals'] = all_signals
root['actions'] = all_actions
root['robot_state'] = all_robot_states
root['episode_id'] = all_episode_ids
root['timestep'] = all_timesteps
root['success'] = all_success
root['done'] = all_done

# Metadata
root.attrs['num_episodes'] = num_episodes
root.attrs['total_steps'] = len(all_signals)
root.attrs['successes'] = successes
root.attrs['failures'] = failures
root.attrs['signal_dim'] = signal_dim
root.attrs['action_dim'] = 6
root.attrs['collection_date'] = timestamp
root.attrs['synthetic'] = True

print(f"✅ Saved to {data_path}")

print(f"\n{'='*70}")
print("SYNTHETIC DATA COMPLETE")
print(f"{'='*70}")
print(f"\nStatistics:")
print(f"   Episodes: {num_episodes}")
print(f"   Total steps: {len(all_signals):,}")
print(f"   Successes: {successes} ({successes/num_episodes*100:.1f}%)")
print(f"   Failures: {failures} ({failures/num_episodes*100:.1f}%)")
print(f"\n✅ Ready to train SALUS: python train_salus_local.py")
print(f"{'='*70}\n")
