"""
Verify collected data quality
"""

import zarr
import numpy as np
from pathlib import Path
import json

data_dir = Path("data/mvp_episodes/20260103_082357")
zarr_path = data_dir / "data.zarr"

print("="*70)
print("DATA QUALITY VERIFICATION")
print("="*70)

# Load zarr
root = zarr.open_group(str(zarr_path), mode='r')

print(f"\n1. Zarr Structure:")
print(f"   Keys: {list(root.keys())}")

print(f"\n2. Array Shapes:")
for key in root.keys():
    if isinstance(root[key], zarr.Array):
        print(f"   {key}: {root[key].shape}, dtype={root[key].dtype}")

print(f"\n3. Episode-wise Data:")
num_episodes = root['images'].shape[0]
for ep in range(num_episodes):
    ep_len = root['images'][ep].shape[0]

    # Load metadata
    metadata_str = str(root['episode_metadata'][ep])
    metadata = json.loads(metadata_str)

    print(f"\n   Episode {ep}:")
    print(f"      Length: {ep_len} steps")
    print(f"      Success: {metadata['success']}")
    print(f"      Failure type: {metadata.get('failure_type', 'N/A')}")

print(f"\n4. Signal Statistics (checking for real values):")
signals = root['signals'][:]  # Load all signals

for ep in range(num_episodes):
    ep_signals = signals[ep]  # (T, 6)

    print(f"\n   Episode {ep}:")
    print(f"      Signal shape: {ep_signals.shape}")
    print(f"      Signal means: {np.mean(ep_signals, axis=0)}")
    print(f"      Signal stds: {np.std(ep_signals, axis=0)}")
    print(f"      Signal ranges: [{np.min(ep_signals, axis=0).round(3)}, {np.max(ep_signals, axis=0).round(3)}]")

    # Check if signals are all zeros (would indicate a bug)
    if np.allclose(ep_signals, 0):
        print(f"      ⚠️  WARNING: All signals are zero!")
    else:
        print(f"      ✅ Signals have variation")

print(f"\n5. Action Statistics (checking for real VLA output):")
actions = root['actions'][:]

for ep in range(num_episodes):
    ep_actions = actions[ep]  # (T, 7)

    print(f"\n   Episode {ep}:")
    print(f"      Action shape: {ep_actions.shape}")
    print(f"      Action means: {np.mean(ep_actions, axis=0).round(3)}")
    print(f"      Action stds: {np.std(ep_actions, axis=0).round(3)}")

    # Check if actions are all the same (would indicate frozen model)
    if np.allclose(ep_actions, ep_actions[0]):
        print(f"      ⚠️  WARNING: All actions are identical!")
    else:
        print(f"      ✅ Actions have variation")

print(f"\n" + "="*70)
print("DATA QUALITY CHECK COMPLETE")
print("="*70)
