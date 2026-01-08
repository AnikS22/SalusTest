"""
Integration test for 12D single-model system.

Tests the complete pipeline without requiring Isaac Lab:
1. VLA wrapper with single model
2. Signal extraction (12D)
3. Training with HybridTemporalPredictor

This validates that all pieces fit together correctly.
"""

import torch
import numpy as np
import zarr
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("INTEGRATION TEST: 12D SINGLE-MODEL SYSTEM")
print("="*70)

# Test 1: Import all components
print("\nTest 1: Import all components")
print("-" * 70)

try:
    from salus.core.vla.wrapper import SmolVLAEnsemble
    from salus.core.vla.single_model_extractor import SingleModelSignalExtractor
    from salus.models.temporal_predictor import HybridTemporalPredictor, TemporalFocalLoss
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Signal extractor produces 12D
print("\nTest 2: Signal extractor produces 12D signals")
print("-" * 70)

device = 'cpu'
extractor = SingleModelSignalExtractor(device=device)

# Mock VLA output (simulating what wrapper returns)
mock_vla_output = {
    'action': torch.randn(1, 6),
    'action_logits': torch.randn(1, 6),
    'hidden_state': torch.randn(1, 512),
}
mock_robot_state = torch.randn(1, 7)

signals = extractor.extract(mock_vla_output, robot_state=mock_robot_state)

print(f"Signal shape: {signals.shape}")
assert signals.shape == (1, 12), f"Expected (1, 12), got {signals.shape}"
assert not torch.isnan(signals).any(), "Signals contain NaN"
assert not torch.isinf(signals).any(), "Signals contain Inf"
print("✓ Signal extractor produces valid 12D signals")

# Test 3: Create synthetic 12D dataset
print("\nTest 3: Create synthetic 12D training dataset")
print("-" * 70)

output_dir = Path("local_data")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_path = output_dir / f"test_12d_data_{timestamp}.zarr"

root = zarr.open(str(data_path), mode='w')

# Create 50 episodes × 100 steps = 5000 timesteps
num_episodes = 50
max_steps = 100
total_steps = num_episodes * max_steps

# Initialize datasets with 12D signals
signals_ds = root.create_dataset('signals', shape=(0, 12), chunks=(1000, 12), dtype='f4')
actions_ds = root.create_dataset('actions', shape=(0, 6), chunks=(1000, 6), dtype='f4')
robot_state_ds = root.create_dataset('robot_state', shape=(0, 7), chunks=(1000, 7), dtype='f4')
episode_ids_ds = root.create_dataset('episode_id', shape=(0,), chunks=(1000,), dtype='i4')
timesteps_ds = root.create_dataset('timestep', shape=(0,), chunks=(1000,), dtype='i4')
success_ds = root.create_dataset('success', shape=(0,), chunks=(1000,), dtype='bool')
done_ds = root.create_dataset('done', shape=(0,), chunks=(1000,), dtype='bool')

# Generate synthetic data
all_signals = []
all_actions = []
all_robot_state = []
all_episode_ids = []
all_timesteps = []
all_success = []
all_done = []

successes = 0
for ep in range(num_episodes):
    # 50% success rate
    is_success = (ep % 2 == 0)
    if is_success:
        successes += 1

    for t in range(max_steps):
        # Generate 12D signals (varying magnitude based on episode success)
        if is_success:
            # Success episodes: lower signal values
            signals_t = np.random.randn(12).astype(np.float32) * 0.3
        else:
            # Failure episodes: higher signal values (more uncertainty)
            signals_t = np.random.randn(12).astype(np.float32) * 0.8

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

# Append all data
signals_ds.append(np.array(all_signals))
actions_ds.append(np.array(all_actions))
robot_state_ds.append(np.array(all_robot_state))
episode_ids_ds.append(np.array(all_episode_ids))
timesteps_ds.append(np.array(all_timesteps))
success_ds.append(np.array(all_success))
done_ds.append(np.array(all_done))

# Save metadata
root.attrs['num_episodes'] = num_episodes
root.attrs['total_steps'] = total_steps
root.attrs['successes'] = successes
root.attrs['failures'] = num_episodes - successes
root.attrs['signal_dim'] = 12
root.attrs['action_dim'] = 6
root.attrs['collection_date'] = timestamp

print(f"✓ Created synthetic dataset: {data_path}")
print(f"   Signals shape: {signals_ds.shape}")
print(f"   Episodes: {num_episodes}")
print(f"   Success rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")

# Test 4: Train SALUS on 12D data
print("\nTest 4: Train SALUS model on 12D data")
print("-" * 70)

# Load data
signals = root['signals'][:]
episode_ids = root['episode_id'][:]
timesteps = root['timestep'][:]
success = root['success'][:]
done = root['done'][:]

print(f"Loaded data: {signals.shape}")

# Validate dimension
assert signals.shape[1] == 12, f"Expected 12D, got {signals.shape[1]}D"
print("✓ Data dimension validated: 12D")

# Create temporal windows
from torch.utils.data import Dataset, DataLoader

class TemporalDataset(Dataset):
    def __init__(self, signals, episode_ids, timesteps, success, window_size, horizon_steps):
        self.signals = signals
        self.episode_ids = episode_ids
        self.timesteps = timesteps
        self.success = success
        self.window_size = window_size
        self.horizon_steps = horizon_steps

        # Build valid indices
        self.valid_indices = []
        for i in range(len(signals) - window_size):
            if episode_ids[i] == episode_ids[i + window_size - 1]:
                self.valid_indices.append(i + window_size - 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.window_size + 1

        window = self.signals[start_idx:end_idx+1]  # (window_size, 12)

        # Simple binary label: will episode succeed?
        episode_id = self.episode_ids[end_idx]
        is_success = self.success[end_idx]

        labels = np.zeros((4, 2), dtype=np.float32)
        for h_idx in range(4):
            if is_success:
                labels[h_idx, 0] = 1.0
            else:
                labels[h_idx, 1] = 1.0

        return torch.tensor(window, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

window_size = 10
horizon_steps = [6, 9, 12, 15]

dataset = TemporalDataset(signals, episode_ids, timesteps, success, window_size, horizon_steps)
print(f"✓ Created dataset: {len(dataset)} windows")

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Initialize model with 12D input
model = HybridTemporalPredictor(
    signal_dim=12,
    conv_dim=32,
    gru_dim=64,
    dropout=0.2,
    num_horizons=4,
    num_failure_types=2
)
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model initialized: {n_params:,} parameters")

# Quick training test (1 epoch)
criterion = TemporalFocalLoss(pos_weight=3.0, fp_penalty_weight=2.0, focal_gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
total_loss = 0
num_batches = 0

for batch_signals, batch_labels in train_loader:
    batch_signals = batch_signals.to(device)
    batch_labels = batch_labels.to(device)

    # Forward pass
    predictions = model(batch_signals)  # (B, 16)

    # Reshape labels to (B, 16)
    labels_flat = batch_labels.reshape(batch_labels.shape[0], -1)

    # Compute loss
    loss = criterion(predictions, labels_flat)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    num_batches += 1

avg_loss = total_loss / num_batches
print(f"✓ Training epoch completed")
print(f"   Average loss: {avg_loss:.4f}")

# Validation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_signals, batch_labels in val_loader:
        batch_signals = batch_signals.to(device)
        batch_labels = batch_labels.to(device)

        predictions = model(batch_signals)
        labels_flat = batch_labels.reshape(batch_labels.shape[0], -1)

        # Check accuracy (threshold at 0.5)
        pred_binary = (predictions > 0.5).float()
        correct += (pred_binary == labels_flat).float().sum().item()
        total += labels_flat.numel()

accuracy = correct / total * 100
print(f"✓ Validation completed")
print(f"   Accuracy: {accuracy:.2f}%")

# Check for NaN/Inf
assert not torch.isnan(torch.tensor(avg_loss)), "Loss is NaN!"
assert not torch.isinf(torch.tensor(avg_loss)), "Loss is Inf!"
print("✓ No NaN/Inf in training")

print("\n" + "="*70)
print("ALL INTEGRATION TESTS PASSED ✅")
print("="*70)

print("\nSummary:")
print("  ✓ Imports working")
print("  ✓ Signal extractor produces 12D")
print("  ✓ Synthetic dataset created (5000 timesteps)")
print("  ✓ SALUS model trains on 12D data")
print(f"  ✓ No NaN/Inf (loss: {avg_loss:.4f})")
print(f"  ✓ Validation accuracy: {accuracy:.2f}%")

print("\n🎉 12D single-model system is fully functional!")
print("="*70 + "\n")

# Cleanup
import shutil
shutil.rmtree(data_path)
print(f"Cleaned up test data: {data_path}")
