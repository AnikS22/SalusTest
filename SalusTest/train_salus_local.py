"""
Train SALUS on locally collected data.

Trains the HybridTemporalPredictor to predict failures from 12D signals.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import zarr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from salus.models.temporal_predictor import HybridTemporalPredictor, TemporalFocalLoss

print("\n" + "="*70)
print("TRAIN SALUS LOCALLY")
print("="*70)

# Find most recent data file
data_dir = Path("local_data")
data_files = sorted(data_dir.glob("salus_data_*.zarr"))

if not data_files:
    print(f"\n❌ No data files found in {data_dir}/")
    print(f"   Run collect_local_data.py first!")
    sys.exit(1)

data_path = data_files[-1]
print(f"\nData: {data_path}")

# Load data
print(f"\nLoading data from Zarr...")
root = zarr.open(str(data_path), mode='r')

signals = root['signals'][:]
episode_ids = root['episode_id'][:]
timesteps = root['timestep'][:]
success = root['success'][:]
done = root['done'][:]

print(f"✅ Data loaded")
print(f"   Signals shape: {signals.shape}")
print(f"   Episodes: {root.attrs.get('num_episodes', 'unknown')}")
print(f"   Success rate: {root.attrs.get('successes', 0)}/{root.attrs.get('num_episodes', 0)}")

# Validate signal dimension
expected_signal_dim = 12
actual_signal_dim = signals.shape[1]
if actual_signal_dim != expected_signal_dim:
    print(f"\n❌ ERROR: Data has {actual_signal_dim}D signals, but expected {expected_signal_dim}D!")
    print(f"   This data was collected with the old signal extractor.")
    print(f"   Please recollect data with updated collect_local_data.py")
    sys.exit(1)

# Create temporal windows
print(f"\nCreating temporal windows...")
window_size = 10  # 333ms at 30Hz
horizon_steps = [6, 9, 12, 15]  # 200ms, 300ms, 400ms, 500ms

class TemporalDataset(Dataset):
    def __init__(self, signals, episode_ids, timesteps, success, done, window_size, horizon_steps):
        self.signals = signals
        self.episode_ids = episode_ids
        self.timesteps = timesteps
        self.success = success
        self.done = done
        self.window_size = window_size
        self.horizon_steps = horizon_steps

        # Build valid indices (where we have full window)
        self.valid_indices = []
        for i in range(len(signals) - window_size):
            # Check if this is a valid window (same episode)
            if episode_ids[i] == episode_ids[i + window_size - 1]:
                self.valid_indices.append(i + window_size - 1)

        print(f"   Valid windows: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.window_size + 1

        # Get window
        window = self.signals[start_idx:end_idx+1]  # (window_size, 12)

        # Get labels (failure at each horizon)
        episode_id = self.episode_ids[end_idx]
        current_timestep = self.timesteps[end_idx]

        # Find where this episode ends
        episode_mask = self.episode_ids == episode_id
        episode_steps = self.timesteps[episode_mask]
        episode_success = self.success[episode_mask]
        max_timestep = episode_steps.max()
        is_success = episode_success[0]  # All same for episode

        # Label: will this episode fail within each horizon?
        labels = np.zeros((4, 2), dtype=np.float32)  # (4 horizons, 2 classes: success/failure)

        for h_idx, horizon in enumerate(self.horizon_steps):
            future_timestep = current_timestep + horizon

            if future_timestep >= max_timestep:
                # Episode will end before this horizon
                if is_success:
                    labels[h_idx, 0] = 1.0  # Success
                else:
                    labels[h_idx, 1] = 1.0  # Failure
            else:
                # Episode continues - no failure yet
                labels[h_idx, 0] = 1.0  # Continuing (treat as success)

        return torch.tensor(window, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

dataset = TemporalDataset(signals, episode_ids, timesteps, success, done, window_size, horizon_steps)

print(f"✅ Dataset ready: {len(dataset)} windows")

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"   Train: {len(train_dataset)}")
print(f"   Val: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
print(f"\nInitializing SALUS model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = HybridTemporalPredictor(
    signal_dim=12,
    conv_dim=32,
    gru_dim=64,
    dropout=0.2,
    num_horizons=4,
    num_failure_types=2  # success/failure binary
)
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model initialized ({n_params:,} parameters)")

# Loss and optimizer
criterion = TemporalFocalLoss(pos_weight=3.0, fp_penalty_weight=2.0, focal_gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"\n{'='*70}")
print("TRAINING")
print(f"{'='*70}\n")

best_val_loss = float('inf')
n_epochs = 50

for epoch in range(n_epochs):
    # Train
    model.train()
    train_loss = 0
    for batch_idx, (windows, labels) in enumerate(train_loader):
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(windows)

        # Reshape outputs and labels
        outputs = outputs.view(-1, 4, 2)  # (B, 4 horizons, 2 classes)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for windows, labels in val_loader:
            windows = windows.to(device)
            labels = labels.to(device)

            outputs = model(windows)
            outputs = outputs.view(-1, 4, 2)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(outputs, dim=-1)  # (B, 4)
            targets = torch.argmax(labels, dim=-1)  # (B, 4)
            correct += (preds == targets).sum().item()
            total += preds.numel()

    val_loss /= len(val_loader)
    val_acc = correct / total * 100

    scheduler.step(val_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{n_epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.2f}%")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "salus_best_local.pth")

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
print(f"\nBest val loss: {best_val_loss:.4f}")
print(f"Model saved: salus_best_local.pth")

# Test on validation set
print(f"\n{'='*70}")
print("TESTING FAILURE PREDICTION")
print(f"{'='*70}\n")

model.load_state_dict(torch.load("salus_best_local.pth"))
model.eval()

# Compute metrics per horizon
horizon_names = ["200ms", "300ms", "400ms", "500ms"]
with torch.no_grad():
    for h_idx, horizon_name in enumerate(horizon_names):
        correct = 0
        total = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        for windows, labels in val_loader:
            windows = windows.to(device)
            labels = labels.to(device)

            outputs = model(windows)
            outputs = outputs.view(-1, 4, 2)

            # Get predictions and targets for this horizon
            preds = torch.argmax(outputs[:, h_idx], dim=-1)  # (B,)
            targets = torch.argmax(labels[:, h_idx], dim=-1)  # (B,)

            correct += (preds == targets).sum().item()
            total += len(preds)

            # Confusion matrix (1 = failure)
            tp += ((preds == 1) & (targets == 1)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            tn += ((preds == 0) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()

        accuracy = correct / total * 100
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{horizon_name} horizon:")
        print(f"   Accuracy:  {accuracy:.2f}%")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1 Score:  {f1:.3f}")
        print(f"   TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")

print(f"{'='*70}")
print("SALUS IS TRAINED AND READY!")
print(f"{'='*70}\n")
