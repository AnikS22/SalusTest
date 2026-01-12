"""
Simple training script for SALUS Predictor
No IsaacLab imports needed - pure PyTorch training
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import zarr
import numpy as np
from tqdm import tqdm

# Import SALUS modules
from salus.core.predictor import SALUSPredictor, MultiHorizonFocalLoss

def main():
    print("="*70)
    print("SALUS Predictor Training")
    print("="*70)

    # Configuration
    # Use massive dataset (5000 episodes, 1M timesteps)
    zarr_path = project_root / "paper_data" / "massive_collection" / "20260109_215258" / "data_20260109_215321.zarr"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\nConfiguration:")
    print(f"  Dataset: {zarr_path}")
    print(f"  Device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    store = zarr.open(str(zarr_path), mode='r')

    signals = np.nan_to_num(store['signals'][:], nan=0.0)
    labels = store['horizon_labels'][:]
    actions = store['actions'][:]

    # Find valid episodes
    num_episodes = 0
    for i in range(signals.shape[0]):
        if actions[i].max() != 0:
            num_episodes += 1
        else:
            break

    signals = signals[:num_episodes]
    labels = labels[:num_episodes]
    actions = actions[:num_episodes]

    print(f"  Episodes: {num_episodes}")
    print(f"  Signals shape: {signals.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Flatten to timesteps
    masks = (np.abs(actions).sum(axis=-1) > 0)
    signals_flat = []
    labels_flat = []

    for ep in range(num_episodes):
        for t in range(signals.shape[1]):
            if masks[ep, t]:
                signals_flat.append(signals[ep, t])
                labels_flat.append(labels[ep, t])  # Keep as (4, 4) shape

    signals_flat = torch.FloatTensor(np.array(signals_flat))
    labels_flat = torch.FloatTensor(np.array(labels_flat))

    print(f"  Flattened: {len(signals_flat)} timesteps")
    # labels_flat shape is (N, 16) where 16 = 4 horizons × 4 types
    # Check if any failure at any horizon/type
    print(f"  Failure rate: {(labels_flat.sum(dim=1) > 0).float().mean():.2%}")

    # Split dataset
    train_size = int(0.8 * len(signals_flat))
    val_size = int(0.1 * len(signals_flat))

    indices = torch.randperm(len(signals_flat))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(signals_flat[train_idx], labels_flat[train_idx])
    val_dataset = torch.utils.data.TensorDataset(signals_flat[val_idx], labels_flat[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Create model
    print("\nCreating model...")
    model = SALUSPredictor(
        signal_dim=12,
        hidden_dims=[128, 256, 128],
        num_horizons=4,
        num_failure_types=4,
        dropout=0.2
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = MultiHorizonFocalLoss()

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')
    num_epochs = 100  # More epochs for larger dataset

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0

        for signals_batch, labels_batch in train_loader:
            signals_batch = signals_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(signals_batch)
            logits = outputs['logits']  # (B, 16)
            # Reshape labels from (B, 16) to (B, 4, 4) for loss
            labels_reshaped = labels_batch.reshape(-1, 4, 4)
            loss, loss_dict = criterion(logits, labels_reshaped)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for signals_batch, labels_batch in val_loader:
                signals_batch = signals_batch.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(signals_batch)
                logits = outputs['logits']
                labels_reshaped = labels_batch.reshape(-1, 4, 4)
                loss, loss_dict = criterion(logits, labels_reshaped)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = project_root.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "salus_predictor_massive.pth")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} (Best: {best_val_loss:.4f})")

    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {checkpoint_dir / 'salus_predictor_massive.pth'}")

    # Save final model too
    torch.save(model.state_dict(), checkpoint_dir / 'salus_predictor_final.pth')

if __name__ == '__main__':
    main()
