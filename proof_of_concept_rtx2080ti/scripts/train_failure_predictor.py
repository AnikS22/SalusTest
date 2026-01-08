"""
Train SALUS Failure Predictor
Proof of concept on 50 episodes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import zarr
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from salus.models.failure_predictor import FailurePredictor, FailurePredictorLoss


class SALUSDataset(Dataset):
    """Dataset for SALUS failure prediction."""

    def __init__(self, zarr_path: str, use_temporal: bool = False):
        """
        Initialize dataset.

        Args:
            zarr_path: Path to zarr data
            use_temporal: If True, return full sequences; if False, return individual timesteps
        """
        self.store = zarr.open(str(zarr_path), mode='r')
        self.use_temporal = use_temporal

        # Load data
        self.signals = self.store['signals'][:]  # (N, T, 12)
        self.labels = self.store['horizon_labels'][:]  # (N, T, 16)
        self.actions = self.store['actions'][:]  # (N, T, 7)

        # Count valid episodes
        self.num_episodes = 0
        for i in range(self.signals.shape[0]):
            if self.actions[i].max() != 0:
                self.num_episodes += 1
            else:
                break

        print(f"Dataset: {self.num_episodes} episodes")
        print(f"  Signals: {self.signals.shape}")
        print(f"  Labels: {self.labels.shape}")

        # Use only valid episodes
        self.signals = self.signals[:self.num_episodes]
        self.labels = self.labels[:self.num_episodes]
        self.actions = self.actions[:self.num_episodes]

        # Handle NaN values in signals
        self.signals = np.nan_to_num(self.signals, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute valid masks (non-zero actions = valid timesteps)
        self.masks = (np.abs(self.actions).sum(axis=-1) > 0).astype(np.float32)  # (N, T)

        print(f"  Valid timesteps: {self.masks.sum():.0f} / {self.masks.size}")

    def __len__(self):
        if self.use_temporal:
            return self.num_episodes
        else:
            # Return number of valid timesteps across all episodes
            return int(self.masks.sum())

    def __getitem__(self, idx):
        if self.use_temporal:
            # Return full sequence
            return {
                'signals': torch.FloatTensor(self.signals[idx]),  # (T, 12)
                'labels': torch.FloatTensor(self.labels[idx]),  # (T, 16)
                'mask': torch.FloatTensor(self.masks[idx])  # (T,)
            }
        else:
            # Return single timestep
            # Map flat index to (episode, timestep)
            cumsum = np.cumsum(self.masks.sum(axis=1))
            episode_idx = np.searchsorted(cumsum, idx, side='right')
            timestep_idx = idx - (cumsum[episode_idx - 1] if episode_idx > 0 else 0)

            # Get valid timestep index
            valid_indices = np.where(self.masks[episode_idx])[0]
            timestep_idx = valid_indices[int(timestep_idx)]

            return {
                'signals': torch.FloatTensor(self.signals[episode_idx, timestep_idx]),  # (12,)
                'labels': torch.FloatTensor(self.labels[episode_idx, timestep_idx])  # (16,)
            }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        signals = batch['signals'].to(device)
        labels = batch['labels'].to(device)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(device)

        # Forward pass
        predictions = model(signals)
        loss = criterion(predictions, labels, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            signals = batch['signals'].to(device)
            labels = batch['labels'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass
            predictions = model(signals)
            loss = criterion(predictions, labels, mask)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions and labels
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Binary predictions (threshold = 0.5)
    pred_binary = (all_predictions > 0.5).float()

    # Compute metrics
    tp = ((pred_binary == 1) & (all_labels == 1)).sum().item()
    fp = ((pred_binary == 1) & (all_labels == 0)).sum().item()
    tn = ((pred_binary == 0) & (all_labels == 0)).sum().item()
    fn = ((pred_binary == 0) & (all_labels == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

    return metrics


def main():
    print("=" * 70)
    print("SALUS Failure Predictor Training - Proof of Concept")
    print("=" * 70)

    # Configuration
    zarr_path = project_root / "paper_data" / "training" / "data_run2" / "20260105_072308" / "data.zarr"
    save_dir = project_root / "paper_data" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Configuration:")
    print(f"   Data: {zarr_path}")
    print(f"   Device: {device}")
    print(f"   Save dir: {save_dir}")

    # Check if labels exist
    store = zarr.open(str(zarr_path), mode='r')
    if 'horizon_labels' not in store or store['horizon_labels'][:].sum() == 0:
        print(f"\n⚠️  Horizon labels not found or empty!")
        print(f"   Computing labels from episode data...")

        from salus.data.preprocess_labels import compute_all_labels
        labels, stats = compute_all_labels(str(zarr_path))
        print(f"   ✅ Labels computed and saved")
    else:
        print(f"   ✅ Horizon labels found")

    # Load dataset
    print(f"\n📦 Loading Dataset...")
    dataset = SALUSDataset(str(zarr_path), use_temporal=False)

    # Split dataset (80% train, 10% val, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Create model
    print(f"\n🤖 Creating Model...")
    model = FailurePredictor(
        input_dim=12,
        hidden_dims=[64, 128, 128, 64],
        num_horizons=4,
        num_failure_types=4,
        dropout=0.2
    ).to(device)

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = FailurePredictorLoss(pos_weight=3.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print(f"\n🚀 Starting Training...")
    num_epochs = 50
    best_val_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])

        # Print metrics
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")

        # Scheduler step
        scheduler.step(val_metrics['loss'])

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, save_dir / 'best_predictor.pt')
            print(f"   ✅ Best model saved (F1: {best_val_f1:.4f})")

    # Test on best model
    print(f"\n📊 Testing Best Model...")
    checkpoint = torch.load(save_dir / 'best_predictor.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*70}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*70}")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1 Score: {test_metrics['f1']:.4f}")
    print(f"{'='*70}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(zarr_path),
        'num_episodes': dataset.num_episodes,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'history': history
    }

    with open(save_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best model: {save_dir / 'best_predictor.pt'}")
    print(f"   Results: {save_dir / 'training_results.json'}")


if __name__ == "__main__":
    main()
