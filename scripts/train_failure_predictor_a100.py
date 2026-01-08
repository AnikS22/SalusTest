"""
A100-Optimized Training for SALUS Failure Predictor
Uses FP16 mixed precision, larger batches, and larger model.
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
from torch.cuda.amp import autocast, GradScaler
import zarr
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from salus.models.failure_predictor import FailurePredictor, FailurePredictorLoss


class SALUSDataset(Dataset):
    """Dataset for SALUS failure prediction (optimized for A100)."""

    def __init__(self, zarr_path: str, use_temporal: bool = False, cache_in_memory: bool = True):
        """
        Initialize dataset.

        Args:
            zarr_path: Path to zarr data
            use_temporal: If True, return full sequences
            cache_in_memory: Cache data in RAM for faster access
        """
        self.store = zarr.open(str(zarr_path), mode='r')
        self.use_temporal = use_temporal
        self.cache_in_memory = cache_in_memory

        # Load shapes
        self.num_episodes = self.store['actions'].shape[0]
        self.max_episode_length = self.store['actions'].shape[1]

        print(f"Dataset: {self.num_episodes} episodes")
        print(f"  Max episode length: {self.max_episode_length}")

        # Cache data in memory for faster training
        if cache_in_memory:
            print(f"  Loading data into RAM...")
            self.signals = self.store['signals'][:]
            self.labels = self.store['horizon_labels'][:]
            self.actions = self.store['actions'][:]

            # Handle NaN values
            self.signals = np.nan_to_num(self.signals, nan=0.0, posinf=0.0, neginf=0.0)

            # Compute valid masks
            self.masks = (np.abs(self.actions).sum(axis=-1) > 0).astype(np.float32)
            print(f"  ✅ Data cached in RAM")
        else:
            self.signals = self.store['signals']
            self.labels = self.store['horizon_labels']
            self.actions = self.store['actions']
            self.masks = None

        print(f"  Valid timesteps: {self.masks.sum() if self.masks is not None else 'computing...'}")

    def __len__(self):
        if self.use_temporal:
            return self.num_episodes
        else:
            return int(self.masks.sum()) if self.masks is not None else self.num_episodes * self.max_episode_length

    def __getitem__(self, idx):
        if self.use_temporal:
            return {
                'signals': torch.FloatTensor(self.signals[idx]),
                'labels': torch.FloatTensor(self.labels[idx]),
                'mask': torch.FloatTensor(self.masks[idx]) if self.masks is not None else torch.ones(self.max_episode_length)
            }
        else:
            # Map flat index to (episode, timestep)
            if self.masks is not None:
                cumsum = np.cumsum(self.masks.sum(axis=1))
                episode_idx = np.searchsorted(cumsum, idx, side='right')
                timestep_idx = idx - (cumsum[episode_idx - 1] if episode_idx > 0 else 0)
                valid_indices = np.where(self.masks[episode_idx])[0]
                timestep_idx = valid_indices[int(timestep_idx)]
            else:
                episode_idx = idx // self.max_episode_length
                timestep_idx = idx % self.max_episode_length

            return {
                'signals': torch.FloatTensor(self.signals[episode_idx, timestep_idx]),
                'labels': torch.FloatTensor(self.labels[episode_idx, timestep_idx])
            }


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_amp=True):
    """Train for one epoch with FP16 mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        signals = batch['signals'].to(device)
        labels = batch['labels'].to(device)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if use_amp and scaler is not None:
            with autocast():
                predictions = model(signals)
                loss = criterion(predictions, labels, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(signals)
            loss = criterion(predictions, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device, use_amp=True):
    """Evaluate model with FP16."""
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

            # Mixed precision inference
            if use_amp:
                with autocast():
                    predictions = model(signals)
                    loss = criterion(predictions, labels, mask)
            else:
                predictions = model(signals)
                loss = criterion(predictions, labels, mask)

            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    pred_binary = (all_predictions > 0.5).float()

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
    print("SALUS Failure Predictor Training - A100 Optimized")
    print("=" * 70)

    # Configuration for A100
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to zarr data")
    parser.add_argument("--save_dir", type=str, default="a100_checkpoints", help="Save directory")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (A100: 1024-2048)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model_size", type=str, default="large", choices=["small", "medium", "large"], help="Model size")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use FP16 mixed precision")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    args = parser.parse_args()

    zarr_path = Path(args.data_path)
    save_dir = project_root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Configuration:")
    print(f"   Data: {zarr_path}")
    print(f"   Device: {device}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Model size: {args.model_size}")
    print(f"   Mixed precision (FP16): {args.use_amp}")
    print(f"   Save dir: {save_dir}")

    # Load dataset
    print(f"\n📦 Loading Dataset...")
    dataset = SALUSDataset(str(zarr_path), use_temporal=False, cache_in_memory=True)

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

    # Create dataloaders with more workers for A100
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )

    # Create model (larger for A100)
    print(f"\n🤖 Creating Model ({args.model_size})...")
    model_configs = {
        'small': [64, 128, 128, 64],  # 35K params (baseline)
        'medium': [128, 256, 256, 128, 64],  # ~100K params
        'large': [128, 256, 256, 256, 128, 64]  # ~200K params
    }

    model = FailurePredictor(
        input_dim=12,
        hidden_dims=model_configs[args.model_size],
        num_horizons=4,
        num_failure_types=4,
        dropout=0.3  # More dropout for larger model
    ).to(device)

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = FailurePredictorLoss(pos_weight=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None

    # Training loop
    print(f"\n🚀 Starting Training...")
    best_val_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, args.use_amp)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, args.use_amp)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])

        # Print metrics
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")

        # Scheduler step
        scheduler.step()
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history,
                'config': vars(args)
            }, save_dir / 'best_predictor_a100.pt')
            print(f"   ✅ Best model saved (F1: {best_val_f1:.4f})")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, save_dir / f'checkpoint_epoch{epoch+1}.pt')

    # Test on best model
    print(f"\n📊 Testing Best Model...")
    checkpoint = torch.load(save_dir / 'best_predictor_a100.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device, args.use_amp)

    print(f"\n{'='*70}")
    print(f"FINAL TEST RESULTS (A100)")
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
        'model_size': args.model_size,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'use_amp': args.use_amp,
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'history': history
    }

    with open(save_dir / 'training_results_a100.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ A100 Training complete!")
    print(f"   Best model: {save_dir / 'best_predictor_a100.pt'}")
    print(f"   Results: {save_dir / 'training_results_a100.json'}")


if __name__ == "__main__":
    main()
