"""
Causal Ablation Study for SALUS Signal Groups

This script performs ablation experiments to measure the importance of each
signal group (temporal, internal, uncertainty, physics) for failure prediction.

Usage:
    python scripts/ablate_signals.py --data_path paper_data/dataset.zarr --epochs 50
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import zarr
import numpy as np
from tqdm import tqdm
import json
import csv
from datetime import datetime
from sklearn.metrics import roc_auc_score
from salus.core.predictor import SALUSPredictor, MultiHorizonFocalLoss


# Signal group definitions (12D signal indices)
SIGNAL_GROUPS = {
    'full': None,  # Baseline (all signals)
    'temporal': [0, 1, 2, 3, 11],  # Temporal action dynamics + consistency
    'internal': [4, 5, 6],  # VLA internal stability
    'uncertainty': [7, 8],  # Model uncertainty
    'physics': [9, 10],  # Physics reality checks
}

# Ablation configurations (removing specific groups)
ABLATION_CONFIGS = {
    'full': None,  # No ablation
    'no_temporal': [4, 5, 6, 7, 8, 9, 10],  # Remove temporal
    'no_internal': [0, 1, 2, 3, 7, 8, 9, 10, 11],  # Remove internal
    'no_uncertainty': [0, 1, 2, 3, 4, 5, 6, 9, 10, 11],  # Remove uncertainty
    'no_physics': [0, 1, 2, 3, 4, 5, 6, 7, 8, 11],  # Remove physics
    'only_uncertainty': [7, 8],  # Only uncertainty
    'only_temporal': [0, 1, 2, 3, 11],  # Only temporal
}


class AblatedSALUSDataset(torch.utils.data.Dataset):
    """Dataset with signal ablation support."""

    def __init__(self, zarr_path: str, ablate_signals=None):
        """
        Args:
            zarr_path: Path to zarr data
            ablate_signals: List of signal indices to KEEP (others set to 0)
                           If None, use all signals
        """
        self.store = zarr.open(str(zarr_path), mode='r')
        self.ablate_signals = ablate_signals

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

        # Use only valid episodes
        self.signals = self.signals[:self.num_episodes]
        self.labels = self.labels[:self.num_episodes]
        self.actions = self.actions[:self.num_episodes]

        # Handle NaN values
        self.signals = np.nan_to_num(self.signals, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply ablation (mask out signals not in ablate_signals)
        if self.ablate_signals is not None:
            mask = np.zeros(12, dtype=bool)
            mask[self.ablate_signals] = True
            self.signals[:, :, ~mask] = 0.0

        # Compute valid masks
        self.masks = (np.abs(self.actions).sum(axis=-1) > 0).astype(np.float32)

        print(f"  Episodes: {self.num_episodes}")
        print(f"  Valid timesteps: {int(self.masks.sum())}")
        if self.ablate_signals is not None:
            print(f"  Using signals: {self.ablate_signals}")

    def __len__(self):
        return int(self.masks.sum())

    def __getitem__(self, idx):
        # Map flat index to (episode, timestep)
        cumsum = np.cumsum(self.masks.sum(axis=1))
        episode_idx = np.searchsorted(cumsum, idx, side='right')
        timestep_idx = idx - (cumsum[episode_idx - 1] if episode_idx > 0 else 0)

        # Get valid timestep index
        valid_indices = np.where(self.masks[episode_idx])[0]
        timestep_idx = valid_indices[int(timestep_idx)]

        return {
            'signals': torch.FloatTensor(self.signals[episode_idx, timestep_idx]),
            'labels': torch.FloatTensor(self.labels[episode_idx, timestep_idx])
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        signals = batch['signals'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(signals)
        logits = outputs['logits']  # (B, 16)

        # Reshape labels from (B, 16) to (B, 4, 4) for loss
        labels_reshaped = labels.reshape(-1, 4, 4)
        loss, loss_dict = criterion(logits, labels_reshaped)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_model(model, dataloader, device):
    """Evaluate model and compute metrics."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            signals = batch['signals'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(signals)
            probs = outputs['probs']  # (B, 4, 4)

            # Flatten predictions and labels
            predictions = probs.flatten(1).cpu()  # (B, 16)
            labels_flat = labels.cpu()

            all_predictions.append(predictions)
            all_labels.append(labels_flat)

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Binary predictions
    pred_binary = (all_predictions > 0.5).astype(np.float32)

    # Compute metrics
    tp = ((pred_binary == 1) & (all_labels == 1)).sum()
    fp = ((pred_binary == 1) & (all_labels == 0)).sum()
    tn = ((pred_binary == 0) & (all_labels == 0)).sum()
    fn = ((pred_binary == 0) & (all_labels == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Compute AUROC
    try:
        auroc = roc_auc_score(all_labels.flatten(), all_predictions.flatten())
    except:
        auroc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def train_ablation(ablation_name, ablate_signals, args, device):
    """Train model with specific ablation."""
    print(f"\n{'='*70}")
    print(f"Ablation: {ablation_name}")
    print(f"{'='*70}")

    # Load dataset with ablation
    print("Loading dataset...")
    dataset = AblatedSALUSDataset(args.data_path, ablate_signals=ablate_signals)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # Create model
    model = SALUSPredictor(
        signal_dim=12,
        hidden_dims=[128, 256, 128],
        num_horizons=4,
        num_failure_types=4,
        dropout=0.2
    ).to(device)

    # Optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = MultiHorizonFocalLoss()

    # Training loop
    best_val_auroc = 0
    best_metrics = None

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = evaluate_model(model, val_loader, device)

        # Update best
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_metrics = val_metrics

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val AUROC={val_metrics['auroc']:.4f}, "
                  f"Val Recall={val_metrics['recall']:.4f}")

    # Final test evaluation
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate_model(model, test_loader, device)

    print(f"\nTest Metrics:")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='SALUS Signal Ablation Study')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to zarr dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per ablation')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Epochs per ablation: {args.epochs}")

    # Store results
    results = []

    # Run ablations
    for ablation_name, ablate_signals in ABLATION_CONFIGS.items():
        metrics = train_ablation(ablation_name, ablate_signals, args, device)

        results.append({
            'ablation': ablation_name,
            'signals_used': str(ablate_signals) if ablate_signals else 'all',
            'auroc': metrics['auroc'],
            'recall': metrics['recall'],
            'precision': metrics['precision'],
            'f1': metrics['f1'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn']
        })

    # Compute degradation relative to full model
    full_auroc = next(r['auroc'] for r in results if r['ablation'] == 'full')
    for result in results:
        result['auroc_drop'] = result['auroc'] - full_auroc
        result['auroc_drop_pct'] = (result['auroc_drop'] / full_auroc) * 100 if full_auroc > 0 else 0

    # Save results to CSV
    csv_path = output_dir / 'ablation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {csv_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Ablation':<20} {'AUROC':>8} {'Recall':>8} {'F1':>8} {'Drop':>8}")
    print(f"{'-'*70}")

    for result in results:
        print(f"{result['ablation']:<20} "
              f"{result['auroc']:>8.4f} "
              f"{result['recall']:>8.4f} "
              f"{result['f1']:>8.4f} "
              f"{result['auroc_drop']:>8.4f}")

    # Identify most critical signals
    print(f"\n{'='*70}")
    print("CRITICAL SIGNAL GROUPS (by AUROC drop)")
    print(f"{'='*70}")

    ablation_drops = [(r['ablation'], r['auroc_drop'])
                     for r in results if r['ablation'].startswith('no_')]
    ablation_drops.sort(key=lambda x: x[1])

    for ablation, drop in ablation_drops:
        signal_group = ablation.replace('no_', '')
        importance = 'HIGH' if drop < -0.1 else 'Medium' if drop < -0.05 else 'Low'
        print(f"  {signal_group:<15} Drop: {drop:>7.4f}  Importance: {importance}")

    print(f"\n✓ Ablation study complete!")


if __name__ == '__main__':
    main()
