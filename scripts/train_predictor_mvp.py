"""
SALUS MVP Predictor Training
Simple training script for the MVP failure predictor

Usage:
    python scripts/train_predictor_mvp.py --data data/mvp_episodes/20260102_120000
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import json

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from salus.core.predictor_mvp import SALUSPredictorMVP, SimpleBCELoss
from salus.data.dataset_mvp import create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for signals, labels in pbar:
        signals = signals.to(device)
        labels = labels.to(device)

        # Forward
        output = model(signals)
        loss = criterion(output['logits'], labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # Metrics
    correct = 0
    total = 0
    tp = torch.zeros(4)  # True positives per class
    fp = torch.zeros(4)  # False positives
    fn = torch.zeros(4)  # False negatives

    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward
            output = model(signals)
            loss = criterion(output['logits'], labels)

            total_loss += loss.item()
            num_batches += 1

            # Predictions
            probs = output['probs']  # (B, 4)
            pred_labels = (probs > 0.5).float()  # Threshold at 0.5

            # Compute metrics
            for i in range(4):
                tp[i] += ((pred_labels[:, i] == 1) & (labels[:, i] == 1)).sum().item()
                fp[i] += ((pred_labels[:, i] == 1) & (labels[:, i] == 0)).sum().item()
                fn[i] += ((pred_labels[:, i] == 0) & (labels[:, i] == 1)).sum().item()

            # Overall accuracy (any correct prediction)
            correct += (pred_labels == labels).all(dim=1).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)

    # Compute per-class metrics
    precision = []
    recall = []
    f1 = []

    for i in range(4):
        p = tp[i] / (tp[i] + fp[i] + 1e-8)
        r = tp[i] / (tp[i] + fn[i] + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)

        precision.append(p.item())
        recall.append(r.item())
        f1.append(f.item())

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_f1': sum(f1) / 4
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train SALUS MVP Predictor')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data directory (with data.zarr)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/mvp',
                        help='Where to save checkpoints')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Limit number of episodes (for testing)')

    args = parser.parse_args()

    print("="*70)
    print("SALUS MVP Predictor Training")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Data: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Device: {args.device}")

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Checkpoints: {checkpoint_dir}")

    # Setup tensorboard
    writer = SummaryWriter(log_dir=checkpoint_dir / 'logs')

    # Save config
    config = vars(args)
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print(f"\n💾 Loading Data...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        train_ratio=0.8,
        max_episodes=args.max_episodes,
        num_workers=2
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Create model
    print(f"\n🧠 Creating Model...")
    device = torch.device(args.device)

    model = SALUSPredictorMVP(
        signal_dim=6,
        hidden_dim=64,
        num_failure_types=4
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # Create loss and optimizer
    # With 6% positive samples, need pos_weight ~16 to balance
    criterion = SimpleBCELoss(pos_weight=16.0)  # Weight failures MUCH higher for 94/6 imbalance
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\n🚀 Starting Training...")
    print("="*70 + "\n")

    best_val_loss = float('inf')
    best_val_f1 = 0.0

    try:
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # Validate
            val_metrics = validate(model, val_loader, criterion, device)

            # Log
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val Accuracy: {val_metrics['accuracy']:.3f}")
            print(f"   Val Mean F1: {val_metrics['mean_f1']:.3f}")
            print(f"   Per-class F1: {[f'{f:.3f}' for f in val_metrics['f1']]}")

            # Tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('F1/mean', val_metrics['mean_f1'], epoch)

            for i, f1 in enumerate(val_metrics['f1']):
                writer.add_scalar(f'F1/class_{i}', f1, epoch)

            # Learning rate schedule
            scheduler.step(val_metrics['loss'])

            # Save best model (by loss)
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_f1': val_metrics['mean_f1']
                }, checkpoint_dir / 'best_loss.pth')
                print(f"   💾 Saved best model (loss)")

            # Save best model (by F1)
            if val_metrics['mean_f1'] > best_val_f1:
                best_val_f1 = val_metrics['mean_f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_f1': val_metrics['mean_f1']
                }, checkpoint_dir / 'best_f1.pth')
                print(f"   💾 Saved best model (F1)")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_f1': val_metrics['mean_f1']
                }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')

            print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    finally:
        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_dir / 'final.pth')

        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"\n📊 Best Results:")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Best Val F1: {best_val_f1:.3f}")
        print(f"\n💾 Saved to: {checkpoint_dir}")
        print(f"\n📋 Next steps:")
        print(f"   1. Evaluate: python scripts/evaluate_mvp.py --checkpoint {checkpoint_dir}/best_f1.pth --data {args.data}")
        print(f"   2. Deploy: Integrate predictor into control loop")

        writer.close()


if __name__ == "__main__":
    main()
