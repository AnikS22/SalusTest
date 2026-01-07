"""
SALUS Temporal Predictor Training

Training script for the enhanced temporal forecasting system with:
1. Hybrid Conv+GRU temporal predictor
2. Temporal windows (not single timesteps)
3. Hard negative mining
4. Temporal smoothness regularization
5. Optional latent compression

Usage:
    # Basic hybrid predictor
    python scripts/train_temporal_predictor.py \\
        --data_dir ~/salus_a100/a100_data_temporal \\
        --epochs 100 \\
        --batch_size 64

    # With latent compression
    python scripts/train_temporal_predictor.py \\
        --data_dir ~/salus_a100/a100_data_temporal \\
        --use_latent_encoder \\
        --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from salus.models.temporal_predictor import (
    HybridTemporalPredictor,
    TemporalFocalLoss,
    TemporalSmoothnessLoss,
    compute_temporal_stability
)
from salus.models.latent_encoder import LatentTemporalPredictor
from salus.data.temporal_dataset import create_temporal_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train SALUS temporal predictor")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with data.zarr")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Temporal window size (default: 10 = 333ms at 30Hz)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to load (for debugging)")

    # Model
    parser.add_argument("--use_latent_encoder", action="store_true",
                        help="Use latent compression (item 4 from plan)")
    parser.add_argument("--signal_dim", type=int, default=12,
                        help="Signal dimension (default: 12)")
    parser.add_argument("--conv_dim", type=int, default=32,
                        help="Conv1d channels (default: 32)")
    parser.add_argument("--gru_dim", type=int, default=64,
                        help="GRU hidden dimension (default: 64)")
    parser.add_argument("--latent_dim", type=int, default=6,
                        help="Latent dimension if using encoder (default: 6)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use mixed precision training")

    # Loss weights
    parser.add_argument("--pos_weight", type=float, default=3.0,
                        help="Positive class weight for focal loss")
    parser.add_argument("--fp_penalty_weight", type=float, default=2.0,
                        help="False positive penalty weight")
    parser.add_argument("--smoothness_weight", type=float, default=0.1,
                        help="Temporal smoothness weight")
    parser.add_argument("--use_hard_negatives", action="store_true",
                        help="Use hard negative mining")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints/temporal",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")

    return parser.parse_args()


def train_epoch(
    model,
    train_loader,
    main_criterion,
    smoothness_criterion,
    optimizer,
    scaler,
    device,
    epoch,
    use_fp16=False,
    use_latent=False
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_main_loss = 0.0
    total_smooth_loss = 0.0
    num_batches = 0

    # For latent models
    total_aux_losses = {'reconstruction': 0.0, 'predictive': 0.0, 'contrastive': 0.0}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    prev_predictions = None

    for batch in pbar:
        signals = batch['signals'].to(device)  # (B, T, signal_dim)
        labels = batch['labels'].to(device)  # (B, 16)
        episode_success = batch['episode_success'].to(device)  # (B,)

        # Forward pass
        if use_fp16:
            with autocast():
                if use_latent:
                    # Latent model computes all losses internally
                    total_batch_loss, loss_dict = model.compute_total_loss(
                        signals, labels, main_criterion
                    )
                    predictions = model(signals, return_latent=False)[0]
                else:
                    # Standard temporal predictor
                    predictions = model(signals)  # (B, 16)

                    # Main prediction loss
                    main_loss = main_criterion(predictions, labels, episode_success)

                    # Smoothness loss
                    smooth_loss = smoothness_criterion(predictions, prev_predictions)

                    # Total loss
                    total_batch_loss = main_loss + smooth_loss
        else:
            if use_latent:
                total_batch_loss, loss_dict = model.compute_total_loss(
                    signals, labels, main_criterion
                )
                predictions = model(signals, return_latent=False)[0]
            else:
                predictions = model(signals)
                main_loss = main_criterion(predictions, labels, episode_success)
                smooth_loss = smoothness_criterion(predictions, prev_predictions)
                total_batch_loss = main_loss + smooth_loss

        # Backward
        optimizer.zero_grad()
        if use_fp16:
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            optimizer.step()

        # Update previous predictions for smoothness
        prev_predictions = predictions.detach()

        # Stats
        total_loss += total_batch_loss.item()
        num_batches += 1

        if use_latent:
            pbar.set_postfix({
                'loss': total_batch_loss.item(),
                'main': loss_dict['main'],
                'recon': loss_dict.get('reconstruction', 0)
            })
            for key in ['reconstruction', 'predictive', 'contrastive']:
                if key in loss_dict:
                    total_aux_losses[key] += loss_dict[key]
        else:
            total_main_loss += main_loss.item()
            total_smooth_loss += smooth_loss.item()
            pbar.set_postfix({
                'loss': total_batch_loss.item(),
                'main': main_loss.item(),
                'smooth': smooth_loss.item()
            })

    avg_loss = total_loss / max(num_batches, 1)
    results = {'total_loss': avg_loss}

    if use_latent:
        for key in total_aux_losses:
            results[key] = total_aux_losses[key] / max(num_batches, 1)
    else:
        results['main_loss'] = total_main_loss / max(num_batches, 1)
        results['smooth_loss'] = total_smooth_loss / max(num_batches, 1)

    return results


def validate(model, val_loader, main_criterion, device, use_latent=False):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # Metrics
    all_predictions = []
    all_labels = []
    all_episode_success = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            signals = batch['signals'].to(device)
            labels = batch['labels'].to(device)
            episode_success = batch['episode_success'].to(device)

            # Forward
            if use_latent:
                predictions, _ = model(signals, return_latent=False)
            else:
                predictions = model(signals)

            # Loss
            loss = main_criterion(predictions, labels, episode_success)
            total_loss += loss.item()
            num_batches += 1

            # Collect for metrics
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_episode_success.append(episode_success.cpu())

    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)  # (N, 16)
    all_labels = torch.cat(all_labels, dim=0)  # (N, 16)
    all_episode_success = torch.cat(all_episode_success, dim=0)  # (N,)

    # Binary predictions (threshold 0.5)
    pred_binary = (all_predictions > 0.5).float()

    # Per-horizon metrics
    num_horizons = 4
    num_failure_types = 4

    # Reshape: (N, 16) → (N, 4 horizons, 4 types)
    pred_reshaped = pred_binary.reshape(-1, num_horizons, num_failure_types)
    labels_reshaped = all_labels.reshape(-1, num_horizons, num_failure_types)

    # Compute F1 per horizon
    horizon_f1 = []
    for h in range(num_horizons):
        pred_h = pred_reshaped[:, h, :].flatten()
        labels_h = labels_reshaped[:, h, :].flatten()

        tp = ((pred_h == 1) & (labels_h == 1)).sum().item()
        fp = ((pred_h == 1) & (labels_h == 0)).sum().item()
        fn = ((pred_h == 0) & (labels_h == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        horizon_f1.append(f1)

    # Overall F1
    tp = ((pred_binary == 1) & (all_labels == 1)).sum().item()
    fp = ((pred_binary == 1) & (all_labels == 0)).sum().item()
    fn = ((pred_binary == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Temporal stability (sample first 100 windows)
    sample_preds = all_predictions[:100].numpy()
    stability = compute_temporal_stability(sample_preds)

    avg_loss = total_loss / max(num_batches, 1)

    return {
        'loss': avg_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'horizon_f1': horizon_f1,
        'temporal_stability': stability
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("SALUS Temporal Predictor Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data_dir}")
    print(f"  Window size: {args.window_size} timesteps ({args.window_size * 1000 / 30:.0f}ms)")
    print(f"  Model: {'Latent+Temporal' if args.use_latent_encoder else 'Hybrid Conv+GRU'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hard negatives: {args.use_hard_negatives}")
    print(f"  Mixed precision: {args.use_fp16}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_temporal_dataloaders(
        data_dir=args.data_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        use_hard_negative_mining=args.use_hard_negatives,
        num_workers=2
    )

    # Create model
    print(f"\nInitializing model...")
    if args.use_latent_encoder:
        model = LatentTemporalPredictor(
            signal_dim=args.signal_dim,
            latent_dim=args.latent_dim,
            conv_dim=args.conv_dim // 2,  # Smaller for latent space
            gru_dim=args.gru_dim // 2
        ).to(device)
        print(f"  Using LatentTemporalPredictor (12D → {args.latent_dim}D latent → 16D)")
    else:
        model = HybridTemporalPredictor(
            signal_dim=args.signal_dim,
            conv_dim=args.conv_dim,
            gru_dim=args.gru_dim
        ).to(device)
        print(f"  Using HybridTemporalPredictor (Conv+GRU)")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")

    # Loss functions
    main_criterion = TemporalFocalLoss(
        pos_weight=args.pos_weight,
        fp_penalty_weight=args.fp_penalty_weight
    )

    smoothness_criterion = TemporalSmoothnessLoss(
        smoothness_weight=args.smoothness_weight
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Mixed precision scaler
    scaler = GradScaler() if args.use_fp16 else None

    # Tensorboard
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = save_dir / f"logs_{timestamp}"
    writer = SummaryWriter(log_dir)

    print(f"\nSaving to: {save_dir}")
    print(f"Tensorboard: tensorboard --logdir {log_dir}")

    # Training loop
    print("\nStarting training...")
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_results = train_epoch(
            model, train_loader, main_criterion, smoothness_criterion,
            optimizer, scaler, device, epoch,
            use_fp16=args.use_fp16,
            use_latent=args.use_latent_encoder
        )

        # Validate
        val_results = validate(
            model, val_loader, main_criterion, device,
            use_latent=args.use_latent_encoder
        )

        # Log
        writer.add_scalar('Loss/train', train_results['total_loss'], epoch)
        writer.add_scalar('Loss/val', val_results['loss'], epoch)
        writer.add_scalar('Metrics/F1', val_results['f1'], epoch)
        writer.add_scalar('Metrics/Precision', val_results['precision'], epoch)
        writer.add_scalar('Metrics/Recall', val_results['recall'], epoch)

        for h_idx, f1 in enumerate(val_results['horizon_f1']):
            writer.add_scalar(f'Horizon_F1/horizon_{h_idx}', f1, epoch)

        writer.add_scalar('Stability/variance', val_results['temporal_stability']['variance'], epoch)
        writer.add_scalar('Stability/autocorr', val_results['temporal_stability']['mean_autocorr'], epoch)

        # Learning rate
        scheduler.step(val_results['f1'])

        # Print results
        print(f"\nResults:")
        print(f"  Train loss: {train_results['total_loss']:.4f}")
        print(f"  Val loss: {val_results['loss']:.4f}")
        print(f"  F1: {val_results['f1']:.4f}")
        print(f"  Precision: {val_results['precision']:.4f}")
        print(f"  Recall: {val_results['recall']:.4f}")
        print(f"  Horizon F1: {[f'{f1:.3f}' for f1 in val_results['horizon_f1']]}")
        print(f"  Temporal stability: {val_results['temporal_stability']['mean_autocorr']:.4f}")

        # Save checkpoint
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            checkpoint_path = save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
                'args': vars(args)
            }, checkpoint_path)
            print(f"\n✅ Saved best model (F1: {best_f1:.4f})")

        if epoch % args.save_interval == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': val_results['f1'],
                'args': vars(args)
            }, checkpoint_path)

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best F1: {best_f1:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pt'}")

    writer.close()


if __name__ == "__main__":
    main()
