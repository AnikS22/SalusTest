"""
Temporal Forecasting Evaluation for SALUS
Analyzes prediction performance at different time horizons.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import zarr
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from salus.models.failure_predictor import FailurePredictor


def load_model_and_data(model_path: str, data_path: str, device: str = 'cuda:0'):
    """Load trained model and test data."""

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    # Determine model size from checkpoint config
    if 'config' in checkpoint:
        hidden_dims = checkpoint['config'].get('hidden_dims', [64, 128, 128, 64])
    else:
        # Try to infer from state dict
        state_dict = checkpoint['model_state_dict']
        if 'network.4.weight' in state_dict:
            hidden_dims = [64, 128, 128, 64]
        else:
            hidden_dims = [128, 256, 256, 256, 128, 64]

    model = FailurePredictor(
        input_dim=12,
        hidden_dims=hidden_dims,
        num_horizons=4,
        num_failure_types=4
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load data
    store = zarr.open(str(data_path), mode='r')
    signals = store['signals'][:]
    labels = store['horizon_labels'][:]
    actions = store['actions'][:]

    # Handle NaN
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

    # Valid mask
    masks = (np.abs(actions).sum(axis=-1) > 0).astype(np.float32)

    print(f"✅ Data loaded: {signals.shape[0]} episodes")

    return model, signals, labels, masks


def evaluate_per_horizon(
    model, signals, labels, masks,
    horizon_names=['200ms', '300ms', '400ms', '500ms'],
    device='cuda:0'
):
    """
    Evaluate model performance at each prediction horizon.

    Returns per-horizon metrics.
    """
    num_episodes, max_timesteps, _ = signals.shape
    num_horizons = 4
    num_failure_types = 4

    # Initialize per-horizon metrics
    horizon_metrics = []

    with torch.no_grad():
        for horizon_idx in range(num_horizons):
            print(f"\nEvaluating Horizon {horizon_idx+1}/4 ({horizon_names[horizon_idx]})...")

            all_preds = []
            all_labels = []

            # Process in batches for memory efficiency
            batch_size = 32

            for ep_start in tqdm(range(0, num_episodes, batch_size), desc=f"  {horizon_names[horizon_idx]}"):
                ep_end = min(ep_start + batch_size, num_episodes)

                # Get batch
                batch_signals = torch.FloatTensor(signals[ep_start:ep_end]).to(device)
                batch_labels = torch.FloatTensor(labels[ep_start:ep_end]).to(device)
                batch_masks = torch.FloatTensor(masks[ep_start:ep_end]).to(device)

                # Reshape: (B, T, 12) -> (B*T, 12)
                B, T, D = batch_signals.shape
                batch_signals_flat = batch_signals.reshape(B * T, D)

                # Forward pass
                predictions = model(batch_signals_flat)  # (B*T, 16)

                # Reshape back: (B*T, 16) -> (B, T, 4, 4)
                predictions = predictions.reshape(B, T, num_horizons, num_failure_types)
                batch_labels = batch_labels.reshape(B, T, num_horizons, num_failure_types)

                # Extract predictions for this horizon
                pred_horizon = predictions[:, :, horizon_idx, :]  # (B, T, 4)
                label_horizon = batch_labels[:, :, horizon_idx, :]  # (B, T, 4)
                mask_horizon = batch_masks  # (B, T)

                # Flatten and apply mask
                pred_flat = pred_horizon.reshape(-1, num_failure_types)  # (B*T, 4)
                label_flat = label_horizon.reshape(-1, num_failure_types)  # (B*T, 4)
                mask_flat = mask_horizon.reshape(-1)  # (B*T,)

                # Keep only valid timesteps
                valid_idx = mask_flat > 0
                pred_valid = pred_flat[valid_idx]
                label_valid = label_flat[valid_idx]

                all_preds.append(pred_valid.cpu())
                all_labels.append(label_valid.cpu())

            # Concatenate all batches
            all_preds = torch.cat(all_preds, dim=0)  # (N, 4)
            all_labels = torch.cat(all_labels, dim=0)  # (N, 4)

            # Compute metrics (considering any failure type)
            pred_any_failure = (all_preds.max(dim=1)[0] > 0.5).float()
            label_any_failure = (all_labels.max(dim=1)[0] > 0.5).float()

            tp = ((pred_any_failure == 1) & (label_any_failure == 1)).sum().item()
            fp = ((pred_any_failure == 1) & (label_any_failure == 0)).sum().item()
            tn = ((pred_any_failure == 0) & (label_any_failure == 0)).sum().item()
            fn = ((pred_any_failure == 0) & (label_any_failure == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            horizon_metrics.append({
                'horizon': horizon_names[horizon_idx],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'total_samples': len(all_preds)
            })

            print(f"  F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    return horizon_metrics


def plot_temporal_performance(horizon_metrics, save_path):
    """Plot performance across time horizons."""

    horizons = [m['horizon'] for m in horizon_metrics]
    f1_scores = [m['f1'] for m in horizon_metrics]
    precisions = [m['precision'] for m in horizon_metrics]
    recalls = [m['recall'] for m in horizon_metrics]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(horizons))
    width = 0.25

    ax.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.8)
    ax.bar(x, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + width, recalls, width, label='Recall', alpha=0.8)

    ax.set_xlabel('Prediction Horizon', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('SALUS Temporal Forecasting Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {save_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate temporal forecasting")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--save_dir", type=str, default="a100_results", help="Save directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SALUS Temporal Forecasting Evaluation")
    print("=" * 70)
    print(f"\nModel: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Device: {args.device}")

    # Load model and data
    model, signals, labels, masks = load_model_and_data(
        args.model_path, args.data_path, args.device
    )

    # Evaluate per horizon
    horizon_metrics = evaluate_per_horizon(
        model, signals, labels, masks,
        horizon_names=['200ms', '300ms', '400ms', '500ms'],
        device=args.device
    )

    # Print summary
    print("\n" + "=" * 70)
    print("TEMPORAL FORECASTING RESULTS")
    print("=" * 70)
    print(f"\n{'Horizon':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10}")
    print("-" * 70)
    for m in horizon_metrics:
        print(f"{m['horizon']:<10} {m['f1']:<8.4f} {m['precision']:<10.4f} {m['recall']:<8.4f} {m['accuracy']:<10.4f}")
    print("=" * 70)

    # Compute average
    avg_f1 = np.mean([m['f1'] for m in horizon_metrics])
    print(f"\nAverage F1 across horizons: {avg_f1:.4f}")

    # Check degradation
    f1_200ms = horizon_metrics[0]['f1']
    f1_500ms = horizon_metrics[3]['f1']
    degradation = (f1_200ms - f1_500ms) / f1_200ms * 100 if f1_200ms > 0 else 0
    print(f"Performance degradation (200ms→500ms): {degradation:.1f}%")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'data_path': args.data_path,
        'horizon_metrics': horizon_metrics,
        'average_f1': avg_f1,
        'degradation_percent': degradation
    }

    results_path = save_dir / 'temporal_forecasting_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved: {results_path}")

    # Plot
    plot_path = save_dir / 'temporal_forecasting_performance.png'
    plot_temporal_performance(horizon_metrics, plot_path)

    print("\n✅ Temporal forecasting evaluation complete!")


if __name__ == "__main__":
    main()
