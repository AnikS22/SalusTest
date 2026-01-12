"""
Evaluate trained SALUS predictor on test set
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import zarr
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import json
import argparse

from salus.core.predictor import SALUSPredictor


def load_dataset(zarr_path):
    """Load signals and labels from zarr dataset."""
    store = zarr.open(str(zarr_path), mode='r')

    signals = store['signals'][:]  # (N, T, 12)
    labels = store['horizon_labels'][:]  # (N, T, 16)
    actions = store['actions'][:]  # (N, T, 7)

    # Count valid episodes
    num_episodes = 0
    for i in range(signals.shape[0]):
        if actions[i].max() != 0:
            num_episodes += 1
        else:
            break

    # Use only valid episodes
    signals = signals[:num_episodes]
    labels = labels[:num_episodes]
    actions = actions[:num_episodes]

    # Handle NaN
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute valid masks
    masks = (np.abs(actions).sum(axis=-1) > 0)

    print(f"Dataset loaded:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Valid timesteps: {masks.sum()}")

    # Flatten to (num_timesteps, signal_dim) and (num_timesteps, 16)
    valid_signals = []
    valid_labels = []

    for ep in range(num_episodes):
        for t in range(signals.shape[1]):
            if masks[ep, t]:
                valid_signals.append(signals[ep, t])
                valid_labels.append(labels[ep, t])

    valid_signals = np.array(valid_signals)  # (num_timesteps, 12)
    valid_labels = np.array(valid_labels)  # (num_timesteps, 16)

    # Binary labels (any failure at any horizon)
    binary_labels = (valid_labels.sum(axis=1) > 0).astype(np.float32)

    print(f"  Failure rate: {binary_labels.mean():.2%}")

    return valid_signals, valid_labels, binary_labels


def evaluate_model(model, signals, labels, binary_labels, device):
    """Evaluate model and return metrics."""
    model.eval()

    # Convert to tensors
    signals_tensor = torch.FloatTensor(signals).to(device)
    labels_tensor = torch.FloatTensor(labels).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(signals_tensor)
        probs = outputs['probs']  # (N, 4, 4)
        max_prob = outputs['max_prob']  # (N,)

    # Use max probability as failure score
    scores = max_prob.cpu().numpy()

    # Compute AUROC
    try:
        auroc = roc_auc_score(binary_labels, scores)
    except:
        auroc = 0.0

    # Find optimal threshold (maximize F1)
    thresholds = np.percentile(scores, [70, 75, 80, 85, 90, 95, 98, 99])
    best_f1 = 0
    best_threshold = thresholds[0] if len(thresholds) > 0 else 0.5
    best_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'tp': 0,
        'fp': 0,
        'tn': 0,
        'fn': 0,
        'threshold': float(best_threshold)
    }

    for threshold in thresholds:
        predictions = (scores > threshold).astype(np.float32)

        tp = ((predictions == 1) & (binary_labels == 1)).sum()
        fp = ((predictions == 1) & (binary_labels == 0)).sum()
        tn = ((predictions == 0) & (binary_labels == 0)).sum()
        fn = ((predictions == 0) & (binary_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'threshold': float(threshold)
            }

    # Compute false alarm rate
    false_alarm_rate = best_metrics['fp'] / (best_metrics['fp'] + best_metrics['tn']) if (best_metrics['fp'] + best_metrics['tn']) > 0 else 0

    results = {
        'method': 'SALUS',
        'auroc': float(auroc),
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'false_alarm_rate': float(false_alarm_rate),
        'threshold': best_threshold,
        'tp': best_metrics['tp'],
        'fp': best_metrics['fp'],
        'tn': best_metrics['tn'],
        'fn': best_metrics['fn']
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SALUS Predictor')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to zarr dataset')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/mpcr/Desktop/SalusV3/checkpoints/salus_predictor_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, default='results/salus_results.json',
                       help='Output path for results')
    args = parser.parse_args()

    print("="*70)
    print("SALUS Predictor Evaluation")
    print("="*70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load dataset
    print("\nLoading dataset...")
    signals, multi_labels, binary_labels = load_dataset(args.data_path)

    # Load model
    print("\nLoading model...")
    model = SALUSPredictor(
        signal_dim=12,
        hidden_dims=[128, 256, 128],
        num_horizons=4,
        num_failure_types=4,
        dropout=0.2
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"  Loaded from: {checkpoint_path}")

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(model, signals, multi_labels, binary_labels, device)

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"  AUROC: {results['auroc']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  False Alarm Rate: {results['false_alarm_rate']:.2%}")
    print(f"  Optimal threshold: {results['threshold']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {results['tp']}, FP: {results['fp']}")
    print(f"    FN: {results['fn']}, TN: {results['tn']}")

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
