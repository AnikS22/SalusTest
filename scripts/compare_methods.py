"""
Compare SALUS vs. Baseline Methods

Runs all methods (baselines + SALUS) on the same dataset and generates
a comprehensive comparison table for the paper.

Usage:
    python scripts/compare_methods.py \\
        --data_path paper_data/dataset.zarr \\
        --checkpoint checkpoints/salus_predictor_best.pth
"""

import sys
from pathlib import Path
import argparse
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import zarr
import numpy as np
from sklearn.metrics import roc_auc_score
import csv
import json
from salus.models.failure_predictor import FailurePredictor


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


def evaluate_method(predictions, labels, method_name, inference_time=None):
    """Evaluate a method and compute metrics."""
    # Ensure predictions are 1D
    if len(predictions.shape) > 1:
        # For multi-horizon, take max across horizons
        predictions = predictions.max(axis=1)

    # Compute AUROC
    try:
        auroc = roc_auc_score(labels, predictions)
    except:
        auroc = 0.0

    # Find optimal threshold (maximize F1)
    thresholds = np.percentile(predictions, [70, 75, 80, 85, 90, 95, 98, 99])
    best_f1 = 0
    best_metrics = {}

    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(np.float32)

        tp = ((pred_binary == 1) & (labels == 1)).sum()
        fp = ((pred_binary == 1) & (labels == 0)).sum()
        tn = ((pred_binary == 0) & (labels == 0)).sum()
        fn = ((pred_binary == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }

    # False alarm rate
    false_alarm_rate = best_metrics['fp'] / (best_metrics['fp'] + best_metrics['tn']) if (best_metrics['fp'] + best_metrics['tn']) > 0 else 0

    return {
        'method': method_name,
        'auroc': float(auroc),
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'false_alarm_rate': float(false_alarm_rate),
        'inference_time_ms': float(inference_time * 1000) if inference_time else None,
        'tp': best_metrics['tp'],
        'fp': best_metrics['fp'],
        'tn': best_metrics['tn'],
        'fn': best_metrics['fn']
    }


def main():
    parser = argparse.ArgumentParser(description='Compare SALUS vs Baselines')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to zarr dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/salus_predictor_best.pth',
                       help='Path to SALUS checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    args = parser.parse_args()

    print("="*70)
    print("Method Comparison: SALUS vs Baselines")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    signals, multi_labels, binary_labels = load_dataset(args.data_path)

    results = []

    # =========================================================================
    # Method 1: Random Predictor (sanity check)
    # =========================================================================
    print("\n" + "="*70)
    print("Method 1: Random Predictor (sanity check)")
    print("="*70)

    np.random.seed(42)
    random_scores = np.random.rand(len(binary_labels))

    start_time = time.time()
    # Simulate inference time
    _ = random_scores.copy()
    inference_time = (time.time() - start_time) / len(binary_labels)

    random_results = evaluate_method(random_scores, binary_labels, "Random", inference_time)
    results.append(random_results)

    print(f"  AUROC: {random_results['auroc']:.4f}")
    print(f"  Recall: {random_results['recall']:.4f}")
    print(f"  Precision: {random_results['precision']:.4f}")
    print(f"  False Alarm Rate: {random_results['false_alarm_rate']:.2%}")
    print(f"  Inference: {random_results['inference_time_ms']:.3f} ms/sample")

    # =========================================================================
    # Method 2: Entropy Threshold
    # =========================================================================
    print("\n" + "="*70)
    print("Method 2: Entropy Threshold")
    print("="*70)

    entropy_scores = signals[:, 7]  # Softmax entropy

    start_time = time.time()
    _ = entropy_scores.copy()
    inference_time = (time.time() - start_time) / len(binary_labels)

    entropy_results = evaluate_method(entropy_scores, binary_labels, "Entropy Threshold", inference_time)
    results.append(entropy_results)

    print(f"  AUROC: {entropy_results['auroc']:.4f}")
    print(f"  Recall: {entropy_results['recall']:.4f}")
    print(f"  Precision: {entropy_results['precision']:.4f}")
    print(f"  False Alarm Rate: {entropy_results['false_alarm_rate']:.2%}")
    print(f"  Inference: {entropy_results['inference_time_ms']:.3f} ms/sample")

    # =========================================================================
    # Method 3: Action Variance Threshold
    # =========================================================================
    print("\n" + "="*70)
    print("Method 3: Action Variance Threshold")
    print("="*70)

    variance_scores = signals[:, 2]  # Action variance

    start_time = time.time()
    _ = variance_scores.copy()
    inference_time = (time.time() - start_time) / len(binary_labels)

    variance_results = evaluate_method(variance_scores, binary_labels, "Action Variance", inference_time)
    results.append(variance_results)

    print(f"  AUROC: {variance_results['auroc']:.4f}")
    print(f"  Recall: {variance_results['recall']:.4f}")
    print(f"  Precision: {variance_results['precision']:.4f}")
    print(f"  False Alarm Rate: {variance_results['false_alarm_rate']:.2%}")
    print(f"  Inference: {variance_results['inference_time_ms']:.3f} ms/sample")

    # =========================================================================
    # Method 4: SALUS (ours)
    # =========================================================================
    print("\n" + "="*70)
    print("Method 4: SALUS (ours)")
    print("="*70)

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FailurePredictor(
        input_dim=12,
        hidden_dims=[64, 128, 128, 64],
        num_horizons=4,
        num_failure_types=4,
        dropout=0.2
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
        print(f"  Using randomly initialized model (results will be poor)")

    model.eval()

    # Run inference
    print("  Running inference...")
    start_time = time.time()

    with torch.no_grad():
        signals_tensor = torch.FloatTensor(signals).to(device)

        # Process in batches for efficiency
        batch_size = 1024
        salus_predictions = []

        for i in range(0, len(signals_tensor), batch_size):
            batch = signals_tensor[i:i+batch_size]
            preds = model(batch)
            salus_predictions.append(preds.cpu().numpy())

        salus_predictions = np.concatenate(salus_predictions, axis=0)

    inference_time = (time.time() - start_time) / len(signals)

    salus_results = evaluate_method(salus_predictions, binary_labels, "SALUS (ours)", inference_time)
    results.append(salus_results)

    print(f"  AUROC: {salus_results['auroc']:.4f}")
    print(f"  Recall: {salus_results['recall']:.4f}")
    print(f"  Precision: {salus_results['precision']:.4f}")
    print(f"  False Alarm Rate: {salus_results['false_alarm_rate']:.2%}")
    print(f"  Inference: {salus_results['inference_time_ms']:.3f} ms/sample")

    # =========================================================================
    # Compute improvements
    # =========================================================================
    best_baseline = max([r for r in results if r['method'] != 'SALUS (ours)'],
                       key=lambda x: x['auroc'])

    salus_results['auroc_improvement'] = salus_results['auroc'] - best_baseline['auroc']
    salus_results['auroc_improvement_pct'] = (salus_results['auroc_improvement'] / best_baseline['auroc']) * 100 if best_baseline['auroc'] > 0 else 0

    salus_results['recall_improvement'] = salus_results['recall'] - best_baseline['recall']
    salus_results['recall_improvement_pct'] = (salus_results['recall_improvement'] / best_baseline['recall']) * 100 if best_baseline['recall'] > 0 else 0

    if best_baseline['false_alarm_rate'] > 0:
        salus_results['far_reduction'] = best_baseline['false_alarm_rate'] / salus_results['false_alarm_rate'] if salus_results['false_alarm_rate'] > 0 else float('inf')
    else:
        salus_results['far_reduction'] = 1.0

    # =========================================================================
    # Save results
    # =========================================================================

    # Save detailed JSON
    json_path = output_dir / 'method_comparison_detailed.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = output_dir / 'method_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("METHOD COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'AUROC':>8} {'Recall':>8} {'Prec':>8} {'FAR':>8} {'Time(ms)':>10}")
    print(f"{'-'*70}")

    for result in results:
        time_str = f"{result['inference_time_ms']:.3f}" if result['inference_time_ms'] else "N/A"
        print(f"{result['method']:<20} "
              f"{result['auroc']:>8.4f} "
              f"{result['recall']:>8.4f} "
              f"{result['precision']:>8.4f} "
              f"{result['false_alarm_rate']:>7.2%} "
              f"{time_str:>10}")

    # Improvements
    print(f"\n{'='*70}")
    print("SALUS IMPROVEMENTS (vs best baseline)")
    print(f"{'='*70}")
    print(f"  AUROC: +{salus_results['auroc_improvement']:.4f} (+{salus_results['auroc_improvement_pct']:.1f}%)")
    print(f"  Recall: +{salus_results['recall_improvement']:.4f} (+{salus_results['recall_improvement_pct']:.1f}%)")
    if salus_results['far_reduction'] != float('inf'):
        print(f"  False Alarm Reduction: {salus_results['far_reduction']:.1f}× fewer alarms")
    else:
        print(f"  False Alarm Reduction: Infinite (baseline has 0 FAR)")

    print(f"\n✓ Method comparison complete!")


if __name__ == '__main__':
    main()
