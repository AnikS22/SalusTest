"""
Baseline Threshold Methods for Failure Prediction

Implements simple baseline methods to compare against SALUS:
1. Entropy threshold (predict failure if entropy > threshold)
2. Action variance threshold (predict failure if action variance > threshold)
3. Random predictor (sanity check)

Usage:
    python scripts/evaluate_baseline_threshold.py --data_path paper_data/dataset.zarr
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import zarr
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import json


def load_dataset(zarr_path):
    """Load signals, labels, and actions from zarr dataset."""
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
    valid_actions = []

    for ep in range(num_episodes):
        for t in range(signals.shape[1]):
            if masks[ep, t]:
                valid_signals.append(signals[ep, t])
                valid_labels.append(labels[ep, t])
                valid_actions.append(actions[ep, t])

    valid_signals = np.array(valid_signals)  # (num_timesteps, 12)
    valid_labels = np.array(valid_labels)  # (num_timesteps, 16)
    valid_actions = np.array(valid_actions)  # (num_timesteps, action_dim)

    # Binary labels (any failure at any horizon)
    binary_labels = (valid_labels.sum(axis=1) > 0).astype(np.float32)

    print(f"  Failure rate: {binary_labels.mean():.2%}")

    return valid_signals, valid_labels, binary_labels, valid_actions


def compute_action_entropy(actions):
    """Compute a softmax entropy proxy from raw action vectors."""
    action_logits = actions - actions.max(axis=-1, keepdims=True)
    exp_logits = np.exp(action_logits)
    probs = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-8)
    return -np.sum(probs * np.log(probs + 1e-10), axis=-1)


def evaluate_threshold_method(scores, labels, method_name):
    """Evaluate a threshold-based method."""
    # Compute AUROC
    try:
        auroc = roc_auc_score(labels, scores)
    except:
        auroc = 0.0

    # Find optimal threshold (maximize F1)
    thresholds = np.percentile(scores, [70, 75, 80, 85, 90, 95, 98, 99])
    best_f1 = 0
    best_threshold = thresholds[0] if len(thresholds) > 0 else 0
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

        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'threshold': threshold
            }

    # Compute false alarm rate
    false_alarm_rate = best_metrics['fp'] / (best_metrics['fp'] + best_metrics['tn']) if (best_metrics['fp'] + best_metrics['tn']) > 0 else 0

    results = {
        'method': method_name,
        'auroc': auroc,
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'false_alarm_rate': false_alarm_rate,
        'threshold': best_threshold,
        'tp': best_metrics['tp'],
        'fp': best_metrics['fp'],
        'tn': best_metrics['tn'],
        'fn': best_metrics['fn']
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Baseline Threshold Evaluation')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to zarr dataset')
    parser.add_argument('--output_path', type=str, default='results/baseline_results.json',
                       help='Output path for results')
    args = parser.parse_args()

    print("="*70)
    print("Baseline Threshold Method Evaluation")
    print("="*70)

    # Load dataset
    print("\nLoading dataset...")
    signals, multi_labels, binary_labels, actions = load_dataset(args.data_path)

    # Signal indices (from single_model_extractor.py):
    # [0] Temporal Action Volatility
    # [1] Action Magnitude
    # [2] Action Acceleration
    # [3] Trajectory Divergence
    # [4] Latent Drift
    # [5] Latent Norm Spike
    # [6] OOD Distance
    # [7] Softmax Entropy (PRIMARY UNCERTAINTY)
    # [8] Max Softmax Probability
    # [9] Execution Mismatch
    # [10] Constraint Margin
    # [11] Rolling std of action volatility

    results = []

    # Method 1: Random Baseline (sanity check)
    print("\n" + "="*70)
    print("Method 1: Random Predictor (sanity check)")
    print("="*70)
    np.random.seed(42)
    random_scores = np.random.rand(len(binary_labels))
    random_results = evaluate_threshold_method(random_scores, binary_labels, "Random")
    results.append(random_results)

    print(f"  AUROC: {random_results['auroc']:.4f}")
    print(f"  Recall: {random_results['recall']:.4f}")
    print(f"  Precision: {random_results['precision']:.4f}")
    print(f"  False Alarm Rate: {random_results['false_alarm_rate']:.2%}")

    # Method 2: Entropy Threshold
    print("\n" + "="*70)
    print("Method 2: Entropy Threshold")
    print("="*70)

    # Use softmax entropy as uncertainty score
    entropy_scores = signals[:, 7]  # Signal index 7 = Softmax Entropy
    if np.nanstd(entropy_scores) < 1e-6:
        print("  ⚠️  Entropy signal is near-constant; using action-entropy proxy.")
        entropy_scores = compute_action_entropy(actions)
    entropy_results = evaluate_threshold_method(entropy_scores, binary_labels, "Entropy Threshold")
    results.append(entropy_results)

    print(f"  AUROC: {entropy_results['auroc']:.4f}")
    print(f"  Recall: {entropy_results['recall']:.4f}")
    print(f"  Precision: {entropy_results['precision']:.4f}")
    print(f"  False Alarm Rate: {entropy_results['false_alarm_rate']:.2%}")
    print(f"  Optimal threshold: {entropy_results['threshold']:.4f}")

    # Method 3: Action Variance Threshold
    print("\n" + "="*70)
    print("Method 3: Action Variance Threshold")
    print("="*70)

    # Use action variance as uncertainty score
    variance_scores = signals[:, 2]  # Signal index 2 = Action Variance
    if np.nanstd(variance_scores) < 1e-6:
        print("  ⚠️  Action variance signal is near-constant; using variance from raw actions.")
        variance_scores = np.var(actions, axis=-1)
    variance_results = evaluate_threshold_method(variance_scores, binary_labels, "Action Variance")
    results.append(variance_results)

    print(f"  AUROC: {variance_results['auroc']:.4f}")
    print(f"  Recall: {variance_results['recall']:.4f}")
    print(f"  Precision: {variance_results['precision']:.4f}")
    print(f"  False Alarm Rate: {variance_results['false_alarm_rate']:.2%}")
    print(f"  Optimal threshold: {variance_results['threshold']:.4f}")

    # Method 4: Combined Score (entropy + variance)
    print("\n" + "="*70)
    print("Method 4: Combined Score (Entropy + Variance)")
    print("="*70)

    # Normalize scores to [0, 1]
    entropy_norm = (entropy_scores - entropy_scores.min()) / (entropy_scores.max() - entropy_scores.min() + 1e-8)
    variance_norm = (variance_scores - variance_scores.min()) / (variance_scores.max() - variance_scores.min() + 1e-8)

    combined_scores = 0.7 * entropy_norm + 0.3 * variance_norm
    combined_results = evaluate_threshold_method(combined_scores, binary_labels, "Combined (Entropy + Variance)")
    results.append(combined_results)

    print(f"  AUROC: {combined_results['auroc']:.4f}")
    print(f"  Recall: {combined_results['recall']:.4f}")
    print(f"  Precision: {combined_results['precision']:.4f}")
    print(f"  False Alarm Rate: {combined_results['false_alarm_rate']:.2%}")

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("BASELINE EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'AUROC':>8} {'Recall':>8} {'Precision':>8} {'FAR':>8}")
    print(f"{'-'*70}")

    for result in results:
        print(f"{result['method']:<30} "
              f"{result['auroc']:>8.4f} "
              f"{result['recall']:>8.4f} "
              f"{result['precision']:>8.4f} "
              f"{result['false_alarm_rate']:>7.2%}")

    print(f"\n✓ Baseline evaluation complete!")


if __name__ == '__main__':
    main()
