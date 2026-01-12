#!/usr/bin/env python3
"""
Compute missing metrics for SALUS paper:
1. Multi-horizon breakdown (per-horizon performance)
2. Per-failure-type breakdown (drops, collisions, kinematic, task)
3. Inference latency benchmark

Usage:
    python compute_missing_metrics.py --checkpoint ../models/salus_predictor_massive.pth
"""

import json
import time
import argparse
import numpy as np
import torch
import zarr
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from salus.core.predictor import SALUSPredictor


def compute_multi_horizon_metrics(model, zarr_path, device='cuda'):
    """Compute per-horizon performance metrics."""
    print("\n" + "="*60)
    print("Computing Multi-Horizon Breakdown")
    print("="*60)

    # Load test data
    data = zarr.open(zarr_path, mode='r')
    signals = torch.tensor(np.array(data['signals'][:]), dtype=torch.float32)
    labels = torch.tensor(np.array(data['horizon_labels'][:]), dtype=torch.float32)

    # Reshape labels: (episodes, steps, 16) -> (episodes, steps, 4, 4)
    labels = labels.reshape(-1, 4, 4)  # (N, horizons, failure_types)

    # Use last 10% as test set
    n_test = len(signals) // 10
    test_signals = signals[-n_test:].to(device)
    test_labels = labels[-n_test:]

    # Flatten to (batch*steps, ...)
    test_signals = test_signals.reshape(-1, test_signals.shape[-1])
    test_labels = test_labels.reshape(-1, 4, 4)

    model.eval()
    with torch.no_grad():
        outputs = model(test_signals)
        probs = torch.sigmoid(outputs['logits']).cpu().numpy()

    horizon_names = ['200ms', '300ms', '400ms', '500ms']
    horizon_results = {}

    for h_idx, h_name in enumerate(horizon_names):
        # Get predictions and labels for this horizon (aggregate over failure types)
        h_probs = probs[:, h_idx, :].max(axis=1)  # Max over failure types
        h_labels = test_labels[:, h_idx, :].max(axis=1)  # Any failure at this horizon

        # Remove NaNs
        valid_mask = ~(np.isnan(h_probs) | np.isnan(h_labels))
        h_probs = h_probs[valid_mask]
        h_labels = h_labels[valid_mask]

        if len(np.unique(h_labels)) < 2:
            print(f"  ⚠ {h_name}: Insufficient data (no failures)")
            continue

        # Compute metrics
        auroc = roc_auc_score(h_labels, h_probs)

        # Threshold at 0.5 for other metrics
        h_preds = (h_probs > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            h_labels, h_preds, average='binary', zero_division=0
        )

        horizon_results[h_name] = {
            'auroc': float(auroc),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
            'lead_time_ms': int(h_name.replace('ms', ''))
        }

        print(f"  {h_name}: AUROC={auroc:.3f}, Recall={recall:.3f}, "
              f"Precision={precision:.3f}, F1={f1:.3f}")

    return horizon_results


def compute_failure_type_metrics(model, zarr_path, device='cuda'):
    """Compute per-failure-type performance metrics."""
    print("\n" + "="*60)
    print("Computing Per-Failure-Type Breakdown")
    print("="*60)

    # Load test data
    data = zarr.open(zarr_path, mode='r')
    signals = torch.tensor(np.array(data['signals'][:]), dtype=torch.float32)
    labels = torch.tensor(np.array(data['horizon_labels'][:]), dtype=torch.float32)

    # Reshape labels: (episodes, steps, 16) -> (episodes, steps, 4, 4)
    labels = labels.reshape(-1, 4, 4)

    # Use last 10% as test set
    n_test = len(signals) // 10
    test_signals = signals[-n_test:].to(device)
    test_labels = labels[-n_test:]

    # Flatten
    test_signals = test_signals.reshape(-1, test_signals.shape[-1])
    test_labels = test_labels.reshape(-1, 4, 4)

    model.eval()
    with torch.no_grad():
        outputs = model(test_signals)
        probs = torch.sigmoid(outputs['logits']).cpu().numpy()

    failure_types = ['Object Drop', 'Collision', 'Kinematic Violation', 'Task Failure']
    type_results = {}

    for t_idx, t_name in enumerate(failure_types):
        # Get predictions and labels for this type (aggregate over horizons)
        t_probs = probs[:, :, t_idx].max(axis=1)  # Max over horizons
        t_labels = test_labels[:, :, t_idx].max(axis=1)  # Any horizon

        # Remove NaNs
        valid_mask = ~(np.isnan(t_probs) | np.isnan(t_labels))
        t_probs = t_probs[valid_mask]
        t_labels = t_labels[valid_mask]

        if len(np.unique(t_labels)) < 2:
            print(f"  ⚠ {t_name}: Insufficient data")
            continue

        # Compute metrics
        auroc = roc_auc_score(t_labels, t_probs)

        t_preds = (t_probs > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            t_labels, t_preds, average='binary', zero_division=0
        )

        type_results[t_name] = {
            'auroc': float(auroc),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1)
        }

        print(f"  {t_name}: AUROC={auroc:.3f}, Recall={recall:.3f}, "
              f"Precision={precision:.3f}")

    return type_results


def benchmark_inference_latency(model, device='cuda', n_iterations=1000):
    """Benchmark forward pass latency."""
    print("\n" + "="*60)
    print("Benchmarking Inference Latency")
    print("="*60)

    model.eval()

    # Dummy input (batch=1, signal_dim=12)
    dummy_input = torch.randn(1, 12, device=device)

    # Warmup
    print("  Warming up GPU...")
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark
    print(f"  Running {n_iterations} iterations...")
    latencies = []

    with torch.no_grad():
        for _ in tqdm(range(n_iterations), desc="Benchmarking"):
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()  # Wait for GPU to finish
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    results = {
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'n_iterations': n_iterations
    }

    print(f"\n  Mean: {results['mean_latency_ms']:.2f} ms")
    print(f"  Std:  {results['std_latency_ms']:.2f} ms")
    print(f"  P50:  {results['p50_latency_ms']:.2f} ms")
    print(f"  P95:  {results['p95_latency_ms']:.2f} ms")
    print(f"  P99:  {results['p99_latency_ms']:.2f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute missing SALUS metrics')
    parser.add_argument('--checkpoint', type=str,
                       default='../models/salus_predictor_massive.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str,
                       default='../data/massive_collection/20260109_215258/data_20260109_215321.zarr',
                       help='Path to Zarr dataset')
    parser.add_argument('--output_dir', type=str,
                       default='../results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("SALUS Missing Metrics Computation")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_path}")
    print(f"Device: {args.device}")

    # Load model
    print("\nLoading model...")
    device = torch.device(args.device)
    model = SALUSPredictor(
        signal_dim=12,
        hidden_dims=[128, 256, 128],
        num_horizons=4,
        num_failure_types=4,
        dropout=0.2
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded")

    # Compute all metrics
    all_results = {}

    # 1. Multi-horizon breakdown
    all_results['multi_horizon'] = compute_multi_horizon_metrics(
        model, data_path, device
    )

    # 2. Per-failure-type breakdown
    all_results['failure_types'] = compute_failure_type_metrics(
        model, data_path, device
    )

    # 3. Inference latency
    all_results['inference_latency'] = benchmark_inference_latency(
        model, device
    )

    # Save results
    output_file = output_dir / 'missing_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print(f"✓ All metrics computed and saved to: {output_file}")
    print("="*60)

    print("\nNext steps:")
    print("  1. Review metrics in results/missing_metrics.json")
    print("  2. Update paper with new metrics")
    print("  3. Run: python create_all_figures.py")


if __name__ == '__main__':
    main()
