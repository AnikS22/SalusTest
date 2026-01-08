"""
Production Metrics Computation for SALUS Paper

Computes deployment-critical metrics with proper calibration analysis:
1. Per-horizon AUROC/AUPRC breakdown
2. Calibration curves (ECE, reliability diagrams)
3. Precision-recall curves for threshold selection
4. Lead time distribution
5. False alarms per minute at multiple thresholds
6. Miss rate analysis

CRITICAL: Exposes calibration issues and threshold sensitivity for honest
evaluation suitable for safety-critical deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import zarr
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("\n" + "="*70)
print("SALUS PRODUCTION METRICS ANALYSIS")
print("="*70)

# Configuration
DATA_PATH = Path("local_data/salus_data_20260107_215201.zarr")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FPS = 30
HORIZONS = [200, 300, 400, 500]  # ms
HORIZON_STEPS = [int(h * FPS / 1000) for h in HORIZONS]

# ============================================================================
# LOAD DATA AND TRAIN MODEL
# ============================================================================

print(f"\nLoading data from: {DATA_PATH}")
root = zarr.open(str(DATA_PATH), mode='r')
signals = torch.tensor(root['signals'][:], dtype=torch.float32)
success_labels = torch.tensor(root['success'][:], dtype=torch.bool)

# Create temporal windows
WINDOW_SIZE = 10
windows = []
labels = []

for i in range(len(signals) - WINDOW_SIZE):
    window = signals[i:i+WINDOW_SIZE]
    label = success_labels[i+WINDOW_SIZE]
    windows.append(window)
    labels.append(label)

windows = torch.stack(windows)
labels = torch.stack(labels)

# Split
val_size = int(0.2 * len(windows))
val_windows = windows[-val_size:]
val_labels = labels[-val_size:]

print(f"✓ Validation set: {len(val_windows)} windows")

# Create multi-horizon labels
def create_multi_horizon_labels(labels, horizon_steps):
    N = len(labels)
    num_horizons = len(horizon_steps)
    num_types = 4
    multi_labels = torch.zeros(N, num_horizons, num_types)

    for i in range(N):
        for h_idx, h_steps in enumerate(horizon_steps):
            end_idx = min(i + h_steps, N)
            if (~labels[i:end_idx]).any():
                multi_labels[i, h_idx, 0] = 1.0

    return multi_labels

val_multi_labels = create_multi_horizon_labels(val_labels, HORIZON_STEPS)

# Load trained model (using HybridTemporalPredictor)
from salus.models.temporal_predictor import HybridTemporalPredictor

model = HybridTemporalPredictor(
    signal_dim=12,
    conv_dim=64,
    gru_dim=128,
    num_horizons=4,
    num_failure_types=4
)

# Quick train for demo (in practice, load checkpoint)
print("\nTraining model for evaluation...")
from test_temporal_leakage import train_model, create_multi_horizon_labels as create_labels

train_windows = windows[:-val_size]
train_labels = labels[:-val_size]
train_multi_labels = create_labels(train_labels, HORIZON_STEPS)

model, _, _ = train_model(
    model, train_windows, train_multi_labels,
    val_windows, val_multi_labels,
    epochs=20, verbose=False
)

print("✓ Model trained")

# Get predictions
model.eval()
with torch.no_grad():
    val_data = val_windows.to(DEVICE)
    outputs = model(val_data)
    probs = torch.sigmoid(outputs).cpu().numpy()

# Reshape: (N, 16) → (N, 4 horizons, 4 types)
probs_shaped = probs.reshape(len(probs), 4, 4)
labels_shaped = val_multi_labels.numpy()

print(f"✓ Generated predictions: {probs_shaped.shape}")

# ============================================================================
# METRIC 1: PER-HORIZON AUROC/AUPRC BREAKDOWN
# ============================================================================

print("\n" + "="*70)
print("[METRIC 1] Per-Horizon Performance Breakdown")
print("="*70)

per_horizon_metrics = []

for h_idx, horizon_ms in enumerate(HORIZONS):
    y_true = labels_shaped[:, h_idx, :].flatten()
    y_pred = probs_shaped[:, h_idx, :].flatten()

    # Skip if not enough positive samples
    if y_true.sum() < 10:
        continue

    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    # Compute F1 at threshold 0.5
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    per_horizon_metrics.append({
        'horizon_ms': horizon_ms,
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    })

print("\nHorizon | AUROC | AUPRC |  F1   | Prec  | Recall")
print("-" * 60)
for m in per_horizon_metrics:
    print(f"{m['horizon_ms']:4d}ms | {m['auroc']:.3f} | {m['auprc']:.3f} | "
          f"{m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f}")

# ============================================================================
# METRIC 2: CALIBRATION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("[METRIC 2] Calibration Analysis")
print("="*70)

# Use 500ms horizon for calibration analysis
h_idx = 3  # 500ms
y_true_cal = labels_shaped[:, h_idx, :].flatten()
y_pred_cal = probs_shaped[:, h_idx, :].flatten()

# Compute Expected Calibration Error (ECE)
def compute_ece(y_true, y_pred, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'count': in_bin.sum()
            })
        else:
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'confidence': (bin_lower + bin_upper) / 2,
                'accuracy': 0.0,
                'count': 0
            })

    return ece, bin_data

ece, bin_data = compute_ece(y_true_cal, y_pred_cal, n_bins=10)

print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
print("\nCalibration bins:")
print("Bin Range     | Confidence | Accuracy | Count | Gap")
print("-" * 60)
for b in bin_data:
    if b['count'] > 0:
        gap = abs(b['confidence'] - b['accuracy'])
        print(f"[{b['bin_lower']:.1f}, {b['bin_upper']:.1f}] | "
              f"{b['confidence']:.3f}      | {b['accuracy']:.3f}   | "
              f"{b['count']:5d} | {gap:.3f}")

# Create reliability diagram
fig, ax = plt.subplots(figsize=(8, 6))

# Plot perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

# Plot model calibration
confidences = [b['confidence'] for b in bin_data if b['count'] > 0]
accuracies = [b['accuracy'] for b in bin_data if b['count'] > 0]
ax.plot(confidences, accuracies, 'bo-', label=f'SALUS (ECE={ece:.3f})', linewidth=2, markersize=8)

ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Observed Frequency', fontsize=12)
ax.set_title('Reliability Diagram (500ms Horizon)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('calibration_diagram.png', dpi=150)
print("\n✓ Saved calibration diagram: calibration_diagram.png")

# ============================================================================
# METRIC 3: PRECISION-RECALL CURVE FOR THRESHOLD SELECTION
# ============================================================================

print("\n" + "="*70)
print("[METRIC 3] Precision-Recall Analysis")
print("="*70)

precisions, recalls, thresholds = precision_recall_curve(y_true_cal, y_pred_cal)

# Find optimal threshold (max F1)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"\nOptimal threshold (max F1): τ = {optimal_threshold:.3f}")
print(f"  → F1 score: {optimal_f1:.3f}")
print(f"  → Precision: {precisions[optimal_idx]:.3f}")
print(f"  → Recall: {recalls[optimal_idx]:.3f}")

# Compute false alarms per minute at different thresholds
test_thresholds = [0.3, 0.5, 0.7, optimal_threshold]

print("\nThreshold Analysis:")
print("Threshold |  F1   | Precision | Recall | FA/min | Miss Rate")
print("-" * 70)

for thresh in sorted(set(test_thresholds)):
    y_pred_binary = (y_pred_cal > thresh).astype(int)

    f1 = f1_score(y_true_cal, y_pred_binary, zero_division=0)
    prec = precision_score(y_true_cal, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_cal, y_pred_binary, zero_division=0)

    # False alarms per minute
    fp = np.sum((y_pred_binary == 1) & (y_true_cal == 0))
    total_time_min = len(y_pred_cal) / FPS / 60
    fa_per_min = fp / total_time_min if total_time_min > 0 else 0

    # Miss rate
    failures = np.sum(y_true_cal == 1)
    if failures > 0:
        missed = np.sum((y_true_cal == 1) & (y_pred_binary == 0))
        miss_rate = missed / failures * 100
    else:
        miss_rate = 0.0

    marker = " *" if abs(thresh - optimal_threshold) < 0.01 else ""
    print(f"  {thresh:.3f}   | {f1:.3f} | {prec:.3f}    | {rec:.3f} | "
          f"{fa_per_min:6.2f} | {miss_rate:5.1f}%{marker}")

# Create precision-recall curve
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(recalls, precisions, 'b-', linewidth=2, label=f'SALUS (AUPRC={auprc:.3f})')
ax.plot(recalls[optimal_idx], precisions[optimal_idx], 'ro', markersize=10,
        label=f'Optimal τ={optimal_threshold:.3f}')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve (500ms Horizon)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=150)
print("\n✓ Saved precision-recall curve: precision_recall_curve.png")

# ============================================================================
# METRIC 4: LEAD TIME DISTRIBUTION
# ============================================================================

print("\n" + "="*70)
print("[METRIC 4] Lead Time Analysis")
print("="*70)

# Simplified lead time computation
# For each failure, find first alert before it
lead_times = []

failure_indices = np.where(y_true_cal == 1)[0]
print(f"\nAnalyzing {len(failure_indices)} failure cases...")

for fail_idx in failure_indices:
    # Look backwards for first alert
    for lookback in range(1, min(fail_idx, 50)):  # Look back up to ~1.5 seconds
        past_idx = fail_idx - lookback
        if y_pred_cal[past_idx] > optimal_threshold:
            lead_time_ms = lookback * (1000 / FPS)
            lead_times.append(lead_time_ms)
            break

if lead_times:
    print(f"\n✓ Found {len(lead_times)} predicted failures")
    print(f"  Mean lead time: {np.mean(lead_times):.1f}ms")
    print(f"  Median lead time: {np.median(lead_times):.1f}ms")
    print(f"  Std dev: {np.std(lead_times):.1f}ms")
    print(f"  Min: {np.min(lead_times):.1f}ms")
    print(f"  Max: {np.max(lead_times):.1f}ms")

    # Lead time distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(lead_times, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(lead_times), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(lead_times):.1f}ms')
    ax.axvline(np.median(lead_times), color='orange', linestyle='--', linewidth=2,
               label=f'Median = {np.median(lead_times):.1f}ms')
    ax.set_xlabel('Lead Time (ms)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Lead Time Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('lead_time_distribution.png', dpi=150)
    print("\n✓ Saved lead time distribution: lead_time_distribution.png")
else:
    print("⚠️  No predicted failures found (model may have low recall)")

# ============================================================================
# SUMMARY: PRODUCTION READINESS ASSESSMENT
# ============================================================================

print("\n" + "="*70)
print("PRODUCTION READINESS ASSESSMENT")
print("="*70)

# Define safety requirements
REQUIREMENTS = {
    'AUROC (500ms)': {'value': per_horizon_metrics[-1]['auroc'], 'threshold': 0.90, 'unit': ''},
    'AUPRC (500ms)': {'value': per_horizon_metrics[-1]['auprc'], 'threshold': 0.80, 'unit': ''},
    'ECE': {'value': ece, 'threshold': 0.10, 'unit': '', 'inverted': True},
    'Mean Lead Time': {'value': np.mean(lead_times) if lead_times else 0,
                       'threshold': 200, 'unit': 'ms'},
    'FA/min (optimal τ)': {'value': None, 'threshold': 1.0, 'unit': '/min'},
    'Miss Rate (optimal τ)': {'value': None, 'threshold': 15.0, 'unit': '%', 'inverted': True}
}

# Compute FA/min and miss rate at optimal threshold
y_pred_opt = (y_pred_cal > optimal_threshold).astype(int)
fp_opt = np.sum((y_pred_opt == 1) & (y_true_cal == 0))
fa_per_min_opt = fp_opt / (len(y_pred_cal) / FPS / 60)
failures_opt = np.sum(y_true_cal == 1)
missed_opt = np.sum((y_true_cal == 1) & (y_pred_opt == 0))
miss_rate_opt = (missed_opt / failures_opt * 100) if failures_opt > 0 else 0

REQUIREMENTS['FA/min (optimal τ)']['value'] = fa_per_min_opt
REQUIREMENTS['Miss Rate (optimal τ)']['value'] = miss_rate_opt

print("\nMetric                  | Value      | Threshold  | Status")
print("-" * 70)

all_pass = True
for metric_name, req in REQUIREMENTS.items():
    value = req['value']
    threshold = req['threshold']
    unit = req['unit']
    inverted = req.get('inverted', False)

    # Check if passes
    if inverted:
        passes = value <= threshold
    else:
        passes = value >= threshold

    status = "✅ PASS" if passes else "❌ FAIL"
    all_pass = all_pass and passes

    print(f"{metric_name:23s} | {value:6.2f}{unit:4s} | {threshold:6.2f}{unit:4s} | {status}")

print("\n" + "="*70)
if all_pass:
    print("✅ OVERALL: System meets all production requirements")
    print("   → Ready for pilot deployment with monitoring")
else:
    print("⚠️  OVERALL: System does NOT meet all requirements")
    print("   → Requires calibration and threshold tuning")
    print("   → Recommend extended validation before deployment")

# Save all metrics
import json

def convert_to_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

metrics_output = {
    'per_horizon': [convert_to_serializable(m) for m in per_horizon_metrics],
    'calibration': {
        'ece': float(ece),
        'bin_data': [convert_to_serializable(b) for b in bin_data]
    },
    'threshold_analysis': {
        'optimal_threshold': float(optimal_threshold),
        'optimal_f1': float(optimal_f1),
        'false_alarms_per_min': float(fa_per_min_opt),
        'miss_rate_pct': float(miss_rate_opt)
    },
    'lead_time': {
        'mean_ms': float(np.mean(lead_times)) if lead_times else 0,
        'median_ms': float(np.median(lead_times)) if lead_times else 0,
        'std_ms': float(np.std(lead_times)) if lead_times else 0
    },
    'production_readiness': {
        metric: {'value': float(req['value']), 'threshold': req['threshold'],
                 'passes': bool(req['value'] <= req['threshold'] if req.get('inverted')
                           else req['value'] >= req['threshold'])}
        for metric, req in REQUIREMENTS.items()
    }
}

with open('production_metrics.json', 'w') as f:
    json.dump(metrics_output, f, indent=2)

print(f"\n✓ All metrics saved to: production_metrics.json")
print("✓ Figures saved: calibration_diagram.png, precision_recall_curve.png, lead_time_distribution.png")
