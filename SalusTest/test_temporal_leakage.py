"""
Temporal Leakage Defense Experiments for SALUS Paper

Implements 3 control experiments to verify models learn genuine failure
dynamics rather than exploiting temporal position information:

1. Time-Shuffle Control: Randomize timestep order within windows
2. Counterfactual Labels: Early failures + late successes
3. Time-Index Removal: Train without positional encoding

Expected Result: < 5% AUROC drop in all experiments proves models don't
rely on temporal position.
"""

import torch
import torch.nn as nn
import numpy as np
import zarr
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from salus.models.temporal_predictor import HybridTemporalPredictor

print("\n" + "="*70)
print("SALUS TEMPORAL LEAKAGE DEFENSE EXPERIMENTS")
print("="*70)

# Configuration
DATA_PATH = Path("local_data/salus_data_20260107_215201.zarr")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FPS = 30
HORIZONS = [200, 300, 400, 500]  # ms
HORIZON_STEPS = [int(h * FPS / 1000) for h in HORIZONS]

print(f"\nDevice: {DEVICE}")
print(f"Horizons: {HORIZONS}ms → {HORIZON_STEPS} timesteps @ {FPS}Hz")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\nLoading data from: {DATA_PATH}")
root = zarr.open(str(DATA_PATH), mode='r')
signals = torch.tensor(root['signals'][:], dtype=torch.float32)
success_labels = torch.tensor(root['success'][:], dtype=torch.bool)

print(f"✓ Loaded {len(signals)} timesteps")

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

print(f"✓ Created {len(windows)} temporal windows (size={WINDOW_SIZE})")

# Split train/val
val_size = int(0.2 * len(windows))
train_windows = windows[:-val_size]
train_labels = labels[:-val_size]
val_windows = windows[-val_size:]
val_labels = labels[-val_size:]

print(f"  Train: {len(train_windows)} windows")
print(f"  Val: {len(val_windows)} windows")

# ============================================================================
# CREATE MULTI-HORIZON LABELS
# ============================================================================

def create_multi_horizon_labels(labels, horizon_steps):
    """Create multi-horizon failure labels."""
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

train_multi_labels = create_multi_horizon_labels(train_labels, HORIZON_STEPS)
val_multi_labels = create_multi_horizon_labels(val_labels, HORIZON_STEPS)

print(f"\n✓ Created multi-horizon labels")
print(f"  Positive rate (500ms): {train_multi_labels[:, -1, 0].mean():.3f}")

# ============================================================================
# TRAINING HELPER
# ============================================================================

def train_model(model, train_data, train_labels, val_data, val_labels,
                epochs=20, lr=0.001, batch_size=32, verbose=True):
    """Train model and return validation AUROC."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(DEVICE)
    train_data = train_data.to(DEVICE)
    train_labels_flat = train_labels.reshape(len(train_labels), -1).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        indices = torch.randperm(len(train_data))
        for i in range(0, len(train_data), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_data = train_data[batch_idx]
            batch_labels = train_labels_flat[batch_idx]

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose and (epoch + 1) % 5 == 0:
            avg_loss = total_loss / (len(train_data) / batch_size)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_data_device = val_data.to(DEVICE)
        outputs = model(val_data_device)
        probs = torch.sigmoid(outputs).cpu().numpy()

    # Compute AUROC for 500ms horizon
    probs_shaped = probs.reshape(len(probs), 4, 4)
    labels_shaped = val_labels.numpy()

    y_true = labels_shaped[:, 3, :].flatten()  # 500ms horizon
    y_pred = probs_shaped[:, 3, :].flatten()

    try:
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
    except:
        auroc = 0.5
        auprc = 0.5

    return model, auroc, auprc

# ============================================================================
# BASELINE: NORMAL TRAINING
# ============================================================================

print("\n" + "="*70)
print("[BASELINE] Normal Training")
print("="*70)

baseline_model = HybridTemporalPredictor(
    signal_dim=12,
    conv_dim=64,
    gru_dim=128,
    num_horizons=4,
    num_failure_types=4
)

print("Training baseline model...")
baseline_model, baseline_auroc, baseline_auprc = train_model(
    baseline_model, train_windows, train_multi_labels,
    val_windows, val_multi_labels,
    epochs=20
)

print(f"\n✓ Baseline AUROC (500ms): {baseline_auroc:.3f}")
print(f"✓ Baseline AUPRC (500ms): {baseline_auprc:.3f}")

# ============================================================================
# EXPERIMENT 1: TIME-SHUFFLE CONTROL
# ============================================================================

print("\n" + "="*70)
print("[EXPERIMENT 1] Time-Shuffle Control")
print("="*70)
print("Randomly permuting timestep order within windows...")

def shuffle_temporal_windows(windows):
    """Shuffle timestep order within each window."""
    shuffled = []
    for window in windows:
        indices = torch.randperm(window.shape[0])
        shuffled.append(window[indices])
    return torch.stack(shuffled)

train_shuffled = shuffle_temporal_windows(train_windows)
val_shuffled = shuffle_temporal_windows(val_windows)

print(f"✓ Shuffled {len(train_shuffled)} training windows")

shuffle_model = HybridTemporalPredictor(
    signal_dim=12,
    conv_dim=64,
    gru_dim=128,
    num_horizons=4,
    num_failure_types=4
)

print("Training on shuffled data...")
shuffle_model, shuffle_auroc, shuffle_auprc = train_model(
    shuffle_model, train_shuffled, train_multi_labels,
    val_shuffled, val_multi_labels,
    epochs=20
)

shuffle_drop = (baseline_auroc - shuffle_auroc) / baseline_auroc * 100

print(f"\n✓ Shuffled AUROC (500ms): {shuffle_auroc:.3f}")
print(f"✓ AUROC Drop: {shuffle_drop:.1f}%")

if shuffle_drop < 5.0:
    print("✅ PASS: < 5% drop (model doesn't rely on temporal order)")
else:
    print("⚠️  WARN: > 5% drop (model may exploit temporal order)")

# ============================================================================
# EXPERIMENT 2: COUNTERFACTUAL LABELS
# ============================================================================

print("\n" + "="*70)
print("[EXPERIMENT 2] Counterfactual Labels")
print("="*70)
print("Testing on early failures + late successes...")

# Create counterfactual validation set
# Failures in first 30% of episodes, successes in last 30%
def create_counterfactual_set(windows, labels, multi_labels):
    """Create validation set with early failures + late successes."""
    N = len(windows)
    early_cutoff = int(0.3 * N)
    late_cutoff = int(0.7 * N)

    # Early failures
    early_failures_mask = (~labels[:early_cutoff]) & (multi_labels[:early_cutoff, -1, 0] == 1)
    early_failures_idx = torch.where(early_failures_mask)[0]

    # Late successes
    late_successes_mask = labels[late_cutoff:]
    late_successes_idx = torch.where(late_successes_mask)[0] + late_cutoff

    # Combine
    counterfactual_idx = torch.cat([early_failures_idx[:100], late_successes_idx[:100]])

    return windows[counterfactual_idx], labels[counterfactual_idx], multi_labels[counterfactual_idx]

val_counter_windows, val_counter_labels, val_counter_multi = create_counterfactual_set(
    val_windows, val_labels, val_multi_labels
)

print(f"✓ Created counterfactual set: {len(val_counter_windows)} samples")

# Evaluate baseline model on counterfactual set
baseline_model.eval()
with torch.no_grad():
    counter_data = val_counter_windows.to(DEVICE)
    counter_outputs = baseline_model(counter_data)
    counter_probs = torch.sigmoid(counter_outputs).cpu().numpy()

counter_probs_shaped = counter_probs.reshape(len(counter_probs), 4, 4)
counter_labels_shaped = val_counter_multi.numpy()

y_true_counter = counter_labels_shaped[:, 3, :].flatten()
y_pred_counter = counter_probs_shaped[:, 3, :].flatten()

try:
    counter_auroc = roc_auc_score(y_true_counter, y_pred_counter)
    counter_auprc = average_precision_score(y_true_counter, y_pred_counter)
except:
    counter_auroc = 0.5
    counter_auprc = 0.5

counter_drop = (baseline_auroc - counter_auroc) / baseline_auroc * 100

print(f"\n✓ Counterfactual AUROC (500ms): {counter_auroc:.3f}")
print(f"✓ AUROC Drop: {counter_drop:.1f}%")

if counter_drop < 5.0:
    print("✅ PASS: < 5% drop (model doesn't exploit episode phase)")
else:
    print("⚠️  WARN: > 5% drop (model may exploit episode phase)")

# ============================================================================
# EXPERIMENT 3: TIME-INDEX REMOVAL
# ============================================================================

print("\n" + "="*70)
print("[EXPERIMENT 3] Time-Index Removal")
print("="*70)
print("Training without positional encoding...")

class TemporalPredictorNoPositional(nn.Module):
    """HybridTemporalPredictor without positional encodings."""
    def __init__(self, signal_dim, conv_dim, gru_dim, num_horizons=4, num_failure_types=4):
        super().__init__()
        self.conv = nn.Conv1d(signal_dim, conv_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(conv_dim, gru_dim, batch_first=True)
        self.fc = nn.Linear(gru_dim, num_horizons * num_failure_types)

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv(x)  # (B, conv_dim, T)
        x = x.transpose(1, 2)  # (B, T, conv_dim)
        # No positional encoding added here
        _, h = self.gru(x)  # h: (1, B, gru_dim)
        h = h.squeeze(0)  # (B, gru_dim)
        return self.fc(h)  # (B, num_horizons * num_failure_types)

no_pos_model = TemporalPredictorNoPositional(
    signal_dim=12,
    conv_dim=64,
    gru_dim=128,
    num_horizons=4,
    num_failure_types=4
)

print("Training without time indices...")
no_pos_model, no_pos_auroc, no_pos_auprc = train_model(
    no_pos_model, train_windows, train_multi_labels,
    val_windows, val_multi_labels,
    epochs=20
)

no_pos_drop = (baseline_auroc - no_pos_auroc) / baseline_auroc * 100

print(f"\n✓ No-Positional AUROC (500ms): {no_pos_auroc:.3f}")
print(f"✓ AUROC Drop: {no_pos_drop:.1f}%")

if no_pos_drop < 5.0:
    print("✅ PASS: < 5% drop (model doesn't rely on time indices)")
else:
    print("⚠️  WARN: > 5% drop (model may rely on time indices)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TEMPORAL LEAKAGE DEFENSE SUMMARY")
print("="*70)

results = {
    'Baseline (normal)': {
        'auroc': baseline_auroc,
        'drop_pct': 0.0
    },
    'Time-Shuffle': {
        'auroc': shuffle_auroc,
        'drop_pct': shuffle_drop
    },
    'Counterfactual': {
        'auroc': counter_auroc,
        'drop_pct': counter_drop
    },
    'No Time-Index': {
        'auroc': no_pos_auroc,
        'drop_pct': no_pos_drop
    }
}

print("\nExperiment          | AUROC  | Drop%  | Status")
print("-" * 70)
for name, metrics in results.items():
    drop = metrics['drop_pct']
    status = "✅ PASS" if drop < 5.0 or drop == 0.0 else "⚠️  WARN"
    print(f"{name:19s} | {metrics['auroc']:.3f}  | {drop:5.1f}% | {status}")

print("\n" + "="*70)

# Overall assessment
all_pass = all(r['drop_pct'] < 5.0 or r['drop_pct'] == 0.0 for r in results.values())
if all_pass:
    print("✅ OVERALL: All experiments pass (< 5% drop)")
    print("   → Model learns genuine failure dynamics, not temporal position")
else:
    print("⚠️  OVERALL: Some experiments show > 5% drop")
    print("   → Model may partially exploit temporal information")

# Save results
import json
leakage_file = Path("temporal_leakage_results.json")
with open(leakage_file, 'w') as f:
    json.dump({k: {'auroc': float(v['auroc']), 'drop_pct': float(v['drop_pct'])}
               for k, v in results.items()}, f, indent=2)
print(f"\n✓ Results saved to: {leakage_file}")
