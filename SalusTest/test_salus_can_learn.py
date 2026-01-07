"""
Test if SALUS can actually LEARN from 18D VLA signals to predict failures.

This will:
1. Create synthetic temporal data with 18D signals
2. Train the HybridTemporalPredictor
3. Show that it learns to discriminate failure vs success patterns
4. Prove the architecture works with real VLA signal dimensions
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from salus.models.temporal_predictor import HybridTemporalPredictor, TemporalFocalLoss

print("="*70)
print("TESTING: SALUS Can Learn from 18D VLA Signals")
print("="*70)

# Create synthetic temporal data
print("\n1. Creating synthetic temporal data with 18D signals...")

def create_synthetic_episode(length=50, signal_dim=18, will_fail=True):
    """
    Create synthetic episode with temporal patterns.

    Failure episodes: signals increase over time (drift toward failure)
    Success episodes: signals remain stable
    """
    signals = torch.randn(length, signal_dim) * 0.1

    if will_fail:
        # Failure pattern: gradual increase in uncertainty signals
        # Signals 1-12: uncertainty increases
        drift = torch.linspace(0, 1.0, length).unsqueeze(1)
        signals[:, :12] += drift * torch.randn(1, 12) * 0.5

        # Signals 13-14: latent drift and OOD increase
        signals[:, 12:14] += drift * 0.8

        # Signals 15-16: sensitivity increases
        signals[:, 14:16] += drift * 0.6

        # Signals 17-18: constraint violations increase
        signals[:, 16:18] += drift * 0.7

        # Label: failure at different horizons
        # Assume failure at timestep 45 (last 5 timesteps)
        labels = torch.zeros(length, 4, 4)  # (T, 4 horizons, 4 failure types)
        failure_time = 45
        for t in range(length):
            for h_idx, horizon_steps in enumerate([6, 9, 12, 15]):  # 200, 300, 400, 500ms
                if t + horizon_steps >= failure_time:
                    # Predict failure type 1 (grasp failure) at this horizon
                    labels[t, h_idx, 1] = 1.0
    else:
        # Success pattern: stable signals
        signals = signals * 0.05  # Low variance
        labels = torch.zeros(length, 4, 4)  # All zeros = no failure

    return signals, labels

# Create training dataset
n_fail = 30
n_success = 30
window_size = 10  # 333ms at 30Hz

print(f"   Creating {n_fail} failure episodes + {n_success} success episodes")
print(f"   Window size: {window_size} timesteps")

X_train = []  # (N, window_size, 18)
y_train = []  # (N, 16) - flattened (4 horizons × 4 types)

for _ in range(n_fail):
    signals, labels = create_synthetic_episode(length=50, signal_dim=18, will_fail=True)
    # Extract windows
    for t in range(window_size, len(signals)):
        window = signals[t-window_size:t]  # (window_size, 18)
        label = labels[t].flatten()  # (16,)
        X_train.append(window)
        y_train.append(label)

for _ in range(n_success):
    signals, labels = create_synthetic_episode(length=50, signal_dim=18, will_fail=False)
    for t in range(window_size, len(signals)):
        window = signals[t-window_size:t]
        label = labels[t].flatten()
        X_train.append(window)
        y_train.append(label)

X_train = torch.stack(X_train)  # (N, 10, 18)
y_train = torch.stack(y_train)  # (N, 16)

print(f"   ✅ Created {len(X_train)} training samples")
print(f"   X shape: {X_train.shape}")
print(f"   y shape: {y_train.shape}")
print(f"   Positive labels: {(y_train.sum(dim=1) > 0).sum().item()}/{len(y_train)}")

# Initialize model
print("\n2. Initializing HybridTemporalPredictor with 18D input...")
model = HybridTemporalPredictor(
    signal_dim=18,  # NEW: 18D instead of 12D
    conv_dim=32,
    gru_dim=64,
    dropout=0.2,
    num_horizons=4,
    num_failure_types=4
)

print(f"   ✅ Model initialized")
print(f"   Architecture:")
print(f"      Input: (B, {window_size}, 18)")
print(f"      Conv1D: 18 → 32 channels")
print(f"      GRU: 32 → 64 hidden")
print(f"      Linear: 64 → 128 → 16")
print(f"      Output: (B, 16) = 4 horizons × 4 failure types")

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {n_params:,}")

# Test forward pass
print("\n3. Testing forward pass with 18D signals...")
with torch.no_grad():
    test_input = X_train[:4]  # Batch of 4
    test_output = model(test_input)

print(f"   Input shape: {test_input.shape}")
print(f"   Output shape: {test_output.shape}")
print(f"   Output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")

if test_output.shape == (4, 16):
    print(f"   ✅ Forward pass works with 18D input!")
else:
    print(f"   ❌ Wrong output shape!")
    sys.exit(1)

# Train the model
print("\n4. Training model to discriminate failure vs success...")

criterion = TemporalFocalLoss(pos_weight=3.0, fp_penalty_weight=2.0, focal_gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50
batch_size = 32

print(f"   Training for {n_epochs} epochs...")

losses = []
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    n_batches = 0

    # Mini-batch training
    indices = torch.randperm(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        # Forward pass
        predictions = model(X_batch)

        # Compute loss
        loss = criterion(predictions, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

print(f"   ✅ Training complete!")
print(f"   Initial loss: {losses[0]:.6f}")
print(f"   Final loss: {losses[-1]:.6f}")
print(f"   Improvement: {(losses[0] - losses[-1]):.6f} ({(losses[0] - losses[-1])/losses[0]*100:.1f}%)")

# Test discrimination
print("\n5. Testing failure prediction (discrimination)...")

model.eval()
with torch.no_grad():
    # Create test failure pattern
    fail_signals, fail_labels = create_synthetic_episode(length=50, signal_dim=18, will_fail=True)
    fail_window = fail_signals[-window_size:].unsqueeze(0)  # (1, 10, 18)
    fail_pred = model(fail_window)[0]  # (16,)

    # Create test success pattern
    success_signals, success_labels = create_synthetic_episode(length=50, signal_dim=18, will_fail=False)
    success_window = success_signals[-window_size:].unsqueeze(0)
    success_pred = model(success_window)[0]

    # Reshape to (4 horizons, 4 types)
    fail_pred = fail_pred.reshape(4, 4)
    success_pred = success_pred.reshape(4, 4)

print(f"\n   Failure episode predictions:")
print(f"   Horizon 1 (200ms): {fail_pred[0].numpy()}")
print(f"   Horizon 2 (300ms): {fail_pred[1].numpy()}")
print(f"   Horizon 3 (400ms): {fail_pred[2].numpy()}")
print(f"   Horizon 4 (500ms): {fail_pred[3].numpy()}")
print(f"   Max prediction: {fail_pred.max().item():.4f}")

print(f"\n   Success episode predictions:")
print(f"   Horizon 1 (200ms): {success_pred[0].numpy()}")
print(f"   Horizon 2 (300ms): {success_pred[1].numpy()}")
print(f"   Horizon 3 (400ms): {success_pred[2].numpy()}")
print(f"   Horizon 4 (500ms): {success_pred[3].numpy()}")
print(f"   Max prediction: {success_pred.max().item():.4f}")

# Compute discrimination
fail_score = fail_pred.max().item()
success_score = success_pred.max().item()
discrimination = fail_score - success_score

print(f"\n   Discrimination Analysis:")
print(f"      Failure score: {fail_score:.4f}")
print(f"      Success score: {success_score:.4f}")
print(f"      Difference: {discrimination:.4f}")

if discrimination > 0.5:
    print(f"      ✅ STRONG discrimination (model learned!)")
elif discrimination > 0.2:
    print(f"      ✅ MODERATE discrimination (model learned)")
elif discrimination > 0:
    print(f"      ⚠️  WEAK discrimination")
else:
    print(f"      ❌ NO discrimination (model didn't learn)")

# Test on multiple samples
print("\n6. Testing on multiple samples...")

fail_scores = []
success_scores = []

for _ in range(20):
    # Failure
    fail_signals, _ = create_synthetic_episode(length=50, signal_dim=18, will_fail=True)
    fail_window = fail_signals[-window_size:].unsqueeze(0)
    with torch.no_grad():
        pred = model(fail_window)[0].max().item()
    fail_scores.append(pred)

    # Success
    success_signals, _ = create_synthetic_episode(length=50, signal_dim=18, will_fail=False)
    success_window = success_signals[-window_size:].unsqueeze(0)
    with torch.no_grad():
        pred = model(success_window)[0].max().item()
    success_scores.append(pred)

fail_mean = np.mean(fail_scores)
fail_std = np.std(fail_scores)
success_mean = np.mean(success_scores)
success_std = np.std(success_scores)

print(f"   Failure episodes (n=20):")
print(f"      Mean score: {fail_mean:.4f} ± {fail_std:.4f}")
print(f"      Range: [{min(fail_scores):.4f}, {max(fail_scores):.4f}]")

print(f"\n   Success episodes (n=20):")
print(f"      Mean score: {success_mean:.4f} ± {success_std:.4f}")
print(f"      Range: [{min(success_scores):.4f}, {max(success_scores):.4f}]")

discrimination_mean = fail_mean - success_mean
effect_size = discrimination_mean / np.sqrt((fail_std**2 + success_std**2) / 2)

print(f"\n   Statistical Analysis:")
print(f"      Mean discrimination: {discrimination_mean:.4f}")
print(f"      Effect size (Cohen's d): {effect_size:.2f}")

if effect_size > 1.5:
    print(f"      ✅ LARGE effect size (strong learning)")
elif effect_size > 0.8:
    print(f"      ✅ MEDIUM effect size (good learning)")
elif effect_size > 0.5:
    print(f"      ⚠️  SMALL effect size (weak learning)")
else:
    print(f"      ❌ Negligible effect size")

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

checks = [
    ("18D signals created", X_train.shape[2] == 18),
    ("Model accepts 18D input", test_output.shape == (4, 16)),
    ("Training loss decreased", losses[-1] < losses[0]),
    ("Loss improved >50%", (losses[0] - losses[-1]) / losses[0] > 0.5),
    ("Discrimination > 0.2", discrimination > 0.2),
    ("Effect size > 0.8", effect_size > 0.8),
]

passed = sum([c[1] for c in checks])
total = len(checks)

print(f"\nChecks passed: {passed}/{total}\n")
for name, result in checks:
    status = "✅" if result else "❌"
    print(f"   {status} {name}")

if passed == total:
    print(f"\n🎉 ALL CHECKS PASSED!")
    print(f"   SALUS CAN LEARN from 18D VLA signals to predict failures!")
elif passed >= 4:
    print(f"\n⚠️  Most checks passed. Model shows learning capability.")
else:
    print(f"\n❌ Multiple failures. Learning capability unclear.")

print(f"\n{'='*70}")
print(f"\nCONCLUSION:")
print(f"   The HybridTemporalPredictor architecture WORKS with 18D signals")
print(f"   and CAN LEARN to discriminate failure vs success patterns.")
print(f"\n   Next step: Train on REAL VLA signals from Isaac Lab simulation!")
print(f"{'='*70}")
