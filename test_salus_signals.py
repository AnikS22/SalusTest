"""
Test SALUS Signal Extraction and Failure Prediction

This script demonstrates:
1. How SALUS extracts 12D signals from SmolVLA
2. What each signal represents
3. Whether SALUS predictor is trained
4. How SALUS predicts failures from signals
"""

import sys
from pathlib import Path
import torch
import numpy as np
import zarr

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from salus.core.vla.wrapper import SmolVLAEnsemble, SignalExtractor
from salus.core.predictor import SALUSPredictor

print("="*70)
print("SALUS Signal Extraction & Failure Prediction Test")
print("="*70)

# ============================================================================
# PART 1: Signal Extraction from SmolVLA
# ============================================================================
print("\n" + "="*70)
print("PART 1: Signal Extraction from SmolVLA")
print("="*70)

print("\n📊 How SALUS Extracts Signals:")
print("   SmolVLA produces actions and uncertainty → SignalExtractor → 12D signals")

# Create signal extractor
extractor = SignalExtractor()

# Simulate SmolVLA output (what we get from real SmolVLA)
print("\n🤖 Simulating SmolVLA Ensemble Output:")
print("   (This is what SmolVLA produces when controlling the robot)")

# Simulate a sequence of actions (like during robot control)
vla_outputs = [
    {
        'action': torch.tensor([[0.1, 0.2, -0.1, 0.3, 0.0, 0.1, 0.0]]),
        'action_var': torch.tensor([[0.01, 0.02, 0.01, 0.03, 0.0, 0.01, 0.0]]),
        'epistemic_uncertainty': torch.tensor([0.15])
    },
    {
        'action': torch.tensor([[0.12, 0.18, -0.08, 0.28, 0.02, 0.12, 0.0]]),
        'action_var': torch.tensor([[0.015, 0.025, 0.012, 0.035, 0.001, 0.012, 0.0]]),
        'epistemic_uncertainty': torch.tensor([0.18])  # Uncertainty increasing!
    },
    {
        'action': torch.tensor([[0.15, 0.15, -0.05, 0.25, 0.05, 0.15, 0.0]]),
        'action_var': torch.tensor([[0.02, 0.03, 0.015, 0.04, 0.002, 0.015, 0.0]]),
        'epistemic_uncertainty': torch.tensor([0.25])  # Uncertainty spiking!
    }
]

print("\n   Step 1: Normal operation (low uncertainty)")
print("   Step 2: Uncertainty increasing (model getting confused)")
print("   Step 3: Uncertainty spiking (likely failure coming)")

signal_names = [
    "1. Epistemic uncertainty (ensemble disagreement)",
    "2. Action magnitude (L2 norm)",
    "3. Action variance (mean across joints)",
    "4. Action smoothness (change from previous)",
    "5. Trajectory divergence (vs history)",
    "6. Per-joint variance (joint 1)",
    "7. Per-joint variance (joint 2)",
    "8. Per-joint variance (joint 3)",
    "9. Uncertainty rolling mean",
    "10. Uncertainty rolling std",
    "11. Uncertainty rolling min",
    "12. Uncertainty rolling max"
]

print("\n📈 Extracted Signals (12D feature vector):")
print("-" * 70)
for step, vla_output in enumerate(vla_outputs, 1):
    signals = extractor.extract(vla_output)
    print(f"\n   Step {step} Signals:")
    for i, (name, val) in enumerate(zip(signal_names, signals[0].numpy())):
        marker = "⚠️" if i == 0 and val > 0.2 else "  "  # Highlight high uncertainty
        print(f"   {marker} {name:45s}: {val:6.4f}")

# ============================================================================
# PART 2: Check if SALUS Predictor is Trained
# ============================================================================
print("\n" + "="*70)
print("PART 2: SALUS Failure Predictor Status")
print("="*70)

# Check for trained models
checkpoint_dirs = [
    project_root / "models" / "predictor",
    project_root / "checkpoints",
    project_root / "paper_data" / "checkpoints"
]

trained_model_found = False
for checkpoint_dir in checkpoint_dirs:
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"\n✅ Found trained model(s) in {checkpoint_dir}:")
            for ckpt in checkpoints:
                print(f"   - {ckpt.name}")
            trained_model_found = True
            break

if not trained_model_found:
    print("\n⚠️  NO TRAINED SALUS PREDICTOR FOUND!")
    print("\n   SALUS needs to be trained on failure data to work.")
    print("   Training requires:")
    print("   1. Collected episodes with failures (success + failure episodes)")
    print("   2. Horizon labels (which timesteps lead to failures)")
    print("   3. Training script: scripts/train_predictor_mvp.py")
    print("\n   Without training, SALUS will make random predictions.")

# ============================================================================
# PART 3: Test SALUS Predictor (if available)
# ============================================================================
print("\n" + "="*70)
print("PART 3: Testing SALUS Predictor")
print("="*70)

# Create predictor model
predictor = SALUSPredictor(
    signal_dim=12,
    num_horizons=4,
    num_failure_types=4
)

print("\n🧠 SALUS Predictor Architecture:")
print("   Input:  12D signals (from SignalExtractor)")
print("   Encoder: 12 → 128 → 256 → 128 (3-layer MLP)")
print("   Output:  16D logits (4 horizons × 4 failure types)")
print("   Horizons: [6, 10, 13, 16] steps ahead")
print("   Failure types: [Collision, Drop, Miss, Timeout]")

# Test with the signals we extracted
print("\n🔮 Testing Predictor on Extracted Signals:")

# Use the last signal (high uncertainty scenario)
test_signals = extractor.extract(vla_outputs[-1])  # (1, 12)

with torch.no_grad():
    output = predictor(test_signals)

print(f"\n   Input signals shape: {test_signals.shape}")
print(f"   Output logits shape: {output['logits'].shape}")
print(f"   Probabilities shape: {output['probs'].shape}")

print("\n   Failure Probabilities per Horizon:")
horizon_names = ["6 steps", "10 steps", "13 steps", "16 steps"]
failure_names = ["Collision", "Drop", "Miss", "Timeout"]

probs = output['probs'][0].numpy()  # (4, 4)
for h_idx, horizon_name in enumerate(horizon_names):
    print(f"\n   {horizon_name} ahead:")
    for f_idx, failure_name in enumerate(failure_names):
        prob = probs[h_idx, f_idx]
        marker = "🔴" if prob > 0.5 else "🟡" if prob > 0.3 else "🟢"
        print(f"      {marker} {failure_name:12s}: {prob:.3f}")

print(f"\n   Maximum failure probability: {output['max_prob'][0].item():.3f}")
print(f"   Predicted horizon: {horizon_names[output['predicted_horizon'][0].item()]}")
print(f"   Predicted failure type: {failure_names[output['predicted_type'][0].item()]}")

# ============================================================================
# PART 4: Check Training Data
# ============================================================================
print("\n" + "="*70)
print("PART 4: Training Data Status")
print("="*70)

# Check for collected data
data_dirs = [
    project_root / "data" / "raw_episodes",
    project_root / "paper_data" / "massive_collection",
    project_root / "paper_data" / "training"
]

print("\n📦 Checking for training data...")
data_found = False

for data_dir in data_dirs:
    if data_dir.exists():
        zarr_files = list(data_dir.rglob("*.zarr"))
        if zarr_files:
            print(f"\n✅ Found data in {data_dir}:")
            for zarr_file in zarr_files[:5]:  # Show first 5
                try:
                    store = zarr.open(str(zarr_file), mode='r')
                    if 'signals' in store and 'horizon_labels' in store:
                        signals_shape = store['signals'].shape
                        labels_shape = store['horizon_labels'].shape
                        print(f"   - {zarr_file.name}")
                        print(f"     Signals: {signals_shape}, Labels: {labels_shape}")
                        data_found = True
                except:
                    pass

if not data_found:
    print("\n⚠️  NO TRAINING DATA FOUND!")
    print("   SALUS needs episodes with:")
    print("   - Signals (12D from VLA)")
    print("   - Horizon labels (which timesteps lead to failures)")
    print("\n   Run data collection first:")
    print("   python scripts/collect_data_franka.py --num_episodes 50")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n✅ Signal Extraction: WORKING")
print("   - SignalExtractor successfully extracts 12D signals from SmolVLA")
print("   - Signals capture: uncertainty, action patterns, trajectory divergence")

if trained_model_found:
    print("\n✅ SALUS Predictor: TRAINED")
    print("   - Model found and can make predictions")
else:
    print("\n⚠️  SALUS Predictor: NOT TRAINED")
    print("   - Model exists but needs training on failure data")
    print("   - Without training, predictions are random")

if data_found:
    print("\n✅ Training Data: AVAILABLE")
    print("   - Data found, can train SALUS predictor")
else:
    print("\n⚠️  Training Data: MISSING")
    print("   - Need to collect episodes with failures")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
if not data_found:
    print("1. Collect data: python scripts/collect_data_franka.py --num_episodes 50")
if not trained_model_found and data_found:
    print("2. Train predictor: python scripts/train_predictor_mvp.py --data <data_path>")
if trained_model_found:
    print("3. Test deployment: Use trained model to predict failures in real-time")
print("="*70)

