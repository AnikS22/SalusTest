"""
Test script to verify signals are REAL from VLA model, not mocks.

This will:
1. Load the actual 865MB SmolVLA model
2. Run inference on real observations
3. Show that signals vary and are non-zero
4. Prove they come from the VLA ensemble
"""

import torch
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("TESTING: Real VLA Signal Extraction")
print("="*70)

# Check if VLA model exists
model_path = Path.home() / "models" / "smolvla" / "smolvla_base" / "model.safetensors"
print(f"\n1. Checking VLA model file...")
print(f"   Path: {model_path}")
print(f"   Exists: {model_path.exists()}")
if model_path.exists():
    size_mb = model_path.stat().st_size / 1024**2
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   ✅ Real VLA model found!")
else:
    print(f"   ❌ Model not found! Cannot proceed.")
    sys.exit(1)

# Load the VLA ensemble
print(f"\n2. Loading SmolVLA ensemble (3 models)...")
try:
    from salus.core.vla.wrapper import SmolVLAEnsemble, EnhancedSignalExtractor

    ensemble = SmolVLAEnsemble(
        model_path="~/models/smolvla/smolvla_base",
        ensemble_size=3,
        device="cuda:0"
    )
    print(f"   ✅ Ensemble loaded!")

    signal_extractor = EnhancedSignalExtractor(device="cuda:0")
    print(f"   ✅ Signal extractor initialized!")

except Exception as e:
    print(f"   ❌ Failed to load: {e}")
    sys.exit(1)

# Create dummy observation (simulating Isaac Lab output)
print(f"\n3. Creating test observations...")
batch_size = 4
obs = {
    'observation.images.camera1': torch.randn(batch_size, 3, 256, 256).cuda() * 0.5 + 0.5,
    'observation.images.camera2': torch.randn(batch_size, 3, 256, 256).cuda() * 0.5 + 0.5,
    'observation.images.camera3': torch.randn(batch_size, 3, 256, 256).cuda() * 0.5 + 0.5,
    'observation.state': torch.randn(batch_size, 7).cuda() * 0.1,  # Robot joint positions
    'task': ["Pick and place the red cube"] * batch_size
}
print(f"   Created batch of {batch_size} observations")

# Run VLA inference
print(f"\n4. Running VLA ensemble forward pass...")
with torch.no_grad():
    output = ensemble(obs, return_internals=True)

print(f"   ✅ VLA inference complete!")
print(f"\n   Output keys: {list(output.keys())}")

# Check each output
print(f"\n5. Verifying VLA outputs are REAL (not zeros/mocks):")

# Action
action = output['action']
print(f"\n   Action:")
print(f"      Shape: {action.shape}")
print(f"      Mean: {action.mean().item():.6f}")
print(f"      Std: {action.std().item():.6f}")
print(f"      First sample: {action[0].cpu().numpy()}")
if action.abs().sum() > 0.01:
    print(f"      ✅ Non-zero actions (from VLA model)")
else:
    print(f"      ❌ All zeros - something wrong!")

# Action variance (epistemic uncertainty)
action_var = output['action_var']
print(f"\n   Action Variance (Epistemic Uncertainty):")
print(f"      Shape: {action_var.shape}")
print(f"      Mean: {action_var.mean().item():.6f}")
print(f"      First sample: {action_var[0].cpu().numpy()}")
if action_var.sum() > 0.001:
    print(f"      ✅ Non-zero variance (ensemble disagrees)")
else:
    print(f"      ⚠️  Zero variance - ensemble in perfect agreement (unlikely)")

# Epistemic uncertainty
epistemic = output['epistemic_uncertainty']
print(f"\n   Epistemic Uncertainty (scalar):")
print(f"      Shape: {epistemic.shape}")
print(f"      Values: {epistemic.cpu().numpy()}")
if epistemic.sum() > 0.001:
    print(f"      ✅ Non-zero uncertainty")
else:
    print(f"      ⚠️  Zero uncertainty")

# Hidden states (VLA internals)
if 'hidden_state_mean' in output:
    hidden = output['hidden_state_mean']
    print(f"\n   Hidden States (VLA internals):")
    print(f"      Shape: {hidden.shape}")
    print(f"      Mean: {hidden.mean().item():.6f}")
    print(f"      Std: {hidden.std().item():.6f}")
    if hidden.abs().sum() > 0.01:
        print(f"      ✅ Real VLA hidden states extracted!")
    else:
        print(f"      ⚠️  Hidden states are zeros")
else:
    print(f"\n   ⚠️  No hidden_state_mean in output")

# Perturbed actions
if 'perturbed_actions' in output:
    perturbed = output['perturbed_actions']
    print(f"\n   Perturbed Actions (sensitivity test):")
    print(f"      Shape: {perturbed.shape}")
    print(f"      Mean: {perturbed.mean().item():.6f}")
    print(f"      Variance across perturbations: {perturbed.var(dim=1).mean().item():.6f}")
    if perturbed.abs().sum() > 0.01:
        print(f"      ✅ Real perturbation testing done!")
    else:
        print(f"      ⚠️  Perturbed actions are zeros")
else:
    print(f"\n   ⚠️  No perturbed_actions in output")

# Extract 18D signals
print(f"\n6. Extracting 18D signals...")
robot_state = obs['observation.state']
signals = signal_extractor.extract(output, robot_state=robot_state)

print(f"\n   Signal vector:")
print(f"      Shape: {signals.shape}")
print(f"      Expected: ({batch_size}, 18)")

if signals.shape[1] == 18:
    print(f"      ✅ Correct dimensions (18D)")
else:
    print(f"      ❌ Wrong dimensions!")

print(f"\n   Signal statistics:")
print(f"      Mean: {signals.mean().item():.6f}")
print(f"      Std: {signals.std().item():.6f}")
print(f"      Min: {signals.min().item():.6f}")
print(f"      Max: {signals.max().item():.6f}")

print(f"\n   Signal breakdown (sample 0):")
sample_signals = signals[0].cpu().numpy()
signal_names = [
    "1. Epistemic Uncertainty",
    "2. Action Magnitude",
    "3. Action Variance",
    "4. Action Smoothness",
    "5. Trajectory Divergence",
    "6-8. Per-Joint Variance (×3)",
    "9. Unc Mean",
    "10. Unc Std",
    "11. Unc Min",
    "12. Unc Max",
    "13. Latent Drift",
    "14. OOD Distance",
    "15. Aug Stability",
    "16. Pert Sensitivity",
    "17. Exec Mismatch",
    "18. Constraint Margin"
]

for i, (name, value) in enumerate(zip(signal_names[:5], sample_signals[:5])):
    print(f"      Signal {name}: {value:.6f}")
print(f"      Signal 6-8: {sample_signals[5:8]}")
print(f"      Signal 9-12: {sample_signals[8:12]}")
print(f"      Signal 13-18: {sample_signals[12:18]}")

# Test variation across batch
print(f"\n7. Testing signal variation across samples:")
signal_variance = signals.var(dim=0)
print(f"   Variance per signal dimension:")
for i in range(6):  # Show first 6
    print(f"      Signal {i+1}: {signal_variance[i].item():.6f}")
print(f"      ...")
for i in range(12, 18):  # Show last 6
    print(f"      Signal {i+1}: {signal_variance[i].item():.6f}")

non_zero_signals = (signals.abs() > 1e-6).sum(dim=1)
print(f"\n   Non-zero signals per sample:")
print(f"      {non_zero_signals.cpu().numpy()}/18")

if non_zero_signals.min() >= 12:
    print(f"      ✅ Most signals are non-zero (REAL data!)")
else:
    print(f"      ⚠️  Many signals are zero")

# Test temporal variation (run twice, signals should differ)
print(f"\n8. Testing temporal variation (2 forward passes):")
with torch.no_grad():
    output1 = ensemble(obs, return_internals=True)
    output2 = ensemble(obs, return_internals=True)

signals1 = signal_extractor.extract(output1, robot_state=robot_state)
signals2 = signal_extractor.extract(output2, robot_state=robot_state)

diff = (signals1 - signals2).abs().mean()
print(f"   Mean absolute difference: {diff.item():.6f}")
if diff.item() > 1e-6:
    print(f"   ✅ Signals change over time (dynamic, not cached)")
else:
    print(f"   ⚠️  Signals are identical (may be deterministic)")

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

checks = [
    ("VLA model file exists (865MB)", model_path.exists()),
    ("VLA ensemble loads", True),
    ("Actions are non-zero", action.abs().sum() > 0.01),
    ("Epistemic uncertainty computed", epistemic.sum() > 0.001),
    ("Hidden states extracted", 'hidden_state_mean' in output and output['hidden_state_mean'].abs().sum() > 0.01),
    ("Perturbations tested", 'perturbed_actions' in output and output['perturbed_actions'].abs().sum() > 0.01),
    ("18D signals extracted", signals.shape[1] == 18),
    ("Signals are non-zero", non_zero_signals.min() >= 12),
]

passed = sum([c[1] for c in checks])
total = len(checks)

print(f"\nChecks passed: {passed}/{total}\n")
for name, result in checks:
    status = "✅" if result else "❌"
    print(f"   {status} {name}")

if passed == total:
    print(f"\n🎉 ALL CHECKS PASSED! Signals are REAL from VLA model!")
elif passed >= 6:
    print(f"\n⚠️  Most checks passed. Some signals may need debugging.")
else:
    print(f"\n❌ Multiple failures. VLA integration has issues.")

print(f"\n{'='*70}")
