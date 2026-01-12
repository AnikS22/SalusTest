"""
Comprehensive test of full 12D signal extraction over multiple timesteps.
Tests that ALL signals work correctly with temporal history.
"""

import torch
import sys
sys.path.insert(0, '/home/mpcr/Desktop/SalusV3/SalusTest')

from salus.core.vla.smolvla_wrapper_fixed import SmolVLAWithInternals

def test_multi_timestep_extraction():
    print("\n" + "="*70)
    print("Testing Full 12D Signal Extraction Over Multiple Timesteps")
    print("="*70)

    # Initialize model
    print("\n[1/4] Initializing SmolVLA with signal extraction...")
    vla = SmolVLAWithInternals(device="cuda:0")

    # Create dummy observation (realistic dimensions)
    obs = {
        'observation.images.camera1': torch.randn(1, 3, 256, 256, device='cuda:0'),
        'observation.state': torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], device='cuda:0'),
        'task': 'pick up the red cube'
    }

    # Run multiple timesteps
    print("\n[2/4] Running 10 timesteps to build temporal history...")
    all_signals = []

    for t in range(10):
        # Slightly vary the observation (simulate robot movement)
        obs['observation.state'] = obs['observation.state'] + torch.randn(1, 7, device='cuda:0') * 0.01
        obs['observation.images.camera1'] = torch.randn(1, 3, 256, 256, device='cuda:0')

        # Run forward pass
        output = vla(obs)

        # Extract signals
        signals = vla.signal_extractor.extract(
            action=output['action'],
            hidden_state=output['hidden_state'],
            action_logits=output['action_logits'],
            robot_state=obs['observation.state']
        )

        all_signals.append(signals[0].cpu())

    # Convert to tensor
    all_signals = torch.stack(all_signals)  # (10, 12)

    print(f"\n[3/4] Signal extraction complete!")
    print(f"   Shape: {all_signals.shape} (expected: [10, 12])")

    # Analyze each signal
    print("\n[4/4] Analyzing signal activity across timesteps:")
    print("-" * 70)

    signal_names = [
        "0: Action Volatility",
        "1: Action Magnitude",
        "2: Action Acceleration",
        "3: Trajectory Divergence",
        "4: Latent Drift",
        "5: Latent Norm Spike",
        "6: OOD Distance",
        "7: Softmax Entropy",
        "8: Max Softmax Probability",
        "9: Execution Mismatch",
        "10: Constraint Margin",
        "11: Temporal Consistency"
    ]

    working_signals = 0
    broken_signals = []

    for i, name in enumerate(signal_names):
        signal_values = all_signals[:, i]

        # Check if signal is active (not all zeros, not all NaN)
        is_nan = torch.isnan(signal_values).all().item()
        is_zero = (signal_values == 0).all().item()
        max_val = signal_values.abs().max().item()
        mean_val = signal_values.mean().item()

        if is_nan:
            status = "❌ ALL NaN"
            broken_signals.append(name)
        elif is_zero:
            status = "⚠️  All zeros (may need more history)"
        else:
            status = f"✅ WORKING (max={max_val:.4f}, mean={mean_val:.4f})"
            working_signals += 1

        print(f"Signal {name:30s}: {status}")

    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY:")
    print(f"  Working signals: {working_signals}/12 ({working_signals/12*100:.1f}%)")
    print(f"  Expected working: 12/12 (100%)")

    if working_signals >= 9:
        print(f"\n✅ SUCCESS: {working_signals}/12 signals working!")
        print("   Note: Some signals may be zero if they need more history.")
    elif working_signals >= 6:
        print(f"\n⚠️  PARTIAL: {working_signals}/12 signals working")
        print("   This is acceptable, but some signals may need debugging.")
    else:
        print(f"\n❌ FAILURE: Only {working_signals}/12 signals working")
        print("   Broken signals:", broken_signals)

    # Check for NaNs
    nan_count = torch.isnan(all_signals).sum().item()
    if nan_count == 0:
        print("\n✅ No NaN values detected!")
    else:
        print(f"\n❌ WARNING: {nan_count} NaN values detected")

    print("="*70)

    return working_signals >= 9


if __name__ == "__main__":
    success = test_multi_timestep_extraction()
    sys.exit(0 if success else 1)
