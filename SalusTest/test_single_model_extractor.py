"""
Unit tests for SingleModelSignalExtractor
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from salus.core.vla.single_model_extractor import SingleModelSignalExtractor


def test_basic_extraction():
    """Test basic signal extraction with minimal inputs."""
    print("\n" + "="*70)
    print("TEST 1: Basic Signal Extraction")
    print("="*70)

    extractor = SingleModelSignalExtractor(device='cpu')

    # Mock VLA output (minimal)
    vla_output = {
        'action': torch.randn(1, 6),
    }
    robot_state = torch.randn(1, 7)

    signals = extractor.extract(vla_output, robot_state)

    print(f"✓ Output shape: {signals.shape}")
    assert signals.shape == (1, 12), f"Expected (1, 12), got {signals.shape}"

    print(f"✓ No NaN values: {not torch.isnan(signals).any()}")
    assert not torch.isnan(signals).any(), "Signals contain NaN"

    print(f"✓ No Inf values: {not torch.isinf(signals).any()}")
    assert not torch.isinf(signals).any(), "Signals contain Inf"

    print(f"\nSignal values: {signals[0].numpy()}")
    print("✅ TEST 1 PASSED")


def test_full_extraction_with_internals():
    """Test extraction with all VLA internals available."""
    print("\n" + "="*70)
    print("TEST 2: Full Extraction (with hidden states & logits)")
    print("="*70)

    extractor = SingleModelSignalExtractor(device='cpu')

    # Mock VLA output (complete)
    vla_output = {
        'action': torch.randn(1, 6),
        'hidden_state': torch.randn(1, 512),
        'action_logits': torch.randn(1, 6),
    }
    robot_state = torch.randn(1, 7)

    signals = extractor.extract(vla_output, robot_state)

    print(f"✓ Output shape: {signals.shape}")
    print(f"✓ No NaN values: {not torch.isnan(signals).any()}")
    print(f"✓ No Inf values: {not torch.isinf(signals).any()}")

    # Check specific signals
    signal_names = extractor.get_signal_names()
    print(f"\nAll 12 signals extracted:")
    for i, (name, value) in enumerate(zip(signal_names, signals[0])):
        print(f"   {name}: {value.item():.6f}")

    # Softmax entropy (signal 8) should be non-zero
    assert signals[0, 7] > 0, "Softmax entropy should be positive"
    print(f"\n✓ Softmax entropy (signal 8) is positive: {signals[0, 7]:.4f}")

    # Max softmax probability (signal 9) should be in [0, 1]
    assert 0 <= signals[0, 8] <= 1, "Max softmax prob should be in [0, 1]"
    print(f"✓ Max softmax prob (signal 9) in range: {signals[0, 8]:.4f}")

    print("\n✅ TEST 2 PASSED")


def test_temporal_dynamics():
    """Test temporal signals over multiple timesteps."""
    print("\n" + "="*70)
    print("TEST 3: Temporal Dynamics (5 timesteps)")
    print("="*70)

    extractor = SingleModelSignalExtractor(device='cpu')

    # Feed 5 timesteps
    for t in range(5):
        vla_output = {
            'action': torch.randn(1, 6) * 0.5,  # Moderate action scale
            'hidden_state': torch.randn(1, 512),
            'action_logits': torch.randn(1, 6),
        }
        robot_state = torch.randn(1, 7) * 0.5

        signals = extractor.extract(vla_output, robot_state)

        print(f"\nTimestep {t}:")
        print(f"   Signal 1 (Volatility):    {signals[0, 0]:.6f}")
        print(f"   Signal 3 (Acceleration):  {signals[0, 2]:.6f}")
        print(f"   Signal 5 (Latent Drift):  {signals[0, 4]:.6f}")
        print(f"   Signal 12 (Consistency):  {signals[0, 11]:.6f}")

    # After 5 timesteps, temporal signals should be computed
    assert signals[0, 0] >= 0, "Volatility should be non-negative"
    assert signals[0, 2] >= 0, "Acceleration should be non-negative"

    print("\n✓ Temporal signals updated over time")
    print("✅ TEST 3 PASSED")


def test_batch_processing():
    """Test extraction with batch size > 1."""
    print("\n" + "="*70)
    print("TEST 4: Batch Processing (batch_size=4)")
    print("="*70)

    extractor = SingleModelSignalExtractor(device='cpu')

    batch_size = 4
    vla_output = {
        'action': torch.randn(batch_size, 6),
        'hidden_state': torch.randn(batch_size, 512),
        'action_logits': torch.randn(batch_size, 6),
    }
    robot_state = torch.randn(batch_size, 7)

    signals = extractor.extract(vla_output, robot_state)

    print(f"✓ Output shape: {signals.shape}")
    assert signals.shape == (batch_size, 12), f"Expected ({batch_size}, 12)"

    print(f"✓ All batch elements have valid signals")
    assert not torch.isnan(signals).any()
    assert not torch.isinf(signals).any()

    print(f"\nSignal ranges across batch:")
    for i in range(12):
        min_val = signals[:, i].min().item()
        max_val = signals[:, i].max().item()
        mean_val = signals[:, i].mean().item()
        print(f"   Signal {i+1:2d}: min={min_val:7.4f}, max={max_val:7.4f}, mean={mean_val:7.4f}")

    print("\n✅ TEST 4 PASSED")


def test_reset_functionality():
    """Test reset() clears history."""
    print("\n" + "="*70)
    print("TEST 5: Reset Functionality")
    print("="*70)

    extractor = SingleModelSignalExtractor(device='cpu')

    # Feed some timesteps
    for t in range(3):
        vla_output = {'action': torch.randn(1, 6)}
        robot_state = torch.randn(1, 7)
        extractor.extract(vla_output, robot_state)

    print(f"✓ Before reset - action history size: {len(extractor.action_history)}")
    assert len(extractor.action_history) == 3, "Should have 3 actions in history"

    # Reset
    extractor.reset()

    print(f"✓ After reset - action history size: {len(extractor.action_history)}")
    assert len(extractor.action_history) == 0, "History should be cleared"
    assert extractor.prev_action is None, "prev_action should be None"

    # Extract again - should not crash
    vla_output = {'action': torch.randn(1, 6)}
    robot_state = torch.randn(1, 7)
    signals = extractor.extract(vla_output, robot_state)

    print(f"✓ Extraction after reset works: {signals.shape}")

    print("\n✅ TEST 5 PASSED")


def test_graceful_degradation():
    """Test extractor handles missing signals gracefully."""
    print("\n" + "="*70)
    print("TEST 6: Graceful Degradation (missing hidden/logits)")
    print("="*70)

    extractor = SingleModelSignalExtractor(device='cpu')

    # Only action, no hidden states or logits
    vla_output = {
        'action': torch.randn(1, 6),
    }
    robot_state = torch.randn(1, 7)

    signals = extractor.extract(vla_output, robot_state)

    print(f"✓ Output shape: {signals.shape}")
    assert signals.shape == (1, 12), "Should still output 12D"

    # Signals 5-9 should be zeros (hidden states & logits missing)
    print(f"\nSignals when internals missing:")
    print(f"   Signal 5 (Latent Drift):   {signals[0, 4]:.6f} (should be 0)")
    print(f"   Signal 6 (Norm Spike):     {signals[0, 5]:.6f} (should be 0)")
    print(f"   Signal 7 (OOD):            {signals[0, 6]:.6f} (should be 0)")
    print(f"   Signal 8 (Entropy):        {signals[0, 7]:.6f} (should be 0)")
    print(f"   Signal 9 (Max Prob):       {signals[0, 8]:.6f} (should be 0)")

    # But signals 1-4 (temporal) should still work
    print(f"\nTemporal signals still work:")
    print(f"   Signal 1 (Volatility):     {signals[0, 0]:.6f}")
    print(f"   Signal 2 (Magnitude):      {signals[0, 1]:.6f}")

    assert signals[0, 1] > 0, "Action magnitude should be positive"

    print("\n✓ Extractor degrades gracefully with missing features")
    print("✅ TEST 6 PASSED")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SINGLE-MODEL SIGNAL EXTRACTOR UNIT TESTS")
    print("="*70)

    try:
        test_basic_extraction()
        test_full_extraction_with_internals()
        test_temporal_dynamics()
        test_batch_processing()
        test_reset_functionality()
        test_graceful_degradation()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        print("\nSingleModelSignalExtractor is ready for deployment!")
        print("Next: Modify VLA wrapper to use single model (Phase 2)")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
