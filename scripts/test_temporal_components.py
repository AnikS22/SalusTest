"""
Test script for SALUS temporal forecasting components.

This validates that all new components work together correctly:
1. HybridTemporalPredictor (Conv+GRU)
2. LatentHealthStateEncoder
3. TemporalFocalLoss
4. TemporalSmoothnessLoss
5. HardNegativeSampler
6. Label generation with anti-leakage

Usage:
    python scripts/test_temporal_components.py
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from salus.models.temporal_predictor import (
    HybridTemporalPredictor,
    TemporalFocalLoss,
    TemporalSmoothnessLoss,
    HardNegativeSampler,
    compute_temporal_stability
)
from salus.models.latent_encoder import (
    LatentHealthStateEncoder,
    LatentTemporalPredictor
)
from salus.data.preprocess_labels import (
    compute_failure_labels,
    compute_failure_labels_with_randomization,
    compute_soft_temporal_labels
)


def test_hybrid_predictor():
    """Test HybridTemporalPredictor."""
    print("\n" + "=" * 60)
    print("Test 1: HybridTemporalPredictor (Conv+GRU)")
    print("=" * 60)

    batch_size = 8
    window_size = 10
    signal_dim = 12

    # Create dummy temporal windows
    signals = torch.randn(batch_size, window_size, signal_dim)

    # Initialize model
    model = HybridTemporalPredictor(
        signal_dim=signal_dim,
        conv_dim=32,
        gru_dim=64
    )

    # Forward pass
    predictions = model(signals)

    print(f"✓ Input shape: {signals.shape}")
    print(f"✓ Output shape: {predictions.shape}")
    print(f"✓ Expected output: ({batch_size}, 16)")
    print(f"✓ Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    # Test per-horizon prediction
    horizon_pred = model.predict_at_horizon(signals, horizon_idx=2)
    print(f"✓ Horizon 2 predictions: {horizon_pred.shape}")
    print(f"✓ Expected: ({batch_size}, 4)")

    assert predictions.shape == (batch_size, 16), "Incorrect output shape!"
    assert horizon_pred.shape == (batch_size, 4), "Incorrect horizon shape!"
    assert predictions.min() >= 0 and predictions.max() <= 1, "Predictions not in [0, 1]!"

    print("\n✅ HybridTemporalPredictor test PASSED!")
    return True


def test_latent_encoder():
    """Test LatentHealthStateEncoder."""
    print("\n" + "=" * 60)
    print("Test 2: LatentHealthStateEncoder")
    print("=" * 60)

    batch_size = 8
    window_size = 10
    signal_dim = 12
    latent_dim = 6

    # Create dummy signals
    signals = torch.randn(batch_size, window_size, signal_dim)

    # Initialize encoder
    encoder = LatentHealthStateEncoder(
        signal_dim=signal_dim,
        latent_dim=latent_dim
    )

    # Encode
    latent = encoder(signals)
    print(f"✓ Input shape: {signals.shape}")
    print(f"✓ Latent shape: {latent.shape}")
    print(f"✓ Expected: ({batch_size}, {window_size}, {latent_dim})")

    # Decode
    reconstructed = encoder.decode(latent)
    print(f"✓ Reconstructed shape: {reconstructed.shape}")

    # Test auxiliary losses
    dummy_labels = torch.randint(0, 2, (batch_size, 16)).float()
    aux_losses = encoder.compute_aux_losses(signals, latent, dummy_labels)

    print(f"✓ Auxiliary losses:")
    for key, value in aux_losses.items():
        print(f"    {key}: {value.item():.4f}")

    assert latent.shape == (batch_size, window_size, latent_dim), "Incorrect latent shape!"
    assert reconstructed.shape == signals.shape, "Incorrect reconstruction shape!"
    assert 'reconstruction' in aux_losses, "Missing reconstruction loss!"

    print("\n✅ LatentHealthStateEncoder test PASSED!")
    return True


def test_latent_temporal_predictor():
    """Test combined LatentTemporalPredictor."""
    print("\n" + "=" * 60)
    print("Test 3: LatentTemporalPredictor (Latent + Hybrid)")
    print("=" * 60)

    batch_size = 8
    window_size = 10
    signal_dim = 12
    latent_dim = 6

    # Create dummy signals
    signals = torch.randn(batch_size, window_size, signal_dim)

    # Initialize model
    model = LatentTemporalPredictor(
        signal_dim=signal_dim,
        latent_dim=latent_dim
    )

    # Forward pass
    predictions, latent = model(signals, return_latent=True)

    print(f"✓ Input shape: {signals.shape}")
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Latent shape: {latent.shape}")
    print(f"✓ Expected predictions: ({batch_size}, 16)")
    print(f"✓ Expected latent: ({batch_size}, {window_size}, {latent_dim})")

    assert predictions.shape == (batch_size, 16), "Incorrect predictions shape!"
    assert latent.shape == (batch_size, window_size, latent_dim), "Incorrect latent shape!"

    print("\n✅ LatentTemporalPredictor test PASSED!")
    return True


def test_temporal_focal_loss():
    """Test TemporalFocalLoss."""
    print("\n" + "=" * 60)
    print("Test 4: TemporalFocalLoss")
    print("=" * 60)

    batch_size = 8

    # Create dummy predictions and labels
    predictions = torch.rand(batch_size, 16)  # Random predictions [0, 1]
    targets = torch.randint(0, 2, (batch_size, 16)).float()  # Binary labels
    episode_success = torch.tensor([True, False, True, False, True, False, True, False])

    # Initialize loss
    criterion = TemporalFocalLoss(
        pos_weight=3.0,
        fp_penalty_weight=2.0,
        focal_gamma=2.0
    )

    # Compute loss
    loss = criterion(predictions, targets, episode_success)

    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Targets shape: {targets.shape}")
    print(f"✓ Episode success shape: {episode_success.shape}")
    print(f"✓ Loss value: {loss.item():.4f}")

    # Test without episode success mask
    loss_no_mask = criterion(predictions, targets, None)
    print(f"✓ Loss without mask: {loss_no_mask.item():.4f}")

    assert loss.item() > 0, "Loss should be positive!"
    assert not torch.isnan(loss), "Loss is NaN!"

    print("\n✅ TemporalFocalLoss test PASSED!")
    return True


def test_temporal_smoothness_loss():
    """Test TemporalSmoothnessLoss."""
    print("\n" + "=" * 60)
    print("Test 5: TemporalSmoothnessLoss")
    print("=" * 60)

    # Test with temporal sequence
    batch_size = 8
    time_steps = 10
    predictions_seq = torch.rand(batch_size, time_steps, 16)

    criterion = TemporalSmoothnessLoss(smoothness_weight=0.1)

    loss = criterion(predictions_seq)
    print(f"✓ Predictions shape: {predictions_seq.shape}")
    print(f"✓ Smoothness loss: {loss.item():.6f}")

    # Test with previous predictions
    predictions_t = torch.rand(batch_size, 16)
    predictions_t_prev = torch.rand(batch_size, 16)

    loss_prev = criterion(predictions_t, predictions_t_prev)
    print(f"✓ Loss with prev: {loss_prev.item():.6f}")

    assert loss.item() >= 0, "Smoothness loss should be non-negative!"
    assert not torch.isnan(loss), "Loss is NaN!"

    print("\n✅ TemporalSmoothnessLoss test PASSED!")
    return True


def test_label_generation():
    """Test label generation with anti-leakage."""
    print("\n" + "=" * 60)
    print("Test 6: Label Generation with Anti-Leakage")
    print("=" * 60)

    # Failure episode metadata
    episode_metadata = {
        'success': False,
        'failure_type': 1,  # Drop
        'episode_length': 50
    }

    horizons = [6, 9, 12, 15]
    max_length = 100

    # Test 1: Standard labels
    labels_standard = compute_failure_labels(
        episode_metadata, 50, max_length, horizons, 4
    )

    print(f"✓ Standard labels shape: {labels_standard.shape}")
    print(f"✓ Expected: ({max_length}, {len(horizons)}, 4)")
    print(f"✓ Positive labels: {(labels_standard > 0).sum()}")

    # Test 2: Randomized labels (anti-leakage)
    labels_random = compute_failure_labels_with_randomization(
        episode_metadata, 50, max_length, horizons, 4, time_noise_steps=5
    )

    print(f"✓ Randomized labels shape: {labels_random.shape}")
    print(f"✓ Positive labels: {(labels_random > 0).sum()}")

    # Test 3: Soft labels
    labels_soft = compute_soft_temporal_labels(
        episode_metadata, 50, max_length, horizons, 4, decay_type='exponential'
    )

    print(f"✓ Soft labels shape: {labels_soft.shape}")
    print(f"✓ Label range: [{labels_soft.min():.3f}, {labels_soft.max():.3f}]")
    print(f"✓ Mean positive label: {labels_soft[labels_soft > 0].mean():.3f}")

    # Verify shapes
    assert labels_standard.shape == (max_length, len(horizons), 4), "Incorrect shape!"
    assert labels_random.shape == (max_length, len(horizons), 4), "Incorrect shape!"
    assert labels_soft.shape == (max_length, len(horizons), 4), "Incorrect shape!"

    # Verify failure type is set correctly
    assert labels_standard[:, :, 1].sum() > 0, "Failure type not set!"

    print("\n✅ Label generation test PASSED!")
    return True


def test_temporal_stability():
    """Test temporal stability metrics."""
    print("\n" + "=" * 60)
    print("Test 7: Temporal Stability Metrics")
    print("=" * 60)

    # Create dummy predictions over time
    time_steps = 100
    predictions = np.random.rand(time_steps, 16)

    # Compute stability
    stability = compute_temporal_stability(predictions)

    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Stability metrics:")
    for key, value in stability.items():
        print(f"    {key}: {value:.4f}")

    assert 'variance' in stability, "Missing variance metric!"
    assert 'autocorr_lag1' in stability, "Missing autocorrelation!"

    print("\n✅ Temporal stability test PASSED!")
    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# SALUS Temporal Forecasting Components Test")
    print("#" * 60)

    tests = [
        ("HybridTemporalPredictor", test_hybrid_predictor),
        ("LatentHealthStateEncoder", test_latent_encoder),
        ("LatentTemporalPredictor", test_latent_temporal_predictor),
        ("TemporalFocalLoss", test_temporal_focal_loss),
        ("TemporalSmoothnessLoss", test_temporal_smoothness_loss),
        ("Label Generation", test_label_generation),
        ("Temporal Stability", test_temporal_stability),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED!")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print("\n" + "-" * 60)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready.")
        print("\nNext steps:")
        print("1. Collect training data with temporal forecasting:")
        print("   python scripts/collect_data_parallel_a100_fixed.py \\")
        print("       --num_episodes 500 \\")
        print("       --num_envs 4 \\")
        print("       --save_dir ~/salus_data_temporal")
        print("\n2. Train the temporal predictor:")
        print("   python scripts/train_temporal_predictor.py \\")
        print("       --data_dir ~/salus_data_temporal \\")
        print("       --epochs 100 \\")
        print("       --use_hard_negatives")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix errors before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
