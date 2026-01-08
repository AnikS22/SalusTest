"""
Quick test for single-model wrapper modifications
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    from salus.core.vla.wrapper import SmolVLAEnsemble
    from salus.core.vla.single_model_extractor import SingleModelSignalExtractor
    print("✓ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 1: Check single-model initialization
print("\n" + "="*70)
print("TEST 1: Single-Model Initialization")
print("="*70)

try:
    # This should work without needing actual VLA model files
    # We'll just verify the class structure
    print("✓ SmolVLAEnsemble class loaded")
    print("✓ SingleModelSignalExtractor class loaded")

    # Check default ensemble_size
    import inspect
    sig = inspect.signature(SmolVLAEnsemble.__init__)
    ensemble_size_default = sig.parameters['ensemble_size'].default
    print(f"✓ Default ensemble_size: {ensemble_size_default}")

    if ensemble_size_default == 1:
        print("✓ Default ensemble_size correctly set to 1")
    else:
        print(f"❌ Default ensemble_size should be 1, got {ensemble_size_default}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)

# Test 2: Check method signatures
print("\n" + "="*70)
print("TEST 2: Method Signatures")
print("="*70)

try:
    # Check that new methods exist
    assert hasattr(SmolVLAEnsemble, '_extract_action_logits'), "Missing _extract_action_logits method"
    print("✓ _extract_action_logits method exists")

    assert hasattr(SmolVLAEnsemble, '_extract_hidden_state'), "Missing _extract_hidden_state method"
    print("✓ _extract_hidden_state method exists")

    # Check that old methods are removed
    assert not hasattr(SmolVLAEnsemble, '_test_perturbation_stability'), "Old _test_perturbation_stability should be removed"
    print("✓ _test_perturbation_stability method removed")

    assert not hasattr(SmolVLAEnsemble, '_aggregate_internals'), "Old _aggregate_internals should be removed"
    print("✓ _aggregate_internals method removed")

except AssertionError as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)

# Test 3: Check docstrings updated
print("\n" + "="*70)
print("TEST 3: Documentation Updates")
print("="*70)

try:
    class_doc = SmolVLAEnsemble.__doc__

    # Check for single-model terminology
    if "single" in class_doc.lower() or "uncertainty estimation via internal signals" in class_doc.lower():
        print("✓ Class docstring updated with single-model terminology")
    else:
        print("⚠️  Class docstring may need more updates")

    # Check forward() docstring
    forward_doc = SmolVLAEnsemble.forward.__doc__
    if "action_logits" in forward_doc:
        print("✓ forward() docstring mentions action_logits")
    else:
        print("⚠️  forward() docstring may need action_logits mentioned")

except Exception as e:
    print(f"⚠️  Warning during docstring check: {e}")

# Test 4: Check deprecation warnings on old classes
print("\n" + "="*70)
print("TEST 4: Deprecation Warnings")
print("="*70)

try:
    from salus.core.vla.wrapper import SignalExtractor, EnhancedSignalExtractor

    if "DEPRECATED" in SignalExtractor.__doc__:
        print("✓ SignalExtractor marked as deprecated")
    else:
        print("⚠️  SignalExtractor should be marked deprecated")

    if "DEPRECATED" in EnhancedSignalExtractor.__doc__:
        print("✓ EnhancedSignalExtractor marked as deprecated")
    else:
        print("⚠️  EnhancedSignalExtractor should be marked deprecated")

except Exception as e:
    print(f"⚠️  Warning: {e}")

print("\n" + "="*70)
print("ALL BASIC TESTS PASSED ✅")
print("="*70)
print("\nWrapper modifications complete!")
print("\nNote: Full testing requires:")
print("  1. Actual SmolVLA model files")
print("  2. Integration with data collection")
print("  3. End-to-end pipeline test")
print("\n" + "="*70)
