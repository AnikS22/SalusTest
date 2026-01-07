#!/usr/bin/env python3
"""
HPC Phase 1 Validation Script

Runs quick validation tests on HPC to ensure temporal system works before
committing to long data collection.

Usage:
    python scripts/test_hpc_phase1.py

Expected runtime: 5-10 minutes
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()

    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print(f"\n✅ {description} - PASSED")
        return True
    else:
        print(f"\n❌ {description} - FAILED")
        return False

def main():
    print("\n" + "="*60)
    print("SALUS HPC Phase 1 Validation")
    print("="*60)

    # Check we're in the right directory
    if not Path("salus/models/temporal_predictor.py").exists():
        print("❌ Error: Run this script from the SalusTest root directory")
        return 1

    # Get project root
    project_root = Path.cwd()

    tests = []

    # Test 1: Check Python and CUDA
    print("\n📋 Test 1: Checking Python and CUDA availability...")
    tests.append(run_command(
        "python -c 'import torch; v=torch.__version__; c=torch.cuda.is_available(); g=torch.cuda.device_count(); print(f\"Python OK\\nPyTorch: {v}\\nCUDA: {c}\\nGPUs: {g}\")'",
        "Python and CUDA Check"
    ))

    # Test 2: Check imports
    print("\n📋 Test 2: Checking all imports...")
    tests.append(run_command(
        "python -c 'from salus.models.temporal_predictor import HybridTemporalPredictor; from salus.models.latent_encoder import LatentHealthStateEncoder; from salus.data.temporal_dataset import TemporalSALUSDataset; print(\"All imports OK\")'",
        "Import Check"
    ))

    # Test 3: Component tests
    print("\n📋 Test 3: Running component tests...")
    tests.append(run_command(
        "python scripts/test_temporal_components.py",
        "Component Tests (7 tests)"
    ))

    # Test 4: Quick proof test
    print("\n📋 Test 4: Running quick proof test...")
    tests.append(run_command(
        "python scripts/quick_proof_test.py",
        "Quick Proof Test (validates temporal learning)"
    ))

    # Summary
    print("\n" + "="*60)
    print("PHASE 1 VALIDATION SUMMARY")
    print("="*60)

    passed = sum(tests)
    total = len(tests)

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe temporal forecasting system is working correctly on HPC.")
        print("\nYou can now proceed to:")
        print("  1. Phase 2: Small-scale data collection (50 episodes)")
        print("  2. Phase 3: Full training (500 episodes)")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease review the failed tests above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
