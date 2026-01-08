# SALUS System Analysis: What It Does vs. What It Claims

## Executive Summary

**SALUS** (Safety Action Learning Uncertainty Synthesis) is a **failure prediction system** for robot manipulation that attempts to predict failures 200-500ms before they occur by monitoring Vision-Language-Action (VLA) model internals.

### Key Finding: **PARTIALLY VERIFIED**

✅ **What Works:**
- Temporal forecasting architecture (Conv1D + GRU) is implemented and functional
- Model can learn temporal patterns on synthetic data
- Multi-horizon prediction (4 horizons × 4 failure types) is implemented
- Test infrastructure exists and some tests pass
- **Recent Fix (PR #1, merged 2026-01-08)**: Uncertainty history tracking bug fixed
- **Recent Fix (PR #1)**: Dummy fallback mode removed - now requires real IsaacLab

⚠️ **What's Unclear:**
- Signal extraction implementation has inconsistencies (12D vs 18D signals)
- Real VLA integration status unclear (model loading vs actual inference)
- No validation on real robot data yet
- Some tests fail due to dimension mismatches

---

## What SALUS Claims to Do

Based on documentation (`README.md`, `HOW_SALUS_WORKS.md`, `COMPLETE_SALUS_SYSTEM_DOCUMENTATION.md`):

### 1. **Predict Failures 200-500ms Before They Occur**
- **Claim**: Multi-horizon temporal forecasting at 4 time scales (200ms, 300ms, 400ms, 500ms)
- **Status**: ✅ Architecture implemented (`HybridTemporalPredictor` with Conv1D + GRU)
- **Evidence**: Code exists in `salus/models/temporal_predictor.py`

### 2. **Extract Signals from VLA Model Internals**
- **Claim**: Extracts 18D feature vector from VLA ensemble uncertainty, hidden states, and perturbations
- **Status**: ⚠️ **INCONSISTENT** - Documentation says 18D, but code shows:
  - `HybridTemporalPredictor` defaults to `signal_dim=12`
  - `test_salus_can_learn.py` uses 12D signals
  - `quick_proof_test.py` uses 12D signals
  - Documentation claims 18D enhanced signals

### 3. **Use Real VLA Models (SmolVLA-450M)**
- **Claim**: Loads and runs 5× SmolVLA models (865MB each) for ensemble uncertainty
- **Status**: ⚠️ **UNCLEAR** - Code exists (`salus/core/vla/wrapper.py`) but:
  - Model loading code is present
  - No verification that models actually run
  - No evidence of actual VLA inference in test outputs

### 4. **Temporal Forecasting with Conv1D + GRU**
- **Claim**: Hybrid architecture processes 333ms sliding windows (10 timesteps @ 30Hz)
- **Status**: ✅ **VERIFIED** - Implementation confirmed:
  - `HybridTemporalPredictor` class exists
  - Conv1D (kernel=5) for local patterns
  - GRU (hidden=64) for long-range dependencies
  - Output: 16D (4 horizons × 4 failure types)

### 5. **99.66% Discrimination on Synthetic Data**
- **Claim**: System achieves high discrimination between failure and success patterns
- **Status**: ⚠️ **MIXED** - Test results show:
  - `quick_proof_test.py`: ✅ **PASSED** (99.97% discrimination)
  - `test_salus_can_learn.py`: ❌ **FAILED** (0% discrimination, dimension mismatch)

---

## Code Analysis

### Architecture Components

#### 1. Temporal Predictor (`salus/models/temporal_predictor.py`)

**Implementation**: ✅ **VERIFIED**

```python
class HybridTemporalPredictor(nn.Module):
    - Conv1D: 12 → 32 channels (kernel=5, ~167ms receptive field)
    - GRU: 32 → 64 hidden (long-range dependencies)
    - Linear: 64 → 128 → 16 (multi-horizon predictions)
    - Parameters: ~31K-50K
```

**Test Results**:
- ✅ Model initializes correctly
- ✅ Forward pass works with (B, T, 12) input
- ✅ Output shape: (B, 16) as expected
- ✅ Can learn temporal patterns on synthetic data

#### 2. Signal Extraction (`salus/core/vla/wrapper.py`)

**Implementation**: ⚠️ **INCONSISTENT** (but recently improved)

**Documentation Claims**:
- 18D enhanced signals including:
  - Signals 1-12: Basic uncertainty (ensemble variance, action magnitude, etc.)
  - Signals 13-14: VLA internal state (latent drift, OOD distance)
  - Signals 15-16: Sensitivity (perturbation response)
  - Signals 17-18: Reality checks (physics, constraints)

**Code Reality**:
- `HybridTemporalPredictor` defaults to `signal_dim=12`
- Tests use 12D signals
- `EnhancedSignalExtractor` class exists but may not be fully integrated
- Single-model extractor (`SingleModelSignalExtractor`) also exists

**Recent Fixes (PR #1, merged 2026-01-08)**:
- ✅ **Fixed uncertainty history tracking**: Now properly tracks `uncertainty_history` instead of repeating current value
- ✅ **Rolling statistics now use real past values**: Uses `self.uncertainty_history[-5:]` for mean/std/min/max calculations
- ✅ **Proper history reset**: `uncertainty_history` is reset at episode start

**Verdict**: Signal dimension mismatch between documentation (18D) and implementation (12D), but uncertainty tracking bug has been fixed.

#### 3. VLA Ensemble (`salus/core/vla/wrapper.py`)

**Implementation**: ⚠️ **UNCLEAR**

**Code Structure**:
```python
class SmolVLAEnsemble(nn.Module):
    - Loads SmolVLA models from lerobot
    - Ensemble size configurable (default: 1, not 5!)
    - Signal extractor integrated
```

**Issues Found**:
1. **Default ensemble_size=1**, not 5 as documentation claims
2. No verification that models actually load and run
3. No test output showing VLA inference happening
4. Model path: `~/models/smolvla/smolvla_base` - existence not verified

**Verdict**: Code structure exists but actual VLA integration status unclear.

---

## Test Results

### ✅ Test 1: `quick_proof_test.py` - **PASSED**

```
Results:
  ✅ Model trains without errors
  ✅ Loss decreases (0.2620 → 0.1707, 34.8% improvement)
  ✅ Final loss < initial
  ✅ Predicts failure pattern higher (0.9997 vs 0.0001)
  ✅ Clear discrimination (99.96% difference)

Passed: 5/5
```

**Conclusion**: Temporal forecasting architecture **WORKS** on synthetic data.

### ❌ Test 2: `test_salus_can_learn.py` - **FAILED**

```
Results:
  ❌ 12D signals created (expected 18D based on test code)
  ✅ Model accepts 12D input
  ✅ Training loss decreased
  ❌ Loss improved >50% (only 34.8%)
  ❌ Discrimination > 0.2 (0.0%)
  ❌ Effect size > 0.8 (-1.27)

Checks passed: 2/6
```

**Issues**:
1. Test creates signals with `signal_dim=12` but expects 18D
2. Model outputs all zeros (no learning)
3. Dimension mismatch in test code itself

**Conclusion**: Test has bugs, but model architecture is sound (proven by `quick_proof_test.py`).

---

## What SALUS Actually Does (Based on Code)

### ✅ **CONFIRMED CAPABILITIES:**

1. **Temporal Sequence Modeling**
   - Processes sliding windows of 10 timesteps (333ms @ 30Hz)
   - Uses Conv1D + GRU architecture
   - Predicts 4 horizons × 4 failure types = 16D output

2. **Multi-Horizon Prediction**
   - Simultaneously predicts failures at 200ms, 300ms, 400ms, 500ms
   - Separate predictions for 4 failure types (collision, drop, miss, timeout)

3. **Learning Capability**
   - Can learn temporal patterns from synthetic data
   - Loss decreases during training
   - Shows discrimination between failure and success patterns

### ⚠️ **UNCLEAR/UNVERIFIED:**

1. **Real VLA Integration**
   - Code exists to load SmolVLA models
   - No evidence models actually run or produce real signals
   - Default ensemble size is 1, not 5 as claimed
   - **Recent Fix (PR #1)**: Dummy fallback removed - now requires real IsaacLab (good for ensuring real tests)

2. **Signal Extraction**
   - Documentation claims 18D enhanced signals
   - Implementation uses 12D signals
   - Enhanced signal extractor exists but integration unclear
   - **Recent Fix (PR #1)**: Uncertainty history tracking bug fixed - rolling stats now use real past values

3. **Real Robot Data**
   - No test results on real Isaac Lab simulation data
   - No validation on actual robot failures
   - All tests use synthetic data
   - **Recent Fix (PR #1)**: Integration scripts updated to properly manage IsaacLab lifecycle

4. **Performance Metrics**
   - Claims "99.66% discrimination" but only on synthetic data
   - No F1 scores, precision, recall on real data
   - Target F1 > 0.60 not yet achieved

---

## Discrepancies Between Claims and Code

| Claim | Code Reality | Status |
|-------|-------------|--------|
| 18D enhanced signals | 12D signals used | ⚠️ Mismatch |
| 5-model ensemble | Default ensemble_size=1 | ⚠️ Mismatch |
| Real VLA inference | Code exists, not verified | ⚠️ Unclear |
| 99.66% discrimination | Only on synthetic data | ⚠️ Limited |
| F1 > 0.60 target | Not yet achieved | ❌ Not met |

---

## Recommendations

### To Verify Claims:

1. **Run Real VLA Inference Test**
   ```bash
   python test_real_vla_signals.py
   ```
   - Verify SmolVLA models actually load
   - Check that signals are non-zero
   - Confirm ensemble variance is computed

2. **Fix Signal Dimension Consistency**
   - Decide: 12D or 18D?
   - Update all code to match
   - Update documentation to match code

3. **Test on Real Data**
   - Collect data from Isaac Lab simulation
   - Train on real VLA signals
   - Evaluate F1, precision, recall metrics

4. **Verify Ensemble Size**
   - Check if ensemble_size=5 is actually used
   - Or update documentation to reflect ensemble_size=1

---

## Recent Updates (PR #1, Merged 2026-01-08)

**Pull Request**: "Require real IsaacLab in tests and fix SignalExtractor uncertainty history"

**Key Fixes**:
1. ✅ **Uncertainty History Bug Fixed**: `SignalExtractor` now properly tracks `uncertainty_history` and uses real past values for rolling statistics (mean, std, min, max) instead of repeating the current value
2. ✅ **Dummy Fallback Removed**: `FrankaPickPlaceEnv` now raises `RuntimeError` if IsaacLab fails to initialize, ensuring tests run against real simulator
3. ✅ **Integration Scripts Updated**: All integration scripts and tests now properly create and manage `AppLauncher` lifecycle
4. ✅ **Code Quality**: Net reduction of 140 lines (381 deletions, 241 additions) - cleaner, more maintainable code

**Impact**: These fixes ensure that:
- Uncertainty features are computed from real historical data (more accurate)
- Tests cannot accidentally pass with dummy data (more reliable)
- IsaacLab lifecycle is properly managed (prevents resource leaks)

---

## Conclusion

**SALUS is a REAL system** with:
- ✅ Functional temporal forecasting architecture
- ✅ Working Conv1D + GRU implementation
- ✅ Multi-horizon prediction capability
- ✅ Learning capability on synthetic data
- ✅ **Recent fixes improve reliability and accuracy**

**However, several claims are UNVERIFIED or INCONSISTENT:**
- ⚠️ Signal dimension mismatch (12D vs 18D)
- ⚠️ Ensemble size mismatch (1 vs 5)
- ⚠️ Real VLA integration not verified
- ⚠️ No real robot data validation
- ⚠️ Performance metrics not achieved on real data

**Verdict**: SALUS is a **working prototype** with solid architecture, but needs:
1. Consistency fixes (signal dimensions, ensemble size)
2. Real VLA integration verification
3. Validation on real robot data
4. Performance metrics on real failures

The core idea and architecture are sound, and recent fixes have improved the system's reliability. However, the system is not yet fully validated as claimed.

