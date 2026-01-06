# Code Analysis: fix_horizon_labels.py and SALUS Claims

## What `fix_horizon_labels.py` Does

### Purpose
This is a **post-processing data cleaning script** that fixes training labels AFTER data collection.

### What It Actually Does:

1. **Reads collected episode data** from Zarr format
2. **Computes multi-horizon labels** retroactively:
   - For each failed episode, it assumes failure happened at the LAST timestep
   - It then labels previous timesteps as "will fail in X steps"
   - Creates labels for 4 horizons: 5, 10, 15, 20 timesteps ahead
3. **Writes corrected labels** back to the Zarr dataset

### Key Code Logic:

```python
def compute_horizon_labels(episode_length, success, failure_type, horizons=[5, 10, 15, 20]):
    # If episode failed:
    if not success:
        failure_time = T - 1  # Assume failure at last timestep
        
        # For each timestep before failure:
        for t in range(failure_time):
            steps_until_failure = failure_time - t
            
            # Label if within prediction horizon
            for h_idx, horizon in enumerate(horizons):
                if steps_until_failure <= horizon:
                    horizon_labels[t, h_idx, failure_type_int] = 1.0
```

**Important**: This is **retroactive labeling** - it's creating training labels based on known outcomes, not actually predicting anything.

---

## What SALUS Claims

### From README.md:

> **SALUS** predicts failures **200-500ms before they occur** using:
> - **Temporal forecasting**: Multi-horizon prediction (200ms, 300ms, 400ms, 500ms ahead)
> - **Internal signal analysis**: Extracts 12D feature vectors from VLA internals
> - **Safety manifolds**: Learns the geometry of safe action space
> - **Model predictive control**: Synthesizes safe actions in <30ms

### Claims:
1. ✅ Predicts failures **BEFORE** they happen (200-500ms ahead)
2. ✅ Multi-horizon prediction (4 different time horizons)
3. ✅ Real-time intervention (<30ms)
4. ✅ Continuous learning

---

## What SALUS Actually Implements

### Current MVP Implementation:

#### 1. Predictor (`salus/core/predictor_mvp.py`)

**What it does:**
```python
class SALUSPredictorMVP:
    Input: 6D signals (not 12D)
    Output: 4D probabilities (failure types only)
    # NO multi-horizon - just predicts IF failure will happen
```

**Reality Check:**
- ❌ **NO multi-horizon prediction** - only predicts failure types (collision, drop, miss, timeout)
- ❌ **NO temporal forecasting** - doesn't predict WHEN, only WHAT type
- ✅ Does predict failure types from signals
- ✅ Uses uncertainty signals from VLA ensemble

**Gap**: Claims "200-500ms ahead" but MVP doesn't predict time horizons!

#### 2. Full Predictor (`salus/core/predictor.py`)

**What it does:**
```python
class SALUSPredictor:
    Input: 12D signals
    Output: 16D logits (4 horizons × 4 failure types)
    # HAS multi-horizon prediction heads
```

**Reality Check:**
- ✅ **HAS multi-horizon prediction** (4 horizons)
- ✅ **HAS temporal forecasting** (predicts WHEN failures will occur)
- ⚠️ **Status**: Code exists but may not be fully tested/integrated

**Gap**: Full version exists but MVP (currently used) doesn't use it!

#### 3. Data Labels (`fix_horizon_labels.py`)

**What it does:**
- Creates multi-horizon labels (5, 10, 15, 20 timesteps)
- But MVP predictor doesn't use these labels!

**Gap**: Labels are multi-horizon, but MVP predictor ignores them!

---

## Does SALUS Do What It Claims?

### Short Answer: **PARTIALLY**

| Claim | MVP Status | Full Version Status |
|-------|-----------|---------------------|
| **Predicts failures before they happen** | ✅ YES (but not WHEN) | ✅ YES (with time horizons) |
| **Multi-horizon prediction (200-500ms)** | ❌ NO | ✅ YES (code exists) |
| **12D signal extraction** | ⚠️ PARTIAL (6D only) | ✅ YES |
| **Safety manifold** | ❌ NO | ⚠️ Placeholder |
| **MPC synthesis** | ❌ NO | ⚠️ Placeholder |
| **Real-time intervention** | ❌ NO | ⚠️ Not implemented |
| **Continuous learning** | ❌ NO | ⚠️ Not implemented |

---

## The Disconnect

### What's Happening:

1. **Data Collection**: Creates multi-horizon labels (via `fix_horizon_labels.py`)
2. **MVP Predictor**: Ignores horizons, only predicts failure types
3. **Full Predictor**: Has multi-horizon code but may not be used
4. **Documentation**: Claims full multi-horizon system

### The Problem:

```
fix_horizon_labels.py creates:
  horizon_labels: (episodes, timesteps, 4 horizons, 4 types)

MVP predictor expects:
  labels: (episodes, timesteps, 4 types)  ← NO horizons!

Result: Horizon labels are created but not used!
```

---

## What `fix_horizon_labels.py` Actually Does

### It's a Data Preparation Script:

1. **Fixes missing labels**: Original data collection may have created dummy (all-zero) labels
2. **Creates proper training labels**: Computes "will fail in X steps" labels from episode outcomes
3. **Prepares for future use**: Creates multi-horizon labels for when full predictor is used

### It Does NOT:
- ❌ Predict anything (it's post-processing)
- ❌ Run at runtime (it's offline data processing)
- ❌ Actually prevent failures (it just fixes labels)

---

## Honest Assessment

### What SALUS MVP Actually Does:

✅ **Works:**
- Extracts uncertainty signals from VLA ensemble
- Predicts failure types (collision, drop, miss, timeout)
- Can identify when VLA is uncertain/confused
- Infrastructure for data collection and training

❌ **Doesn't Work (Yet):**
- Multi-horizon prediction (claims 200-500ms, but MVP doesn't predict WHEN)
- Real-time intervention (no safety manifold or MPC)
- Continuous learning (not implemented)

### What the Code Shows:

1. **MVP is simplified**: Only predicts failure types, not horizons
2. **Full version exists**: Multi-horizon code is written but may not be integrated
3. **Labels are prepared**: `fix_horizon_labels.py` creates labels for future use
4. **Gap between claims and implementation**: Documentation claims more than MVP delivers

---

## Recommendations

### To Make SALUS Match Its Claims:

1. **Use Full Predictor**: Switch from `predictor_mvp.py` to `predictor.py`
2. **Use Horizon Labels**: Train on the multi-horizon labels created by `fix_horizon_labels.py`
3. **Implement Intervention**: Add safety manifold and MPC synthesis
4. **Update Documentation**: Clarify MVP vs Full version capabilities

### Current State:

- **MVP**: Proof-of-concept that can predict failure types
- **Full Version**: Code exists but needs integration and testing
- **Claims**: Describe full system, not current MVP

---

## Bottom Line

**Does SALUS do what it claims?**

- **MVP (current)**: ❌ NO - Only predicts failure types, not WHEN (no multi-horizon)
- **Full Version (code exists)**: ⚠️ PARTIALLY - Has multi-horizon code but needs integration
- **Claims**: Describe full system, not current implementation

**What `fix_horizon_labels.py` does:**
- ✅ Fixes training labels (post-processing)
- ✅ Creates multi-horizon labels for future use
- ❌ Does NOT predict anything (it's data preparation)
- ❌ Does NOT prevent failures (it's offline)

**The script is preparing data for a multi-horizon predictor that the MVP doesn't currently use.**

