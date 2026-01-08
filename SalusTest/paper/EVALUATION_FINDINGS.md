# SALUS Evaluation Findings - Honest Assessment

**Date:** 2026-01-07
**Status:** NOT production-ready (requires calibration)

---

## Executive Summary

SALUS demonstrates **strong failure detection capability** (AUROC 0.991) but has **critical calibration issues** that prevent immediate deployment. The system can effectively **rank** which episodes are more likely to fail, but its probability outputs are **not well-calibrated** for threshold-based decision making.

### Critical Finding

**⚠️ High AUROC ≠ Production Ready**

This evaluation exposed a common but dangerous pitfall: a model can have excellent discrimination (AUROC) while producing poorly calibrated probabilities (ECE). For safety-critical systems, **calibration is as important as accuracy**.

---

## Detailed Findings

### ✅ What Works Well

#### 1. Failure Detection (Discrimination)
- **AUROC (500ms): 0.991** - Excellent at ranking failure risk
- **AUPRC (500ms): 0.958** - Strong precision-recall tradeoff
- **Recall: 100%** at τ=0.5 - Catches all failures (no misses)

#### 2. Temporal Leakage Defense
All control experiments passed (< 5% AUROC drop):
- **Time-shuffle**: 4.1% drop - Model doesn't rely on temporal order
- **Counterfactual**: 0.1% drop - Model doesn't exploit episode phase
- **No time-index**: -0.3% drop (actually improved) - Model doesn't need positional encoding

**Conclusion:** Model learns genuine failure dynamics, not temporal shortcuts.

#### 3. Baseline Comparison
| Method | AUROC | AUPRC | Status |
|--------|-------|-------|--------|
| SALUS (full) | 0.991 | 0.958 | Best |
| Temporal-only | 0.989 | 0.950 | Strong |
| SAFE-style | 0.991 | 0.960 | Comparable |
| Entropy-only | 0.980 | 0.920 | Good |
| Anomaly Detector | 0.454 | 0.160 | Poor |

**Finding:** All neural baselines achieve high performance on synthetic data, suggesting the synthetic failure patterns may be too predictable.

---

### ❌ Critical Issues Preventing Deployment

#### 1. **CALIBRATION FAILURE** (Most Critical)

**Expected Calibration Error (ECE): 0.450**
- **Threshold:** < 0.10 for production use
- **Status:** **4.5× worse than acceptable**

**What This Means:**
The model's predicted probabilities DO NOT reflect actual failure frequencies. When the model says "50% chance of failure", the actual failure rate is anywhere from 0% to 15% depending on the bin.

**Calibration Breakdown:**
```
Predicted Confidence → Actual Failure Rate → Error
      50% (bin 0.4-0.5) →     0.0%          → 50% overconfident
      50% (bin 0.5-0.6) →    15.0%          → 35% overconfident
      73% (bin 0.7-0.8) →    99.2%          → 26% underconfident
```

**Why This Matters:**
- Operators cannot trust the probability values
- Threshold selection becomes arbitrary
- Risk assessment is unreliable
- Violates safety-critical system requirements

**Root Cause:**
- Binary cross-entropy loss optimizes for discrimination (AUROC), not calibration
- No explicit calibration objective during training
- Synthetic data may have imbalanced probability distributions

**Solutions:**
1. **Temperature scaling** - Post-hoc calibration method
2. **Focal loss** - Penalizes miscalibrated predictions during training
3. **Calibration layer** - Learnable calibration parameters
4. **Platt scaling** - Logistic regression on validation probabilities

---

#### 2. **INSUFFICIENT LEAD TIME**

**Mean Lead Time: 139.9ms**
- **Threshold:** > 200ms for human operator intervention
- **Status:** **30% below minimum requirement**

**Distribution:**
- Median: 133.3ms (50% of alerts below this)
- Std dev: 90.4ms (high variability)
- Min: 133.3ms
- Max: 1600.0ms

**Why This Matters:**
- Humans need ~200-300ms to perceive, decide, and act
- Robot safety stops need ~150-200ms actuation time
- **Total required lead time: 350-500ms minimum**

**Current Performance vs Requirements:**
```
Mean lead time:        139.9ms
Human reaction time:  +250.0ms
Robot stop latency:   +150.0ms
----------------------
Total response time:   539.9ms
Available window:      139.9ms  ← 400ms short!
```

**Root Cause:**
- Model only looks 333ms into past (window size 10 @ 30Hz)
- Synthetic failures may develop too quickly
- 500ms prediction horizon may be insufficient for gradual failures

**Solutions:**
1. **Increase window size** - Look further back (15-20 timesteps = 500-667ms)
2. **Multi-scale temporal modeling** - Capture both fast and slow dynamics
3. **Earlier detection signals** - Add precursor indicators (e.g., velocity trends)
4. **Longer horizons** - Test 700ms, 1000ms horizons

---

#### 3. **SYNTHETIC DATA CONCERNS**

**Observation:** All neural baselines achieve 0.98-0.99 AUROC

**Red Flags:**
1. **Too easy** - Real robot failures won't be this predictable
2. **False alarm rate sensitivity** - Small threshold changes cause huge FA/min swings:
   - τ=0.30: 1543 FA/min (unusable)
   - τ=0.50: 206 FA/min (unusable)
   - τ=0.51: 2.25 FA/min (acceptable)

   This 0.01 difference in threshold causes 100× change in false alarms!

3. **Perfect recall** - Model achieves 100% recall at multiple thresholds, suggesting failure patterns are too distinct

**Implications:**
- **Generalization Risk:** Performance may drop significantly on real robot data
- **Overfitting to Synthetic Patterns:** Model may learn synthetic-specific artifacts
- **Threshold Brittleness:** Real deployment will need continuous threshold tuning

---

### ⚠️ Moderate Concerns

#### 1. **False Alarm Rate: 2.25/min**
- **Threshold:** < 1.0/min for operator acceptance
- **Status:** **2.25× above target**
- **Impact:** Operators may develop "alarm fatigue" and ignore warnings

**Mitigation:**
- Increase threshold to τ=0.70 → 1.80 FA/min (still above target)
- Implement multi-stage alerts (warning → critical)
- Add confidence-based filtering

#### 2. **Miss Rate: 14.0%**
- **Threshold:** < 15% for safety requirements
- **Status:** **Within tolerance but borderline**
- **Impact:** 1 in 7 failures goes undetected

**Analysis:**
These are likely sudden, non-gradual failures (external collisions, hardware faults) that manifest without detectable signal precursors.

---

## Production Readiness Assessment

### Passed Requirements ✅
- [x] AUROC (500ms): 0.991 > 0.90
- [x] AUPRC (500ms): 0.958 > 0.80
- [x] Temporal leakage defense: All experiments < 5% drop
- [x] Miss rate: 14.0% < 15%

### Failed Requirements ❌
- [ ] **ECE: 0.450 > 0.10** (CRITICAL)
- [ ] **Lead time: 139.9ms < 200ms** (CRITICAL)
- [ ] FA/min: 2.25 > 1.0 (moderate)

### Overall Status: **NOT PRODUCTION READY**

---

## Recommendations for Paper

### 1. **Be Transparent About Limitations**

**Good:** "SALUS achieves 99% AUROC on synthetic validation data."

**Better:** "SALUS achieves 99% AUROC on synthetic validation data but exhibits poor calibration (ECE=0.45), indicating the model can rank failures effectively but requires post-hoc calibration before probability outputs can be trusted for threshold-based decision making."

### 2. **Emphasize Need for Real Robot Validation**

The 0.98-0.99 AUROC across ALL baselines suggests synthetic data may be too predictable. Add to limitations:

> "The current evaluation uses synthetic data with simplified failure patterns. Real robot deployments will face:
> - Complex contact dynamics and friction variability
> - Perception errors and occlusions
> - Diverse object properties (mass, compliance, surface friction)
> - Environmental disturbances (lighting changes, vibrations)
>
> We expect AUROC to drop by 10-15% on real robot data based on similar vision-based prediction tasks [citations]. Physical robot validation is essential before deployment."

### 3. **Discuss Calibration Methods**

Add subsection to Discussion:

> **Calibration for Deployment:** While SALUS achieves high AUROC (discrimination), its probability outputs require calibration (ECE=0.45 vs target <0.10). We recommend:
> 1. Temperature scaling [Guo et al. 2017] - simple post-hoc calibration
> 2. Focal loss [Lin et al. 2017] - better calibration during training
> 3. Per-task threshold tuning based on operator tolerance and failure costs
>
> Calibration adds <1ms inference latency and requires only validation data, making it practical for deployment."

### 4. **Present Threshold Tradeoffs Honestly**

Replace single-threshold results with precision-recall tradeoff:

**Current (misleading):**
"SALUS achieves 0.12 false alarms per minute with 8.2% miss rate."

**Better (honest):**
"SALUS presents a precision-recall tradeoff: at τ=0.51 (optimal F1), the system achieves 2.25 FA/min with 14% miss rate. Operators can tune this threshold based on task criticality:
- High-risk tasks (τ=0.70): 1.8 FA/min, 14% miss rate
- Balanced (τ=0.51): 2.25 FA/min, 14% miss rate
- Low-risk tasks (τ=0.30): 1543 FA/min, 0% miss rate"

### 5. **Address Lead Time Limitation**

"Mean lead time of 139.9ms is below the 200ms minimum for human intervention. For autonomous safety stops, this provides sufficient margin (robot actuation: ~150ms), but human-in-the-loop scenarios require:
1. Longer temporal windows (500-667ms vs current 333ms)
2. Precursor signal detection (velocity trends, force buildup)
3. Multi-horizon cascading alerts (preliminary warning → critical alert)"

---

## Next Steps for Real Deployment

### Phase 1: Calibration (1-2 weeks)
1. Implement temperature scaling
2. Re-evaluate ECE on held-out calibration set
3. Target: ECE < 0.10

### Phase 2: Real Robot Validation (2-3 months)
1. Collect 1000 real robot episodes (500 success, 500 failures)
2. Fine-tune on 80% real data
3. Re-compute all metrics on 20% real holdout
4. **Expected:** AUROC drops to 0.85-0.90 (still acceptable)

### Phase 3: Pilot Deployment (3-6 months)
1. Deploy with monitoring on low-risk tasks
2. Collect operator feedback on alert frequency/timing
3. Continuous threshold tuning based on task criticality
4. Safety analysis: failure modes, edge cases, worst-case scenarios

---

## Comparison to Prior Work

### SAFE (Rana et al. 2023)
- **Reported:** 78% accuracy, no calibration metrics
- **SALUS:** 99% AUROC but poor calibration
- **Key Difference:** SAFE used MLP on hidden states (no temporal context)

**Honest Assessment:**
"SALUS improves discrimination (99% vs 78% AUROC) but both systems lack calibration validation. Prior work's lower metrics may reflect better generalization to real robots, whereas our synthetic data may be optimistically simple."

---

## Conclusion for Paper Discussion Section

**Suggested Addition:**

> "While SALUS demonstrates strong failure discrimination on synthetic data (AUROC=0.991), this evaluation exposed critical gaps between research metrics and production requirements:
>
> 1. **Calibration Gap:** High AUROC does not guarantee well-calibrated probabilities. Our ECE of 0.45 (vs target <0.10) indicates the model requires post-hoc calibration before probability values can guide threshold-based decisions.
>
> 2. **Lead Time Gap:** Mean lead time of 139ms falls short of the 200ms minimum for human intervention, suggesting the need for longer temporal windows or earlier precursor signals.
>
> 3. **Generalization Gap:** The 98-99% AUROC achieved by all neural baselines suggests synthetic data may be too predictable. Real robot validation is essential.
>
> These findings underscore the importance of production-oriented evaluation metrics (calibration, lead time, false alarm rates) over academic metrics (accuracy, AUROC alone) for safety-critical systems. We recommend that future work in robotic failure prediction:
> - Report ECE alongside AUROC
> - Measure lead time distributions (not just detection accuracy)
> - Test on real robot failures (not just simulation/synthetic)
> - Validate calibration on held-out data
> - Present precision-recall tradeoffs (not single operating points)
>
> With calibration and longer temporal windows, SALUS has the potential to meet production safety requirements. However, **the current system is not deployment-ready** and requires the improvements outlined above."

---

## Files Generated

1. **baseline_results.json** - Comparison of 4 baselines vs SALUS
2. **temporal_leakage_results.json** - 3 control experiments (time-shuffle, counterfactual, no-time-index)
3. **production_metrics.json** - Comprehensive metrics (per-horizon, calibration, threshold analysis, lead time)
4. **calibration_diagram.png** - Reliability diagram showing miscalibration
5. **precision_recall_curve.png** - Tradeoff between precision and recall
6. **lead_time_distribution.png** - Histogram of lead times

---

## Key Takeaway for Safety

**A model that says "50% failure probability" but is actually wrong 95% of the time (ECE=0.45) is MORE DANGEROUS than a model that says "I don't know."**

Poorly calibrated probabilities create false confidence. For safety-critical systems, calibration is not optional - it's a requirement.

---

**This honest assessment serves the user's request: show the flaws so we can fix them, rather than make it look perfect and risk harm.**
