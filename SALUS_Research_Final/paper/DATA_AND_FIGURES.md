# SALUS Data Collection & Figure Planning

## Current Data Status (as of Jan 12, 2026)

### ✅ **Data We Have Collected**

#### 1. **Massive Dataset** (COMPLETE)
**Location**: `paper_data/massive_collection/20260109_215258/data_20260109_215321.zarr`

**Statistics**:
- **Episodes**: 5,000
- **Total Timesteps**: 1,000,000
- **Failure Rate**: 8.0%
- **Episode Length**: 200 timesteps max
- **Control Frequency**: 30Hz

**Data Arrays**:
- `signals` (5000, 200, 12): 12D uncertainty signals
- `horizon_labels` (5000, 200, 16): Multi-horizon failure labels (4 horizons × 4 types)
- `actions` (5000, 200, 7): Robot actions (6-DOF end-effector + gripper)
- `states` (5000, 200, N): Proprioceptive states
- `images` (5000, 200, 256, 256, 3): RGB camera observations (3 cameras available)

**Failure Type Breakdown**:
- Object Drops: 35% of failures
- Collisions: 28%
- Kinematic Violations: 22%
- Task Failures: 15%

#### 2. **Trained SALUS Model** (COMPLETE)
**Location**: `/home/mpcr/Desktop/SalusV3/checkpoints/salus_predictor_massive.pth`

**Model Details**:
- Parameters: 70,672
- Architecture: [128, 256, 128] MLP
- Training: 100 epochs
- Best Val Loss: 0.0653 (epoch 61)
- File Size: 284KB

#### 3. **SALUS Evaluation Results** (COMPLETE)
**Location**: `results/salus_results_massive.json`

**Metrics** (on test set):
- **AUROC**: 0.8833
- **Precision**: 41.25%
- **Recall**: 51.56%
- **F1 Score**: 45.83%
- **False Alarm Rate**: 6.39% (per timestep)
- **Confusion Matrix**:
  - True Positives: 41,249
  - False Positives: 58,751
  - True Negatives: 861,249
  - False Negatives: 38,751

#### 4. **Baseline Evaluation Results** (PARTIAL)
**Location**: `results/baseline_results_massive.json`

**Status**: Random baseline works (AUROC=0.5006), but Entropy/Action Variance baselines have zero results (not properly implemented)

**What's Available**:
- Random predictor: AUROC=0.5006, Recall=30.3%, FAR=30.0%

**What's Missing**:
- Proper Entropy Threshold implementation
- Proper Action Variance implementation
- Combined baseline

---

### ⏳ **Data Currently Being Generated**

#### 5. **Ablation Study** (RUNNING - 4-5 hours remaining)
**Target Location**: `results/ablation/ablation_results.csv`

**Current Progress**:
- Started: Jan 12, 10:15 AM
- Currently on: Ablation 3/7 (no_internal), Epoch 1/30
- Estimated Completion: ~6-7 PM today

**Ablation Configurations**:
1. ✅ **Full Model** (12D) - COMPLETE
2. ✅ **No Temporal** (7D) - COMPLETE
3. 🔄 **No Internal** (9D) - IN PROGRESS (54% through epoch 1)
4. ⏳ **No Uncertainty** (10D) - QUEUED
5. ⏳ **No Physics** (10D) - QUEUED
6. ⏳ **Only Uncertainty** (2D) - QUEUED
7. ⏳ **Only Temporal** (5D) - QUEUED

**Output Format** (CSV):
```
ablation,signals_used,auroc,recall,precision,f1,tp,fp,tn,fn,auroc_drop,auroc_drop_pct
full,all,0.XXX,0.XXX,0.XXX,0.XXX,X,X,X,X,0.000,0.0
no_temporal,[4,5,6,7,8,9,10],0.XXX,...
...
```

---

### ❌ **Data We're Missing**

#### 6. **Multi-Horizon Breakdown** (NOT YET COMPUTED)
**What We Need**: Per-horizon performance metrics

**Current Status**: Model predicts 4 horizons (200ms, 300ms, 400ms, 500ms), but we only evaluated aggregate performance

**To Generate**:
```python
# scripts/compute_horizon_metrics.py
for horizon_idx in [0, 1, 2, 3]:  # 200ms, 300ms, 400ms, 500ms
    horizon_predictions = model_outputs[:, horizon_idx, :]
    horizon_labels = labels[:, horizon_idx, :]
    compute_auroc, recall, precision for this horizon
```

**Output**:
```json
{
  "200ms": {"auroc": 0.XXX, "recall": 0.XXX, "precision": 0.XXX},
  "300ms": {"auroc": 0.XXX, ...},
  "400ms": {"auroc": 0.XXX, ...},
  "500ms": {"auroc": 0.XXX, ...}
}
```

**Time Estimate**: 30 minutes to implement + 10 minutes to run

#### 7. **Per-Failure-Type Breakdown** (NOT YET COMPUTED)
**What We Need**: Per-type performance (drops, collisions, kinematic, task)

**Current Status**: Model predicts 4 failure types, but we only evaluated aggregate

**To Generate**:
```python
# scripts/compute_failure_type_metrics.py
for failure_type_idx in [0, 1, 2, 3]:  # drop, collision, kinematic, task
    type_predictions = model_outputs[:, :, failure_type_idx]
    type_labels = labels[:, :, failure_type_idx]
    compute_auroc, recall, precision for this type
```

**Output**:
```json
{
  "Object Drop": {"auroc": 0.XXX, "recall": 0.XXX, "precision": 0.XXX},
  "Collision": {"auroc": 0.XXX, ...},
  "Kinematic Violation": {"auroc": 0.XXX, ...},
  "Task Failure": {"auroc": 0.XXX, ...}
}
```

**Time Estimate**: 30 minutes to implement + 10 minutes to run

#### 8. **Inference Latency Benchmark** (NOT YET MEASURED)
**What We Need**: Forward pass timing

**To Generate**:
```python
# scripts/benchmark_latency.py
import time
model.eval()
with torch.no_grad():
    for i in range(1000):
        start = time.time()
        output = model(random_signals)
        end = time.time()
        latencies.append((end - start) * 1000)  # ms

mean_latency = np.mean(latencies)
std_latency = np.std(latencies)
```

**Output**:
```json
{
  "mean_latency_ms": 2.1,
  "std_latency_ms": 0.3,
  "min_latency_ms": 1.8,
  "max_latency_ms": 3.2,
  "p50_latency_ms": 2.0,
  "p95_latency_ms": 2.5,
  "p99_latency_ms": 2.8
}
```

**Time Estimate**: 15 minutes to implement + 5 minutes to run

#### 9. **Training Time Measurement** (NOT RECORDED)
**What We Need**: Total training time for 100 epochs

**Current Status**: Not logged during training

**Estimate from logs**: ~2-3 hours for 100 epochs (need to verify)

**Time Estimate**: Can extract from logs if available, or re-run 1 epoch and extrapolate

#### 10. **Baseline Implementation** (INCOMPLETE)
**What We Need**: Proper Entropy Threshold and Action Variance baselines

**Current Status**: Return all zeros (not implemented correctly)

**To Fix**: Implement proper thresholding logic in `evaluate_baseline_threshold.py`

**Time Estimate**: 1 hour to implement + 30 minutes to run

---

## Figure Planning

Based on the SafeVLA paper style, here are the figures we can create:

### **Figure 1: System Overview** (Can Create Now)
**Type**: Architecture diagram
**Content**:
- VLA policy block
- 12D signal extraction
- Multi-horizon predictor
- Intervention decision
- Adaptive response

**Tools**: Draw.io, TikZ, or Python matplotlib
**Time Estimate**: 2-3 hours for clean diagram

**Similar to SafeVLA Figure 1** - shows the full pipeline

---

### **Figure 2: Signal Extraction Illustration** (Can Create Now)
**Type**: Multi-panel visualization
**Content**:
- Panel 1: Temporal signals (action volatility over time)
- Panel 2: Internal signals (latent drift heatmap)
- Panel 3: Uncertainty signals (entropy timeline)
- Panel 4: Physics signals (constraint margin)

**Data Source**: Sample episode from `paper_data/massive_collection/.../data.zarr`
**Tools**: Python matplotlib/seaborn
**Time Estimate**: 3-4 hours

**Script**:
```python
# scripts/visualize_signals.py
episode_idx = 42  # Pick interesting episode with failure
signals = zarr.open(...)['signals'][episode_idx]  # (200, 12)
plot_4_panels(signals)
```

---

### **Figure 3: Main Results Comparison** (Need Baseline Data)
**Type**: Bar chart with error bars
**Content**:
- X-axis: Methods (Random, Entropy, Action Var, SALUS)
- Y-axis dual: AUROC (left), False Alarm Rate (right)
- Show SALUS improvement clearly

**Data Source**:
- ✅ `results/salus_results_massive.json`
- ❌ `results/baseline_results_massive.json` (needs fixing)

**Tools**: Python matplotlib/seaborn
**Time Estimate**: 1 hour (after baseline data available)

**Similar to SafeVLA Table 1** - but as a figure

---

### **Figure 4: ROC Curves** (Can Create Now)
**Type**: ROC curve comparison
**Content**:
- SALUS ROC curve (AUROC=0.8833)
- Random baseline (diagonal)
- Optional: Entropy/Action Variance if we fix them

**Data Source**: Can re-generate from model predictions
**Tools**: sklearn.metrics.roc_curve + matplotlib
**Time Estimate**: 1 hour

**Script**:
```python
# scripts/plot_roc_curves.py
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label=f'SALUS (AUC={auc:.3f})')
```

---

### **Figure 5: Ablation Results** (Need Ablation Data - 4-5 hours)
**Type**: Bar chart showing AUROC drop
**Content**:
- X-axis: Ablation configurations
- Y-axis: AUROC
- Highlight full model
- Show drop when removing each group

**Data Source**: ⏳ `results/ablation/ablation_results.csv` (currently running)

**Tools**: Python matplotlib/seaborn
**Time Estimate**: 1 hour (after ablation completes)

**Similar to SafeVLA Figure** - showing component importance

---

### **Figure 6: Multi-Horizon Performance** (Need Horizon Data)
**Type**: Line plot or bar chart
**Content**:
- X-axis: Prediction horizon (200ms, 300ms, 400ms, 500ms)
- Y-axis: AUROC, Recall, Precision
- Show optimal horizon (likely 400ms)

**Data Source**: ❌ Need to compute from model
**Tools**: Python matplotlib
**Time Estimate**: 1 hour (after computing horizon metrics)

---

### **Figure 7: Per-Failure-Type Performance** (Need Type Data)
**Type**: Grouped bar chart
**Content**:
- X-axis: Failure types (Drop, Collision, Kinematic, Task)
- Y-axis: AUROC, Recall, Precision
- Show which failures are easiest/hardest to predict

**Data Source**: ❌ Need to compute from model
**Tools**: Python matplotlib/seaborn
**Time Estimate**: 1 hour (after computing type metrics)

---

### **Figure 8: Failure Prediction Timeline** (Can Create Now)
**Type**: Timeline visualization
**Content**:
- Real episode showing:
  - Predicted failure probability over time
  - Actual failure occurrence
  - Intervention threshold
  - Lead time demonstration

**Data Source**: Sample from test set predictions
**Tools**: Python matplotlib
**Time Estimate**: 2 hours

**Script**:
```python
# scripts/visualize_failure_timeline.py
episode_with_failure = ...
predictions_over_time = model.predict(episode_signals)
plt.plot(timesteps, predictions[:, horizon_idx, failure_type])
plt.axvline(failure_timestep, color='red', label='Actual Failure')
plt.axhline(threshold, color='orange', label='Alert Threshold')
```

---

### **Figure 9: Confusion Matrix Heatmap** (Can Create Now)
**Type**: Heatmap
**Content**:
- 2x2 confusion matrix
- Show TP, FP, TN, FN proportions

**Data Source**: ✅ Already have in results JSON
**Tools**: seaborn heatmap
**Time Estimate**: 30 minutes

---

### **Figure 10: Training Curves** (Can Create if Logs Available)
**Type**: Line plots
**Content**:
- Panel 1: Train/Val Loss over epochs
- Panel 2: Train/Val AUROC over epochs
- Show early stopping point

**Data Source**: Training logs (if saved)
**Tools**: Python matplotlib
**Time Estimate**: 1 hour (if logs exist)

---

## Timeline to Complete All Data & Figures

### Today (Jan 12)
- **6:00 PM**: Ablation study completes
- **6:30 PM**: Fix baseline implementations
- **7:00 PM**: Run baselines
- **7:30 PM**: Compute multi-horizon metrics
- **8:00 PM**: Compute per-failure-type metrics
- **8:15 PM**: Benchmark inference latency

### Tomorrow (Jan 13)
- **Morning**: Create all figures (8-10 hours of work)
  - Figure 1: System overview (3 hours)
  - Figure 2: Signal visualization (3 hours)
  - Figure 3: Main results (1 hour)
  - Figure 4: ROC curves (1 hour)
  - Figure 5: Ablation (1 hour)
  - Figure 6: Multi-horizon (1 hour)
  - Figure 7: Per-type (1 hour)
  - Figure 8: Timeline (2 hours)

- **Afternoon**: Fill results into paper (2 hours)
  - Run `python paper/fill_results.py`
  - Manually verify all values
  - Check consistency

- **Evening**: Final paper review (2 hours)
  - Compile PDF
  - Review all sections
  - Check figure references
  - Polish abstract

### Total Time to Submission-Ready
**~24 hours from now** (end of day Jan 13)

---

## Priority Order (What to Do First)

### **Priority 1: Finish Running Experiments** (4-5 hours - automated)
- ✅ Ablation study (running, just wait)

### **Priority 2: Compute Missing Metrics** (2 hours - your work)
1. Fix baseline implementations (1 hour)
2. Run baselines (30 min)
3. Compute multi-horizon breakdown (30 min)
4. Compute per-failure-type breakdown (30 min)
5. Benchmark inference latency (15 min)

### **Priority 3: Create Figures** (8-10 hours - your work)
1. System overview diagram (3 hours) - **Most Important**
2. Signal visualization (3 hours)
3. Main results comparison (1 hour)
4. ROC curves (1 hour)
5. Ablation results (1 hour) - after ablation completes
6. Multi-horizon (1 hour)
7. Per-failure-type (1 hour)
8. Failure timeline (2 hours)

### **Priority 4: Fill Paper Results** (2 hours)
1. Run `fill_results.py`
2. Manually verify tables
3. Add inline numbers

### **Priority 5: Final Polish** (2 hours)
1. Compile to PDF
2. Review formatting
3. Check figure placement
4. Proofread text

---

## Minimal Viable Paper (If Time Constrained)

If you need to submit faster, here's the minimal set:

### **Required Figures** (Must Have):
1. **Figure 1**: System overview (architecture diagram)
2. **Figure 3**: Main results (SALUS vs baselines)
3. **Figure 5**: Ablation study (signal importance)

### **Required Data** (Must Have):
1. ✅ SALUS performance (already have)
2. ⏳ Ablation results (running now)
3. ❌ Baseline comparison (need to fix)

### **Optional but Valuable**:
- ROC curves
- Multi-horizon breakdown
- Per-failure-type analysis
- Signal visualization
- Timeline visualization

With this minimal set, you could have a submission-ready paper by **tomorrow evening** (Jan 13, 8 PM).

---

## Next Immediate Steps

1. **Let ablation run** (automatic, ~4 hours)
2. **Fix baseline implementation** (1 hour work)
3. **Start creating Figure 1** (system overview) - can do while waiting
4. **When ablation finishes**: Generate Figure 5
5. **Fill remaining metrics**: horizons, types, latency
6. **Create remaining figures**: Based on priority
7. **Fill paper results**: Use `fill_results.py`
8. **Final review and submit**

You're very close! The main bottleneck is the ablation study finishing (automatic) and creating the figures (manual work).
