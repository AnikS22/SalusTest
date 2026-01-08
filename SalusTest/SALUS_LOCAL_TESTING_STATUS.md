# SALUS Local Testing Status

## ✅ What's Fixed and Working

### 1. VLA Integration ✅
- **Multi-GPU bug fixed**: VLA now loads on single GPU (cuda:0)
- **6D→7D conversion working**: Environment handles VLA's 6D actions properly
- **Confirmed working**: Test showed VLA outputs real actions and all 12D signals extracted

### 2. Signal Extraction ✅
All 12D signals verified working:
```
1-12:  Basic uncertainty (from ensemble)
13-14: VLA internals (latent drift, OOD)
15-16: Sensitivity (perturbation response)
17-18: Reality checks (execution, constraints)
```

Example output from test:
- Model uncertainty: 0.061
- Action magnitude: 1.365 rad
- Latent drift: extracted from transformer (norm=1.23)
- Perturbation sensitivity: 0.517

---

## 🔄 What's Running NOW

### Data Collection (PID 318022)
```
Status: RUNNING
Runtime: 2+ minutes
CPU: 150% (multi-threaded)
Memory: 5.9 GB (VLA loaded)
Progress: Collecting 50 episodes with VLA control
```

**What it's doing**:
1. VLA controls robot to pick up red cube
2. Every step: extracts 12D signals from VLA
3. Labels each episode as success/failure
4. Saves all data to Zarr format

**Expected time**: ~15-25 minutes total (depends on episode lengths)

**Output**: `local_data/salus_data_YYYYMMDD_HHMMSS.zarr`

---

## 📋 Next Steps (Automated)

### 1. Wait for Data Collection ⏳
- 50 episodes @ ~100 steps each = ~5,000 timesteps
- Each timestep: VLA inference + signal extraction
- Will save: signals, actions, robot states, labels

### 2. Train SALUS 🎯
**Script ready**: `train_salus_local.py`

Will train HybridTemporalPredictor:
- **Input**: 10-step windows of 12D signals
- **Output**: Failure predictions at 4 horizons (200ms, 300ms, 400ms, 500ms)
- **Architecture**: Conv1D + GRU + Linear
- **Loss**: TemporalFocalLoss (handles imbalance)

Training config:
- 50 epochs
- 80/20 train/val split
- Batch size: 32
- Learning rate: 0.001 with scheduler

### 3. Test Predictions 🔍
Script will show:
- Accuracy per horizon
- Precision/Recall/F1
- Confusion matrix (TP, FP, TN, FN)

**Key question**: Can SALUS predict failures before they happen?

---

## 🎯 Success Criteria

### Data Collection
- ✅ 50 episodes collected
- ✅ Mix of success and failure episodes
- ✅ All 12D signals non-zero and varying
- ✅ Proper labels attached

### SALUS Training
- ✅ Training loss decreases
- ✅ Validation accuracy > 60% (better than random)
- ✅ Can distinguish failure vs success patterns
- ✅ F1 score > 0.5 on at least one horizon

### If SALUS Works
Metrics showing it CAN predict failures:
- Accuracy > 70% on any horizon
- Recall > 0.5 (catches most failures)
- Precision > 0.5 (few false alarms)
- F1 score > 0.6

### If SALUS Doesn't Work
Possible reasons:
- Not enough training data (50 episodes may be too few)
- VLA too consistent (no variation in signals)
- Failure modes not predictable from signals
- Need more sophisticated model

---

## 📊 What We'll Learn

### Question 1: Do 12D signals contain failure information?
- If yes: Signals will differ between success/failure episodes
- If no: Signals will be similar regardless of outcome

### Question 2: Can temporal patterns predict failures?
- If yes: Early signals will correlate with later failures
- If no: Failures happen too suddenly to predict

### Question 3: Which signals matter most?
- Training will show which of the 18 dimensions are informative
- May reveal that some signals are redundant

### Question 4: How far ahead can we predict?
- 200ms horizon: Easiest (failure imminent)
- 500ms horizon: Hardest (more uncertainty)

---

## 🔧 System Architecture

```
Isaac Lab Environment
    ↓
Robot executes VLA actions
    ↓
SmolVLA Ensemble (3× 865MB)
    ├─ Processes camera images
    ├─ Outputs 6D actions
    └─ Exposes transformer hidden states
    ↓
EnhancedSignalExtractor
    ├─ Computes model uncertainty (internal uncertainty signals)
    ├─ Extracts latent drift (hidden state changes)
    ├─ Tests perturbation sensitivity (3× extra VLA runs)
    └─ Checks physics constraints
    ↓
18D Signal Vector (every 33ms @ 30Hz)
    ↓
Zarr Storage
    ├─ signals: (N, 18)
    ├─ episode_id: (N,)
    ├─ success: (N,)
    └─ done: (N,)
    ↓
TemporalDataset
    └─ Creates 10-step windows
    ↓
HybridTemporalPredictor
    ├─ Conv1D: Local patterns
    ├─ GRU: Temporal dynamics
    └─ Linear: Multi-horizon prediction
    ↓
Failure Predictions (4 horizons × 2 classes)
```

---

## 💾 Data Format

**Zarr structure**:
```
salus_data_YYYYMMDD_HHMMSS.zarr/
├── signals/        # (N, 18) float32 - 12D signal vectors
├── actions/        # (N, 6) float32 - VLA actions
├── robot_state/    # (N, 7) float32 - Joint angles
├── episode_id/     # (N,) int32 - Which episode
├── timestep/       # (N,) int32 - Step within episode
├── success/        # (N,) bool - Episode outcome
└── done/           # (N,) bool - Episode termination

Attributes:
- num_episodes: 50
- total_steps: ~5000
- successes: X
- failures: Y
- signal_dim: 18
- action_dim: 6
```

---

## ⏱️ Timeline Estimate

### Current (16:12): Data Collection Started
- VLA loading: 1 minute ✅ DONE
- Episode 1-10: ~5 minutes
- Episode 11-20: ~5 minutes
- Episode 21-30: ~5 minutes
- Episode 31-40: ~5 minutes
- Episode 41-50: ~5 minutes

**Expected completion**: ~16:35 (23 minutes)

### After Collection: Training
- Load data: 10 seconds
- Train 50 epochs: 2-5 minutes
- Evaluation: 10 seconds

**Total training time**: ~5 minutes

### Final: Testing
- Load best model: 1 second
- Compute metrics: 10 seconds
- Print results: immediate

**Total time from now**: ~30 minutes to full SALUS evaluation

---

## 🎯 What This Proves

If SALUS works (>70% accuracy):
1. ✅ VLA signals contain failure information
2. ✅ Failures are predictable from temporal patterns
3. ✅ Real-time failure prediction is feasible
4. ✅ System ready for HPC deployment

If SALUS doesn't work (<60% accuracy):
1. ⚠️ Need more data (try 200-500 episodes)
2. ⚠️ Need different signals (add more features)
3. ⚠️ Need better model (try Transformer instead of GRU)
4. ⚠️ Task may be too hard to predict

---

## 📝 Files Status

### ✅ Working
- `salus/core/vla/wrapper.py` - VLA ensemble with device fix
- `salus/simulation/franka_pick_place_env.py` - 6D→7D conversion
- `salus/models/temporal_predictor.py` - SALUS model (18D input)

### 🔄 Running
- `collect_local_data.py` - Collecting 50 episodes

### ⏸️ Ready
- `train_salus_local.py` - Training script ready to run

### 📊 Output
- `local_data/salus_data_*.zarr` - Will contain training data
- `salus_best_local.pth` - Will contain trained model

---

## 🚀 After Local Testing

If SALUS works locally:
1. Sync code to Athene HPC
2. Collect 500 episodes on HPC (better GPUs, more data)
3. Train production SALUS model
4. Deploy for real-time failure prediction

If SALUS needs work:
1. Analyze which signals are informative
2. Try different model architectures
3. Collect more diverse failure modes
4. Iterate on signal extraction

---

**Current Status**: Waiting for data collection to complete (~20 more minutes)

**Next Action**: Automatically train SALUS when data is ready

**Expected Result**: Know if SALUS can predict failures in ~30 minutes
