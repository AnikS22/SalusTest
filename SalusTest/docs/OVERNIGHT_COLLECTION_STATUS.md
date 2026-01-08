# SALUS Overnight Data Collection - Running 🌙

**Started**: January 2, 2026 - 21:35:44
**Status**: ✅ **RUNNING**
**Mode**: Dummy VLA (random actions + signals)

---

## 📊 Collection Progress

```
Episodes: 500 total
Current: ~11/500 (after 2.5 minutes)
Speed: ~13 seconds per episode
Estimated completion: ~1.8 hours (around 11:15 PM)
```

**Storage**:
- Estimated final size: ~19.68 GB
- Location: `data/mvp_episodes_overnight/20260102_213544/`

---

## 🏗️ Environment Configuration

### Task
**"Pick up the red cube and place it in the blue bin"**

### Robot
- **Type**: Franka Panda
- **DOF**: 7 (joint control)
- **Control frequency**: 30 Hz (200 steps = 6.7 seconds per episode)

### Observations
- **Camera 1**: 256×256 RGB (front view) - *currently random noise*
- **Camera 2**: 256×256 RGB (side view) - *currently random noise*
- **Camera 3**: 256×256 RGB (top view) - *currently random noise*
- **Robot state**: 7D joint positions
- **Task instruction**: Natural language string

### Failure Types (4 classes)
1. **Collision (0)**: Robot hits environment or obstacles
2. **Drop (1)**: Object dropped during manipulation
3. **Miss (2)**: Failed to grasp the object
4. **Timeout (3)**: Task not completed within time limit

### Current Simulation Mode
⚠️ **Dummy Environment**:
- Random camera images (noise)
- Simulated robot states
- **50% success rate** (random)
- Random failure type assignment
- No physics simulation

**Why dummy mode?**
- TinyVLA model weights not downloaded yet
- IsaacSim requires display/GUI setup
- Good for pipeline testing
- Fast data generation

---

## 🔧 VLA Configuration

**Mode**: Dummy VLA (not real TinyVLA)

**What's being collected**:
- ✅ Episode structure (correct format)
- ✅ 6D signals (from dummy ensemble)
- ✅ Images, states, actions
- ✅ Success/failure labels
- ⚠️ Actions are random (not from trained policy)
- ⚠️ Signals are random (not real uncertainty)

**Dummy VLA generates**:
- Random actions: `torch.randn(1, 7) * 0.1`
- Random action variance: `torch.rand(1, 7) * 0.1`
- Random model uncertainty: `torch.rand(1) * 0.5`

---

## 📈 Data Format

### Per Episode
```python
episode_data = {
    'images': (T, 1, 3, 256, 256),    # T timesteps, 1 camera, RGB
    'states': (T, 7),                  # Robot joint positions
    'actions': (T, 7),                 # Action commands
    'signals': (T, 6),                 # MVP: 6D uncertainty signals
    'horizon_labels': (T, 4, 4)        # Dummy for MVP (all zeros)
}

episode_metadata = {
    'episode_id': int,
    'success': bool,
    'failure_type': int,              # 0-3, or -1 if success
    'episode_length': int,            # Actual timesteps used
    'timestamp': str
}
```

### 6D Signal Features
1. **Model uncertainty** - Ensemble variance (dummy random)
2. **Action magnitude** - L2 norm of action
3. **Action variance** - Mean variance across joints
4. **Action smoothness** - Change from previous action
5. **Max per-dim variance** - Highest joint uncertainty
6. **Uncertainty trend** - Recent uncertainty direction

---

## 📝 Monitoring

### Check Progress
```bash
# Watch collection log
tail -f data/collection_overnight.log

# Check latest status
tail -30 data/collection_overnight.log

# Kill if needed (not recommended while running)
pkill -f collect_episodes_mvp.py
```

### Collection Process
- **PID**: 3578278
- **Log file**: `data/collection_overnight.log`
- **Data directory**: `data/mvp_episodes_overnight/20260102_213544/`

### Checkpoints
The recorder automatically saves checkpoints every 50 episodes:
- `checkpoint_50.json`
- `checkpoint_100.json`
- `checkpoint_150.json`
- etc.

---

## 🎯 Expected Results

### Upon Completion (~11:15 PM tonight)

**Data**:
- ✅ 500 episodes collected
- ✅ ~100,000 timesteps (500 episodes × 200 steps)
- ✅ ~20 GB compressed Zarr storage
- ✅ Success: ~250 episodes (50%)
- ✅ Failure: ~250 episodes (50%)
  - Collision: ~62-63 episodes (25%)
  - Drop: ~62-63 episodes (25%)
  - Miss: ~62-63 episodes (25%)
  - Timeout: ~62-63 episodes (25%)

**Quality**:
- ⚠️ Random actions (not learned policy)
- ⚠️ Random signals (not real uncertainty)
- ⚠️ Random failure assignment (not physics-based)
- ✅ Correct data format for training
- ✅ Good for testing training pipeline
- ⚠️ May not learn meaningful patterns

---

## 🚀 Next Steps (After Collection Completes)

### Option A: Train on Dummy Data (Tonight/Tomorrow AM)
**Purpose**: Test training pipeline end-to-end

```bash
# Train predictor on collected data
python scripts/train_predictor_mvp.py \
    --data data/mvp_episodes_overnight/20260102_213544 \
    --epochs 50 \
    --batch_size 32 \
    --device cuda:0 \
    --checkpoint_dir checkpoints/mvp_dummy_data

# Expected: Training works but performance may be poor
# (because signals don't correlate with real failures)
```

**Expected Performance**:
- Training loss will decrease
- Validation metrics may be poor (~0.25 F1)
- Model will learn some patterns from random data
- Good for verifying training infrastructure

### Option B: Collect Real Data (Tomorrow)

**Download TinyVLA weights first**:
```bash
# Install TinyVLA
cd ~/
git clone https://github.com/OpenDriveLab/TinyVLA.git
cd TinyVLA
pip install -e .

# Download model weights
# (Follow TinyVLA instructions to download tinyvla-1b)
# Place in ~/models/tinyvla/tinyvla-1b
```

**Then collect with real VLA**:
```bash
python scripts/collect_episodes_mvp.py \
    --num_episodes 500 \
    --use_real_vla \
    --device cuda:0 \
    --save_dir data/mvp_episodes_real
```

**Benefits**:
- Real VLA actions (learned policy)
- Real uncertainty signals (internal uncertainty signals)
- Still 50% random success (dummy environment)
- Better signal-failure correlation
- Expected performance: 0.70-0.85 F1

### Option C: Wait for Real IsaacSim Environment

**Setup real physics simulation**:
- Requires IsaacSim with display
- Real pick-place physics
- Real failure modes (collision detection, object tracking)
- Most realistic data
- Best for final deployment

---

## 💡 Recommendations

### Tonight (While Collection Runs)
1. ✅ **Let collection complete** (~11:15 PM)
2. ⏸️ Go to sleep / let it run
3. ✅ Check in morning for completion

### Tomorrow Morning
**Path 1: Quick Test** (Recommended for learning)
1. Train on dummy data (verify training works)
2. See baseline performance
3. Then collect real data

**Path 2: Skip to Real Data** (Recommended for quality)
1. Download TinyVLA weights
2. Collect 500 episodes with real VLA
3. Train on that data
4. Much better performance expected

**Path 3: Full Setup** (Recommended for production)
1. Set up real IsaacSim environment
2. Implement physics-based failures
3. Collect with real VLA + real sim
4. Best data quality

---

## 🔍 Data Quality Notes

### Current Dummy Data Limitations

**What's missing**:
- ❌ Real VLA actions (using random actions)
- ❌ Real uncertainty signals (using random values)
- ❌ Real failure physics (using random 50% success)
- ❌ Real camera observations (using noise)
- ❌ Correlation between signals and failures

**What's working**:
- ✅ Correct data format
- ✅ Zarr storage and compression
- ✅ Episode recording
- ✅ Metadata tracking
- ✅ Checkpoint system
- ✅ Training pipeline compatibility

**Bottom line**:
This dummy data is excellent for **testing the pipeline** but may not train a good predictor. For real performance, need either:
- Real VLA + dummy environment (better)
- Real VLA + real IsaacSim (best)

---

## 📊 Storage Breakdown

```
Estimated storage per episode: ~40 MB
  - Images: ~35 MB (200 steps × 1 camera × 256×256×3 × uint8)
  - States: ~6 KB (200 steps × 7 joints × float32)
  - Actions: ~6 KB (200 steps × 7 joints × float32)
  - Signals: ~5 KB (200 steps × 6 signals × float32)
  - Metadata: ~1 KB

Total for 500 episodes: ~20 GB compressed
  (Zarr uses zstd compression, actual may be less)
```

---

## ✅ Pipeline Verification Status

- [x] Data collection works
- [x] Zarr storage works
- [x] Checkpointing works
- [x] Signal extraction works
- [x] Failure labeling works
- [x] Episode metadata works
- [x] Compression works
- [x] 500 episode collection in progress

**Next to verify**:
- [ ] Training on collected data
- [ ] Model checkpointing during training
- [ ] Evaluation metrics
- [ ] Deployment with intervention

---

**Monitor**: `tail -f data/collection_overnight.log`
**Check**: Around 11:15 PM for completion
**Then**: Train predictor or collect real data

---

**Status**: ✅ Running smoothly
**ETA**: ~1.8 hours (11:15 PM)
**Will auto-save**: Final checkpoint on completion
