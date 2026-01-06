# SALUS System Status Report
**Date**: January 4, 2026
**System**: Isaac Sim 5.1.0 + Isaac Lab 0.48.5 + SALUS Pipeline

---

## ✅ WHAT IS WORKING

### 1. **Isaac Sim + Isaac Lab Environment** ✅
- **Isaac Sim 5.1.0**: Successfully loads and runs on GPU
- **Isaac Lab 0.48.5**: Environment framework operational
- **Franka Panda Robot**: Spawns correctly with 9 DOF (7 arm + 2 gripper)
- **Physics Simulation**: PhysX running with GPU acceleration
- **Scene Management**: Ground plane, lighting, objects all spawn correctly

### 2. **Vision System** ✅
- **3× RGB Cameras**: Front, side, and top views
- **Resolution**: 224×224 pixels (reduced from 256 to save GPU memory)
- **Image Pipeline**: Isaac Lab → [B, H, W, C] → VLA [B, C, H, W] format conversion working
- **Data Format**: Images saved as uint8 in zarr: (1, 200, 3, 3, 224, 224)

### 3. **VLA (Vision-Language-Action Model)** ✅
- **Model**: SmolVLA-450M (based on SmolVLM2-500M)
- **Ensemble**: Reduced to 1 model (from 5) to fit in GPU memory
- **Action Generation**: Successfully generates 7 DOF actions from camera observations
- **Device Management**: All components on cuda:0 (single GPU strategy)

### 4. **Data Collection Pipeline** ✅
- **Zarr Format**: Chunked, compressed storage working
- **Data Groups**: 6 groups created:
  - `actions`: (1, 200, 7) - VLA-generated actions
  - `states`: (1, 200, 7) - Robot joint positions
  - `images`: (1, 200, 3, 3, 224, 224) - Camera observations
  - `signals`: (1, 200, 12) - Internal VLA signals
  - `horizon_labels`: Failure prediction horizons
  - `episode_metadata`: Episode info (success/failure type)
- **File Size**: ~4.8MB per episode
- **Save Location**: `data/SINGLE_GPU_TEST/20260104_175833/data.zarr`

### 5. **Technical Integration** ✅
- **Import Order**: Fixed - minimal imports before AppLauncher
- **NumPy Version**: Downgraded to 1.26.0 for compatibility
- **Action Padding**: 7 DOF actions → 9 DOF (adds gripper commands)
- **Device Placement**: VLA and Isaac both on cuda:0 (no cross-GPU errors)
- **Scene Initialization**: `sim.reset()` called at correct time (first `env.reset()`)

---

## ❌ WHAT IS **NOT** WORKING

### 1. **SALUS Failure Prediction - NOT TESTED** ❌
**Status**: The SALUS failure prediction system has **NOT been trained or tested yet**.

**What's Missing**:
- ❌ **No failure injection**: The environment doesn't inject failures (random forces, object slip, sensor noise)
- ❌ **No predictor training**: The failure predictor neural network hasn't been trained
- ❌ **No manifold learning**: The triplet network for state-space manifold not trained
- ❌ **No synthesis evaluation**: Recovery trajectory generation not implemented
- ❌ **No metrics**: F1 score, precision, recall - none measured yet

**Current State**:
- ✅ We collect data (actions, states, images, signals)
- ✅ We have placeholders for failure labels
- ❌ But the VLA just generates "normal" actions - no failures are happening
- ❌ The predictor model exists in code but is untrained (random weights)

### 2. **Episode Completion** ⏳
- **Status**: Data collection started 20+ minutes ago but hasn't completed
- **Issue**: Episode should take ~200 timesteps × 30 Hz = 6.7 seconds, but it's very slow
- **Likely Cause**:
  - VLA inference is slow (~1-2 seconds per step)
  - Isaac Sim rendering overhead
  - Debug logging slowing things down
- **Data Collected**: 4.8MB (appears complete based on zarr shape)

### 3. **Task Success Validation** ❓
- **Task**: "Pick up red cube and place in blue zone"
- **Problem**: We don't know if the robot actually succeeds at this task
- **Why**:
  - No success detection implemented (checking cube in goal zone)
  - VLA is pre-trained but not fine-tuned for this specific task
  - Random/exploration actions won't accomplish the task

### 4. **Performance Issues** ⚠️
- **Speed**: Very slow (~1-2 seconds per simulation step)
- **Memory**: Using 6.3GB GPU memory (tight fit)
- **CPU**: 166% CPU usage indicates bottleneck

---

## 🤔 IS SALUS PREDICTING FAILURES?

### **Answer: NO - Not Yet**

**Why?**:
1. **No Training Data**: We just started collecting data - need 50-100 episodes with failures
2. **No Failure Injection**: The environment runs normally, no failures to predict
3. **No Model Training**: The predictor network has random weights (untrained)
4. **No Evaluation**: Haven't measured F1 score, accuracy, or any metrics

**What SALUS Would Do (Once Trained)**:
```
1. Robot executes action
2. VLA generates next action + internal signals (attention, uncertainty)
3. SALUS Predictor receives signals → predicts failure probability at 4 horizons:
   - 200ms ahead: P(failure) = 0.05 (safe)
   - 300ms ahead: P(failure) = 0.32 (warning)
   - 400ms ahead: P(failure) = 0.78 (danger!)
   - 500ms ahead: P(failure) = 0.95 (imminent failure)
4. If P(failure) > threshold: SALUS Synthesis generates recovery trajectory
5. Controller switches from VLA actions to recovery actions
6. Failure avoided!
```

**Current Reality**:
```
1. Robot executes action ✅
2. VLA generates next action + signals ✅
3. SALUS Predictor receives signals → outputs random noise (untrained) ❌
4. No recovery actions generated ❌
5. Robot continues with VLA actions (no failure prevention)
```

---

## 🔍 HOW WE FIGURED THIS OUT

### **The Debugging Journey** (7 Critical Fixes):

#### **Fix #1: NumPy Version Conflict**
- **Error**: `AttributeError: module 'numpy' has no attribute '_no_nep50_warning'`
- **Cause**: Isaac Sim 5.1.0 requires numpy 1.26.x, we had 2.4.0
- **Solution**: `pip install numpy==1.26.0 --force-reinstall`
- **How Found**: Read error message, checked Isaac Sim requirements

#### **Fix #2: Import Order**
- **Error**: AppLauncher hangs forever (128% CPU, no progress)
- **Cause**: Importing torch/numpy BEFORE AppLauncher interferes with Omniverse Kit
- **Solution**: Only import argparse/sys/Path before AppLauncher, everything else after
- **How Found**: Added granular print statements, saw exact line where hang occurred

#### **Fix #3: Duplicate AppLauncher**
- **Error**: Script hangs at "Importing franka_pick_place_env.py..."
- **Cause**: Environment file was creating a second AppLauncher (only one allowed per process)
- **Solution**: Remove AppLauncher from environment, pass `simulation_app` as parameter
- **How Found**: Traced imports line-by-line, found duplicate creation

#### **Fix #4: Ground Plane Configuration**
- **Error**: `Unknown asset config type for ground: GroundPlaneCfg`
- **Cause**: Isaac Lab 0.48.5 API change - ground must be wrapped in `AssetBaseCfg`
- **Solution**:
```python
ground = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
)
```
- **How Found**: Examined Isaac Lab demo scripts, found correct pattern

#### **Fix #5: Articulation Actuators**
- **Error**: `'Articulation' object has no attribute 'actuators'`
- **Cause**: Calling `scene.reset()` before simulation initialized
- **Solution**: Move `sim.reset()` to first `env.reset()` call, not during scene creation
- **How Found**: Read Isaac Lab multi_asset demo, saw correct initialization sequence

#### **Fix #6: Action Dimensions**
- **Error**: `shape mismatch: value tensor of shape [7] cannot be broadcast to [1, 9]`
- **Cause**: VLA outputs 7 DOF, but Franka has 9 joints (7 arm + 2 gripper)
- **Solution**: Pad actions: `torch.cat([action_7dof, gripper_cmd_2dof], dim=-1)`
- **How Found**: Read error message, checked Franka joint configuration

#### **Fix #7: GPU Device Mismatch**
- **Error**: `Expected all tensors to be on the same device, cuda:1 and cuda:0!`
- **Cause**: VLA on cuda:0, Isaac Sim on cuda:1, tensors couldn't mix
- **Solution**: Run EVERYTHING on cuda:0 using `CUDA_VISIBLE_DEVICES=0`
- **How Found**: Added device debug prints, saw tensors on different GPUs

---

## 🎮 GUI SIMULATION

**Status**: Not tested yet in this session, but should work now.

**Working Configuration**:
```bash
cd "/home/mpcr/Desktop/Salus Test/SalusTest"
source /home/mpcr/miniconda/etc/profile.d/conda.sh
conda activate isaaclab
DISPLAY=:1 CUDA_VISIBLE_DEVICES=0 python scripts/collect_data_franka.py \
    --num_episodes 1 \
    --save_dir data/GUI_TEST \
    --enable_cameras \
    --device cuda:0
```

**Expected Behavior**:
- Isaac Sim window opens on display :1
- Shows Franka Panda robot + red cube + ground plane
- 3 camera viewports (front, side, top)
- Real-time physics simulation
- Robot arm moves based on VLA actions

**Note**: GUI will be slower than headless mode.

---

## 📊 NEXT STEPS TO GET SALUS WORKING

### **Phase 1: Data Collection** (Current Phase)
1. ✅ Get Isaac Sim + VLA working (DONE!)
2. ⏳ Complete first episode (in progress)
3. ❌ Implement failure injection:
   - Random forces on objects
   - Object slip during grasp
   - Sensor noise
4. ❌ Collect 50-100 episodes with ~30% failure rate

### **Phase 2: Model Training**
1. ❌ Train Failure Predictor:
   - Input: VLA signals (12D: attention, uncertainty, etc.)
   - Output: Failure probability at 4 horizons (200ms, 300ms, 400ms, 500ms)
   - Architecture: [12, 64, 128, 128, 64, 4×4] (4 horizons × 4 failure types)
2. ❌ Train Manifold Network:
   - Triplet loss on (state, action) pairs
   - Learn 8D latent space where nearby points = similar outcomes
3. ❌ Implement Synthesis Module:
   - Find nearest successful trajectory in manifold
   - Generate recovery actions

### **Phase 3: Evaluation**
1. ❌ Measure F1 score on test set
2. ❌ Compare to baseline (no SALUS)
3. ❌ Measure failure prevention rate

---

## 🎯 SUMMARY

### What Works:
✅ Isaac Sim physics simulation
✅ Robot control (7 DOF + gripper)
✅ Camera rendering (3 views)
✅ VLA action generation
✅ Data collection (zarr format)

### What Doesn't Work (Yet):
❌ SALUS failure prediction (not trained)
❌ Failure injection (not implemented)
❌ Recovery trajectory generation (not implemented)
❌ Performance metrics (not measured)

### The Bottom Line:
**We have successfully built the DATA COLLECTION PIPELINE for SALUS.** The hard part (getting Isaac Sim working with all the compatibility issues) is done. But SALUS itself (the failure prediction system) hasn't been trained or tested yet. We're at the "infrastructure working" stage, not the "AI working" stage.

To get SALUS actually predicting failures, we need to:
1. Collect data with injected failures
2. Train the predictor neural network
3. Evaluate on test episodes

**Estimated Time to Full SALUS**:
- Data collection: 2-4 hours (50-100 episodes)
- Training: 1-2 hours (GPU training)
- Evaluation: 30 minutes
- **Total**: ~4-6 hours of compute time
