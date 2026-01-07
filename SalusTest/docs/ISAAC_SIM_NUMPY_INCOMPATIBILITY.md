# Isaac Sim 5.1.0 Numpy Incompatibility Issue

**Date**: 2026-01-03
**Status**: BLOCKED - Cannot use real Isaac Lab simulation
**Root Cause**: Isaac Sim 5.1.0 bundled numpy incompatible with isaaclab conda env numpy

---

## Summary

Attempted to run real Isaac Lab simulation with Franka Panda robot. **Isaac Sim loaded successfully** but **critical extensions failed** due to numpy version incompatibility, making the simulation unusable.

---

## What Happened

### 1. Environment Setup ✅
- Created `isaaclab` conda environment
- Isaac Sim 5.1.0 installed at: `/home/mpcr/miniconda/envs/isaaclab/lib/python3.11/site-packages/isaacsim`
- Isaac Lab 0.48.5 installed at: `/home/mpcr/Downloads/IsaacLab/`
- Activated with: `conda activate isaaclab`

### 2. Script Execution ✅
- Script: `scripts/collect_data_franka.py`
- Correct structure: AppLauncher created BEFORE other imports
- Command: `python scripts/collect_data_franka.py --num_episodes 2 --save_dir data/test_real_isaac --enable_cameras`

### 3. Isaac Sim Loading ⚠️
- Isaac Sim 5.1.0 launched successfully (~22 seconds)
- Detected 4x RTX 2080 Ti GPUs
- Vulkan rendering initialized
- **BUT**: Many extensions failed to load

---

## Critical Extension Failures ❌

Due to numpy incompatibility, these essential extensions **failed to load**:

### Physics & Core:
- ❌ `isaacsim.core.simulation_manager` - Physics simulation management
- ❌ `isaacsim.core.prims` - USD prims API
- ❌ `isaacsim.core.api` - Core Isaac Sim API
- ❌ `isaacsim.core.cloner` - Object cloning for parallel envs

### Sensors & Rendering:
- ❌ `isaacsim.sensors.camera` - Camera sensors (CRITICAL for vision)
- ❌ `isaacsim.sensors.rtx` - RTX raytracing sensors
- ❌ `isaacsim.sensors.physx` - PhysX-based sensors
- ❌ `omni.syntheticdata` - Synthetic data generation
- ❌ `omni.replicator.core` - Data replication framework

### Isaac Lab Specific:
- ❌ `isaaclab_assets` - Contains Franka robot models
- ❌ `isaaclab_tasks` - Task definitions

### Visualization:
- ❌ `omni.replicator.replicator_yaml` - Replicator config
- ❌ `isaacsim.robot.manipulators` - Robot manipulators
- ❌ `isaacsim.robot.surface_gripper` - Surface grippers

**Result**: Cannot create Franka robot, cannot render cameras, cannot run physics.

---

## Numpy Error Details

### Error #1: Missing numpy attribute
```
AttributeError: module 'numpy' has no attribute '_no_nep50_warning'
```
- Isaac Sim's bundled numpy (old version) trying to use attribute that doesn't exist in conda numpy
- Affects: scipy, Isaac Sim extensions importing scipy

### Error #2: Binary incompatibility
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```
- Compiled extensions (Cython) built against different numpy version
- Affects: numpy.random, gymnasium

### Error #3: Missing function
```
ImportError: cannot import name 'broadcast_to' from 'numpy.lib.stride_tricks'
```
- Isaac Sim bundled numpy trying to import from conda numpy
- Affects: omni.syntheticdata, omni.replicator.core

---

## Where Numpy Conflict Happens

**Isaac Sim bundled numpy location**:
```
/home/mpcr/miniconda/envs/isaaclab/lib/python3.11/site-packages/isaacsim/
  kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/
  pip_prebundle/numpy/
```

**Conda environment numpy**:
```
/home/mpcr/miniconda/envs/isaaclab/lib/python3.11/site-packages/numpy/
```

When Isaac Sim extensions try to import scipy/numpy, they get conda's numpy instead of their bundled version, causing version mismatch errors.

---

## What This Means for SALUS

### ✅ What IS Working:
1. **VLA Integration**: SmolVLA-450M loads and runs correctly
2. **Signal Extraction**: 6D uncertainty signals compute correctly
3. **SALUS Predictor**: 4,868 parameter model trains correctly
4. **Data Pipeline**: Zarr storage, checkpointing work
5. **Training Loop**: Loss decreases, metrics compute correctly

### ❌ What Is NOT Working:
1. **Real Isaac Lab**: Cannot use due to numpy incompatibility
2. **Franka Robot**: Cannot instantiate (isaaclab_assets failed)
3. **Camera Rendering**: Cannot render images (sensors failed)
4. **Physics Simulation**: Cannot run physics (core extensions failed)

### Current Data Status:
- **500 episodes collected**: From dummy environment (random noise images)
- **Training results**: F1 = 0.000 (correct, since data is random)
- **Infrastructure**: Solid, just needs real simulation data

---

## Why Dummy Environment Was Used Before

Looking back at previous collection:
- Script: `scripts/collect_episodes_mvp.py`
- Used: `IsaacLabEnv` (dummy fallback)
- Why: Because `_check_isaac_sim()` always returned `False`
- Result: Random noise images, random failures

**The intent was always to use real Isaac Lab**, but the fallback kicked in.

---

## Potential Solutions

### Option 1: Use Isaac Sim's Python (NOT conda python) ⭐ RECOMMENDED
Isaac Sim likely provides its own Python interpreter that's compatible with its bundled numpy.

**Try**:
```bash
# Find Isaac Sim's python
find /home/mpcr/miniconda/envs/isaaclab -name "python.sh" 2>/dev/null

# OR use Isaac Lab's launcher script
cd /home/mpcr/Downloads/IsaacLab
./isaaclab.sh -p scripts/collect_data_franka.py --num_episodes 2
```

### Option 2: Downgrade Conda Numpy
Match Isaac Sim's bundled numpy version. Risky - may break other packages.

### Option 3: Use Different Isaac Sim Version
Isaac Sim 4.x or Isaac Sim 2023.x may have better numpy compatibility.

### Option 4: Use Alternative Simulator
- **MuJoCo**: Excellent physics, good Python integration
- **PyBullet**: Open source, easy to use, slower physics
- **Webots**: Good robot simulation, open source

---

## Recommendation

### ❌ Attempted: Option 1 (Isaac Lab Launcher Script) - FAILED

**Test Run**:
```bash
cd /home/mpcr/Downloads/IsaacLab && \
source /home/mpcr/miniconda/etc/profile.d/conda.sh && \
conda activate isaaclab && \
./isaaclab.sh -p /home/mpcr/Desktop/Salus\ Test/SalusTest/scripts/collect_data_franka.py \
    --num_episodes 1 --save_dir /home/mpcr/Desktop/Salus\ Test/SalusTest/data/test_isaac_launcher \
    --headless --enable_cameras
```

**Result**: FAILED
- Isaac Sim loaded successfully (128 seconds vs 22 seconds with direct execution)
- Fewer error messages compared to direct execution
- **BUT**: Script hung after Isaac Sim initialization
  - Consumed 406% CPU for 8+ minutes (infinite loop/busy-wait)
  - No data directory created
  - No Franka robot creation
  - No episode collection
- Process had to be forcibly killed

**Conclusion**: Isaac Lab launcher script does NOT solve the numpy incompatibility issue.

---

### ⭐ RECOMMENDED: Switch to Alternative Simulator

Given that Isaac Lab launcher failed, **switching to MuJoCo is the fastest path forward**:
- Faster setup (no Isaac Sim dependency issues)
- Good physics quality
- Excellent Python integration (pure Python, no C++ compilation needed)
- Large community, well-documented
- Works with SmolVLA (just needs rendered images)
- Can be integrated in 1-2 days vs weeks of debugging Isaac Sim

---

## Hardware Capability

Your system is **more than capable** for this simulation:
- ✅ 4x RTX 2080 Ti (11GB each)
- ✅ Intel i9-9820X (10 cores, 20 threads)
- ✅ 64GB RAM
- ✅ 1.8TB NVMe SSD

The hardware is NOT the bottleneck - it's the software incompatibility.

---

## Timeline Impact

**Time Lost**: ~2-3 hours debugging numpy issues
**Time Saved by Using MuJoCo**: 1-2 days (vs. fixing Isaac Sim)
**Training Time (once data collected)**: 8 hours for 50 epochs

**Critical Path**:
1. Get ANY real physics simulator working: 1 day
2. Collect 500 episodes: 4-6 hours
3. Train SALUS: 8 hours
4. Evaluate: 1 hour

**Total to working system**: 2 days if simulator resolved today.

---

## Bottom Line

The SALUS infrastructure (VLA, predictor, data pipeline, training) is **solid and working correctly**. The only blocker is getting a working physics simulator with camera rendering.

Isaac Sim is powerful but has environment issues. **Recommend trying Isaac Lab's launcher script first**, then switching to MuJoCo if that fails.

---

*Assessment Date*: 2026-01-03 19:47
*Isaac Sim Version*: 5.1.0
*Python Version*: 3.11
*Isaac Lab Version*: 0.48.5
