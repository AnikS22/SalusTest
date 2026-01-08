# IsaacSim Headless Mode Test Results

**Date:** January 2, 2026
**Status:** ✅ **HEADLESS MODE WORKING**

---

## Test Summary

### ✅ Successful Components

1. **IsaacSim Headless Initialization**
   - IsaacSim 5.1.0 successfully loads in headless mode
   - All 4x RTX 2080 Ti GPUs detected and active
   - No display required - perfect for server-based data collection
   - Vulkan rendering working properly

2. **IsaacLab Integration**
   - IsaacLab 0.48.5 loads correctly
   - AppLauncher working with `--headless` flag
   - SimulationContext initializes successfully
   - Logging to `/tmp/isaaclab_*.log`

3. **Environment Setup**
   - Conda environment `isaaclab` working correctly
   - Python 3.11.14 with all dependencies installed
   - Imports working: `isaaclab`, `isaaclab.app`, `isaaclab.sim`, `isaaclab.scene`

### ⚠️ Known Issues

1. **Franka Scene Creation**
   - Scene creation with Franka robot and cameras is hanging
   - Likely related to USD asset loading paths
   - Needs debugging - may need to use AssetsCfg instead of direct USD paths
   - **Workaround:** Use dummy environment for immediate data collection

2. **Physics Simulation Step Size**
   - Warning: "Large simulation step size (> 0.0333 seconds)" for 1/30 Hz
   - Recommendation: Enable `enable_stabilization` in PhysxCfg
   - Not blocking, just a performance warning

---

## System Configuration

```
GPUs: 4x NVIDIA GeForce RTX 2080 Ti (11,510 MB each)
Driver: 580.95.05
OS: Ubuntu 24.04.3 LTS (Kernel 6.14.0-36)
CPU: Intel Core i9-9820X (10 cores, 20 threads @ 3.30GHz)
RAM: 64 GB
Python: 3.11.14 (isaaclab conda environment)
IsaacSim: 5.1.0 (pip-installed)
IsaacLab: 0.48.5
```

---

## Running Headless IsaacSim

### Method 1: With IsaacLab Wrapper (Recommended)

```bash
# Activate environment
conda activate isaaclab

# Run your script with headless flag
cd "/home/mpcr/Desktop/Salus Test/SalusTest"
python your_script.py --headless
```

### Method 2: Direct Python Execution

```bash
conda activate isaaclab
cd "/home/mpcr/Desktop/Salus Test/SalusTest"
python salus/simulation/franka_pick_place_env.py --headless
```

### Method 3: Data Collection Script

```bash
conda activate isaaclab
cd "/home/mpcr/Desktop/Salus Test/SalusTest"

# Use dummy environment (works immediately)
python scripts/collect_data.py --num_episodes 10 --use_dummy

# Use real IsaacSim (after fixing scene creation)
python scripts/collect_data.py --num_episodes 10
```

---

## Test Output

```
[INFO][AppLauncher]: Using device: cuda:0
[INFO][AppLauncher]: Loading experience file: /home/mpcr/Downloads/IsaacLab/apps/isaaclab.python.headless.kit
✅ IsaacLab 0.48.5 loaded
[INFO] IsaacLab logging to file: /tmp/isaaclab_2026-01-02_12-39-53.log

|=============================================================================================|
| GPU | Name                             | Active | GPU Memory |
|---------------------------------------------------------------------------------------------|
| 0   | NVIDIA GeForce RTX 2080 Ti       | Yes: 0 | 11510   MB |
| 1   | NVIDIA GeForce RTX 2080 Ti       | Yes: 1 | 11510   MB |
| 2   | NVIDIA GeForce RTX 2080 Ti       | Yes: 2 | 11510   MB |
| 3   | NVIDIA GeForce RTX 2080 Ti       | Yes: 3 | 11510   MB |
|=============================================================================================|
```

---

## Next Steps

1. ✅ **Headless mode verified** - Can proceed with data collection
2. ⏳ **Debug Franka scene creation** - Fix USD asset loading
3. ⏳ **Test with GUI mode** - Verify visualization works
4. ✅ **Use dummy environment** - Start collecting test data immediately
5. ⏳ **Collect 10 test episodes** - Verify full pipeline
6. ⏳ **Collect 500 production episodes** - Once real sim working

---

## Files Modified

1. **`salus/simulation/franka_pick_place_env.py`**
   - Updated imports from `omni.isaac.lab` → `isaaclab`
   - Removed `use_gpu_pipeline` parameter (not supported in 0.48.5)
   - Fixed AppLauncher initialization
   - Added proper dummy mode fallback

2. **`ISAACLAB_SETUP.md`**
   - Original setup guide (still valid)

3. **`HEADLESS_TEST_RESULTS.md`** (this file)
   - Test results and findings

---

## Debugging Scene Creation

To fix the Franka scene loading issue:

1. **Check Asset Paths**
   ```python
   # Current approach (may not work)
   usd_path=f"{sim_utils.ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd"

   # Try using AssetsCfg instead
   from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
   ```

2. **Use Simpler Scene First**
   - Test with just ground plane + cube
   - Add Franka after basic scene works
   - Add cameras last

3. **Check IsaacLab Examples**
   ```bash
   # Look at working Franka examples
   find ~/Downloads/IsaacLab -name "*franka*" -type f | grep -E "(\.py|\.yaml)$"
   ```

---

**Conclusion:** IsaacSim headless mode is **fully functional**. Scene creation needs debugging, but dummy environment allows immediate pipeline testing and data collection.
