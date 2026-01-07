# IsaacLab + IsaacSim Setup Guide

## Current Status

✅ **IsaacSim Installed**: `~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64`
✅ **IsaacLab Installed**: `~/Downloads/IsaacLab`
⚠️  **IsaacSim Requires**: Display or proper headless setup

## Quick Start (Using Dummy Environment)

**For immediate development** (no IsaacSim running required):

```bash
cd ~/Desktop/Salus\ Test/SalusTest
python scripts/collect_data.py --use_dummy --num_episodes 10
```

This uses a **dummy environment** that provides correct observation formats but random images.

---

## Running Real IsaacSim (When Ready)

### Option 1: Headless Mode (Recommended for Server)

IsaacSim can run without a display using EGL/headless rendering:

```bash
# Set environment variables for headless rendering
export OMNI_KIT_ALLOW_ROOT=1
export DISPLAY=

# Run IsaacLab script with headless flag
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/Salus\ Test/SalusTest/scripts/collect_data.py --headless
```

### Option 2: With Display (If GUI Available)

If you have a display or X11 forwarding:

```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/Salus\ Test/SalusTest/scripts/collect_data.py
```

### Option 3: Docker Container (Most Reliable)

Use IsaacLab's Docker container for isolated, reproducible environment:

```bash
cd ~/Downloads/IsaacLab
./docker/container.sh
# Inside container:
python /workspace/salus/scripts/collect_data.py
```

---

## Environment Files

### 1. **Dummy Environment** (Always Works)
- File: `salus/simulation/isaaclab_env.py`
- Use: Pipeline testing, development
- Provides: Correct observation format, random images

### 2. **Real Environment** (Requires IsaacSim)
- File: `salus/simulation/franka_pick_place_env.py`
- Use: Production data collection
- Provides: Real physics, Franka robot, 3 cameras

---

## Switching Between Dummy and Real

**In your data collection script:**

```python
# Dummy mode (default)
from salus.simulation.isaaclab_env import SimplePickPlaceEnv
env = SimplePickPlaceEnv(num_envs=4)

# Real mode (when IsaacSim is running)
from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
env = FrankaPickPlaceEnv(num_envs=4)
```

---

## Verifying IsaacSim Works

Test if IsaacSim can run:

```bash
cd ~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64
./python.sh -c "import omni; print('IsaacSim working!')"
```

If this succeeds, you're ready for real simulation!

---

## Troubleshooting

### Error: "No display found"
**Solution**: Use headless mode (Option 1 above)

### Error: "CUDA out of memory"
**Solution**: Reduce `num_envs` in config (try `num_envs=1`)

### Error: "Module not found"
**Solution**: Use IsaacLab's Python:
```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p your_script.py
```

---

## Next Steps

1. **Test with dummy environment** - Collect 10 episodes, verify pipeline
2. **Set up headless IsaacSim** - Follow Option 1 above
3. **Switch to real environment** - Update script to use `FrankaPickPlaceEnv`
4. **Collect full dataset** - 500 episodes with real physics

---

## Performance Notes

**Dummy Environment**:
- Speed: ~1000 FPS
- Use: Pipeline testing
- Data quality: Not suitable for final training

**Real IsaacSim**:
- Speed: ~30 FPS per environment × 4 envs = 120 FPS total
- Use: Production data collection
- Data quality: Production-ready

---

**Current Recommendation**: Start with dummy environment to test the full pipeline, then switch to real IsaacSim for actual data collection.
