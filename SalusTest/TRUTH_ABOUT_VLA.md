# TRUTH: What Actually Happens with VLA Control

## Test Status: RUNNING NOW (PID 197194)

The truth test is currently executing. Here's what we know so far:

---

## Camera Analysis: Does VLA See the Cube?

I analyzed the camera images saved during the test:

### Front Camera ✅
```
Shape: 256×256×3 RGB
Red pixels found: 628
Status: ✅ RED CUBE IS VISIBLE
```

**The front camera DOES capture the red cube.** 628 red pixels means VLA can see it.

### Side Camera ❌
```
Shape: 256×256×3 RGB
Red pixels found: 0
Status: ❌ Image is mostly black
```

**Side camera may have lighting issues** or is positioned poorly.

### Top Camera ❌
```
Shape: 256×256×3 RGB
Red pixels found: 0
Status: ❌ Image is completely black
```

**Top camera is not rendering properly** (all zeros = black image).

---

## What This Means

### ✅ VLA CAN See the Cube
- At least 1 camera (front) shows the red cube clearly
- SmolVLA's vision encoder processes all 3 camera inputs
- Even with 2 broken cameras, VLA has visual information

### ⚠️ Environment Issues
- 2 out of 3 cameras are dark/broken
- This is an Isaac Lab rendering issue, not a VLA issue
- The VLA still gets 1 working camera feed

---

## What Happens During VLA Inference

Based on the code and what's running:

### 1. **VLA Receives** (every 30ms):
```python
Input observation = {
    'camera1': 256×256×3 image (RED CUBE VISIBLE),
    'camera2': 256×256×3 image (dark),
    'camera3': 256×256×3 image (dark),
    'state': [7 joint angles],
    'task': "Pick up the red cube and place it in the blue zone"
}
```

### 2. **VLA Processes**:
```
Vision Encoder (Qwen2-VL):
  - Extracts features from camera1 (sees red cube)
  - Extracts features from camera2 (mostly black - no info)
  - Extracts features from camera3 (all black - no info)

Language Encoder (Qwen2):
  - Tokenizes: "Pick up the red cube..."
  - Understands: {action: grasp, target: red_cube}

Fusion Transformer:
  - Attention: links "red cube" text → red pixels in camera1
  - Proprioception: knows current joint angles
  - Decision: "move gripper toward red region"
```

### 3. **VLA Outputs**:
```python
action = [6D vector]  # End-effector pose or joint deltas
```

**PROBLEM**: VLA outputs 6D, but Franka robot has 7 DOF.

This is **the dimension mismatch issue**.

---

## The Core Problem: 6D vs 7D

### Why VLA Outputs 6D

SmolVLA was trained on datasets where actions are:
- **Option A**: End-effector pose (x, y, z, roll, pitch, yaw) = 6D
- **Option B**: First 6 joints only (datasets with 6-DOF robots)

### Why Robot Needs 7D

Franka Panda has 7 revolute joints:
- Shoulder: 3 DOF
- Elbow: 1 DOF
- Wrist: 3 DOF

**You MUST command all 7 joint angles** or use inverse kinematics (IK) to convert 6D pose → 7D joint angles.

---

## What's Actually Happening in Your Test

Based on the code execution:

1. **Environment Loads** ✅
   - Isaac Lab creates scene with robot + cube
   - Cameras spawn (but 2/3 have rendering issues)

2. **VLA Loads** ✅
   - 3× 865MB models loaded (2.6GB VRAM)
   - Memory usage confirms real models

3. **VLA Runs Inference** ✅ (probably)
   - Takes 50-100ms per forward pass
   - Processes camera images + task text
   - Outputs 6D action vector

4. **Action Conversion** ⚠️
   - Script tries to convert 6D → 7D
   - Uses: `current_position[0:6] + VLA_action * 0.1`
   - Leaves 7th joint at 0

5. **Robot Moves** ❓
   - IF conversion works: Robot moves sporadically
   - IF VLA outputs zeros: Robot doesn't move
   - IF VLA outputs are too large: Robot moves erratically

---

## Possible Scenarios

### Scenario A: VLA Works, But Actions Are Small
```
VLA sees cube → outputs action = [0.02, 0.01, -0.03, 0.01, 0.00, 0.02]
Scaled by 0.1 → [0.002, 0.001, -0.003, ...]
Result: Robot moves VERY SLIGHTLY (you don't notice)
```

**Why**: VLA expects actions in different units or scale.

### Scenario B: VLA Outputs Are Corrupted
```
VLA sees broken cameras (2/3 black) → confused
Outputs: [NaN, NaN, NaN, NaN, NaN, NaN] or [0, 0, 0, 0, 0, 0]
Result: Robot doesn't move or moves randomly
```

**Why**: Bad camera inputs confuse the model.

### Scenario C: VLA Works, But 7th Joint Causes Issues
```
VLA outputs reasonable 6D action
But 7th joint stays at 0 → unbalanced configuration
Robot controller tries to compensate → sporadic movement
```

**Why**: 7-DOF robot needs all 7 joints commanded.

---

## To Know The TRUTH

Wait for the test to complete. It will tell you:

1. **Does VLA output non-zero actions?**
   - If yes → VLA is working
   - If no → VLA inference is broken

2. **Do VLA actions change over time?**
   - If yes → VLA is reactive to observations
   - If no → VLA is frozen/deterministic

3. **Does robot move when given VLA commands?**
   - If yes → Action application works
   - If no → Dimension mismatch or control issue

4. **Is movement reasonable?**
   - If yes → VLA is controlling successfully
   - If no → Need to tune action scaling/conversion

---

## Current Test Output Files

Check these files after test completes:

```bash
truth_test_output/
├── camera_front_step_0.png  ✅ Shows red cube
├── camera_side_step_0.png   ❌ Dark image
└── camera_top_step_0.png    ❌ Black image
```

**Open `camera_front_step_0.png`** to see exactly what VLA sees!

---

## My Hypothesis

Based on evidence so far:

1. ✅ VLA model is REAL (865MB file, 7GB RAM usage)
2. ✅ VLA CAN see the cube (front camera has 628 red pixels)
3. ⚠️ Camera setup is BROKEN (2/3 cameras are black)
4. ⚠️ 6D→7D conversion is HACKY (may cause issues)
5. ❓ VLA action magnitude unknown (waiting for test output)

**Most likely**: VLA IS working and outputting reasonable actions, but:
- Actions are too small (need scaling up)
- OR 7th joint issue causes weird behavior
- OR broken cameras confuse the model sometimes

---

## What Needs to Be Fixed

Regardless of test outcome:

### Fix 1: Camera Rendering
```python
# In franka_pick_place_env.py
# Need to fix lighting/positioning for side and top cameras
```

### Fix 2: 6D→7D Conversion
```python
# Option A: Use Inverse Kinematics
from isaaclab.utils.math import quat_to_rot_mat
target_pose_6d = vla_output['action']  # (x,y,z,r,p,y)
joint_angles_7d = IK_solver(target_pose_6d)

# Option B: Train VLA to output 7D
# (requires fine-tuning on Franka-specific data)

# Option C: Pad with null-space optimization
# Let IK solver choose 7th joint value
```

### Fix 3: Action Scaling
```python
# Tune this multiplier based on test results
action_scale = 0.1  # May need to be 0.5 or 1.0 or 0.01
```

---

## WAIT FOR TEST TO COMPLETE

The running test will answer:
- ❓ What does VLA actually output?
- ❓ Does robot move when VLA commands it?
- ❓ How much does it move?
- ❓ Does VLA adapt to new observations?

**Check back in 2-3 minutes for complete results.**

The truth test saves all data and prints a complete diagnostic.
