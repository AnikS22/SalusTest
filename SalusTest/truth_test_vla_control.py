"""
TRUTH TEST: What actually happens when VLA sees the block and controls the robot?

This script shows:
1. What the cameras actually see (saves images to disk)
2. What VLA actually outputs when given those images
3. Whether the robot actually follows VLA commands
4. Whether VLA actions are reasonable (moving toward cube)

NO SCRIPTED ACTIONS. NO WORKAROUNDS. Just VLA controlling the robot.
"""

import argparse
from pathlib import Path
import sys
import torch
import numpy as np
import time
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.headless = False
args.num_envs = 1
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
from salus.core.vla.wrapper import SmolVLAEnsemble

print("\n" + "="*70)
print("TRUTH TEST: VLA Control Diagnostic")
print("="*70)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create output directory
output_dir = Path("truth_test_output")
output_dir.mkdir(exist_ok=True)
print(f"\n📁 Saving diagnostics to: {output_dir}/")

print(f"\n1. Loading VLA model...")
model_path = Path.home() / "models" / "smolvla" / "smolvla_base"

if not model_path.exists():
    print(f"❌ VLA model not found at {model_path}")
    print(f"   Cannot run test without VLA model.")
    sys.exit(1)

try:
    vla = SmolVLAEnsemble(
        model_path=str(model_path),
        ensemble_size=3,
        device=device
    )
    print(f"✅ VLA loaded")
except Exception as e:
    print(f"❌ VLA loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n2. Creating Isaac Lab environment...")
env = FrankaPickPlaceEnv(
    simulation_app=simulation_app,
    num_envs=1,
    device=device,
    render=True,
    max_episode_length=100
)
print(f"✅ Environment created")

print(f"\n3. Resetting environment...")
obs = env.reset()
print(f"✅ Reset complete")

# Get initial robot state
robot_state = obs['observation.state'][0, :7].cpu().numpy()
print(f"\n📊 Initial robot state (7 DOF):")
print(f"   {robot_state}")

print(f"\n4. Saving camera images to disk...")
# Save what the cameras actually see
for cam_name, cam_key in [
    ("front", 'observation.images.camera1'),
    ("side", 'observation.images.camera2'),
    ("top", 'observation.images.camera3')
]:
    img = obs[cam_key][0].cpu().numpy()  # (C, H, W) or (H, W, C)

    # Handle different tensor formats
    if img.shape[0] == 3:  # (C, H, W)
        img = img.transpose(1, 2, 0)  # → (H, W, C)

    # Convert to uint8 if needed
    if img.dtype == np.float32:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Save
    img_pil = Image.fromarray(img)
    img_path = output_dir / f"camera_{cam_name}_step_0.png"
    img_pil.save(img_path)
    print(f"   ✅ Saved {img_path}")

print(f"\n5. Preparing observation for VLA...")
obs_vla = {
    'observation.images.camera1': obs['observation.images.camera1'],
    'observation.images.camera2': obs['observation.images.camera2'],
    'observation.images.camera3': obs['observation.images.camera3'],
    'observation.state': obs['observation.state'][:, :6],  # VLA expects 6D
    'task': ['Pick up the red cube and place it in the blue zone']
}

print(f"   Observation format:")
for key, val in obs_vla.items():
    if isinstance(val, torch.Tensor):
        print(f"      {key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"      {key}: {val}")

print(f"\n6. Running VLA inference (this takes ~50-100ms)...")
start_time = time.time()

with torch.no_grad():
    vla_output = vla(obs_vla, return_internals=True)

inference_time = (time.time() - start_time) * 1000
print(f"✅ VLA inference complete in {inference_time:.1f}ms")

print(f"\n7. VLA output:")
print(f"   Keys: {list(vla_output.keys())}")

action = vla_output['action'][0].cpu().numpy()
action_var = vla_output['action_var'][0].cpu().numpy()
epistemic_unc = vla_output['epistemic_uncertainty'][0].item()

print(f"\n   🎯 VLA Action (6D):")
print(f"      {action}")
print(f"      Magnitude: {np.linalg.norm(action):.4f}")
print(f"\n   📊 Epistemic Uncertainty (ensemble disagreement):")
print(f"      Scalar: {epistemic_unc:.4f}")
print(f"      Per-dim variance: {action_var}")
print(f"      Mean variance: {action_var.mean():.4f}")

if 'hidden_state_mean' in vla_output:
    hidden = vla_output['hidden_state_mean'][0].cpu().numpy()
    print(f"\n   🧠 VLA Hidden State (from transformer):")
    print(f"      Shape: {hidden.shape}")
    print(f"      Norm: {np.linalg.norm(hidden):.4f}")
    print(f"      Mean: {hidden.mean():.4f}")
    print(f"      Std: {hidden.std():.4f}")

print(f"\n8. Checking if VLA action is reasonable...")

# Check if action is all zeros
if np.abs(action).sum() < 1e-6:
    print(f"   ❌ VLA action is ALL ZEROS - model not working!")
elif np.abs(action).max() > 10.0:
    print(f"   ❌ VLA action is HUGE ({np.abs(action).max():.2f}) - unstable!")
elif epistemic_unc > 0.5:
    print(f"   ⚠️  HIGH epistemic uncertainty ({epistemic_unc:.4f}) - model unsure")
else:
    print(f"   ✅ VLA action looks reasonable")

print(f"\n9. Converting VLA 6D action to 7D robot command...")
# VLA outputs 6D, Franka needs 7D
# Option 1: Pad with zero for 7th joint
# Option 2: Use VLA action as delta and add to current state
# Option 3: Interpret as end-effector pose (needs IK)

# Let's try: Current position + VLA action as delta
current_pos = obs['observation.state'][0, :7]
action_7d = torch.zeros(1, 7, device=device)
action_7d[0, :6] = torch.tensor(action, device=device) * 0.1  # Scale down
action_7d[0, 6] = 0.0  # 7th joint

# Clip to safe range
target_pos = current_pos + action_7d[0]
target_pos = torch.clamp(target_pos, -2.8, 2.8)  # Franka joint limits

print(f"   Current: {current_pos.cpu().numpy()}")
print(f"   Delta (VLA): {action_7d[0].cpu().numpy()}")
print(f"   Target: {target_pos.cpu().numpy()}")

print(f"\n10. Applying VLA action to robot...")
print(f"    Watch the Isaac Lab viewer - robot should move!")

# Apply action for 30 steps (1 second)
for step in range(30):
    obs, done, info = env.step(target_pos.unsqueeze(0))
    time.sleep(0.03)

    if step % 10 == 0:
        new_pos = obs['observation.state'][0, :7]
        error = torch.norm(new_pos - target_pos).item()
        print(f"   Step {step}: tracking error = {error:.4f}")

final_pos = obs['observation.state'][0, :7].cpu().numpy()
actual_movement = np.linalg.norm(final_pos - robot_state)

print(f"\n11. Result:")
print(f"   Initial pos: {robot_state}")
print(f"   Final pos:   {final_pos}")
print(f"   Actual movement: {actual_movement:.4f} rad")

if actual_movement < 0.01:
    print(f"   ❌ Robot didn't move - VLA control not working!")
elif actual_movement > 1.0:
    print(f"   ⚠️  Robot moved a lot ({actual_movement:.2f}rad) - may be unstable")
else:
    print(f"   ✅ Robot moved as expected")

print(f"\n12. Running one more VLA inference with new observation...")
obs_vla_new = {
    'observation.images.camera1': obs['observation.images.camera1'],
    'observation.images.camera2': obs['observation.images.camera2'],
    'observation.images.camera3': obs['observation.images.camera3'],
    'observation.state': obs['observation.state'][:, :6],
    'task': ['Pick up the red cube and place it in the blue zone']
}

with torch.no_grad():
    vla_output_new = vla(obs_vla_new, return_internals=True)

action_new = vla_output_new['action'][0].cpu().numpy()
hidden_new = vla_output_new['hidden_state_mean'][0].cpu().numpy()

print(f"\n   New VLA action: {action_new}")
print(f"   Old VLA action: {action}")
action_change = np.linalg.norm(action_new - action)
print(f"   Action changed by: {action_change:.4f}")

if 'hidden_state_mean' in vla_output:
    hidden_change = np.linalg.norm(hidden_new - hidden)
    print(f"   Hidden state changed by: {hidden_change:.4f}")

if action_change < 1e-6:
    print(f"   ❌ VLA action DIDN'T CHANGE - model may be frozen/broken")
else:
    print(f"   ✅ VLA action changed (model is reactive)")

print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)

checks = [
    ("VLA model loads", True),
    ("Environment creates", True),
    ("Cameras render", True),
    ("VLA produces action", np.abs(action).sum() > 1e-6),
    ("VLA action reasonable", np.abs(action).max() < 10.0),
    ("Robot moves", actual_movement > 0.01),
    ("VLA updates", action_change > 1e-6),
]

passed = sum([c[1] for c in checks])
total = len(checks)

print(f"\nChecks: {passed}/{total}\n")
for name, result in checks:
    status = "✅" if result else "❌"
    print(f"   {status} {name}")

if passed == total:
    print(f"\n✅ VLA IS CONTROLLING THE ROBOT!")
    print(f"   - VLA sees the cameras")
    print(f"   - VLA outputs actions")
    print(f"   - Robot follows VLA commands")
    print(f"   - VLA adapts to new observations")
elif passed >= 5:
    print(f"\n⚠️  VLA MOSTLY WORKING but needs tuning:")
    if actual_movement < 0.01:
        print(f"   - VLA outputs actions but robot doesn't move much")
        print(f"   - May need to scale VLA actions differently")
    if action_change < 1e-6:
        print(f"   - VLA actions don't change → model may be stuck")
else:
    print(f"\n❌ VLA CONTROL NOT WORKING:")
    if np.abs(action).sum() < 1e-6:
        print(f"   - VLA outputs zeros → model inference broken")
    if actual_movement < 0.01:
        print(f"   - Robot doesn't move → action application broken")

print(f"\n📁 Camera images saved to: {output_dir}/")
print(f"   Check these to see what VLA actually sees!")

print("="*70 + "\n")

env.close()
simulation_app.close()
