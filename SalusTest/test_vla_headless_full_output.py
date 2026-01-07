"""
Headless VLA test with FULL output of everything:
- VLA actions (6D)
- Converted actions (7D)
- All 18D signals from extractor
- Robot state
- Epistemic uncertainty

Fast (headless) and complete diagnostic.
"""

import argparse
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# HEADLESS MODE
args.headless = True
args.enable_cameras = True
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
from salus.core.vla.wrapper import SmolVLAEnsemble, EnhancedSignalExtractor

print("\n" + "="*70)
print("HEADLESS VLA TEST - FULL OUTPUT")
print("="*70)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load VLA
print(f"\n[1/4] Loading VLA ensemble...")
model_path = Path.home() / "models" / "smolvla" / "smolvla_base"

if not model_path.exists():
    print(f"ERROR: VLA not found at {model_path}")
    sys.exit(1)

vla = SmolVLAEnsemble(
    model_path=str(model_path),
    ensemble_size=3,
    device=device
)
print(f"✅ VLA loaded (3× 865MB models)")

# Signal extractor
print(f"\n[2/4] Initializing signal extractor...")
signal_extractor = EnhancedSignalExtractor(device=device)
print(f"✅ Signal extractor ready (18D)")

# Environment
print(f"\n[3/4] Creating environment (headless)...")
env = FrankaPickPlaceEnv(
    simulation_app=simulation_app,
    num_envs=1,
    device=device,
    render=False,
    max_episode_length=100
)
print(f"✅ Environment created")

# Reset
print(f"\n[4/4] Resetting environment...")
obs = env.reset()
signal_extractor.reset()
print(f"✅ Reset complete")

print(f"\n{'='*70}")
print("RUNNING EPISODE - VLA CONTROL")
print(f"{'='*70}\n")

episode_reward = 0
success = False

for t in range(100):
    print(f"\n{'─'*70}")
    print(f"TIMESTEP {t}")
    print(f"{'─'*70}")

    # Current robot state
    robot_state_7d = obs['observation.state'][0, :7].cpu().numpy()
    print(f"\n🤖 Robot State (7D joint angles):")
    print(f"   {robot_state_7d}")
    print(f"   Magnitude: {np.linalg.norm(robot_state_7d):.4f}")

    # Prepare VLA observation
    obs_vla = {
        'observation.images.camera1': obs['observation.images.camera1'],
        'observation.images.camera2': obs['observation.images.camera2'],
        'observation.images.camera3': obs['observation.images.camera3'],
        'observation.state': obs['observation.state'][:, :6],  # VLA expects 6D
        'task': ['Pick up the red cube and place it in the blue zone']
    }

    # VLA inference
    print(f"\n🧠 Running VLA inference...")
    with torch.no_grad():
        vla_output = vla(obs_vla, return_internals=True)

    # VLA action (6D)
    action_6d = vla_output['action'][0].cpu().numpy()
    print(f"\n🎯 VLA Output (6D action):")
    print(f"   {action_6d}")
    print(f"   Magnitude: {np.linalg.norm(action_6d):.4f}")

    # Epistemic uncertainty
    epistemic_unc = vla_output['epistemic_uncertainty'][0].item()
    action_var = vla_output['action_var'][0].cpu().numpy()
    print(f"\n📊 VLA Uncertainty:")
    print(f"   Epistemic (scalar): {epistemic_unc:.6f}")
    print(f"   Per-dim variance: {action_var}")

    # Hidden states
    if 'hidden_state_mean' in vla_output:
        hidden = vla_output['hidden_state_mean'][0].cpu().numpy()
        print(f"\n🧠 VLA Hidden State:")
        print(f"   Shape: {hidden.shape}")
        print(f"   Norm: {np.linalg.norm(hidden):.4f}")
        print(f"   Mean: {hidden.mean():.4f}, Std: {hidden.std():.4f}")

    # Extract 18D signals
    print(f"\n🔍 Extracting 18D signals...")
    robot_state_full = obs['observation.state'].to(device)
    signals = signal_extractor.extract(vla_output, robot_state=robot_state_full)
    signals_np = signals[0].cpu().numpy()

    print(f"\n📊 18D SIGNALS:")
    signal_names = [
        "1. Epistemic Uncertainty",
        "2. Action Magnitude",
        "3. Action Variance",
        "4. Action Smoothness",
        "5. Trajectory Divergence",
        "6. Per-Joint Var (joint 0)",
        "7. Per-Joint Var (joint 1)",
        "8. Per-Joint Var (joint 2)",
        "9. Unc Mean",
        "10. Unc Std",
        "11. Unc Min",
        "12. Unc Max",
        "13. Latent Drift (VLA hidden)",
        "14. OOD Distance",
        "15. Aug Stability",
        "16. Pert Sensitivity",
        "17. Execution Mismatch",
        "18. Constraint Margin",
    ]

    for i, (name, value) in enumerate(zip(signal_names, signals_np)):
        status = ""
        if i == 12 and value > 0.5:
            status = " ⚠️ HIGH"
        elif i == 13 and value > 1.0:
            status = " ⚠️ OOD"
        elif i == 15 and value > 0.5:
            status = " ⚠️ UNSTABLE"
        elif i == 16 and value > 0.3:
            status = " ⚠️ DRIFT"
        elif i == 17 and value > 0.5:
            status = " ⚠️ UNSAFE"

        print(f"   {name:30s}: {value:8.5f}{status}")

    # Apply action (environment converts 6D → 7D)
    print(f"\n⚙️  Applying VLA action (6D → 7D conversion)...")
    action_6d_tensor = torch.tensor([action_6d], device=device, dtype=torch.float32)

    obs, done, info = env.step(action_6d_tensor)

    # Show converted action (what robot actually received)
    new_robot_state = obs['observation.state'][0, :7].cpu().numpy()
    actual_movement = np.linalg.norm(new_robot_state - robot_state_7d)

    print(f"   Actual movement: {actual_movement:.4f} rad")

    # Reward
    reward = info.get('reward', torch.tensor([0.0]))[0].item()
    episode_reward += reward
    print(f"\n💰 Reward: {reward:.4f} (total: {episode_reward:.4f})")

    # Check success/done
    if done[0]:
        success = info.get('success', [False])[0]
        print(f"\n🏁 Episode ended: {'✅ SUCCESS' if success else '❌ FAILURE'}")
        break

    # Print summary every 10 steps
    if t > 0 and t % 10 == 9:
        print(f"\n{'='*70}")
        print(f"SUMMARY (Steps 0-{t})")
        print(f"{'='*70}")
        print(f"Total reward: {episode_reward:.4f}")
        print(f"Avg epistemic unc: {epistemic_unc:.4f}")
        print(f"Avg movement: {actual_movement:.4f}")

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Episode length: {t+1} steps")
print(f"Total reward: {episode_reward:.4f}")
print(f"Success: {'✅' if success else '❌'}")

# Check if VLA is working
if np.abs(action_6d).sum() < 1e-6:
    print(f"\n❌ VLA OUTPUTS ZEROS - Model inference broken!")
elif actual_movement < 0.001:
    print(f"\n❌ ROBOT DOESN'T MOVE - Action application broken!")
elif signals_np.sum() < 0.01:
    print(f"\n❌ SIGNALS ARE ZEROS - Extraction broken!")
else:
    print(f"\n✅ VLA IS WORKING:")
    print(f"   - VLA outputs non-zero actions")
    print(f"   - Robot moves in response")
    print(f"   - Signals extracted from VLA internals")
    if success:
        print(f"   - Task completed successfully!")

print(f"\n{'='*70}\n")

env.close()
simulation_app.close()
