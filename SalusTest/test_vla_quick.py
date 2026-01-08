"""
Quick VLA test - 5 steps with immediate output
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
from salus.core.vla.wrapper import SmolVLAEnsemble
from salus.core.vla.single_model_extractor import SingleModelSignalExtractor

def print_flush(msg):
    print(msg, flush=True)

print_flush("\n" + "="*70)
print_flush("QUICK VLA TEST - 5 STEPS")
print_flush("="*70)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print_flush("\nLoading VLA...")
vla = SmolVLAEnsemble(str(Path.home() / "models/smolvla/smolvla_base"), ensemble_size=1, device=device)
print_flush("✅ VLA loaded")

print_flush("\nCreating environment...")
env = FrankaPickPlaceEnv(simulation_app, 1, device, False, 100)
print_flush("✅ Environment ready")

print_flush("\nInitializing signal extractor...")
signal_extractor = SingleModelSignalExtractor(device)
print_flush("✅ Signal extractor ready")

print_flush("\nResetting...")
obs = env.reset()
signal_extractor.reset()
print_flush("✅ Reset complete\n")

for t in range(5):
    print_flush(f"\n{'='*70}")
    print_flush(f"STEP {t}")
    print_flush(f"{'='*70}")

    robot_state = obs['observation.state'][0, :7].cpu().numpy()
    print_flush(f"\nRobot state (7D): {robot_state}")

    obs_vla = {
        'observation.images.camera1': obs['observation.images.camera1'],
        'observation.images.camera2': obs['observation.images.camera2'],
        'observation.images.camera3': obs['observation.images.camera3'],
        'observation.state': obs['observation.state'][:, :6],
        'task': ['Pick up the red cube']
    }

    print_flush("\nRunning VLA inference...")
    with torch.no_grad():
        vla_output = vla(obs_vla, return_internals=True)

    action_6d = vla_output['action'][0].cpu().numpy()
    print_flush(f"\nVLA action (6D): {action_6d}")
    print_flush(f"Action magnitude: {np.linalg.norm(action_6d):.6f}")

    epistemic = vla_output['epistemic_uncertainty'][0].item()
    print_flush(f"Epistemic uncertainty: {epistemic:.6f}")

    if 'hidden_state_mean' in vla_output:
        hidden = vla_output['hidden_state_mean'][0]
        print_flush(f"Hidden state norm: {torch.norm(hidden).item():.4f}")

    print_flush("\nExtracting 12D signals...")
    signals = signal_extractor.extract(vla_output, robot_state=obs['observation.state'].to(device))
    s = signals[0].cpu().numpy()

    print_flush(f"\n12D SIGNALS:")
    names = ["Epistemic", "ActMag", "ActVar", "ActSmooth", "TrajDiv",
             "JointVar0", "JointVar1", "JointVar2", "UncMean", "UncStd",
             "UncMin", "UncMax", "LatentDrift", "OOD", "AugStab",
             "PertSens", "ExecMis", "ConstraintMar"]
    for i, (name, val) in enumerate(zip(names, s)):
        print_flush(f"   {i+1:2d}. {name:15s}: {val:8.5f}")

    print_flush("\nApplying action...")
    obs, done, info = env.step(torch.tensor([action_6d], device=device))

    new_state = obs['observation.state'][0, :7].cpu().numpy()
    movement = np.linalg.norm(new_state - robot_state)
    print_flush(f"Robot moved: {movement:.6f} rad")

print_flush("\n" + "="*70)
print_flush("TEST COMPLETE")
print_flush("="*70)

env.close()
simulation_app.close()
