"""
GUI Demo: SmolVLA Controlling Franka Robot in Isaac Lab

This script visualizes SmolVLA controlling a Franka Panda robot in real-time.
You'll see:
- Robot moving based on VLA predictions
- Real-time 18D signal values printed to console
- Camera views showing what VLA sees
- Physics simulation in Isaac Lab viewer

Usage:
    python demo_smolvla_gui.py --enable_cameras
"""

# CRITICAL: Only minimal imports before AppLauncher
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Isaac Lab AppLauncher (MUST be before other imports)
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="GUI Demo of SmolVLA control")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to run")
parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps (seconds)")
args = parser.parse_args()

# Override to enable GUI viewer
args.headless = False
args.enable_cameras = True
args.num_envs = 1  # Single environment for visualization

# Create AppLauncher
print(f"\n{'='*70}")
print("🎥 SMOLVLA GUI DEMO")
print(f"{'='*70}")
print(f"\nLaunching Isaac Lab with GUI viewer...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"✅ Isaac Lab viewer opened!")

# NOW safe to import everything else
import torch
import numpy as np
import time
from salus.core.vla.wrapper import SmolVLAEnsemble, EnhancedSignalExtractor
from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv

print("✅ All modules imported")

def print_signals(signals, timestep):
    """Print 18D signals in a readable format."""
    s = signals[0].cpu().numpy()

    print(f"\n{'─'*70}")
    print(f"⏱️  Timestep {timestep:3d}")
    print(f"{'─'*70}")

    print("📊 BASIC UNCERTAINTY (Signals 1-12):")
    print(f"   1. Epistemic Uncertainty:  {s[0]:.4f}")
    print(f"   2. Action Magnitude:       {s[1]:.4f}")
    print(f"   3. Action Variance:        {s[2]:.4f}")
    print(f"   4. Action Smoothness:      {s[3]:.4f}")
    print(f"   5. Trajectory Divergence:  {s[4]:.4f}")
    print(f"   6-8. Per-Joint Variance:   [{s[5]:.4f}, {s[6]:.4f}, {s[7]:.4f}]")
    print(f"   9-12. Unc Stats:           Mean={s[8]:.4f}, Std={s[9]:.4f}, Min={s[10]:.4f}, Max={s[11]:.4f}")

    print("\n🧠 VLA INTERNALS (Signals 13-14):")
    print(f"   13. Latent Drift:          {s[12]:.4f}  {'⚠️ HIGH' if s[12] > 0.5 else '✅'}")
    print(f"   14. OOD Distance:          {s[13]:.4f}  {'⚠️ OOD' if s[13] > 1.0 else '✅'}")

    print("\n🔬 SENSITIVITY (Signals 15-16):")
    print(f"   15. Aug Stability:         {s[14]:.4f}")
    print(f"   16. Pert Sensitivity:      {s[15]:.4f}")

    print("\n⚙️  REALITY CHECK (Signals 17-18):")
    print(f"   17. Execution Mismatch:    {s[16]:.4f}  {'⚠️ DRIFT' if s[16] > 0.3 else '✅'}")
    print(f"   18. Constraint Margin:     {s[17]:.4f}  {'⚠️ UNSAFE' if s[17] > 0.5 else '✅'}")

    # Overall health
    risk_score = (s[0] + s[12] + s[13] + s[16] + s[17]) / 5.0
    if risk_score < 0.3:
        status = "🟢 LOW RISK"
    elif risk_score < 0.6:
        status = "🟡 MODERATE RISK"
    else:
        status = "🔴 HIGH RISK"

    print(f"\n🎯 OVERALL RISK: {risk_score:.3f} - {status}")


def main():
    print(f"\n{'='*70}")
    print("SETUP")
    print(f"{'='*70}")

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    # Load VLA model
    print(f"\n🤖 Loading SmolVLA ensemble (3 models)...")
    model_path = Path.home() / "models" / "smolvla" / "smolvla_base"

    if not model_path.exists():
        print(f"❌ VLA model not found at {model_path}")
        print(f"   Please download SmolVLA model first.")
        print(f"   See MODEL_DOWNLOAD_GUIDE.md for instructions.")
        return

    vla = SmolVLAEnsemble(
        model_path=str(model_path),
        ensemble_size=3,
        device=device
    )
    print(f"✅ VLA loaded! ({model_path})")

    # Signal extractor
    print(f"\n🔍 Initializing signal extractor...")
    signal_extractor = EnhancedSignalExtractor(device=device)
    print(f"✅ Signal extractor ready (18D signals)")

    # Create environment
    print(f"\n🏗️  Creating Franka environment...")
    env = FrankaPickPlaceEnv(
        simulation_app=simulation_app,
        num_envs=1,
        device=device,
        render=True,  # Enable visualization
        max_episode_length=args.max_steps
    )
    print(f"✅ Environment created!")

    print(f"\n{'='*70}")
    print("DEMO STARTING")
    print(f"{'='*70}")
    print(f"\n📹 Watch the Isaac Lab viewer window!")
    print(f"🎯 SmolVLA will control the robot for {args.num_episodes} episodes")
    print(f"📊 18D signals will be printed every step\n")

    # Run episodes
    for episode in range(args.num_episodes):
        print(f"\n{'#'*70}")
        print(f"🎬 EPISODE {episode + 1}/{args.num_episodes}")
        print(f"{'#'*70}")

        # Reset
        obs = env.reset()
        signal_extractor.reset()

        episode_reward = 0.0

        # Run episode
        for t in range(args.max_steps):
            # Prepare VLA observation
            obs_vla = {
                'observation.images.camera1': obs['observation.images.camera1'],
                'observation.images.camera2': obs['observation.images.camera2'],
                'observation.images.camera3': obs['observation.images.camera3'],
                'observation.state': obs['observation.state'][:, :6],  # 6D state
                'task': ['Pick and place the red cube']
            }

            # VLA forward pass
            with torch.no_grad():
                action_dict = vla(obs_vla, return_internals=True)

            action = action_dict['action']

            # Extract 18D signals
            robot_state = obs['observation.state'].to(device)
            signals = signal_extractor.extract(action_dict, robot_state=robot_state)

            # Print signals every 10 steps
            if t % 10 == 0:
                print_signals(signals, t)

            # Apply action to environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0].item()

            # Add delay for visualization
            time.sleep(args.delay)

            # Check if done
            if done[0]:
                success = info.get('success', [False])[0]
                status = "✅ SUCCESS" if success else "❌ FAILURE"
                print(f"\n🏁 Episode ended at step {t+1}: {status}")
                print(f"💰 Total reward: {episode_reward:.2f}")
                break
        else:
            print(f"\n⏱️  Episode ended (max steps reached)")
            print(f"💰 Total reward: {episode_reward:.2f}")

        # Pause between episodes
        if episode < args.num_episodes - 1:
            print(f"\n⏸️  Pausing for 2 seconds before next episode...")
            time.sleep(2)

    print(f"\n{'='*70}")
    print("🎉 DEMO COMPLETE")
    print(f"{'='*70}")
    print(f"\nYou saw SmolVLA controlling the robot with:")
    print(f"   • Real-time visual feedback in Isaac Lab")
    print(f"   • 18D signals extracted from VLA internals")
    print(f"   • Ensemble epistemic uncertainty")
    print(f"   • Hidden state monitoring")
    print(f"   • Perturbation sensitivity testing")
    print(f"\nAll signals are REAL from the 865MB SmolVLA model!")
    print(f"\n{'='*70}\n")

    # Clean shutdown
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        simulation_app.close()
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
