"""
Real-time SALUS Demonstration with GUI
Shows VLA executing pick-and-place with live failure prediction overlay
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from isaaclab.app import AppLauncher

# Parse arguments first (before importing simulation modules)
parser = argparse.ArgumentParser(description="SALUS Real-time Demo")
parser.add_argument("--checkpoint", type=str,
                   default="/home/mpcr/Desktop/SalusV3/checkpoints/salus_predictor_massive.pth",
                   help="Path to trained SALUS model")
parser.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes to run")
parser.add_argument("--max_steps", type=int, default=120,
                   help="Max steps per episode")
parser.add_argument("--alert_threshold", type=float, default=0.7,
                   help="Failure probability threshold for alerts")

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force enable cameras for VLA
if not hasattr(args_cli, 'enable_cameras'):
    args_cli.enable_cameras = True

# Launch IsaacLab
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
args = args_cli  # Use the modified args

# Now import IsaacLab modules (after app launch)
from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
from salus.core.vla.smolvla_wrapper import SmolVLAEnsemble, SimpleSignalExtractor
from salus.core.predictor import SALUSPredictor


def print_banner():
    """Print demo banner"""
    print("\n" + "="*70)
    print("SALUS REAL-TIME FAILURE PREDICTION DEMO")
    print("="*70)
    print("\nControls:")
    print("  - Watch the robot perform pick-and-place tasks")
    print("  - Red text = High failure risk detected")
    print("  - Green text = Normal operation")
    print("  - Press Ctrl+C to stop\n")
    print("="*70 + "\n")


def format_prediction_display(failure_prob, alert_threshold, step, max_steps):
    """Format prediction display with color coding"""
    alert = failure_prob > alert_threshold

    if alert:
        status = f"\033[91m⚠ HIGH RISK\033[0m"  # Red
        bar_color = "\033[91m"  # Red
    else:
        status = f"\033[92m✓ NORMAL\033[0m"  # Green
        bar_color = "\033[92m"  # Green

    # Progress bar
    bar_length = 30
    filled = int(failure_prob * bar_length)
    bar = bar_color + "█" * filled + "\033[0m" + "░" * (bar_length - filled)

    return (
        f"\rStep {step:3d}/{max_steps} │ "
        f"Failure Prob: {failure_prob:.2%} │ "
        f"{bar} │ "
        f"Status: {status}"
    )


def run_episode(env, vla, signal_extractor, predictor, device,
                max_steps, alert_threshold, episode_num):
    """Run one episode with real-time predictions"""
    print(f"\n{'='*70}")
    print(f"EPISODE {episode_num}")
    print(f"{'='*70}\n")

    # Reset environment
    obs = env.reset()
    signal_extractor.reset()

    step = 0
    dones = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    episode_signals = []
    episode_predictions = []
    alerts = 0

    while not dones.any() and step < max_steps:
        # Prepare observation for VLA
        obs_vla = {
            "observation.images.camera1": obs["observation.images.camera1"].float() / 255.0,
            "observation.images.camera2": obs["observation.images.camera2"].float() / 255.0,
            "observation.images.camera3": obs["observation.images.camera3"].float() / 255.0,
            "observation.state": obs["observation.state"],
            "task": obs["task"],
        }

        # Get VLA action and extract signals
        with torch.no_grad():
            action, signals = vla.predict(
                image=torch.stack([
                    obs_vla["observation.images.camera1"][0],
                    obs_vla["observation.images.camera2"][0],
                    obs_vla["observation.images.camera3"][0]
                ], dim=0),  # (3, H, W, C)
                state=obs_vla["observation.state"][0],  # (state_dim,)
                instruction=obs_vla["task"][0]  # str
            )
        signals_cpu = signals[0].detach().cpu().numpy()
        episode_signals.append(signals_cpu)

        # Run SALUS predictor
        with torch.no_grad():
            signals_tensor = torch.FloatTensor(signals_cpu).unsqueeze(0).to(device)
            pred_output = predictor(signals_tensor)
            failure_prob = pred_output['max_prob'].item()

        episode_predictions.append(failure_prob)

        # Display real-time prediction
        display = format_prediction_display(
            failure_prob, alert_threshold, step + 1, max_steps
        )
        print(display, end='', flush=True)

        if failure_prob > alert_threshold:
            alerts += 1

        # Step environment
        obs, dones, infos = env.step(action)
        step += 1

    # Episode summary
    print("\n\n" + "-"*70)
    success = infos["success"][0].item()
    failure_type = infos["failure_type"][0].item()

    if success:
        print(f"Result: \033[92m✓ SUCCESS\033[0m")
    else:
        failure_names = ["collision", "drop", "miss", "timeout"]
        failure_name = failure_names[int(failure_type)] if int(failure_type) < 4 else "unknown"
        print(f"Result: \033[91m✗ FAILURE ({failure_name})\033[0m")

    print(f"Steps: {step}")
    print(f"Alerts triggered: {alerts}")
    print(f"Max failure probability: {max(episode_predictions):.2%}")
    print(f"Mean failure probability: {np.mean(episode_predictions):.2%}")
    print("-"*70)

    return {
        'success': success,
        'steps': step,
        'alerts': alerts,
        'max_prob': max(episode_predictions),
        'mean_prob': np.mean(episode_predictions),
        'signals': episode_signals,
        'predictions': episode_predictions
    }


def main():
    print_banner()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Alert threshold: {args.alert_threshold:.2%}\n")

    # Create environment with GUI
    print("Initializing IsaacLab environment...")
    env = FrankaPickPlaceEnv(
        simulation_app=simulation_app,
        num_envs=1,
        device=str(device),
        render=True,  # Enable GUI
        max_episode_length=args.max_steps,
    )
    print("✓ Environment ready\n")

    # Load VLA
    print("Loading SmolVLA model...")
    vla = SmolVLAEnsemble(
        model_path="lerobot/smolvla_base",  # Use HuggingFace ID
        ensemble_size=1,
        device=str(device),
    )
    signal_extractor = SimpleSignalExtractor()
    print("✓ VLA loaded\n")

    # Load trained SALUS predictor
    print("Loading trained SALUS predictor...")
    predictor = SALUSPredictor(
        signal_dim=12,
        hidden_dims=[128, 256, 128],
        num_horizons=4,
        num_failure_types=4,
        dropout=0.2
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        env.close()
        simulation_app.close()
        return

    predictor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictor.eval()
    print(f"✓ Predictor loaded from: {checkpoint_path}\n")

    # Run episodes
    results = []
    try:
        for ep in range(args.episodes):
            result = run_episode(
                env, vla, signal_extractor, predictor, device,
                args.max_steps, args.alert_threshold, ep + 1
            )
            results.append(result)

            # Brief pause between episodes
            import time
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")

    # Final summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    successes = sum(1 for r in results if r['success'])
    print(f"Episodes completed: {len(results)}/{args.episodes}")
    print(f"Successes: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"Total alerts: {sum(r['alerts'] for r in results)}")
    print(f"Avg alerts per episode: {np.mean([r['alerts'] for r in results]):.1f}")
    print(f"Avg max failure prob: {np.mean([r['max_prob'] for r in results]):.2%}")
    print("="*70 + "\n")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
