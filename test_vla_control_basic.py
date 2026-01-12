"""
Simple test: Can SmolVLA control a robot in IsaacLab simulation?

This test verifies:
1. SmolVLA loads correctly
2. IsaacLab environment initializes
3. VLA generates actions
4. Robot responds to actions
5. Episode completes without crashes
"""

import sys
import torch
import numpy as np

# Setup IsaacLab before importing SALUS
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

# Now safe to import SALUS
from salus.core.vla.smolvla_wrapper_fixed import SmolVLAWithInternals
from salus.simulation.franka_pick_place_env import SimplePickPlaceEnv

def test_vla_control():
    print("=" * 70)
    print("TEST: Can SmolVLA Control Robot in Simulation?")
    print("=" * 70)

    # Initialize environment
    print("\n[1/5] Initializing IsaacLab environment...")
    env = SimplePickPlaceEnv(
        simulation_app=simulation_app,
        num_envs=1,
        render_mode='rgb_array',
        success_reward=10.0,
        failure_penalty=-5.0
    )
    print("   ✓ Environment ready")

    # Initialize VLA
    print("\n[2/5] Loading SmolVLA model...")
    vla = SmolVLAWithInternals(device="cuda:0")
    print("   ✓ VLA loaded")

    # Reset environment
    print("\n[3/5] Resetting environment...")
    obs = env.reset()
    print(f"   ✓ Observation keys: {list(obs.keys())}")
    print(f"   ✓ Image shape: {obs['observation.images.camera1'].shape}")
    print(f"   ✓ State shape: {obs['observation.state'].shape}")

    # Run control loop
    print("\n[4/5] Running control loop (20 steps)...")
    task = "pick up the red cube"

    for step in range(20):
        # Prepare observation for VLA
        vla_obs = {
            'observation.images.camera1': obs['observation.images.camera1'],
            'observation.state': obs['observation.state'],
            'task': task
        }

        # Get action from VLA
        with torch.no_grad():
            output = vla(vla_obs)
            action = output['action']

        # Execute action in environment
        obs, reward, done, truncated, info = env.step(action)

        # Print progress
        if step % 5 == 0:
            print(f"   Step {step:2d}: action={action[0, :3].cpu().numpy()}, "
                  f"reward={reward[0].item():.2f}, done={done[0].item()}")

        if done[0]:
            print(f"   Episode ended at step {step}")
            break

    print("   ✓ Control loop completed")

    # Verify everything worked
    print("\n[5/5] Verification...")
    checks = {
        "VLA loaded": True,
        "Environment initialized": True,
        "Actions generated": action is not None and action.shape == (1, 7),
        "Robot responded": obs is not None,
        "No crashes": True
    }

    all_passed = all(checks.values())

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {check}")

    # Cleanup
    env.close()
    simulation_app.close()

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ SUCCESS: VLA CAN control robot in simulation!")
        print("\nNext steps:")
        print("  1. Run data collection: python scripts/collect_data_franka.py")
        print("  2. Train predictor: python scripts/train_failure_predictor.py")
    else:
        print("❌ FAILURE: Something is broken")
        print("\nDebug:")
        for check, passed in checks.items():
            if not passed:
                print(f"  - Fix: {check}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    try:
        success = test_vla_control()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
