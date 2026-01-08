#!/usr/bin/env python3
"""
SALUS + Isaac Sim Integration

Connects SALUS to Isaac Sim for realistic robot visualization while
the terminal GUI shows all metrics in real-time.

Setup:
    1. Install Isaac Sim 2023.1+
    2. Source Isaac Sim: source ~/.local/share/ov/pkg/isaac_sim-*/setup_python_env.sh
    3. Run: python salus_isaac_sim.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Isaac Sim imports (will be available after sourcing setup_python_env.sh)
try:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.franka import Franka
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.isaac.core.utils.numpy.rotations as rot_utils
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    print("Warning: Isaac Sim not available. Run in demo mode without visualization.")
    ISAAC_SIM_AVAILABLE = False

# SALUS imports
sys.path.append(str(Path(__file__).parent))
from salus_live_demo import SALUSWrapper, SALUSTerminalGUI, MockVLA
import threading
import torch

class IsaacSimEnvironment:
    """Isaac Sim environment with Franka Panda robot"""

    def __init__(self):
        # Create world
        self.world = World(stage_units_in_meters=1.0)

        # Add Franka Panda robot
        self.robot = self.world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                position=np.array([0, 0, 0])
            )
        )

        # Add table
        self._add_table()

        # Add target objects
        self._add_objects()

        # Reset world
        self.world.reset()

        print("✓ Isaac Sim environment initialized")

    def _add_table(self):
        """Add table to scene"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Warning: Could not find Isaac Sim assets")
            return

        table_path = assets_root_path + "/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        add_reference_to_stage(usd_path=table_path, prim_path="/World/Table")

    def _add_objects(self):
        """Add manipulation objects (mugs, blocks, etc.)"""
        from omni.isaac.core.objects import DynamicCuboid

        # Add a simple cube as manipulation target
        self.cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="target_cube",
                position=np.array([0.5, 0.0, 0.05]),
                size=0.05,
                color=np.array([1.0, 0.0, 0.0])
            )
        )

    def get_observation(self):
        """Get current robot observation"""
        # Get robot joint positions
        joint_positions = self.robot.get_joint_positions()

        # Get end-effector pose
        ee_position, ee_orientation = self.robot.end_effector.get_world_pose()

        # Mock camera image (in real system, get from Isaac Sim camera)
        camera_image = torch.randn(1, 3, 224, 224)

        return {
            'joint_positions': joint_positions,
            'ee_position': ee_position,
            'ee_orientation': ee_orientation,
            'camera_image': camera_image
        }

    def apply_action(self, action):
        """
        Apply VLA action to robot.

        Args:
            action: 7D action (joint velocities or delta joint positions)
        """
        if action.shape[0] == 7:
            # Apply as joint position targets
            current_pos = self.robot.get_joint_positions()
            target_pos = current_pos + action * 0.01  # Scale delta
            self.robot.set_joint_position_targets(target_pos)
        else:
            print(f"Warning: Unexpected action shape {action.shape}")

    def step(self):
        """Step simulation forward"""
        self.world.step(render=True)

    def reset(self):
        """Reset environment"""
        self.world.reset()

def run_isaac_sim_demo(model_path='salus_fixed_pipeline.pt'):
    """
    Run SALUS with Isaac Sim visualization + terminal GUI.
    """
    if not ISAAC_SIM_AVAILABLE:
        print("ERROR: Isaac Sim not available!")
        print("Please install Isaac Sim and source setup_python_env.sh")
        return

    print("\n" + "="*60)
    print("SALUS + Isaac Sim Integration")
    print("="*60 + "\n")

    # Initialize environment
    print("Initializing Isaac Sim environment...")
    env = IsaacSimEnvironment()

    # Initialize VLA (replace with real VLA)
    print("Initializing VLA model...")
    vla = MockVLA()

    # Wrap with SALUS
    print("Loading SALUS wrapper...")
    salus = SALUSWrapper(vla, model_path=model_path)

    # Initialize terminal GUI
    print("Starting terminal GUI...")
    gui = SALUSTerminalGUI()

    # Start GUI in separate thread
    gui_thread = threading.Thread(target=gui.run, daemon=True)
    gui_thread.start()

    print("\n✓ System ready! Isaac Sim GUI + Terminal GUI running\n")
    time.sleep(2)

    # Main control loop
    episode = 0
    step_count = 0
    episode_steps = 0

    try:
        while simulation_app.is_running():
            # Get observation from Isaac Sim
            obs = env.get_observation()

            # VLA prediction with SALUS monitoring
            action, signals, risk_scores, alert_result = salus.predict(
                obs['camera_image'],
                language_instruction="pick up the red cube"
            )

            # Check for intervention
            original_action = action.copy() if isinstance(action, np.ndarray) else action.cpu().numpy()
            if salus.should_intervene(alert_result):
                action = salus.apply_intervention(action, 'slow_mode')
                gui.total_interventions += 1

            # Apply action to robot in Isaac Sim
            action_np = action.cpu().numpy() if torch.is_tensor(action) else action
            env.apply_action(action_np)

            # Step simulation
            env.step()

            # Update terminal GUI
            gui.signal_queue.put({
                'signals': signals,
                'alert_state': alert_result['state'],
                'ema_risk': alert_result['risk_ema']
            })
            gui.prediction_queue.put(risk_scores)

            # Update stats
            step_count += 1
            episode_steps += 1
            gui.total_steps = step_count
            gui.total_alerts = salus.alert_count

            # Episode end condition (e.g., cube grasped or timeout)
            if episode_steps >= 200:
                episode += 1
                episode_steps = 0
                gui.episode_count = episode
                env.reset()

                # Check if episode was successful
                # (In real system: check if task completed)
                if np.random.rand() > 0.3:  # Mock success rate
                    pass
                else:
                    gui.failure_count += 1

            success_rate = 1.0 - (gui.failure_count / max(episode, 1))

            gui.stats_queue.put({
                'episode': episode,
                'total_steps': step_count,
                'alerts': salus.alert_count,
                'interventions': gui.total_interventions,
                'failures': gui.failure_count,
                'success_rate': success_rate
            })

            # Control loop timing (10Hz)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SALUS + Isaac Sim Demo")
    parser.add_argument('--model', type=str, default='salus_fixed_pipeline.pt',
                       help='Path to SALUS model checkpoint')

    args = parser.parse_args()

    run_isaac_sim_demo(model_path=args.model)
