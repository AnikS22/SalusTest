"""
Minimal IsaacLab Environment for SALUS VLA Testing
Simple pick-place task with Franka Panda
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Check if IsaacLab is available (requires IsaacSim to be running)
# For now, we'll use a dummy environment that provides correct observation format
ISAAC_LAB_AVAILABLE = False

try:
    isaac_lab_path = Path.home() / "Downloads/IsaacLab/source/isaaclab"
    if isaac_lab_path.exists():
        print(f"✅ IsaacLab found at: {isaac_lab_path}")
        print(f"   ⚠️  Full IsaacLab requires IsaacSim to be running")
        print(f"   Using dummy environment for pipeline testing")
except Exception as e:
    print(f"   IsaacLab path check: {e}")


class SimplePickPlaceEnv:
    """
    Minimal wrapper for Franka pick-place task

    Provides observations compatible with SmolVLA:
    - 3 camera views (256x256 RGB)
    - Robot state (7-DOF joint positions)
    - Task instruction (string)
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cuda:0",
        render: bool = False
    ):
        """
        Initialize environment

        Args:
            num_envs: Number of parallel environments
            device: Device to run on
            render: Enable rendering (slower)
        """
        self.num_envs = num_envs
        self.device = device
        self.render_enabled = render

        print(f"\n🏗️  Initializing IsaacLab Environment...")
        print(f"   Environments: {num_envs}")
        print(f"   Device: {device}")
        print(f"   Render: {render}")

        # For now, create a placeholder
        # Real IsaacLab integration requires IsaacSim to be running
        self.is_sim_available = self._check_isaac_sim()

        if not self.is_sim_available:
            print("   ⚠️  IsaacSim not available - using dummy environment")
            print("   This is OK for testing the pipeline!")
            self._init_dummy_env()
        else:
            print("   ✅ IsaacSim available - initializing real environment")
            self._init_isaac_env()

    def _check_isaac_sim(self) -> bool:
        """Check if IsaacSim is available"""
        # For now, always use dummy environment
        # Real IsaacSim integration requires the simulator to be running
        return False

    def _init_dummy_env(self):
        """Initialize dummy environment for testing"""
        self.current_step = 0
        self.max_steps = 200

        # Dummy robot state
        self.robot_state = torch.zeros(self.num_envs, 7, device=self.device)

        # Dummy camera images (random noise)
        self.camera_images = {
            'camera1': torch.randint(0, 255, (self.num_envs, 3, 256, 256), device=self.device, dtype=torch.uint8),
            'camera2': torch.randint(0, 255, (self.num_envs, 3, 256, 256), device=self.device, dtype=torch.uint8),
            'camera3': torch.randint(0, 255, (self.num_envs, 3, 256, 256), device=self.device, dtype=torch.uint8),
        }

        print("   ✅ Dummy environment initialized")

    def _init_isaac_env(self):
        """Initialize real IsaacLab environment"""
        # TODO: Implement real IsaacLab environment setup
        # This would involve:
        # 1. Creating scene with Franka robot
        # 2. Adding objects (cube, target zone)
        # 3. Setting up cameras
        # 4. Configuring physics

        print("   ⚠️  Real IsaacLab environment not yet implemented")
        print("   Falling back to dummy environment")
        self._init_dummy_env()

    def reset(self):
        """
        Reset environment

        Returns:
            observation: Dict with format compatible with SmolVLA
        """
        self.current_step = 0

        # Reset robot to home position
        if not self.is_sim_available:
            self.robot_state = torch.zeros(self.num_envs, 7, device=self.device)

        return self._get_observation()

    def step(self, action: torch.Tensor):
        """
        Execute action

        Args:
            action: (num_envs, 7) joint position targets

        Returns:
            observation: Dict
            done: (num_envs,) bool tensor
            info: Dict with success/failure labels
        """
        self.current_step += 1

        # Apply action (in dummy mode, just update state)
        if not self.is_sim_available:
            self.robot_state = action.clone()

            # Generate new random images (simulating camera movement)
            for camera in self.camera_images:
                self.camera_images[camera] = torch.randint(
                    0, 255,
                    (self.num_envs, 3, 256, 256),
                    device=self.device,
                    dtype=torch.uint8
                )

        # Check termination
        done = self.current_step >= self.max_steps
        done_tensor = torch.full((self.num_envs,), done, device=self.device, dtype=torch.bool)

        # Dummy success/failure (50% success rate)
        success = torch.rand(self.num_envs, device=self.device) > 0.5 if done else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        info = {
            'success': success,
            'failure_type': torch.randint(0, 4, (self.num_envs,), device=self.device),  # Random failure type
            'episode_length': self.current_step
        }

        return self._get_observation(), done_tensor, info

    def _get_observation(self):
        """
        Get observation in SmolVLA format

        Returns:
            Dict with keys:
                - 'observation.images.camera1': (num_envs, 3, 256, 256)
                - 'observation.images.camera2': (num_envs, 3, 256, 256)
                - 'observation.images.camera3': (num_envs, 3, 256, 256)
                - 'observation.state': (num_envs, 7)
                - 'task': str
        """
        return {
            'observation.images.camera1': self.camera_images['camera1'],
            'observation.images.camera2': self.camera_images['camera2'],
            'observation.images.camera3': self.camera_images['camera3'],
            'observation.state': self.robot_state,
            'task': 'pick up the red cube and place it in the blue bin'
        }

    def close(self):
        """Clean up"""
        print("   ✅ Environment closed")


# Test the environment
if __name__ == "__main__":
    print("Testing IsaacLab Environment...\n")

    # Create environment
    env = SimplePickPlaceEnv(num_envs=2, device="cuda:0")

    # Reset
    print("\n📝 Resetting environment...")
    obs = env.reset()
    print(f"   Observation keys: {list(obs.keys())}")
    print(f"   Camera1 shape: {obs['observation.images.camera1'].shape}")
    print(f"   State shape: {obs['observation.state'].shape}")

    # Test a few steps
    print("\n🔄 Running 5 test steps...")
    for i in range(5):
        # Random action
        action = torch.randn(2, 7, device="cuda:0") * 0.1

        obs, done, info = env.step(action)
        print(f"   Step {i+1}: done={done.cpu().numpy()}")

    env.close()

    print("\n✅ Environment test passed!")
    print("\n💡 Next steps:")
    print("   1. Implement real IsaacLab environment (requires IsaacSim)")
    print("   2. Connect VLA model to environment")
    print("   3. Test full VLA → IsaacLab → Data Recording pipeline")
