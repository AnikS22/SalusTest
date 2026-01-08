"""
Real IsaacLab Environment for SALUS
Franka Panda pick-place task with 3-camera setup

NOTE: This module expects AppLauncher to be created BEFORE importing this module!
Do NOT create AppLauncher here - it must be created only once in the main script.
"""

import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Optional

# IsaacLab should be installed in the isaaclab conda environment
# AppLauncher must be created BEFORE importing this module!


class FrankaPickPlaceEnv:
    """
    Real IsaacLab environment with Franka Panda robot

    Features:
    - Franka Panda 7-DOF robot
    - 3 RGB cameras (front, side, top views)
    - Pick-place task (red cube → blue target zone)
    - Success/failure detection
    - Compatible with SmolVLA observation format
    """

    def __init__(
        self,
        simulation_app=None,
        num_envs: int = 4,
        device: str = "cuda:0",
        render: bool = False,
        max_episode_length: int = 200
    ):
        """
        Initialize IsaacLab environment

        Args:
            simulation_app: The SimulationApp instance (from AppLauncher.app)
            num_envs: Number of parallel environments
            device: CUDA device
            render: Enable visualization (slower)
            max_episode_length: Max steps per episode
        """
        self.num_envs = num_envs
        self.device = device
        self.render_enabled = render
        self.max_episode_length = max_episode_length

        print(f"\n🏗️  Initializing Real IsaacLab Environment...")
        print(f"   Environments: {num_envs}")
        print(f"   Device: {device}")
        print(f"   Max episode length: {max_episode_length}")

        try:
            # Import Isaac Lab modules (must be after AppLauncher is created)
            import isaaclab
            from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
            from isaaclab.sim import SimulationCfg, SimulationContext
            from isaaclab.assets import RigidObjectCfg
            from isaaclab.sensors import CameraCfg
            import isaaclab.sim as sim_utils
            from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

            print(f"   ✅ IsaacLab {isaaclab.__version__} loaded")

            # Store reference to simulation app
            self.simulation_app = simulation_app

            # Store imports for later use
            self.isaaclab = isaaclab
            self.sim_utils = sim_utils
            self.RigidObjectCfg = RigidObjectCfg
            self.CameraCfg = CameraCfg
            self.InteractiveScene = InteractiveScene
            self.InteractiveSceneCfg = InteractiveSceneCfg
            self.FRANKA_CFG = FRANKA_PANDA_HIGH_PD_CFG

            # Debug: Check FRANKA_CFG
            print(f"   DEBUG: Original FRANKA_CFG type = {type(FRANKA_PANDA_HIGH_PD_CFG)}")
            print(f"   DEBUG: Has actuators? {hasattr(FRANKA_PANDA_HIGH_PD_CFG, 'actuators')}")
            if hasattr(FRANKA_PANDA_HIGH_PD_CFG, 'actuators'):
                print(f"   DEBUG: actuators keys = {list(FRANKA_PANDA_HIGH_PD_CFG.actuators.keys())}")

            # Initialize simulation
            self._setup_simulation()

            # Create scene
            self._create_scene()

            # Initialize episode tracking
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int, device=device)
            self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=device)

            print("   ✅ Real IsaacLab environment ready!")

        except Exception as e:
            raise RuntimeError(
                "Failed to initialize IsaacLab. Ensure IsaacSim is running and "
                "AppLauncher was created before importing this module."
            ) from e

    def _setup_simulation(self):
        """Setup Isaac Sim simulation context"""
        from isaaclab.sim import SimulationCfg, SimulationContext

        sim_cfg = SimulationCfg(
            dt=1/30,  # 30 Hz control frequency
            device=self.device
        )

        self.sim = SimulationContext(sim_cfg)
        print("   ✅ Simulation context created")

    def _create_scene(self):
        """Create scene with robot, object, and cameras"""
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        # Store self reference for use in config class
        sim_utils = self.sim_utils
        RigidObjectCfg = self.RigidObjectCfg
        CameraCfg = self.CameraCfg
        FRANKA_CFG = self.FRANKA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

        # Debug: Check actuators
        print(f"   DEBUG: FRANKA_CFG type = {type(FRANKA_CFG)}")
        print(f"   DEBUG: FRANKA_CFG has actuators? {hasattr(FRANKA_CFG, 'actuators')}")
        if hasattr(FRANKA_CFG, 'actuators'):
            print(f"   DEBUG: actuators = {list(FRANKA_CFG.actuators.keys())}")

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            """Configuration for pick-place scene"""

            # Ground plane (using AssetBaseCfg with GroundPlaneCfg spawn)
            from isaaclab.assets import AssetBaseCfg
            ground = AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )

            # Franka Panda robot (using pre-configured settings)
            robot = FRANKA_CFG

            # Red cube (object to pick)
            cube = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.05, 0.05, 0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),  # Red
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5, 0.0, 0.05),  # On table
                ),
            )

            # Cameras (3 views for SmolVLA)
            camera_front = CameraCfg(
                prim_path="/World/envs/env_.*/CameraFront",
                update_period=1/30,
                height=256,
                width=256,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.7, 0.0, 0.5),
                    rot=(0.0, 0.0, 1.0, 0.0),  # Looking at robot
                ),
            )

            camera_side = CameraCfg(
                prim_path="/World/envs/env_.*/CameraSide",
                update_period=1/30,
                height=256,
                width=256,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.7, 0.5),
                    rot=(0.0, 0.0, 0.707, 0.707),  # Looking from side
                ),
            )

            camera_top = CameraCfg(
                prim_path="/World/envs/env_.*/CameraTop",
                update_period=1/30,
                height=256,
                width=256,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 1.2),
                    rot=(0.707, 0.0, 0.0, 0.707),  # Looking down
                ),
            )

        # Create scene
        print("   Creating scene configuration...")
        scene_cfg = SceneCfg(num_envs=self.num_envs, env_spacing=2.0)
        print("   Creating InteractiveScene...")
        self.scene = self.InteractiveScene(scene_cfg)

        print(f"   ✅ Scene created with {self.num_envs} parallel environments")

        # Note: sim.reset() will be called in the first reset() call, not here
        self._scene_initialized = False

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """
        Reset environment(s)

        Args:
            env_ids: Environment indices to reset (None = reset all)

        Returns:
            observation: Dict compatible with SmolVLA
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset episode tracking
        self.episode_length_buf[env_ids] = 0

        # Initialize simulation on first reset
        if not self._scene_initialized:
            print("   Initializing simulation (first reset)...")
            self.sim.reset()
            self._scene_initialized = True
            print("   Simulation initialized!")

        # Real IsaacLab reset
        # Reset robot to home pose (9 joints: 7 arm + 2 gripper fingers)
        home_pos = self.scene["robot"].data.default_joint_pos[env_ids].clone()
        self.scene["robot"].write_joint_state_to_sim(home_pos, torch.zeros_like(home_pos), env_ids=env_ids)

        # Reset cube position (randomize slightly)
        cube_pos = torch.tensor([[0.5, 0.0, 0.05]], device=self.device).repeat(len(env_ids), 1)
        cube_pos[:, :2] += torch.randn(len(env_ids), 2, device=self.device) * 0.05
        self.scene["cube"].write_root_pose_to_sim(
            torch.cat([cube_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(len(env_ids), 1)], dim=-1),
            env_ids=env_ids
        )

        return self._get_observation()

    def step(self, actions: torch.Tensor):
        """
        Execute actions

        Args:
            actions: (num_envs, 7) joint position targets

        Returns:
            observation: Dict
            dones: (num_envs,) bool tensor
            infos: Dict with episode info
        """
        # Increment episode length
        self.episode_length_buf += 1

        # Real IsaacLab step
        # Pad action from 7 DOF (arm) to 9 DOF (arm + 2 gripper fingers)
        if actions.shape[-1] == 7:
            # Add gripper commands (open gripper by default)
            gripper_cmd = torch.ones(actions.shape[0], 2, device=actions.device) * 0.04
            actions_full = torch.cat([actions, gripper_cmd], dim=-1)
        else:
            actions_full = actions

        self.scene["robot"].set_joint_position_target(actions_full)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

        # Check termination
        dones = self.episode_length_buf >= self.max_episode_length

        # Compute success/failure
        infos = self._compute_rewards_and_dones(dones)

        # Reset environments that are done
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        return self._get_observation(), dones, infos

    def _get_observation(self) -> Dict:
        """Get observation in SmolVLA format"""

        # Real IsaacLab observations
        # Get all joint positions (9 joints for Franka: 7 arm + 2 gripper)
        robot_state = self.scene["robot"].data.joint_pos

        # Get camera images from Isaac Lab (format: [batch, height, width, channels])
        cam1_rgb = self.scene["camera_front"].data.output["rgb"]
        cam2_rgb = self.scene["camera_side"].data.output["rgb"]
        cam3_rgb = self.scene["camera_top"].data.output["rgb"]

        # DEBUG: Print shapes
        if not hasattr(self, '_debug_printed'):
            print(f"   DEBUG: cam1_rgb shape = {cam1_rgb.shape}, dtype = {cam1_rgb.dtype}")
            print(f"   DEBUG: cam1_rgb min/max = {cam1_rgb.min()}/{cam1_rgb.max()}")
            self._debug_printed = True

        # Convert from Isaac Lab format [B, H, W, C] to VLA format [B, C, H, W]
        cam1_rgb = cam1_rgb.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        cam2_rgb = cam2_rgb.permute(0, 3, 1, 2)
        cam3_rgb = cam3_rgb.permute(0, 3, 1, 2)

        # Return only the first 7 joints (arm) for VLA compatibility
        return {
            'observation.images.camera1': cam1_rgb,
            'observation.images.camera2': cam2_rgb,
            'observation.images.camera3': cam3_rgb,
            'observation.state': robot_state[:, :7],  # Only arm joints for VLA
            'task': 'pick up the red cube and place it in the blue zone'
        }

    def _compute_rewards_and_dones(self, dones: torch.Tensor) -> Dict:
        """Compute success/failure for each environment"""

        # Real success detection
        cube_pos = self.scene["cube"].data.root_pos_w
        goal_pos = torch.tensor([0.3, 0.5, 0.2], device=self.device)

        # Success: cube within 5cm of goal
        dist_to_goal = torch.norm(cube_pos - goal_pos.unsqueeze(0), dim=-1)
        success = dist_to_goal < 0.05

        # Failure detection
        cube_fell = cube_pos[:, 2] < 0.01  # Below table
        timeout = dones

        failure_type = torch.where(
            cube_fell, 1,  # Drop
            torch.where(timeout, 2, 3)  # Timeout or other
        )

        return {
            'success': success,
            'failure_type': failure_type,
            'episode_length': self.episode_length_buf.clone()
        }

    def close(self):
        """Clean up"""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.close()
        print("   ✅ Environment closed")


# Test
if __name__ == "__main__":
    print("Testing Real IsaacLab Environment...\n")

    try:
        env = FrankaPickPlaceEnv(num_envs=2, device="cuda:0")

        print("\n📝 Running test episode...")
        obs = env.reset()
        print(f"   Observation keys: {list(obs.keys())}")

        for i in range(10):
            action = torch.randn(2, 7, device="cuda:0") * 0.1
            obs, done, info = env.step(action)
            if i % 5 == 0:
                print(f"   Step {i}: done={done.cpu().numpy()}")

        env.close()
        print("\n✅ Test passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
