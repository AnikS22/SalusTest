"""
IsaacLab Environment Entry Point for SALUS

This module intentionally exposes only the real IsaacLab environment.
The previous dummy/fake environment has been removed to prevent running
non-physical simulations by accident.
"""

from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv


class SimplePickPlaceEnv(FrankaPickPlaceEnv):
    """
    Compatibility wrapper that maps the old SimplePickPlaceEnv name
    to the real IsaacLab Franka pick-place environment.
    """

    def __init__(
        self,
        simulation_app=None,
        num_envs: int = 1,
        device: str = "cuda:0",
        render: bool = False,
        max_episode_length: int = 200
    ):
        if simulation_app is None:
            raise ValueError(
                "simulation_app is required. Create an AppLauncher and pass "
                "app_launcher.app before instantiating SimplePickPlaceEnv."
            )
        super().__init__(
            simulation_app=simulation_app,
            num_envs=num_envs,
            device=device,
            render=render,
            max_episode_length=max_episode_length
        )
