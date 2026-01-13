"""
Small-machine sanity checks for the SALUS pipeline.

This script exercises core components without requiring IsaacSim or GPUs by
using dummy tensors. Optional flags enable IsaacSim/VLA checks when available.
"""

import argparse
import importlib.util
from typing import Dict

import numpy as np
import torch

from salus.core.adaptation import AdaptationModule, FailureType
from salus.core.predictor import SALUSPredictor
from salus.core.vla.wrapper import SignalExtractor


def check_signal_extractor() -> None:
    extractor = SignalExtractor()

    action = torch.randn(2, 7)
    action_var = torch.abs(torch.randn(2, 7)) * 0.05
    epistemic = torch.abs(torch.randn(2))

    signals = extractor.extract({
        "action": action,
        "action_var": action_var,
        "epistemic_uncertainty": epistemic,
    })

    assert signals.shape == (2, 12), f"Unexpected signal shape: {signals.shape}"
    assert torch.isfinite(signals).all(), "Signals contain NaNs/Infs"
    print("✓ Signal extractor smoke test passed")


def check_predictor() -> None:
    predictor = SALUSPredictor()
    signals = torch.randn(4, 12)
    output = predictor(signals)

    assert output["logits"].shape == (4, 16)
    assert output["probs"].shape == (4, 4, 4)
    assert output["horizon_probs"].shape == (4, 4)
    assert output["max_prob"].shape == (4,)
    print("✓ Predictor forward pass test passed")

    preds = predictor.predict_failure(signals, threshold=0.1)
    assert preds["failure_predicted"].shape == (4,)
    print("✓ Predictor prediction test passed")


def check_adaptation() -> None:
    module = AdaptationModule()
    prediction: Dict[str, torch.Tensor] = {
        "failure_predicted": torch.tensor([True]),
        "failure_horizon": torch.tensor([0]),
        "failure_type": torch.tensor([FailureType.COLLISION.value]),
        "confidence": torch.tensor([0.95]),
    }

    decision = module.decide_intervention(prediction, current_step=1)
    assert decision.intervention.name == "EMERGENCY_STOP"
    print("✓ Adaptation decision test passed")


def check_isaaclab_env() -> None:
    if importlib.util.find_spec("isaaclab") is None:
        print("⚠️  IsaacLab not available; skipping IsaacSim check")
        return

    if importlib.util.find_spec("isaaclab.app") is None:
        print("⚠️  IsaacLab AppLauncher not available; skipping IsaacSim check")
        return

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args([])
    if not hasattr(args_cli, "enable_cameras"):
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv

    env = FrankaPickPlaceEnv(
        simulation_app=simulation_app,
        num_envs=1,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        render=False,
        max_episode_length=5,
    )

    obs = env.reset()
    action = torch.zeros(1, 7, device=env.device)
    obs, done, info = env.step(action)

    assert "success" in info
    assert "failure_type" in info

    env.close()
    simulation_app.close()
    print("✓ IsaacLab environment smoke test passed")


def check_vla_wrapper() -> None:
    if importlib.util.find_spec("lerobot") is None:
        print("⚠️  lerobot not available; skipping VLA wrapper check")
        return

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available; skipping VLA wrapper check")
        return

    from salus.core.vla.smolvla_wrapper import SmolVLAEnsemble

    vla = SmolVLAEnsemble(ensemble_size=1, device="cuda:0")
    dummy_obs = {
        "observation.images.camera1": torch.randn(1, 3, 256, 256, device=vla.device),
        "observation.images.camera2": torch.randn(1, 3, 256, 256, device=vla.device),
        "observation.images.camera3": torch.randn(1, 3, 256, 256, device=vla.device),
        "observation.state": torch.randn(1, 7, device=vla.device),
        "task": "pick up the red cube",
    }

    output = vla(dummy_obs)
    assert "action" in output
    assert "action_var" in output
    print("✓ VLA wrapper smoke test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="SALUS small-machine sanity checks")
    parser.add_argument("--with-isaac", action="store_true", help="Run IsaacSim environment check")
    parser.add_argument("--with-vla", action="store_true", help="Run SmolVLA wrapper check")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    check_signal_extractor()
    check_predictor()
    check_adaptation()

    if args.with_isaac:
        check_isaaclab_env()
    if args.with_vla:
        check_vla_wrapper()

    print("\n✅ All requested checks completed")


if __name__ == "__main__":
    main()
