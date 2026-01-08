"""
SALUS IsaacLab Failure Prediction Evaluation

Runs a small SmolVLA ensemble in IsaacLab, extracts signals, trains a
multi-horizon SALUS predictor, and reports accuracy without leakage
(episode-level split).
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from isaaclab.app import AppLauncher

from salus.core.predictor import SALUSPredictor
from salus.core.vla.wrapper import SignalExtractor, SmolVLAEnsemble
from salus.simulation.isaaclab_env import SimplePickPlaceEnv


@dataclass
class EpisodeBatch:
    signals: torch.Tensor  # (T, 12)
    labels: torch.Tensor   # (T, 4, 4)


def _create_app_launcher():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    return AppLauncher(args)


def _compute_horizon_labels(labels_list: List[Dict], horizons: List[int]) -> np.ndarray:
    T = len(labels_list)
    horizon_labels = np.zeros((T, len(horizons), 4), dtype=np.float32)

    failure_time = None
    failure_type = None

    for t, label in enumerate(labels_list):
        if not label["success"]:
            failure_time = t
            failure_type = int(label["failure_type"])
            break

    if failure_time is not None:
        for t in range(failure_time):
            steps_until_failure = failure_time - t
            for h_idx, horizon in enumerate(horizons):
                if steps_until_failure <= horizon:
                    horizon_labels[t, h_idx, failure_type] = 1.0

    return horizon_labels


def _collect_episode(
    env: SimplePickPlaceEnv,
    vla: SmolVLAEnsemble,
    signal_extractor: SignalExtractor,
    max_steps: int,
    noise_std: float
) -> EpisodeBatch:
    obs = env.reset()
    signal_extractor.reset()

    signals = []
    labels = []

    step = 0
    done = torch.zeros(1, dtype=torch.bool, device=env.device)

    while not done.any() and step < max_steps:
        obs_vla = {
            "observation.images.camera1": obs["observation.images.camera1"].float() / 255.0,
            "observation.images.camera2": obs["observation.images.camera2"].float() / 255.0,
            "observation.images.camera3": obs["observation.images.camera3"].float() / 255.0,
            "observation.state": obs["observation.state"],
            "task": obs["task"],
        }

        with torch.no_grad():
            vla_output = vla(obs_vla)
            action = vla_output["action"]

        if noise_std > 0.0:
            action = action + torch.randn_like(action) * noise_std

        next_obs, done, info = env.step(action)

        sig = signal_extractor.extract(vla_output)
        signals.append(sig[0].detach().cpu())
        labels.append(
            {
                "success": bool(info["success"][0].item()),
                "failure_type": int(info["failure_type"][0].item()),
            }
        )

        obs = next_obs
        step += 1

    horizons = [6, 10, 13, 16]
    horizon_labels = _compute_horizon_labels(labels, horizons)

    return EpisodeBatch(
        signals=torch.stack(signals, dim=0),
        labels=torch.from_numpy(horizon_labels),
    )


def _split_episodes(episodes: List[EpisodeBatch], test_ratio: float):
    num_episodes = len(episodes)
    test_count = max(1, int(num_episodes * test_ratio))
    train_count = num_episodes - test_count
    train_eps = episodes[:train_count]
    test_eps = episodes[train_count:]
    return train_eps, test_eps


def _flatten_batches(episodes: List[EpisodeBatch]) -> Tuple[torch.Tensor, torch.Tensor]:
    signals = torch.cat([ep.signals for ep in episodes], dim=0)
    labels = torch.cat([ep.labels for ep in episodes], dim=0)
    return signals, labels


def _train_predictor(
    model: SALUSPredictor,
    train_signals: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
) -> SALUSPredictor:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_signals = train_signals.to(device)
    train_labels = train_labels.to(device)
    train_labels = train_labels.view(train_labels.shape[0], -1)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_signals)
        logits = outputs["logits"]
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

    return model


def _evaluate_predictor(
    model: SALUSPredictor,
    test_signals: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    test_signals = test_signals.to(device)
    test_labels = test_labels.to(device).view(test_labels.shape[0], -1)

    with torch.no_grad():
        outputs = model(test_signals)
        probs = torch.sigmoid(outputs["logits"])

    preds = (probs > 0.5).float()
    correct = (preds == test_labels).float().mean().item()
    return {"mean_label_accuracy": correct}


def main():
    parser = argparse.ArgumentParser(description="SALUS failure prediction eval")
    parser.add_argument("--episodes", type=int, default=8, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=120, help="Max steps per episode")
    parser.add_argument("--ensemble_size", type=int, default=1, help="SmolVLA ensemble size")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Action noise std")
    parser.add_argument("--test_ratio", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    app_launcher = _create_app_launcher()
    simulation_app = app_launcher.app

    env = SimplePickPlaceEnv(
        simulation_app=simulation_app,
        num_envs=1,
        device=str(device),
        render=False,
        max_episode_length=args.max_steps,
    )

    vla = SmolVLAEnsemble(
        model_path="~/models/smolvla/smolvla_base",
        ensemble_size=args.ensemble_size,
        device=str(device),
    )
    signal_extractor = SignalExtractor()

    episodes = []
    for _ in range(args.episodes):
        episodes.append(
            _collect_episode(
                env,
                vla,
                signal_extractor,
                max_steps=args.max_steps,
                noise_std=args.noise_std,
            )
        )

    train_eps, test_eps = _split_episodes(episodes, args.test_ratio)
    train_signals, train_labels = _flatten_batches(train_eps)
    test_signals, test_labels = _flatten_batches(test_eps)

    model = SALUSPredictor()
    model = _train_predictor(
        model,
        train_signals,
        train_labels,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    metrics = _evaluate_predictor(model, test_signals, test_labels, device=device)
    print("Evaluation metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
