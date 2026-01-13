"""
Automatic improvement loop for SALUS predictor.

Supports single-shot fine-tuning from a dataset or watching a directory
for new zarr datasets.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import zarr

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from salus.core.online_learning import OnlineImprovementConfig, OnlineImprovementManager
from salus.core.predictor import SALUSPredictor


def load_timesteps(zarr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    store = zarr.open(str(zarr_path), mode="r")

    signals = store["signals"][:]  # (N, T, 12)
    labels = store["horizon_labels"][:]  # (N, T, 16)
    actions = store["actions"][:]  # (N, T, action_dim)

    # Count valid episodes
    num_episodes = 0
    for i in range(signals.shape[0]):
        if actions[i].max() != 0:
            num_episodes += 1
        else:
            break

    signals = signals[:num_episodes]
    labels = labels[:num_episodes]
    actions = actions[:num_episodes]

    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

    masks = (np.abs(actions).sum(axis=-1) > 0)

    valid_signals: List[np.ndarray] = []
    valid_labels: List[np.ndarray] = []

    for ep in range(num_episodes):
        for t in range(signals.shape[1]):
            if masks[ep, t]:
                valid_signals.append(signals[ep, t])
                valid_labels.append(labels[ep, t])

    return np.array(valid_signals), np.array(valid_labels)


def improve_from_dataset(
    dataset_path: Path,
    manager: OnlineImprovementManager,
) -> float:
    signals, labels = load_timesteps(dataset_path)
    manager.add_samples(signals, labels)
    return manager.force_update()


def main() -> None:
    parser = argparse.ArgumentParser(description="SALUS Automatic Improvement")
    parser.add_argument("--checkpoint", type=str, default=None, help="Initial predictor checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/online", help="Checkpoint output dir")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--data_path", type=str, default=None, help="Single zarr dataset to ingest")
    parser.add_argument("--watch_dir", type=str, default=None, help="Watch directory for zarr datasets")
    parser.add_argument("--poll_seconds", type=float, default=10.0, help="Polling interval for watch mode")
    parser.add_argument("--min_samples", type=int, default=2000)
    parser.add_argument("--update_every", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if not args.data_path and not args.watch_dir:
        raise SystemExit("Provide --data_path or --watch_dir")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    predictor = SALUSPredictor().to(device)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            predictor.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    config = OnlineImprovementConfig(
        min_samples=args.min_samples,
        update_every=args.update_every,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=Path(args.output_dir),
    )
    manager = OnlineImprovementManager(predictor, device=device, config=config)

    if args.data_path:
        dataset_path = Path(args.data_path)
        loss = improve_from_dataset(dataset_path, manager)
        print(f"Update complete. Loss: {loss:.6f}")
        latest = manager.latest_checkpoint()
        if latest:
            print(f"Saved checkpoint: {latest}")
        return

    watch_dir = Path(args.watch_dir)
    seen = set()
    print(f"Watching {watch_dir} for new zarr datasets...")

    while True:
        for dataset in sorted(watch_dir.glob("*.zarr")):
            if dataset in seen:
                continue
            print(f"Ingesting: {dataset}")
            loss = improve_from_dataset(dataset, manager)
            print(f"Update complete. Loss: {loss:.6f}")
            latest = manager.latest_checkpoint()
            if latest:
                print(f"Saved checkpoint: {latest}")
            seen.add(dataset)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
