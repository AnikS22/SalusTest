"""
Data Collection Logger for SALUS Paper
Tracks episode outcomes, statistics, and metadata during data collection.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from typing import Dict, List, Any


class PaperDataLogger:
    """Logger for tracking data collection metrics for paper analysis."""

    def __init__(self, save_dir: Path, dataset_type: str = "training"):
        """
        Initialize the paper data logger.

        Args:
            save_dir: Directory to save logs and statistics
            dataset_type: Type of dataset (training/validation/test)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_type = dataset_type
        self.start_time = datetime.now()

        # Episode tracking
        self.episodes: List[Dict[str, Any]] = []
        self.current_episode = 0

        # Statistics
        self.success_count = 0
        self.failure_count = 0
        self.failure_types = {"drop": 0, "timeout": 0, "collision": 0, "other": 0}

        # CSV log file
        self.csv_path = self.save_dir / f"{dataset_type}_episodes.csv"
        self._init_csv()

        # Metadata file
        self.metadata = {
            "dataset_type": dataset_type,
            "start_time": self.start_time.isoformat(),
            "system_info": self._get_system_info()
        }

    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_id",
                "timestamp",
                "success",
                "failure_type",
                "episode_length",
                "episode_duration_sec",
                "final_cube_distance",
                "data_file"
            ])

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        return info

    def start_episode(self, episode_id: int):
        """Mark the start of a new episode."""
        self.current_episode = episode_id
        self.episode_start_time = datetime.now()

    def log_episode(
        self,
        episode_id: int,
        success: bool,
        failure_type: str,
        episode_length: int,
        final_cube_distance: float,
        data_file: str
    ):
        """
        Log a completed episode.

        Args:
            episode_id: Episode number
            success: Whether episode succeeded
            failure_type: Type of failure (if any): drop/timeout/collision/other
            episode_length: Number of timesteps
            final_cube_distance: Final distance from cube to goal (meters)
            data_file: Path to zarr data file
        """
        episode_duration = (datetime.now() - self.episode_start_time).total_seconds()
        timestamp = datetime.now().isoformat()

        # Update statistics
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            self.failure_types[failure_type] = self.failure_types.get(failure_type, 0) + 1

        # Store episode data
        episode_data = {
            "episode_id": episode_id,
            "timestamp": timestamp,
            "success": success,
            "failure_type": failure_type,
            "episode_length": episode_length,
            "episode_duration_sec": episode_duration,
            "final_cube_distance": final_cube_distance,
            "data_file": str(data_file)
        }
        self.episodes.append(episode_data)

        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_id,
                timestamp,
                success,
                failure_type,
                episode_length,
                f"{episode_duration:.2f}",
                f"{final_cube_distance:.4f}",
                data_file
            ])

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        total_episodes = self.success_count + self.failure_count

        stats = {
            "total_episodes": total_episodes,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / total_episodes if total_episodes > 0 else 0,
            "failure_rate": self.failure_count / total_episodes if total_episodes > 0 else 0,
            "failure_types": self.failure_types.copy(),
            "avg_episode_length": np.mean([e["episode_length"] for e in self.episodes]) if self.episodes else 0,
            "avg_episode_duration": np.mean([e["episode_duration_sec"] for e in self.episodes]) if self.episodes else 0,
        }

        return stats

    def save_summary(self):
        """Save final summary to JSON."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        summary = {
            "metadata": self.metadata,
            "collection_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_sec": duration,
                "total_duration_human": f"{duration/3600:.2f} hours"
            },
            "statistics": self.get_statistics(),
            "episodes": self.episodes
        }

        # Save summary JSON
        summary_path = self.save_dir / f"{self.dataset_type}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Data Collection Summary ({self.dataset_type}):")
        print(f"   Total episodes: {summary['statistics']['total_episodes']}")
        print(f"   Success rate: {summary['statistics']['success_rate']*100:.1f}%")
        print(f"   Failure rate: {summary['statistics']['failure_rate']*100:.1f}%")
        print(f"   Failure types: {summary['statistics']['failure_types']}")
        print(f"   Avg episode length: {summary['statistics']['avg_episode_length']:.1f} steps")
        print(f"   Avg episode duration: {summary['statistics']['avg_episode_duration']:.1f} sec")
        print(f"   Total collection time: {summary['collection_info']['total_duration_human']}")
        print(f"   Summary saved to: {summary_path}")

    def print_progress(self, episode_id: int, total_episodes: int):
        """Print progress update."""
        stats = self.get_statistics()
        print(f"\n📈 Progress: {episode_id}/{total_episodes} episodes")
        print(f"   Success: {self.success_count} ({stats['success_rate']*100:.1f}%)")
        print(f"   Failure: {self.failure_count} ({stats['failure_rate']*100:.1f}%)")
        print(f"   Avg episode length: {stats['avg_episode_length']:.1f} steps")
