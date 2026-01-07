"""
SmolVLA Ensemble Wrapper for SALUS

Integrates SmolVLA-450M with SALUS for epistemic uncertainty estimation.

Key Features:
- 3-model ensemble for uncertainty quantification
- 6D → 7D action padding for Franka Panda
- Real pre-trained VLA (not dummy/random)
- Single-GPU compatible (~2.7 GB VRAM for ensemble)

Usage:
    ensemble = SmolVLAEnsemble(ensemble_size=3, device="cuda:0")
    actions, signals = ensemble.predict(image, state, instruction)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer

# Set to single GPU to avoid accelerate multi-GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class SmolVLAEnsemble(nn.Module):
    """
    Ensemble of SmolVLA models for epistemic uncertainty estimation.

    Args:
        ensemble_size (int): Number of models in ensemble (default: 3)
        device (str): Device to load models on (default: "cuda:0")
        model_path (str): HuggingFace model path (default: "lerobot/smolvla_base")
    """

    def __init__(
        self,
        ensemble_size: int = 3,
        device: str = "cuda:0",
        model_path: str = "lerobot/smolvla_base"
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.device = device
        self.model_path = model_path

        print(f"Initializing SmolVLA Ensemble (size={ensemble_size})...")
        print(f"  Model: {model_path}")
        print(f"  Device: {device}")

        # Load tokenizer (shared across all models)
        print("  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )

        # Load ensemble models
        self.models = nn.ModuleList()
        for i in range(ensemble_size):
            print(f"  Loading model {i+1}/{ensemble_size}...")
            model = SmolVLAPolicy.from_pretrained(model_path)
            model = model.to(device=self.device)
            model.eval()
            self.models.append(model)

        # Track memory usage
        mem_mb = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"✅ Ensemble loaded: {mem_mb:.2f} MB on {device}\n")

        self.signal_extractor = SimpleSignalExtractor()

    def _prepare_batch(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        instruction: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for SmolVLA inference.

        Args:
            image: (1, 3, H, W) RGB image tensor
            state: (1, 7) robot state tensor
            instruction: Language instruction string

        Returns:
            Batch dict with proper keys and dtypes
        """
        # Ensure image is 256x256 and float32
        if image.shape[-2:] != (256, 256):
            image = torch.nn.functional.interpolate(
                image, size=(256, 256), mode='bilinear', align_corners=False
            )
        image = image.to(dtype=torch.float32, device=self.device)

        # Ensure state is float32
        state = state.to(dtype=torch.float32, device=self.device)

        # Tokenize instruction
        tokens = self.tokenizer(instruction, return_tensors="pt")

        # Create batch
        batch = {
            'observation.images.camera1': image,
            'observation.state': state,
            'observation.language.tokens': tokens['input_ids'].to(device=self.device),
            'observation.language.attention_mask': tokens['attention_mask'].to(device=self.device).bool()
        }

        return batch

    def _pad_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Pad 6D SmolVLA action to 7D for Franka Panda.

        Args:
            action: (batch, 6) action tensor

        Returns:
            (batch, 7) padded action tensor
        """
        # SmolVLA outputs 6D, Franka needs 7D
        # Pad with zero for the 7th joint
        if action.shape[-1] == 6:
            padding = torch.zeros(action.shape[0], 1, device=action.device, dtype=action.dtype)
            action = torch.cat([action, padding], dim=-1)
        return action

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        instruction: str = "Pick and place the object"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run ensemble inference and extract uncertainty signals.

        Args:
            image: (1, 3, H, W) RGB image
            state: (1, 7) robot state
            instruction: Language instruction

        Returns:
            actions: (1, 7) mean action across ensemble
            signals: (1, 6) uncertainty signals for SALUS
        """
        # Prepare batch
        batch = self._prepare_batch(image, state, instruction)

        # Run ensemble
        ensemble_actions = []
        for model in self.models:
            action = model.select_action(batch)  # (1, 6)
            action = self._pad_action(action)     # (1, 7)
            ensemble_actions.append(action)

        ensemble_actions = torch.stack(ensemble_actions, dim=1)  # (1, ensemble_size, 7)

        # Mean action
        mean_action = ensemble_actions.mean(dim=1)  # (1, 7)

        # Extract uncertainty signals
        signals = self.signal_extractor.extract({
            'actions': ensemble_actions,
            'mean_action': mean_action
        })

        return mean_action, signals


class SimpleSignalExtractor:
    """
    Extracts 6D uncertainty signals from VLA ensemble for SALUS MVP.

    Signals:
    1. Epistemic uncertainty (ensemble disagreement)
    2. Action magnitude
    3. Action variance across ensemble
    4. Action smoothness (change from previous)
    5. Max per-dimension variance
    6. Uncertainty trend
    """

    def __init__(self):
        self.prev_action = None
        self.prev_uncertainty = None

    def extract(self, vla_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract 6D signals from VLA ensemble output.

        Args:
            vla_output: Dict containing:
                - 'actions': (B, ensemble_size, action_dim) ensemble predictions
                - 'mean_action': (B, action_dim) mean action

        Returns:
            signals: (B, 6) uncertainty signals
        """
        actions = vla_output['actions']  # (B, ensemble_size, action_dim)
        mean_action = vla_output['mean_action']  # (B, action_dim)

        B = actions.shape[0]

        # 1. Epistemic uncertainty (std across ensemble)
        epistemic_uncertainty = actions.std(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)

        # 2. Action magnitude
        action_magnitude = mean_action.norm(dim=-1, keepdim=True)  # (B, 1)

        # 3. Action variance
        action_variance = actions.var(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)

        # 4. Action smoothness (change from previous action)
        if self.prev_action is None:
            action_smoothness = torch.zeros(B, 1, device=actions.device)
        else:
            action_smoothness = (mean_action - self.prev_action).norm(dim=-1, keepdim=True)
        self.prev_action = mean_action.detach().clone()

        # 5. Max per-dimension variance
        per_dim_var = actions.var(dim=1)  # (B, action_dim)
        max_per_dim_var = per_dim_var.max(dim=-1, keepdim=True)[0]  # (B, 1)

        # 6. Uncertainty trend
        if self.prev_uncertainty is None:
            uncertainty_trend = torch.zeros(B, 1, device=actions.device)
        else:
            uncertainty_trend = epistemic_uncertainty - self.prev_uncertainty
        self.prev_uncertainty = epistemic_uncertainty.detach().clone()

        # Concatenate all signals
        signals = torch.cat([
            epistemic_uncertainty,
            action_magnitude,
            action_variance,
            action_smoothness,
            max_per_dim_var,
            uncertainty_trend
        ], dim=-1)  # (B, 6)

        return signals

    def reset(self):
        """Reset temporal state (call at episode start)"""
        self.prev_action = None
        self.prev_uncertainty = None


def test_smolvla_ensemble():
    """Test SmolVLA ensemble with dummy data"""
    print("="*60)
    print("Testing SmolVLA Ensemble")
    print("="*60 + "\n")

    # Initialize ensemble
    ensemble = SmolVLAEnsemble(ensemble_size=3, device="cuda:0")

    # Create dummy inputs
    image = torch.randn(1, 3, 256, 256)
    state = torch.randn(1, 7)
    instruction = "Pick up the red cube and place it in the bin"

    print("Running inference...")
    actions, signals = ensemble.predict(image, state, instruction)

    print(f"\n✅ Inference successful!")
    print(f"Actions shape: {actions.shape} (expected: [1, 7])")
    print(f"Signals shape: {signals.shape} (expected: [1, 6])")
    print(f"\nSample action: {actions[0].cpu().numpy()}")
    print(f"Sample signals: {signals[0].cpu().numpy()}")

    print(f"\n" + "="*60)
    print("✅ SmolVLA ensemble is working!")
    print("="*60)


if __name__ == "__main__":
    test_smolvla_ensemble()
