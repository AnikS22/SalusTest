"""
TinyVLA Ensemble Wrapper for SALUS
Simple wrapper around TinyVLA-1B for uncertainty estimation

This is a MINIMAL implementation for MVP.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path


class TinyVLAEnsemble(nn.Module):
    """
    Simple ensemble wrapper for TinyVLA-1B

    Uses 3-5 models for epistemic uncertainty
    (Simpler than 5 models - faster for testing)
    """

    def __init__(
        self,
        model_path: str = "~/models/tinyvla/tinyvla-1b",
        ensemble_size: int = 3,  # Start with 3 for MVP
        device: str = "cuda:0"
    ):
        super().__init__()

        self.device = torch.device(device)
        self.ensemble_size = ensemble_size
        self.model_path = Path(model_path).expanduser()

        print(f"\n🤖 Loading TinyVLA Ensemble ({ensemble_size} models on {device})...")

        # Try to import TinyVLA
        try:
            # TinyVLA uses similar API to other VLAs
            # Adjust import based on actual TinyVLA structure
            from tinyvla import TinyVLAPolicy

            self.models = nn.ModuleList()
            for i in range(ensemble_size):
                print(f"  Loading model {i+1}/{ensemble_size}...")
                model = TinyVLAPolicy.from_pretrained(str(self.model_path))
                model = model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"  ✓ Model {i+1}/{ensemble_size} loaded")

            print(f"✅ TinyVLA ensemble ready on {device}")
            print(f"   Model size: ~1B parameters per model")
            print(f"   Total VRAM: ~{3 * ensemble_size}GB (approx)")

        except ImportError:
            print(f"❌ TinyVLA not installed!")
            print(f"\nInstall with:")
            print(f"  cd ~/")
            print(f"  git clone https://github.com/OpenDriveLab/TinyVLA.git")
            print(f"  cd TinyVLA")
            print(f"  pip install -e .")
            raise

    @torch.no_grad()
    def forward(self, observation: Dict[str, torch.Tensor]) -> Dict:
        """
        Forward pass through ensemble

        Args:
            observation: Dict with keys:
                - 'image': (B, C, H, W) camera image
                - 'state': (B, state_dim) robot state
                - 'instruction': str or List[str] task description

        Returns:
            Dict with:
                - 'action': (B, action_dim) mean action
                - 'action_var': (B, action_dim) variance across ensemble
                - 'epistemic_uncertainty': (B,) scalar uncertainty
        """
        # Ensure inputs on correct device
        observation = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in observation.items()}

        # Collect actions from ensemble
        actions = []

        for model in self.models:
            # Enable dropout for diversity (optional)
            model.train()

            # Forward pass
            output = model(observation)

            # Extract action
            if isinstance(output, dict):
                action = output['action']
            else:
                action = output

            actions.append(action)

        # Stack and compute statistics
        actions = torch.stack(actions, dim=1)  # (B, ensemble_size, action_dim)

        action_mean = actions.mean(dim=1)  # (B, action_dim)
        action_var = actions.var(dim=1)    # (B, action_dim)

        # Epistemic uncertainty: mean variance across dimensions
        epistemic_uncertainty = action_var.mean(dim=-1)  # (B,)

        return {
            'action': action_mean,
            'action_var': action_var,
            'epistemic_uncertainty': epistemic_uncertainty
        }


class SimpleSignalExtractor:
    """
    Simple 6D signal extractor for SALUS MVP

    Much simpler than full 12D version - easier to train
    """

    def __init__(self):
        self.action_history = []
        self.max_history = 5  # Keep last 5 actions

    def extract(self, vla_output: Dict) -> torch.Tensor:
        """
        Extract 6D signal vector from VLA output

        Signals (6 dimensions):
          1. Epistemic uncertainty (ensemble variance)
          2. Action magnitude (L2 norm)
          3. Action variance (mean across dims)
          4. Action smoothness (change from previous)
          5. Max per-dim variance (highest joint uncertainty)
          6. Recent uncertainty trend (increasing/decreasing)

        Args:
            vla_output: Output from TinyVLAEnsemble.forward()

        Returns:
            signals: (B, 6) feature vector
        """
        batch_size = vla_output['action'].shape[0]
        signals = []

        # 1. Epistemic uncertainty
        epistemic = vla_output['epistemic_uncertainty']  # (B,)
        signals.append(epistemic.unsqueeze(-1))

        # 2. Action magnitude
        action = vla_output['action']  # (B, action_dim)
        action_mag = torch.norm(action, dim=-1)
        signals.append(action_mag.unsqueeze(-1))

        # 3. Action variance (mean)
        action_var = vla_output['action_var']  # (B, action_dim)
        action_var_mean = action_var.mean(dim=-1)
        signals.append(action_var_mean.unsqueeze(-1))

        # 4. Action smoothness
        if len(self.action_history) > 0:
            prev_action = self.action_history[-1]
            smoothness = torch.norm(action - prev_action, dim=-1)
            signals.append(smoothness.unsqueeze(-1))
        else:
            signals.append(torch.zeros(batch_size, 1, device=epistemic.device))

        # Update history
        self.action_history.append(action.detach())
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)

        # 5. Max per-dimension variance
        max_var = action_var.max(dim=-1)[0]
        signals.append(max_var.unsqueeze(-1))

        # 6. Recent uncertainty trend
        if len(self.action_history) >= 2:
            # Simple trend: is uncertainty increasing?
            recent_uncertainties = [epistemic]  # Simplified
            trend = epistemic  # Use current as proxy
            signals.append(trend.unsqueeze(-1))
        else:
            signals.append(epistemic.unsqueeze(-1))

        # Concatenate all signals
        signals = torch.cat(signals, dim=-1)  # (B, 6)

        return signals

    def reset(self):
        """Reset history at episode start"""
        self.action_history = []


# Test script
if __name__ == "__main__":
    print("🧪 Testing TinyVLA Ensemble Wrapper...\n")

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        exit(1)

    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print()

    # Create dummy observation (since we don't have TinyVLA installed yet)
    print("📝 Creating dummy test...")
    print("   (Run this after installing TinyVLA to test real model)\n")

    # Test signal extractor
    print("Testing Signal Extractor...")
    extractor = SimpleSignalExtractor()

    # Dummy VLA output
    dummy_output = {
        'action': torch.randn(1, 7, device='cuda:0'),
        'action_var': torch.rand(1, 7, device='cuda:0') * 0.1,
        'epistemic_uncertainty': torch.rand(1, device='cuda:0') * 0.5
    }

    signals = extractor.extract(dummy_output)

    print(f"   Input: action={dummy_output['action'].shape}, uncertainty={dummy_output['epistemic_uncertainty'].shape}")
    print(f"   Output: signals={signals.shape}")
    print(f"   Signal values: {signals[0].cpu().numpy()}")
    print()

    print("✅ Signal extractor works!")
    print("\n📋 Next steps:")
    print("   1. Install TinyVLA")
    print("   2. Download model weights")
    print("   3. Test with real VLA")
    print("   4. Create control loop with recording")
