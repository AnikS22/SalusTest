"""
Test ONLY VLA - no Isaac Lab
Shows VLA outputs and signals
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70, flush=True)
print("VLA ONLY TEST", flush=True)
print("="*70, flush=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}", flush=True)

print("\n1. Loading VLA...", flush=True)
from salus.core.vla.wrapper import SmolVLAEnsemble, EnhancedSignalExtractor

vla = SmolVLAEnsemble(
    str(Path.home() / "models/smolvla/smolvla_base"),
    ensemble_size=3,
    device=device
)
print("✅ VLA loaded", flush=True)

print("\n2. Creating signal extractor...", flush=True)
extractor = EnhancedSignalExtractor(device)
print("✅ Signal extractor ready", flush=True)

print("\n3. Creating fake observation...", flush=True)
obs = {
    'observation.images.camera1': torch.randn(1, 3, 256, 256, device=device),
    'observation.images.camera2': torch.randn(1, 3, 256, 256, device=device),
    'observation.images.camera3': torch.randn(1, 3, 256, 256, device=device),
    'observation.state': torch.randn(1, 6, device=device) * 0.5,
    'task': ['Pick up the red cube']
}
print("✅ Fake observation created", flush=True)

print("\n4. Running VLA inference...", flush=True)
with torch.no_grad():
    output = vla(obs, return_internals=True)
print("✅ VLA inference complete", flush=True)

print("\n5. VLA OUTPUTS:", flush=True)
action = output['action'][0].cpu().numpy()
print(f"\nAction (6D): {action}", flush=True)
print(f"Action magnitude: {(action**2).sum()**0.5:.6f}", flush=True)

epistemic = output['epistemic_uncertainty'][0].item()
print(f"\nEpistemic uncertainty: {epistemic:.6f}", flush=True)

action_var = output['action_var'][0].cpu().numpy()
print(f"Action variance: {action_var}", flush=True)

if 'hidden_state_mean' in output:
    hidden = output['hidden_state_mean'][0]
    print(f"\nHidden state shape: {hidden.shape}", flush=True)
    print(f"Hidden state norm: {torch.norm(hidden).item():.4f}", flush=True)
    print(f"Hidden state mean: {hidden.mean().item():.4f}", flush=True)

if 'perturbed_actions' in output:
    perturbed = output['perturbed_actions']
    print(f"\nPerturbed actions shape: {perturbed.shape}", flush=True)
    print(f"Perturbation variance: {perturbed.var().item():.6f}", flush=True)

print("\n6. Extracting 18D signals...", flush=True)
robot_state = torch.randn(1, 7, device=device) * 0.5
signals = extractor.extract(output, robot_state=robot_state)
s = signals[0].cpu().numpy()

print("\n18D SIGNALS:", flush=True)
names = ["Epistemic", "ActMag", "ActVar", "ActSmooth", "TrajDiv",
         "JointVar0", "JointVar1", "JointVar2", "UncMean", "UncStd",
         "UncMin", "UncMax", "LatentDrift", "OOD", "AugStab",
         "PertSens", "ExecMis", "ConstraintMar"]
for i, (name, val) in enumerate(zip(names, s)):
    print(f"   {i+1:2d}. {name:15s}: {val:8.5f}", flush=True)

print("\n" + "="*70, flush=True)
print("SUCCESS - VLA and signal extraction working!", flush=True)
print("="*70, flush=True)
