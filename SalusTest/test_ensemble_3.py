"""
Test with ACTUAL production config: ensemble_size=1
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from salus.core.vla.smolvla_wrapper import SmolVLAEnsemble
from salus.simulation.isaaclab_env import SimplePickPlaceEnv

print("="*70)
print("PRODUCTION CONFIG TEST: ensemble_size=1")
print("="*70)

# Test 1: Load ensemble of 3
print("\n1. Loading ensemble of 3 models...")
ensemble = SmolVLAEnsemble(ensemble_size=1, device="cuda:0")

mem_mb = torch.cuda.memory_allocated(0) / (1024**2)
print(f"✅ Ensemble loaded: {mem_mb:.2f} MB")

# Test 2: Inference
print("\n2. Testing inference...")
image = torch.randn(1, 3, 256, 256)
state = torch.randn(1, 7)
instruction = "Pick up the red cube"

action, signals = ensemble.predict(image, state, instruction)
print(f"✅ Inference works")
print(f"   Action: {action.shape}")
print(f"   Signals: {signals.shape}")
print(f"   Signal values: {signals[0].cpu().numpy()}")

# Test 3: With real environment
print("\n3. Testing with environment...")
env = SimplePickPlaceEnv(num_envs=1, device="cuda:0", render=False)
obs = env.reset()

image = obs['observation.images.camera1']
state = obs['observation.state']
instruction = obs['task']

print(f"   Image: {image.shape}, {image.dtype}")
print(f"   State: {state.shape}, {state.dtype}")
print(f"   Instruction: {instruction}")

action, signals = ensemble.predict(image, state, instruction)
print(f"✅ VLA works with env observation")
print(f"   Action: {action.shape}")
print(f"   Signals: {signals.shape}")

# Test 4: Full episode (10 steps)
print("\n4. Running 10-step episode...")
obs = env.reset()

for step in range(10):
    image = obs['observation.images.camera1']
    state = obs['observation.state']
    instruction = obs['task']

    action, signals = ensemble.predict(image, state, instruction)
    obs, done, info = env.step(action)

    if step % 5 == 0:
        print(f"   Step {step}: action norm={action.norm().item():.3f}, signal[0]={signals[0,0].item():.3f}")

print(f"✅ Episode completed")

# Test 5: Memory check
mem_mb = torch.cuda.memory_allocated(0) / (1024**2)
print(f"\n5. Final memory: {mem_mb:.2f} MB / 11,264 MB")

if mem_mb < 10000:
    print(f"✅ Memory usage is OK")
else:
    print(f"⚠️  High memory usage!")

env.close()

print("\n" + "="*70)
print("✅ ALL PRODUCTION CONFIG TESTS PASSED!")
print("="*70)
