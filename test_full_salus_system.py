"""
Full SALUS System Test
Tests the complete pipeline: SmolVLA → Signal Extraction → Failure Prediction → Intervention
"""

import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# IsaacLab AppLauncher must be created first
from isaaclab.app import AppLauncher
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Full SALUS System Test")
parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to test")
parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Create AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else
from salus.simulation.franka_pick_place_env import FrankaPickPlaceEnv
from salus.core.vla.wrapper import SmolVLAEnsemble, SignalExtractor
from salus.core.predictor import SALUSPredictor
from salus.core.adaptation import AdaptationModule

print("="*70)
print("FULL SALUS SYSTEM TEST")
print("="*70)
print(f"\nConfiguration:")
print(f"  Episodes: {args.episodes}")
print(f"  Max steps per episode: {args.max_steps}")
print(f"  Device: cuda:0")
print("="*70)

# Initialize components
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 Initializing Components...")

# 1. Environment
print("  1. Initializing IsaacSim environment...")
env = FrankaPickPlaceEnv(
    simulation_app=simulation_app,
    num_envs=1,
    device=str(device),
    render=True,  # GUI enabled
    max_episode_length=args.max_steps
)
print("     ✅ Environment ready")

# 2. SmolVLA Ensemble
print("  2. Loading SmolVLA ensemble...")
try:
    vla = SmolVLAEnsemble(
        model_path="lerobot/smolvla_base",
        ensemble_size=1,  # Use 1 for faster testing
        device=str(device)
    )
    print("     ✅ SmolVLA loaded")
except Exception as e:
    print(f"     ❌ Failed to load SmolVLA: {e}")
    env.close()
    simulation_app.close()
    sys.exit(1)

# 3. Signal Extractor
print("  3. Initializing signal extractor...")
signal_extractor = SignalExtractor()
print("     ✅ Signal extractor ready")

# 4. SALUS Predictor
print("  4. Loading SALUS predictor...")
predictor = SALUSPredictor(
    signal_dim=12,
    num_horizons=4,
    num_failure_types=4
).to(device)
predictor.eval()
print("     ⚠️  Using UNTRAINED predictor (random predictions)")
print("     ✅ Predictor ready")

# 5. Adaptation Module
print("  5. Initializing adaptation module...")
adapter = AdaptationModule(
    emergency_threshold=0.9,
    slow_down_threshold=0.7,
    retry_threshold=0.6,
    enable_logging=True
)
print("     ✅ Adaptation module ready")

print("\n" + "="*70)
print("SYSTEM READY - Starting Test Episodes")
print("="*70)

# Run test episodes
all_results = []

for episode in range(args.episodes):
    print(f"\n{'='*70}")
    print(f"EPISODE {episode + 1}/{args.episodes}")
    print(f"{'='*70}\n")
    
    # Reset
    obs = env.reset()
    signal_extractor.reset()
    adapter.reset()
    
    episode_data = {
        'signals': [],
        'predictions': [],
        'interventions': [],
        'actions': [],
        'steps': 0
    }
    
    step = 0
    done = torch.zeros(1, dtype=torch.bool, device=device)
    
    print("  Step | Uncertainty | Failure Prob | Intervention | Action")
    print("  " + "-"*65)
    
    while not done.any() and step < args.max_steps:
        # Prepare observation for VLA
        obs_vla = {
            'observation.images.camera1': obs['observation.images.camera1'].to(device).float() / 255.0,
            'observation.images.camera2': obs['observation.images.camera2'].to(device).float() / 255.0,
            'observation.images.camera3': obs['observation.images.camera3'].to(device).float() / 255.0,
            'observation.state': obs['observation.state'].to(device),
            'task': obs['task']
        }
        
        # Step 1: SmolVLA generates action
        with torch.no_grad():
            vla_output = vla(obs_vla)
            action = vla_output['action']
            
            # Extract signals
            signals = signal_extractor.extract(vla_output)
            episode_data['signals'].append(signals[0].cpu().numpy())
            
            # Get epistemic uncertainty
            epistemic_uncertainty = vla_output['epistemic_uncertainty'][0].item()
        
        # Step 2: SALUS predicts failure
        with torch.no_grad():
            signals_tensor = signals.to(device)
            pred_output = predictor(signals_tensor)
            failure_prob = pred_output['max_prob'][0].item()
            predicted_horizon = pred_output['predicted_horizon'][0].item()
            predicted_type = pred_output['predicted_type'][0].item()
            
            episode_data['predictions'].append({
                'failure_prob': failure_prob,
                'horizon': predicted_horizon,
                'type': predicted_type
            })
        
        # Step 3: Adaptation module decides intervention
        prediction_dict = {
            'failure_predicted': torch.tensor([failure_prob > 0.5], device=device),
            'failure_horizon': torch.tensor([predicted_horizon], device=device),
            'failure_type': torch.tensor([predicted_type], device=device),
            'confidence': torch.tensor([failure_prob], device=device)
        }
        
        decision = adapter.decide_intervention(prediction_dict, current_step=step)
        
        # Step 4: Apply intervention
        modified_action, should_reset = adapter.apply_intervention(action, decision)
        
        if decision.intervention.value != 0:  # Not NONE
            episode_data['interventions'].append({
                'step': step,
                'type': decision.intervention.name,
                'confidence': failure_prob,
                'reason': decision.reason
            })
        
        # Display
        intervention_str = decision.intervention.name if decision.intervention.value != 0 else "NONE"
        action_str = "STOP" if decision.intervention.value == 1 else "SLOW" if decision.intervention.value == 2 else "NORMAL"
        
        print(f"  {step:4d} | {epistemic_uncertainty:11.3f} | {failure_prob:12.3f} | {intervention_str:12s} | {action_str}")
        
        # Step 5: Execute action in environment
        obs, done, info = env.step(modified_action)
        
        episode_data['actions'].append(modified_action[0].cpu().numpy())
        step += 1
        episode_data['steps'] = step
    
    # Episode summary
    success = info['success'][0].item()
    failure_type = info['failure_type'][0].item() if not success else None
    
    print(f"\n  {'='*65}")
    print(f"  Episode {episode + 1} Summary:")
    print(f"    Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    if not success:
        failure_names = ["Collision", "Drop", "Miss", "Timeout"]
        print(f"    Failure type: {failure_names[int(failure_type)] if int(failure_type) < 4 else 'Unknown'}")
    print(f"    Steps: {step}")
    print(f"    Interventions: {len(episode_data['interventions'])}")
    if episode_data['predictions']:
        avg_failure_prob = np.mean([p['failure_prob'] for p in episode_data['predictions']])
        max_failure_prob = max([p['failure_prob'] for p in episode_data['predictions']])
        print(f"    Avg failure prob: {avg_failure_prob:.3f}")
        print(f"    Max failure prob: {max_failure_prob:.3f}")
    print(f"  {'='*65}")
    
    # Update adapter stats
    adapter.on_episode_end(success=success)
    
    all_results.append({
        'episode': episode + 1,
        'success': success,
        'failure_type': failure_type,
        'steps': step,
        'interventions': len(episode_data['interventions']),
        'avg_failure_prob': avg_failure_prob if episode_data['predictions'] else 0.0,
        'max_failure_prob': max_failure_prob if episode_data['predictions'] else 0.0
    })

# Final summary
print("\n" + "="*70)
print("FULL SYSTEM TEST SUMMARY")
print("="*70)

print(f"\n📊 Results:")
print(f"  Episodes completed: {len(all_results)}/{args.episodes}")
successes = sum(1 for r in all_results if r['success'])
print(f"  Successes: {successes}/{len(all_results)} ({successes/len(all_results)*100:.1f}%)")
print(f"  Total interventions: {sum(r['interventions'] for r in all_results)}")
print(f"  Avg interventions per episode: {np.mean([r['interventions'] for r in all_results]):.1f}")

print(f"\n📈 Prediction Statistics:")
if all_results:
    avg_probs = [r['avg_failure_prob'] for r in all_results]
    max_probs = [r['max_failure_prob'] for r in all_results]
    print(f"  Average failure probability: {np.mean(avg_probs):.3f}")
    print(f"  Maximum failure probability: {np.max(max_probs):.3f}")

print(f"\n🔧 Adaptation Statistics:")
adapter.print_statistics()

print("\n" + "="*70)
print("SYSTEM COMPONENTS TESTED:")
print("="*70)
print("  ✅ SmolVLA Ensemble - Generated actions from observations")
print("  ✅ Signal Extractor - Extracted 12D signals from VLA output")
print("  ✅ SALUS Predictor - Predicted failures from signals")
print("  ✅ Adaptation Module - Decided interventions based on predictions")
print("  ✅ Environment Integration - Executed actions in IsaacSim")
print("  ✅ Full Pipeline - End-to-end operation verified")
print("="*70)

print("\n⚠️  NOTE: Predictor is UNTRAINED (random predictions)")
print("   For real failure prediction, train SALUS first:")
print("   python scripts/train_predictor_mvp.py --data <data_path>")
print("="*70)

# Cleanup
env.close()
simulation_app.close()

print("\n✅ Full system test completed!")

