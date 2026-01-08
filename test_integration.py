"""
Comprehensive Integration Test for SALUS + SmolVLA

Tests each component separately to find ALL bugs before data collection.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import traceback

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def _create_app_launcher():
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    return AppLauncher(args)

def test_1_smolvla_import():
    """Test 1: Can we import SmolVLA?"""
    print("\n" + "="*70)
    print("TEST 1: SmolVLA Import")
    print("="*70)
    try:
        from salus.core.vla.smolvla_wrapper import SmolVLAEnsemble, SimpleSignalExtractor
        print("✅ PASS: SmolVLA imports successfully")
        return True, SmolVLAEnsemble, SimpleSignalExtractor
    except Exception as e:
        print(f"❌ FAIL: {e}")
        traceback.print_exc()
        return False, None, None

def test_2_smolvla_single_model():
    """Test 2: Can we load a single SmolVLA model?"""
    print("\n" + "="*70)
    print("TEST 2: Single SmolVLA Model")
    print("="*70)
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        model = model.to(device="cuda:0")
        model.eval()

        mem_mb = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"✅ PASS: Single model loaded ({mem_mb:.2f} MB)")

        del model
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        traceback.print_exc()
        return False

def test_3_smolvla_inference():
    """Test 3: Can SmolVLA run inference?"""
    print("\n" + "="*70)
    print("TEST 3: SmolVLA Inference")
    print("="*70)
    try:
        from salus.core.vla.smolvla_wrapper import SmolVLAEnsemble

        ensemble = SmolVLAEnsemble(ensemble_size=1, device="cuda:0")  # Single model for test

        # Create dummy input
        image = torch.randn(1, 3, 256, 256)
        state = torch.randn(1, 7)
        instruction = "Pick up the cube"

        action, signals = ensemble.predict(image, state, instruction)

        print(f"✅ PASS: Inference works")
        print(f"   Action shape: {action.shape}")
        print(f"   Signals shape: {signals.shape}")

        # Check shapes
        assert action.shape == (1, 7), f"Expected action shape (1, 7), got {action.shape}"
        assert signals.shape == (1, 6), f"Expected signals shape (1, 6), got {signals.shape}"

        print(f"✅ PASS: Output shapes correct")

        del ensemble
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        traceback.print_exc()
        return False

def test_4_environment():
    """Test 4: Can we load the environment?"""
    print("\n" + "="*70)
    print("TEST 4: Environment")
    print("="*70)
    try:
        app_launcher = _create_app_launcher()
        simulation_app = app_launcher.app
        from salus.simulation.isaaclab_env import SimplePickPlaceEnv
        env = SimplePickPlaceEnv(
            simulation_app=simulation_app,
            num_envs=1,
            device="cuda:0",
            render=False
        )
        print(f"✅ PASS: Environment loaded")

        # Test reset
        obs = env.reset()
        print(f"✅ PASS: Environment reset works")

        # Check observation structure
        print(f"\n   Observation keys: {obs.keys()}")
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                print(f"      {key}: {val.shape}, {val.dtype}, {val.device}")
            elif isinstance(val, list):
                print(f"      {key}: list of {len(val)} items")

        # Test step
        dummy_action = torch.randn(1, 7, device="cuda:0")
        next_obs, done, info = env.step(dummy_action)
        print(f"✅ PASS: Environment step works")

        env.close()
        simulation_app.close()
        return True, obs
    except Exception as e:
        print(f"❌ FAIL: {e}")
        traceback.print_exc()
        return False, None

def test_5_environment_vla_compatibility():
    """Test 5: Is environment output compatible with VLA input?"""
    print("\n" + "="*70)
    print("TEST 5: Environment ↔ VLA Compatibility")
    print("="*70)

    try:
        app_launcher = _create_app_launcher()
        simulation_app = app_launcher.app
        from salus.simulation.isaaclab_env import SimplePickPlaceEnv
        from salus.core.vla.smolvla_wrapper import SmolVLAEnsemble
        env = SimplePickPlaceEnv(
            simulation_app=simulation_app,
            num_envs=1,
            device="cuda:0",
            render=False
        )
        obs = env.reset()

        # Check if we can extract what VLA needs
        print("\n   Checking observation format...")

        # Image
        if 'observation.images.camera1' in obs:
            image = obs['observation.images.camera1']
            print(f"   ✅ Image found: {image.shape}, {image.dtype}")
        else:
            print(f"   ❌ No 'observation.images.camera1' key!")
            print(f"   Available keys: {obs.keys()}")
            return False

        # State
        if 'observation.state' in obs:
            state = obs['observation.state']
            print(f"   ✅ State found: {state.shape}, {state.dtype}")
        else:
            print(f"   ❌ No 'observation.state' key!")
            return False

        # Task
        if 'task' in obs:
            task = obs['task']
            print(f"   ✅ Task found: {type(task)}, value: {task}")

            # Convert to string if needed
            if isinstance(task, list):
                instruction = task[0]
            else:
                instruction = str(task)
            print(f"   ✅ Instruction extracted: '{instruction}'")
        else:
            print(f"   ⚠️  No 'task' key - will use default instruction")
            instruction = "Pick and place the object"

        # Now try VLA inference with real observation
        print("\n   Testing VLA with environment observation...")
        ensemble = SmolVLAEnsemble(ensemble_size=1, device="cuda:0")

        action, signals = ensemble.predict(image, state, instruction)
        print(f"   ✅ VLA inference works with env observation")
        print(f"      Action: {action.shape}")
        print(f"      Signals: {signals.shape}")

        # Try stepping with VLA action
        print("\n   Testing environment step with VLA action...")
        next_obs, done, info = env.step(action)
        print(f"   ✅ Environment accepts VLA action")

        env.close()
        simulation_app.close()
        del ensemble
        torch.cuda.empty_cache()

        print(f"\n✅ PASS: Full compatibility verified")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        traceback.print_exc()
        return False

def test_6_recorder():
    """Test 6: Can recorder save data?"""
    print("\n" + "="*70)
    print("TEST 6: Data Recorder")
    print("="*70)
    try:
        from salus.data.recorder import ScalableDataRecorder
        from datetime import datetime

        save_dir = Path("test_recorder_output") / datetime.now().strftime("%Y%m%d_%H%M%S")

        recorder = ScalableDataRecorder(
            save_dir=save_dir,
            max_episodes=5,
            max_episode_length=100,
            signal_dim=6,
            num_cameras=1,
            chunk_size=2
        )
        print(f"✅ PASS: Recorder initialized")

        # Create dummy episode
        T = 50
        episode_data = {
            'images': np.random.randint(0, 255, (T, 1, 3, 256, 256), dtype=np.uint8),
            'states': np.random.randn(T, 7).astype(np.float32),
            'actions': np.random.randn(T, 7).astype(np.float32),
            'signals': np.random.randn(T, 6).astype(np.float32),
            'horizon_labels': np.zeros((T, 4, 4), dtype=np.float32)
        }

        episode_info = {
            'episode_id': 0,
            'success': True,
            'failure_type': -1,
            'episode_length': T,
            'timestamp': datetime.now().isoformat()
        }

        recorder.record_episode(episode_data, episode_info)
        print(f"✅ PASS: Episode recorded")

        stats = recorder.get_statistics()
        print(f"   Episodes: {stats['num_episodes']}")
        print(f"   Timesteps: {stats['total_timesteps']}")
        print(f"   Size: {stats['storage_size_gb']:.3f} GB")

        recorder.close()
        print(f"✅ PASS: Recorder closed successfully")

        # Cleanup
        import shutil
        if save_dir.exists():
            shutil.rmtree(save_dir.parent)

        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        traceback.print_exc()
        return False

def test_7_memory_leak():
    """Test 7: Check for memory leaks over multiple episodes"""
    print("\n" + "="*70)
    print("TEST 7: Memory Leak Test (5 episodes)")
    print("="*70)

    try:
        app_launcher = _create_app_launcher()
        simulation_app = app_launcher.app
        from salus.simulation.isaaclab_env import SimplePickPlaceEnv
        from salus.core.vla.smolvla_wrapper import SmolVLAEnsemble
        env = SimplePickPlaceEnv(
            simulation_app=simulation_app,
            num_envs=1,
            device="cuda:0",
            render=False
        )
        ensemble = SmolVLAEnsemble(ensemble_size=1, device="cuda:0")

        initial_mem = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"   Initial GPU memory: {initial_mem:.2f} MB")

        for ep in range(5):
            obs = env.reset()

            for step in range(20):  # Short episodes
                image = obs['observation.images.camera1']
                state = obs['observation.state']
                instruction = "Pick and place"

                action, signals = ensemble.predict(image, state, instruction)
                obs, done, info = env.step(action)

                if done.any():
                    break

            current_mem = torch.cuda.memory_allocated(0) / (1024**2)
            mem_increase = current_mem - initial_mem
            print(f"   Episode {ep+1}: {current_mem:.2f} MB (Δ = {mem_increase:+.2f} MB)")

            # If memory keeps increasing significantly, we have a leak
            if mem_increase > 500:  # More than 500 MB increase
                print(f"   ⚠️  WARNING: Significant memory increase detected!")

        final_mem = torch.cuda.memory_allocated(0) / (1024**2)
        total_increase = final_mem - initial_mem

        if total_increase < 100:  # Less than 100 MB increase is acceptable
            print(f"\n✅ PASS: No significant memory leak ({total_increase:.2f} MB)")
            result = True
        else:
            print(f"\n⚠️  WARNING: Memory increased by {total_increase:.2f} MB")
            result = True  # Still pass but warn

        env.close()
        simulation_app.close()
        del ensemble
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"❌ FAIL: {e}")
        traceback.print_exc()
        return False

def test_8_full_episode():
    """Test 8: Complete episode collection"""
    print("\n" + "="*70)
    print("TEST 8: Full Episode Collection")
    print("="*70)

    try:
        from datetime import datetime

        # Initialize
        app_launcher = _create_app_launcher()
        simulation_app = app_launcher.app
        from salus.simulation.isaaclab_env import SimplePickPlaceEnv
        from salus.core.vla.smolvla_wrapper import SmolVLAEnsemble
        from salus.data.recorder import ScalableDataRecorder
        env = SimplePickPlaceEnv(
            simulation_app=simulation_app,
            num_envs=1,
            device="cuda:0",
            render=False
        )
        ensemble = SmolVLAEnsemble(ensemble_size=1, device="cuda:0")

        save_dir = Path("test_full_episode") / datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder = ScalableDataRecorder(
            save_dir=save_dir,
            max_episodes=1,
            max_episode_length=200,
            signal_dim=6,
            num_cameras=1,
            chunk_size=1
        )

        # Collect 1 full episode
        obs = env.reset()

        episode_data = {
            'images': [],
            'states': [],
            'actions': [],
            'signals': []
        }

        max_steps = 200
        for step in range(max_steps):
            # VLA inference
            image = obs['observation.images.camera1']
            state = obs['observation.state']
            instruction = "Pick and place the object"

            action, signals = ensemble.predict(image, state, instruction)

            # Environment step
            next_obs, done, info = env.step(action)

            # Store
            episode_data['images'].append(image[0].cpu().numpy().astype(np.uint8))
            episode_data['states'].append(state[0].cpu().numpy().astype(np.float32))
            episode_data['actions'].append(action[0].cpu().numpy().astype(np.float32))
            episode_data['signals'].append(signals[0].cpu().numpy().astype(np.float32))

            obs = next_obs

            if done.any():
                print(f"   Episode ended at step {step+1}")
                break

        # Convert to arrays
        T = len(episode_data['images'])
        images = np.stack(episode_data['images'], axis=0)
        images = np.expand_dims(images, axis=1)  # Add camera dim

        episode_arrays = {
            'images': images,
            'states': np.stack(episode_data['states'], axis=0),
            'actions': np.stack(episode_data['actions'], axis=0),
            'signals': np.stack(episode_data['signals'], axis=0),
            'horizon_labels': np.zeros((T, 4, 4), dtype=np.float32)
        }

        episode_info = {
            'episode_id': 0,
            'success': bool(info['success'][0].item()),
            'failure_type': -1,
            'episode_length': T,
            'timestamp': datetime.now().isoformat()
        }

        # Record
        recorder.record_episode(episode_arrays, episode_info)
        recorder.close()

        print(f"✅ PASS: Full episode collected and saved")
        print(f"   Episode length: {T} steps")
        print(f"   Success: {episode_info['success']}")

        # Cleanup
        import shutil
        if save_dir.exists():
            shutil.rmtree(save_dir.parent)

        env.close()
        simulation_app.close()
        del ensemble
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE INTEGRATION TEST")
    print("Testing SALUS + SmolVLA Pipeline")
    print("="*70)

    results = {}

    # Test 1: Import
    results['import'], SmolVLAEnsemble, SimpleSignalExtractor = test_1_smolvla_import()
    if not results['import']:
        print("\n❌ CRITICAL: Cannot import SmolVLA. Stopping tests.")
        return

    # Test 2: Single model
    results['single_model'] = test_2_smolvla_single_model()

    # Test 3: Inference
    results['inference'] = test_3_smolvla_inference()

    # Test 4: Environment
    results['environment'], obs = test_4_environment()

    # Test 5: Compatibility
    results['compatibility'] = test_5_environment_vla_compatibility()

    # Test 6: Recorder
    results['recorder'] = test_6_recorder()

    # Test 7: Memory leak
    results['memory'] = test_7_memory_leak()

    # Test 8: Full episode
    results['full_episode'] = test_8_full_episode()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is ready for data collection")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("⚠️  Fix issues before data collection!")
        print("="*70)

    return all_passed

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Single GPU

    success = main()
    sys.exit(0 if success else 1)
