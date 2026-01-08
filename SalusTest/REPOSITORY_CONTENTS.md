# Repository Contents Summary

## 📊 Overview

This repository contains **SALUS MVP implementation** - a runtime safety system for VLA models.

**Total Code**: ~4,175 lines of Python  
**Status**: MVP implementation complete, ready for data collection and training

---

## 📁 Directory Structure

### Core Implementation (`salus/`)

#### ✅ VLA Integration (`salus/core/vla/`)
- **`wrapper.py`** - VLA ensemble wrapper (original implementation)
- **`tinyvla_wrapper.py`** - TinyVLA-specific ensemble wrapper (MVP)
  - single VLA model for model uncertainty
  - Signal extraction (6D features)
  - ~9GB VRAM usage

#### ✅ Predictor (`salus/core/predictor/`)
- **`predictor.py`** - Full multi-horizon predictor (not yet implemented)
- **`predictor_mvp.py`** - ✅ MVP predictor (simple MLP)
  - 6D input → 4D output (failure types)
  - ~4K parameters
  - <1ms inference time

#### ⚠️ Manifold (`salus/core/manifold/`)
- **`__init__.py`** - Empty (placeholder)
- **Status**: Not yet implemented for MVP

#### ⚠️ Synthesis (`salus/core/synthesis/`)
- **`__init__.py`** - Empty (placeholder)
- **Status**: Not yet implemented for MVP

#### ✅ Data Pipeline (`salus/data/`)
- **`recorder.py`** - Episode recording to Zarr format
- **`dataset_mvp.py`** - Dataset loader for training
  - Loads episodes from Zarr
  - Signal extraction
  - Train/val split

#### ✅ Simulation (`salus/simulation/`)
- **`isaaclab_env.py`** - IsaacLab environment integration
- **`franka_pick_place_env.py`** - Franka robot pick-place environment

#### ✅ Utilities (`salus/utils/`)
- **`config.py`** - Configuration management

---

### Scripts (`scripts/`)

#### ✅ Data Collection
- **`collect_episodes_mvp.py`** - ✅ MVP data collection script
  - Runs VLA ensemble in simulation
  - Records episodes with signals
  - Saves to Zarr format
  - Supports real VLA and dummy modes

- **`collect_data.py`** - Full data collection (if implemented)
- **`collect_data_franka.py`** - Franka-specific collection

#### ✅ Training
- **`train_predictor_mvp.py`** - ✅ MVP predictor training
  - Training loop with wandb logging
  - Checkpointing
  - Evaluation metrics

#### ✅ Evaluation
- **`evaluate_mvp.py`** - MVP evaluation script

#### ✅ Testing
- **`test_vla_isaaclab.py`** - IsaacLab integration test

---

### Documentation (`docs/`)

#### Core Documentation
- **`README.md`** - Main project overview
- **`QUICK_START.md`** - Quick start guide
- **`GETTING_STARTED.md`** - Detailed getting started guide
- **`LOCAL_MACHINE_SETUP.md`** - GPU setup instructions
- **`WHAT_ARE_YOU_ACTUALLY_BUILDING.md`** - Plain English explanation

#### Technical Documentation
- **`salus_software_architecture.md`** - System architecture
- **`salus_implementation_guide.md`** - Implementation details
- **`salus_math_explained.md`** - Mathematical foundations
- **`SYSTEM_VALIDATION_AND_PAPER_ALIGNMENT.md`** - Validation details

#### Research Papers (`docs/papers/`)
- **`salus_vla_safety.tex`** - Main paper
- **`salus_vla_safety_v2.tex`** - Extended version
- **`salus_vla_safety_academic.tex`** - Academic submission

#### Other Documentation
- **`salus_ip_strategy.md`** - Intellectual property strategy
- **`RENAMING_COMPLETE_SUMMARY.md`** - Project renaming notes

---

### Status Documents (Root Directory)

#### Implementation Status
- **`SALUS_MVP_STATUS.md`** - ✅ MVP implementation status
- **`SALUS_IMPLEMENTATION_COMPLETE.md`** - Implementation completion notes
- **`SALUS_MVP_PIPELINE_VERIFIED.md`** - Pipeline verification
- **`INTEGRATION_COMPLETE.md`** - Integration status
- **`TRAINING_COMPLETE.md`** - Training completion
- **`FINAL_STATUS.md`** - Final status summary

#### Setup & Configuration
- **`CLAUDE_HANDOFF.md`** - Development handoff document
- **`MODEL_DOWNLOAD_GUIDE.md`** - Model download instructions
- **`HARDWARE_COMPARISON.md`** - Hardware comparison (TinyVLA vs OpenVLA)
- **`ISAACLAB_SETUP.md`** - IsaacLab setup notes
- **`TINYVLA_DOWNLOADED.md`** - TinyVLA download status

#### System Design
- **`HOW_SALUS_WORKS.md`** - System explanation
- **`SALUS_SYSTEM_DESIGN.md`** - Design document
- **`SALUS_MVP_README.md`** - MVP-specific README
- **`SALUS_TRAINING_EXPLAINED.md`** - Training explanation
- **`MULTI_HORIZON_PREDICTION.md`** - Prediction details

---

### Data (`data/`)

#### Collected Episodes
- **`raw_episodes/`** - Multiple data collection runs (timestamps)
  - Format: Zarr archives
  - Contains: RGB images, states, actions, signals, labels

- **`mvp_episodes_test/`** - Test episodes
- **`mvp_episodes_overnight/`** - Overnight collection runs

#### Processed Data
- **`processed/`** - Processed datasets (if any)
- **`labels/`** - Label files
- **`checkpoints/`** - Data checkpoints

---

### Models & Checkpoints (`checkpoints/`)

- **`mvp_test/`** - Test training runs
  - Model checkpoints (best_f1.pth, best_loss.pth, final.pth)
  - Training logs
  - Config files

- **`mvp_500episodes/`** - Training on 500 episodes

---

### Configuration (`configs/`)

- **`base_config.yaml`** - Base configuration file

---

### Logs (`logs/`)

- **`data_collection/`** - Data collection logs
- **`training/`** - Training logs
- **`simulation/`** - Simulation logs
- **`evaluation/`** - Evaluation logs
- **`deployment/`** - Deployment logs

---

## ✅ What's Implemented (MVP)

1. ✅ **TinyVLA Ensemble Wrapper** - single VLA model with signal extraction
2. ✅ **MVP Predictor** - Simple MLP predictor (6D → 4D)
3. ✅ **Data Collection Pipeline** - Episode recording with Zarr storage
4. ✅ **Training Infrastructure** - Dataset loading, training loop, evaluation
5. ✅ **IsaacLab Integration** - Simulation environment setup
6. ✅ **Configuration System** - YAML-based config management

## ⚠️ What's NOT Implemented (Future Work)

1. ⚠️ **Full Multi-Horizon Predictor** - MVP uses simplified version
2. ⚠️ **Safety Manifold** - Contrastive learning (placeholder only)
3. ⚠️ **MPC Synthesizer** - Safe action synthesis (placeholder only)
4. ⚠️ **Continuous Learning** - Online adaptation (placeholder only)

---

## 🚀 Quick Start

```bash
# 1. Setup (already done)
source venv_salus/bin/activate

# 2. Collect data (with real VLA or dummy mode)
python scripts/collect_episodes_mvp.py --num_episodes 500

# 3. Train predictor
python scripts/train_predictor_mvp.py --data_dir data/mvp_episodes_overnight/...

# 4. Evaluate
python scripts/evaluate_mvp.py --checkpoint checkpoints/mvp_test/.../best_f1.pth
```

---

## 📝 Key Files to Understand

1. **`salus/core/vla/tinyvla_wrapper.py`** - VLA ensemble implementation
2. **`salus/core/predictor_mvp.py`** - Predictor architecture
3. **`scripts/collect_episodes_mvp.py`** - Data collection pipeline
4. **`scripts/train_predictor_mvp.py`** - Training loop
5. **`SALUS_MVP_STATUS.md`** - Detailed MVP status

---

## 🔍 File Count Summary

- **Python files**: ~30+ implementation files
- **Documentation**: ~20+ markdown files
- **Data directories**: Multiple collection runs
- **Checkpoints**: Multiple training runs
- **Total lines of code**: ~4,175 lines

---

**Last Updated**: January 2, 2026  
**Status**: MVP Complete, Ready for Data Collection & Training





