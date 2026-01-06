# SALUS A100 Scaling Guide
**From RTX 2080 Ti Proof of Concept to A100 Production**

---

## 📊 Performance Comparison

| Metric | RTX 2080 Ti (PoC) | A100 Target | Improvement |
|--------|-------------------|-------------|-------------|
| **Data Collection** |
| Episodes | 50 | 500-1000 | 10-20× |
| Parallel Envs | 1 | 8 | 8× |
| Collection Time | 11 hours | 50-100 hours | -- |
| Episodes/hour | 4.5 | ~10 (parallel) | 2.2× |
| **Model Training** |
| Batch Size | 256 | 1024 | 4× |
| Model Size | 35K params | 150-200K params | 4-6× |
| Training Time | 3 min | 10-15 min | 3-5× slower (larger model) |
| FP16 Support | No | Yes | 2× speedup |
| **Expected Results** |
| F1 Score | 0.33 | 0.6-0.7 | 2× |
| Precision | 0.28 | 0.5-0.6 | 2× |
| Recall | 0.40 | 0.6-0.7 | 1.5-1.75× |

---

## 🚀 Quick Start

### 1. Setup A100 Environment

```bash
# On A100 machine
cd "/path/to/SalusTest"
conda activate isaaclab

# Verify A100 is detected
nvidia-smi
# Should show: NVIDIA A100-SXM4-80GB or similar
```

### 2. Parallel Data Collection (8 environments)

**Target**: 500 episodes in ~50 hours

```bash
# Start parallel collection
CUDA_VISIBLE_DEVICES=0 python scripts/collect_data_parallel_a100.py \
    --num_episodes 500 \
    --num_envs 8 \
    --save_dir a100_data/training_500eps \
    --config configs/a100_config.yaml \
    --headless \
    --enable_cameras \
    --device cuda:0 \
    > a100_logs/collection_500eps.log 2>&1 &

# Monitor progress
tail -f a100_logs/collection_500eps.log
```

**Estimated Time**: ~50 hours
- 500 episodes / 8 parallel envs = 62.5 batches
- ~13 minutes per episode × 62.5 batches = ~13.5 hours
- With overhead: ~50 hours total

### 3. Train Large Model with FP16

**Target**: F1 > 0.6 in ~15 minutes

```bash
# After data collection completes
python scripts/train_failure_predictor_a100.py \
    --data_path a100_data/training_500eps/[timestamp]/data.zarr \
    --save_dir a100_checkpoints \
    --batch_size 1024 \
    --num_epochs 100 \
    --model_size large \
    --use_amp \
    --num_workers 8 \
    > a100_logs/training_large_model.log 2>&1 &

# Monitor training
tail -f a100_logs/training_large_model.log
```

**Estimated Time**: 10-15 minutes
- 100 epochs × 8 seconds/epoch = 13 minutes

---

## 🏗️ System Architecture

### A100 Resource Allocation

**GPU Memory (80GB total)**:
- VLA Model (SmolVLA-450M): ~1GB
- Isaac Sim (8 parallel envs): ~15-20GB
- Camera rendering (8 × 3 cameras): ~5-8GB
- Physics simulation: ~5-10GB
- **Total Used**: ~30-40GB
- **Available for training**: ~40-50GB

### Parallel Collection Strategy

```
A100 (80GB)
├── VLA Ensemble (1GB)
│   └── SmolVLA-450M on cuda:0
│
├── Isaac Sim Environment Pool (~35GB)
│   ├── Env 0: Franka + Cube + 3 Cameras
│   ├── Env 1: Franka + Cube + 3 Cameras
│   ├── Env 2: Franka + Cube + 3 Cameras
│   ├── Env 3: Franka + Cube + 3 Cameras
│   ├── Env 4: Franka + Cube + 3 Cameras
│   ├── Env 5: Franka + Cube + 3 Cameras
│   ├── Env 6: Franka + Cube + 3 Cameras
│   └── Env 7: Franka + Cube + 3 Cameras
│
└── Data Buffers (~5GB)
    └── Episode storage before zarr write
```

### Training Resource Allocation

```
A100 (80GB)
├── Model (~2GB)
│   └── 200K param predictor
│
├── Optimizer States (~4GB)
│   └── AdamW moment estimates
│
├── Data Batch (~10-15GB)
│   └── 1024 samples × (12 signals + 16 labels)
│
├── Gradients (~2GB)
│   └── FP16 gradients + FP32 master weights
│
└── CUDA Context (~5GB)
    └── CuDNN, kernels, etc.

Total: ~25GB (leaves 55GB free)
```

---

## 📁 File Organization

```
SalusTest/
├── proof_of_concept_rtx2080ti/       # PoC results (archived)
│   ├── data/training_50episodes.zarr  # 66.7 MB
│   ├── models/best_predictor.pt       # 140 KB
│   └── results/                       # Analysis docs
│
├── a100_data/                         # A100 collected data
│   ├── training_500eps/
│   │   └── [timestamp]/
│   │       └── data.zarr              # ~700 MB (500 eps)
│   ├── validation_150eps/
│   │   └── [timestamp]/
│   │       └── data.zarr              # ~210 MB (150 eps)
│   └── test_150eps/
│       └── [timestamp]/
│           └── data.zarr              # ~210 MB (150 eps)
│
├── a100_checkpoints/                  # A100 trained models
│   ├── best_predictor_a100.pt         # 800 KB (200K params)
│   ├── checkpoint_epoch20.pt
│   ├── checkpoint_epoch40.pt
│   └── training_results_a100.json
│
├── a100_logs/                         # Training/collection logs
│   ├── collection_500eps.log
│   ├── training_large_model.log
│   └── evaluation_results.log
│
├── scripts/
│   ├── collect_data_parallel_a100.py  # Parallel collection
│   └── train_failure_predictor_a100.py # FP16 training
│
└── configs/
    └── a100_config.yaml               # A100 configuration
```

---

## ⚙️ Configuration Details

### Data Collection (`configs/a100_config.yaml`)

```yaml
data_collection:
  num_episodes: 500
  num_parallel_envs: 8  # Key: 8× parallelism
  max_episode_length: 200

environment:
  num_envs: 8  # Must match num_parallel_envs
  cameras:
    resolution: [224, 224]  # Reduced for speed

  perturbations:
    enabled: true
    frequency: 0.2  # 20% failure injection
```

### Training (`configs/a100_config.yaml`)

```yaml
predictor:
  hidden_dims: [128, 256, 256, 256, 128, 64]  # Large model
  dropout: 0.3

  training:
    batch_size: 1024  # A100: can handle 2048+
    use_amp: true     # FP16 mixed precision
    num_workers: 8    # Parallel data loading
    pin_memory: true  # Faster GPU transfer

    optimizer: "adamw"
    scheduler: "cosine"  # Better than ReduceLROnPlateau
    warmup_epochs: 5
```

### Temporal Forecasting

```yaml
temporal_forecasting:
  horizons:
    - name: "h1_200ms"
      steps: 6
      weight: 1.0
    - name: "h2_300ms"
      steps: 9
      weight: 1.0
    - name: "h3_400ms"
      steps: 12
      weight: 0.8      # Reduce weight for distant horizons
    - name: "h4_500ms"
      steps: 15
      weight: 0.6

  temporal_consistency:
    enabled: true
    weight: 0.1       # Encourage smooth predictions over time
```

---

## 🎯 Expected Outcomes

### Data Quality

**500 Episodes**:
- Success rate: 30-40% (vs 0% in PoC)
- Failure types:
  - Drop: 20-25%
  - Timeout: 30-35%
  - Collision: 5-10%
  - Other: 5-10%
- Positive labels: ~5-8% (vs 1.31% in PoC)

### Model Performance

**Target Metrics** (vs PoC):
```
F1 Score:    0.60-0.70  (was 0.33)  ↑ 80-110%
Precision:   0.50-0.60  (was 0.28)  ↑ 80-115%
Recall:      0.60-0.70  (was 0.40)  ↑ 50-75%
Accuracy:    >0.95      (was 0.98)  ↓ slight (less imbalance)
```

**Per-Horizon Performance**:
```
Horizon   F1 Score   Interpretation
200ms     0.70-0.75  Best (closest predictions)
300ms     0.65-0.70  Good
400ms     0.60-0.65  Moderate
500ms     0.55-0.60  Hardest (distant future)
```

---

## 🔬 Optimization Tips

### 1. GPU Memory Optimization

**If running out of memory**:

```yaml
# Reduce batch size
batch_size: 512  # instead of 1024

# Reduce number of parallel environments
num_parallel_envs: 4  # instead of 8

# Reduce camera resolution
cameras:
  resolution: [192, 192]  # instead of 224
```

### 2. Speed Optimization

**Faster data collection**:

```yaml
# Disable rendering when not needed
render: false

# Reduce physics substeps
physics:
  substeps: 1  # minimum

# Use smaller VLA ensemble
vla:
  ensemble_size: 1  # single model
```

**Faster training**:

```yaml
# Enable all optimizations
training:
  use_amp: true           # FP16 mixed precision
  num_workers: 8          # Parallel loading
  pin_memory: true        # Fast GPU transfer
  persistent_workers: true # Keep workers alive
  prefetch_factor: 4      # Pre-load batches
```

### 3. Quality Optimization

**Better generalization**:

```yaml
# More data augmentation
augmentation:
  enabled: true
  signal_noise: 0.01      # Add small noise
  temporal_jitter: 2      # Shift by ±2 timesteps

# Stronger regularization
training:
  dropout: 0.4            # More dropout
  weight_decay: 0.0002    # More L2
  label_smoothing: 0.1    # Soften labels
```

---

## 📈 Monitoring and Debugging

### Real-time Monitoring

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Memory usage
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# Collection progress
watch -n 10 'tail -n 20 a100_logs/collection_500eps.log'

# Training progress
watch -n 5 'tail -n 30 a100_logs/training_large_model.log'
```

### Performance Profiling

```bash
# Profile data collection
nsys profile --stats=true python scripts/collect_data_parallel_a100.py ...

# Profile training
python -m torch.profiler scripts/train_failure_predictor_a100.py ...
```

### Common Issues

**Issue 1**: Out of GPU memory during collection
```yaml
# Solution: Reduce parallel environments
num_parallel_envs: 4  # down from 8
```

**Issue 2**: Slow data loading during training
```yaml
# Solution: More workers and pin memory
num_workers: 16       # up from 8
pin_memory: true
prefetch_factor: 8    # up from 4
```

**Issue 3**: Poor F1 score (<0.5)
```yaml
# Solutions:
# 1. Collect more diverse data
num_episodes: 1000    # up from 500

# 2. Stronger augmentation
augmentation:
  enabled: true

# 3. Class balancing
training:
  pos_weight: 5.0     # up from 3.0
```

---

## 🧪 Validation Steps

### After Data Collection

```bash
# 1. Verify data integrity
python << 'EOF'
import zarr
store = zarr.open('a100_data/training_500eps/[timestamp]/data.zarr')
print(f"Episodes: {store['actions'].shape[0]}")
print(f"Non-zero actions: {(store['actions'][:] != 0).any()}")
print(f"Non-zero states: {(store['states'][:] != 0).any()}")
print(f"Images populated: {(store['images'][:].max() > 0)}")
EOF

# 2. Check success/failure distribution
python << 'EOF'
import zarr, numpy as np
store = zarr.open('a100_data/training_500eps/[timestamp]/data.zarr')
labels = store['horizon_labels'][:]
print(f"Positive rate: {(labels > 0).sum() / labels.size * 100:.2f}%")
EOF
```

### After Training

```bash
# 1. Verify model saved
ls -lh a100_checkpoints/best_predictor_a100.pt

# 2. Check results
cat a100_checkpoints/training_results_a100.json

# 3. Verify F1 > 0.6
python << 'EOF'
import json
with open('a100_checkpoints/training_results_a100.json') as f:
    results = json.load(f)
    f1 = results['test_metrics']['f1']
    print(f"Test F1: {f1:.4f}")
    assert f1 > 0.6, f"F1 too low: {f1:.4f}"
    print("✅ F1 target met!")
EOF
```

---

## 🎓 Next Steps After A100 Scaling

### 1. Manifold Network (Week 2)
- Train triplet network on (state, action) pairs
- 16D latent space
- Enable nearest-neighbor recovery search

### 2. Synthesis Module (Week 3)
- Generate recovery trajectories
- Use manifold to find successful states
- Online replanning

### 3. Full System Integration (Week 4)
- VLA + Predictor + Manifold + Synthesis
- Real-time intervention
- Closed-loop failure prevention

### 4. Paper Preparation (Week 5-6)
- Ablation studies
- Baseline comparisons
- Figure generation
- Write manuscript

---

## 📞 Troubleshooting

**Question**: Should I use multiple GPUs?
**Answer**: Not needed. Single A100 80GB is sufficient for 500-1000 episodes. Multi-GPU adds complexity without significant benefit for this dataset size.

**Question**: How long will 1000 episodes take?
**Answer**: ~100 hours with 8 parallel environments. Consider running overnight for 4-5 days.

**Question**: Can I resume interrupted collection?
**Answer**: Yes! The recorder checkpoints every 100 episodes. Just restart with same save_dir and it will continue from last checkpoint.

**Question**: What if F1 < 0.6 after training?
**Answer**:
1. Check data diversity (success rate should be 30-40%)
2. Increase model size to "large" if using "medium"
3. Train for more epochs (200 instead of 100)
4. Collect more data (1000 episodes instead of 500)

---

## ✅ Success Criteria

Before proceeding to publication:
- [ ] 500+ episodes collected with 8 parallel envs
- [ ] Success rate 30-40% in collected data
- [ ] All 4 failure types represented (>5% each)
- [ ] Model trains without OOM errors
- [ ] Test F1 > 0.60
- [ ] Test Precision > 0.50
- [ ] Test Recall > 0.60
- [ ] Per-horizon F1 computed
- [ ] All results documented

---

**Status**: Ready for A100 deployment
**Estimated Total Time**: 50-100 hours data collection + 15 min training
**Expected F1**: 0.6-0.7 (2× improvement from PoC)
