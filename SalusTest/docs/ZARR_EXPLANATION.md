# What is Zarr Storage?

## Quick Answer

**Zarr** is a chunked, compressed storage format for large multi-dimensional arrays (like NumPy arrays). Think of it as "HDF5 but optimized for cloud/distributed computing and very large datasets."

---

## Why SALUS Uses Zarr

SALUS collects **massive amounts of data**:
- **500+ episodes** of robot interactions
- Each episode: ~200 timesteps
- Each timestep: RGB images (224×224×3), robot states, actions, signals
- **Total data**: Can easily be 50-100GB+ uncompressed

### Traditional Formats (HDF5, Pickle, NPY) Problems:
- ❌ Load entire file into memory (doesn't scale)
- ❌ Slow random access
- ❌ Hard to parallelize reads/writes
- ❌ Not efficient for cloud storage

### Zarr Advantages:
- ✅ **Chunked storage** - Only loads chunks you need
- ✅ **Compression** - Reduces storage by 3-5×
- ✅ **Parallel I/O** - Multiple processes can read/write simultaneously
- ✅ **Incremental writing** - Append data as you collect it
- ✅ **Cloud-friendly** - Works with S3, GCS, Azure

---

## How Zarr Works

### Concept: Chunked Arrays

Instead of storing one big array:
```
[████████████████████████████████]  ← One huge file (slow to load)
```

Zarr splits it into chunks:
```
[███][███][███][███][███][███][███][███][███]  ← Many small chunks
```

**Benefits**:
- Only load the chunks you need (fast!)
- Parallel reads/writes (multiple chunks at once)
- Compression per chunk (better compression)

---

## Zarr vs HDF5 in SALUS

### HDF5 (Traditional Approach)
```python
# Save one episode per file
with h5py.File(f"episode_{i}.h5", 'w') as f:
    f.create_dataset('rgb', data=rgb_images)      # (200, 224, 224, 3)
    f.create_dataset('actions', data=actions)      # (200, 7)
    # ... etc

# Problems:
# - 500 episodes = 500 files (hard to manage)
# - To train: Must load all 500 files sequentially (slow)
# - No compression by default
```

### Zarr (SALUS Approach)
```python
# All episodes in one Zarr store
zarr_store = zarr.open("data.zarr", mode='a')

# Structure:
# data.zarr/
#   ├── images: (500, 200, 3, 224, 224)  ← All episodes, chunked
#   ├── actions: (500, 200, 7)           ← All episodes, chunked
#   ├── signals: (500, 200, 12)          ← All episodes, chunked
#   └── metadata: (500,)                 ← Episode labels

# Benefits:
# - Single file/directory (easy to manage)
# - Random access: zarr_store['images'][episode_42] (fast!)
# - Compression: zstd compression (3-5× smaller)
# - Parallel reads: Multiple workers can read simultaneously
```

---

## SALUS Implementation

Looking at `salus/data/recorder.py`:

```python
import zarr

# Create Zarr store
zarr_store = zarr.open("data.zarr", mode='a')

# Create arrays with chunking
zarr_store.create_array(
    'images',
    shape=(500, 200, 3, 224, 224),  # 500 episodes, 200 timesteps
    chunks=(100, 10, 1, 224, 224),   # Chunk size: 100 episodes, 10 timesteps
    dtype='uint8',
    compressor=zarr.Zstd(level=3)    # Compression
)

# Write data incrementally
zarr_store['images'][episode_idx, :timesteps] = episode_images

# Read data (only loads needed chunks)
images = zarr_store['images'][42, :]  # Load episode 42 only
```

---

## Real Example from SALUS

### Data Structure in Zarr:

```
data.zarr/
├── images:      (500, 200, 3, 224, 224)  uint8  [chunked, compressed]
├── states:      (500, 200, 7)            float32
├── actions:     (500, 200, 7)            float32
├── signals:     (500, 200, 12)           float32
├── horizon_labels: (500, 200, 4, 4)      float32
└── episode_metadata: (500,)              object (JSON strings)
```

### Loading for Training (`salus/data/dataset_mvp.py`):

```python
# Open Zarr store
zarr_root = zarr.open("data.zarr", mode='r')

# Load specific episode (only loads chunks containing that episode)
signals = zarr_root['signals'][episode_idx, timestep]  # Fast!

# PyTorch DataLoader can read in parallel
dataset = SALUSMVPDataset("data.zarr")
dataloader = DataLoader(dataset, batch_size=64, num_workers=4)  # 4 workers read chunks in parallel
```

---

## Performance Comparison

| Operation | HDF5 (500 files) | Zarr (1 store) |
|-----------|-----------------|----------------|
| **Write 500 episodes** | ~2 hours | ~1 hour (parallel writes) |
| **Load 1 episode** | ~100ms | ~50ms (chunked) |
| **Load all episodes** | ~30 min | ~5 min (parallel) |
| **Storage size** | 100GB | 25GB (compressed) |
| **Random access** | Slow (file open overhead) | Fast (direct chunk access) |

---

## Key Benefits for SALUS

1. **Scalable**: Handles TB-scale datasets (future-proof)
2. **Efficient**: Only loads what you need (memory-friendly)
3. **Fast Training**: Parallel DataLoader workers read chunks simultaneously
4. **Small Storage**: Compression reduces disk usage
5. **Incremental Collection**: Can append episodes as you collect them

---

## Installation

```bash
pip install zarr
```

Already included in SALUS `requirements.txt`.

---

## Further Reading

- **Zarr Documentation**: https://zarr.readthedocs.io/
- **Zarr Tutorial**: https://zarr.readthedocs.io/en/stable/tutorial.html
- **Why Zarr?**: https://zarr.dev/

---

## Summary

**Zarr** = Chunked + Compressed + Parallel-friendly storage for large arrays

**For SALUS**: Essential for managing 50-100GB+ of robot episode data efficiently.

**Think of it as**: "HDF5 but designed for the cloud era and massive datasets"




