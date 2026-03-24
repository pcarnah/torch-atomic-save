# torch-atomic-save

[![Pytest](https://github.com/pcarnah/torch-atomic-save/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/torch-atomic-save/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An asynchronous, atomic checkpointing utility for PyTorch, optimized for Slurm and Lustre/NFS environments.

## Key Features
* **Atomic Moves:** Prevents corrupted checkpoints during Slurm preemption.
* **Non-Blocking:** Offloads I/O to a background thread pool.
* **Race Condition Protection:** Automatically clones tensors to CPU before background saving.
* **Cross-FS Support:** Handles moves between local SSD and network storage safely.

## Installation
```bash
pip install torch-atomic-save
```


## Usage
```python
from torch_atomic_save import SlurmAtomicManager

# Initialize the manager
manager = SlurmAtomicManager(max_workers=4)

# In your training loop:
if epoch % save_interval == 0:
    manager.save(model, "path/to/checkpoints/model.pt", tmp_dir="/scratch/user/tmp")

# Ensure all I/O is finished before exiting
manager.wait_for_all()
```

## Attribution
If you use this software in your research, please cite it using the "Cite this repository" button or the provided CITATION.cff.