# torch-atomic-save

An asynchronous, atomic checkpointing utility for PyTorch, optimized for Slurm and Lustre/NFS environments.

## Key Features
* **Atomic Moves:** Prevents corrupted checkpoints during Slurm preemption.
* **Non-Blocking:** Offloads I/O to a background thread pool.
* **Race Condition Protection:** Automatically clones tensors to CPU before background saving.
* **Cross-FS Support:** Handles moves between local SSD and network storage safely.

## Installation
```bash
pip install -e .