import os
import shutil
import tempfile
import threading
import concurrent.futures
import logging
import signal
import torch
from typing import Any, Optional, Dict, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class SlurmAtomicManager:
    """
    Manages asynchronous, atomic checkpoint saving across different filesystems.
    Optimized for Slurm/Lustre environments.
    """

    def __init__(self, max_workers: int = 4):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="SlurmSaveWorker"
        )
        # path -> (future, cancellation_event)
        self._registry: Dict[str, Tuple[concurrent.futures.Future, threading.Event]] = {}
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def wait_for_all(self, timeout: Optional[float] = None):
        """Blocks until all currently pending saves are complete."""
        with self._lock:
            futures = [f for f, _ in self._registry.values()]
        concurrent.futures.wait(futures, timeout=timeout)

    def save(self, model: torch.nn.Module, path: Union[str, Path], tmp_dir: Optional[str] = None, half_prec: bool = False):
        """
        Main entry point for saving models.
        Moves tensors to CPU and clones them to prevent race conditions.
        """
        path = os.path.abspath(path)

        # Snapshot weights: Move to CPU and Clone
        if half_prec:
            state_dict = {k: v.half().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if tmp_dir is None:
            # Synchronous fallback
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)
        else:
            self._atomic_save(state_dict, path, tmp_dir)

    def _atomic_save(self, obj: Any, path: str, tmp_dir: str) -> None:
        """
        Saves an object to a staging area, then schedules an atomic cross-FS move.
        """
        if self._shutdown_event.is_set():
            logger.warning(f"Manager is shutting down. Ignoring save request for {path}")
            return

        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.isdir(tmp_dir):
            raise FileNotFoundError(f"Staging directory missing: {tmp_dir}")

        # 1. Setup staging dir
        fd, tmp_src = tempfile.mkstemp(dir=tmp_dir, suffix=".pt.staging")
        os.close(fd)

        # 2. Registry Update & Cancellation
        cancelled = threading.Event()
        with self._lock:
            if path in self._registry:
                prev_f, prev_ev = self._registry[path]
                prev_ev.set()  # Signal background thread to skip the move
                prev_f.cancel()

            future = self._executor.submit(
                self._save_and_copy, obj, tmp_src, path, cancelled
            )
            self._registry[path] = (future, cancelled)

        future.add_done_callback(lambda f: self._cleanup_registry(path, f))

    def _cleanup_registry(self, path: str, future: concurrent.futures.Future) -> None:
        """Removes the future from registry and logs errors."""
        with self._lock:
            current = self._registry.get(path)
            if current and current[0] is future:
                self._registry.pop(path, None)

        if not future.cancelled():
            exc = future.exception()
            if exc:
                logger.error(f"Async save to {path} failed: {exc}", exc_info=exc)

    def _save_and_copy(self, obj, tmp_src, dst, cancelled):
        """Handles both the initial write and the atomic move."""
        try:
            if not cancelled.is_set():
                torch.save(obj, tmp_src)

            self._atomic_copy_and_cleanup(tmp_src, dst, cancelled)
        except Exception as e:
            # Cleanup tmp_src if torch.save failed
            if os.path.exists(tmp_src):
                os.unlink(tmp_src)
            raise

    def _atomic_copy_and_cleanup(self, src: str, dst: str, cancelled: threading.Event, max_retries: int = 5) -> None:
        """The background worker logic."""
        dst_dir = os.path.dirname(os.path.abspath(dst))
        try:
            # Check if same device (ST_DEV)
            if _same_filesystem(src, dst):
                if not cancelled.is_set():
                    os.replace(src, dst)
            else:
                # Cross-FS: Create sibling tmp in destination's parent
                fd, sibling_tmp = tempfile.mkstemp(dir=dst_dir, suffix=".atomic_tmp")
                os.close(fd)
                try:
                    shutil.copy2(src, sibling_tmp)
                    if not cancelled.is_set():
                        os.replace(sibling_tmp, dst)
                    else:
                        os.unlink(sibling_tmp)
                except Exception:
                    try:
                        os.unlink(sibling_tmp)
                    except OSError:
                        pass
                    raise
        finally:
            try:
                os.unlink(src)
            except OSError:
                pass

    def shutdown(self, wait: bool = True):
        """Cleanly shut down the executor."""
        self._shutdown_event.set()
        self._executor.shutdown(wait=wait)


def install_slurm_handler(manager: SlurmAtomicManager):
    """
    Installs a handler for SIGTERM (Slurm preemption) to ensure
    the manager flushes pending IO before the process exits.
    """

    def handle_sigterm(signum, frame):
        logger.info("Received SIGTERM/Preemption signal. Flushing IO...")
        manager.shutdown(wait=True)
        # Optional: Re-raise or exit
        # os._exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)


def _same_filesystem(path_a: str, path_b: str) -> bool:
    stat_b_target = path_b if os.path.exists(path_b) else os.path.dirname(os.path.abspath(path_b))
    return os.stat(path_a).st_dev == os.stat(stat_b_target).st_dev
