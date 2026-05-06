"""
Comprehensive tests for SlurmAtomicManager.
Run with: pytest test_manager.py -v
"""

import logging
import os
import shutil
import tempfile
import threading
import time
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Adjust import to your actual package path
from torch_atomic_save.manager import SlurmAtomicManager, install_slurm_handler


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _make_model():
    model = torch.nn.Linear(10, 1)
    return model

@pytest.fixture()
def manager():
    """Provides a fresh manager for every test, ensuring a clean thread pool."""
    mgr = SlurmAtomicManager(max_workers=4)
    yield mgr
    mgr.shutdown(wait=True)


@pytest.fixture()
def tmp(tmp_path):
    src_dir = tmp_path / "staging"
    dst_dir = tmp_path / "checkpoints"
    src_dir.mkdir()
    dst_dir.mkdir()
    return src_dir, dst_dir


# ===========================================================================
# 1. Basic Correctness & Serialization
# ===========================================================================

class TestBasicSave:
    def test_model_round_trip(self, manager, tmp):
        src_dir, dst_dir = tmp
        model = _make_model()
        dst = dst_dir / "ckpt.pt"

        manager.save(model, str(dst), str(src_dir))

        path_key = os.path.abspath(str(dst))
        manager._registry[path_key][0].result(timeout=10)

        assert dst.exists()
        loaded = torch.load(str(dst))
        # Check that weights match
        for k, v in model.state_dict().items():
            assert torch.equal(v.cpu(), loaded[k])

    def test_half_precision_save(self, manager, tmp):
        src_dir, dst_dir = tmp
        model = _make_model()
        dst = dst_dir / "ckpt_half.pt"

        manager.save(model, dst, str(src_dir), half_prec=True)

        path_key = os.path.abspath(str(dst))
        manager._registry[path_key][0].result(timeout=10)

        loaded = torch.load(str(dst))
        assert loaded['weight'].dtype == torch.float16

    def test_cpu_cloning_race_protection(self, manager, tmp):
        """Verify that modifying the model immediately after save() doesn't corrupt the checkpoint."""
        src_dir, dst_dir = tmp
        model = _make_model()
        dst = dst_dir / "race_test.pt"

        # Initialize with known value
        with torch.no_grad():
            model.weight.fill_(1.0)

        # Trigger save
        manager.save(model, str(dst), str(src_dir))

        # IMMEDIATELY change weights in main thread
        with torch.no_grad():
            model.weight.fill_(99.0)

        path_key = os.path.abspath(str(dst))
        manager._registry[path_key][0].result(timeout=10)

        loaded = torch.load(str(dst))
        # Loaded weights should be 1.0, NOT 99.0
        assert torch.all(loaded['weight'] == 1.0)

    def test_synchronous_fallback(self, manager, tmp_path):
        """Synchronous fallback when tmp_dir is None."""
        model = torch.nn.Linear(1, 1)
        dst = tmp_path / "sync_dir" / "model.pt"

        # Passing tmp_dir=None triggers the synchronous torch.save path
        manager.save(model, str(dst), tmp_dir=None)

        assert dst.exists()
        assert torch.load(str(dst))['weight'].shape == model.state_dict()['weight'].shape

# ===========================================================================
# 2. Filesystem & Atomicity
# ===========================================================================

class TestFilesystemLogic:
    def test_same_fs_uses_replace_not_copy(self, manager, tmp):
        """On same filesystem os.replace is called; shutil.copy2 must NOT be."""
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"

        with patch("shutil.copy2") as mock_copy, \
             patch("os.replace", wraps=os.replace) as mock_replace:

            assert os.stat(src_dir).st_dev == os.stat(dst_dir).st_dev, \
                "Both dirs should be on the same FS for this test"

            manager.save(_make_model(), str(dst), str(src_dir))
            manager._registry[os.path.abspath(str(dst))][0].result(timeout=10)

            mock_copy.assert_not_called()
            mock_replace.assert_called_once()

    def test_cross_fs_uses_sibling_copy(self, manager, tmp):
        """When _same_filesystem returns False the sibling-copy path is taken."""
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"

        with patch("torch_atomic_save.manager._same_filesystem", return_value=False), \
             patch("shutil.copy2", wraps=shutil.copy2) as mock_copy, \
             patch("os.replace", wraps=os.replace) as mock_replace:

            manager.save(_make_model(), str(dst), str(src_dir))
            manager._registry[os.path.abspath(str(dst))][0].result(timeout=10)


            mock_copy.assert_called_once()
            mock_replace.assert_called_once()

    def test_cross_fs_sibling_tmp_cleaned_on_success(self, manager, tmp):
        """The .atomic_tmp sibling file must not linger after a successful copy."""
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"

        with patch("torch_atomic_save.manager._same_filesystem", return_value=False):
            manager.save(_make_model(), str(dst), str(src_dir))
            manager._registry[os.path.abspath(str(dst))][0].result(timeout=10)


        leftovers = list(dst_dir.glob("*.atomic_tmp"))
        assert leftovers == [], f"Leftover sibling tmp files: {leftovers}"

    def test_staging_file_cleaned_on_success(self, manager, tmp):
        """The .pt.staging file in tmp_dir must be removed after copy."""
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"
        manager.save(_make_model(), str(dst), str(src_dir))
        manager._registry[os.path.abspath(str(dst))][0].result(timeout=10)

        leftovers = list(src_dir.glob("*.pt.staging"))
        assert leftovers == [], f"Leftover staging files: {leftovers}"


# ===========================================================================
# 3. Directory handling
# ===========================================================================

class TestDirectoryHandling:

    def test_missing_tmp_dir_raises(self, manager, tmp_path):
        """Verify that a non-existent staging directory raises FileNotFoundError immediately."""
        model = _make_model()
        dst = tmp_path / "ckpt.pt"
        staging = tmp_path / "nonexistent_staging"

        with pytest.raises(FileNotFoundError, match="Staging directory missing"):
            manager.save(model, str(dst), str(staging))

    def test_missing_dst_dir_is_created(self, manager, tmp_path):
        """Verify that deep nested destination directories are created automatically."""
        src_dir = tmp_path / "staging"
        src_dir.mkdir()
        # Nested path that doesn't exist yet
        dst = tmp_path / "deep" / "nested" / "ckpt.pt"

        manager.save(_make_model(), str(dst), str(src_dir))

        # Wait for the background task to finish
        path_key = os.path.abspath(str(dst))
        manager._registry[path_key][0].result(timeout=10)

        assert dst.exists()

    def test_existing_dst_dir_is_fine(self, manager, tmp):
        """Verify that saving to a directory that already exists does not cause errors."""
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"

        # Should not raise even if dst_dir already exists from the fixture
        manager.save(_make_model(), str(dst), str(src_dir))

        path_key = os.path.abspath(str(dst))
        manager._registry[path_key][0].result(timeout=10)
        assert dst.exists()

# ===========================================================================
# 4. Cancellation & Registry
# ===========================================================================

class TestCancellation:

    def test_cancelled_copy_does_not_call_replace(self, manager, tmp):
        """A copy whose Event is pre-set must skip os.replace entirely."""
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"
        cancelled = threading.Event()
        cancelled.set()

        # Create a dummy staging file manually for the low-level call
        fd, tmp_src = tempfile.mkstemp(dir=str(src_dir), suffix=".pt.staging")
        os.close(fd)
        torch.save(_make_model().state_dict(), tmp_src)

        with patch("os.replace") as mock_replace:
            manager._atomic_copy_and_cleanup(tmp_src, str(dst), cancelled)
            mock_replace.assert_not_called()

        assert not dst.exists()

    def test_cancelled_sibling_tmp_is_cleaned_up(self, manager, tmp):
        """Even when cancelled after copy2, the sibling_tmp must be removed."""
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"
        cancelled = threading.Event()
        cancelled.set()

        fd, tmp_src = tempfile.mkstemp(dir=str(src_dir), suffix=".pt.staging")
        os.close(fd)
        torch.save(_make_model().state_dict(), tmp_src)

        with patch("torch_atomic_save.manager._same_filesystem", return_value=False):
            manager._atomic_copy_and_cleanup(tmp_src, str(dst), cancelled)

        leftovers = list(dst_dir.glob("*.atomic_tmp"))
        assert leftovers == [], "Sibling tmp not cleaned after cancellation"

    def test_registry_cleanup_on_done(self, manager, tmp):
        src_dir, dst_dir = tmp
        dst = str(dst_dir / "clean_registry.pt")

        manager.save(_make_model(), dst, str(src_dir))
        path_key = os.path.abspath(dst)

        future, _ = manager._registry[path_key]
        future.result(timeout=5)

        # Registry should be empty after callback runs
        assert path_key not in manager._registry

    def test_stalled_copy_respects_cancellation(self, manager, tmp):
        src_dir, dst_dir = tmp
        dst = dst_dir / "stalled.pt"
        path_key = os.path.abspath(str(dst))
        barrier = threading.Barrier(2)

        original_copy2 = shutil.copy2
        stalled_once = False
        lock = threading.Lock()
        def stalling_copy2(src, dst_sibling):
            nonlocal stalled_once
            original_copy2(src, dst_sibling)

            should_stall = False
            with lock:
                if not stalled_once:
                    stalled_once = True
                    should_stall = True

            if should_stall:
                # Worker 1 waits here
                barrier.wait(timeout=5)
                # Worker 2 (and any others) bypass the barrier and finish immediately

        with patch("torch_atomic_save.manager._same_filesystem", return_value=False), \
                patch("shutil.copy2", side_effect=stalling_copy2), \
                patch("os.replace", wraps=os.replace) as mock_replace:
            # 1. Submit first save (will stall at barrier)
            manager.save(_make_model(), str(dst), str(src_dir))
            _, first_event = manager._registry[path_key]

            # 2. Submit second save (sets first_event to cancelled)
            manager.save(_make_model(), str(dst), str(src_dir))
            assert first_event.is_set()

            # 3. Release worker 1
            barrier.wait(timeout=5)

            # Wait for manager to finish
            manager._registry[path_key][0].result(timeout=10)

            # mock_replace should only be called for the second save
            # because the first one should have seen the cancellation event.
            assert mock_replace.call_count == 1


# ===========================================================================
# 5. Error handling and logging
# ===========================================================================

class TestErrorHandling:

    def test_copy_failure_does_not_raise_in_main_thread(self, manager, tmp, caplog):
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"
        model = _make_model()

        # Patch the manager's package path
        with patch("torch_atomic_save.manager.shutil.copy2",
                   side_effect=OSError("Lustre exploded")), \
                patch("torch_atomic_save.manager._same_filesystem", return_value=False), \
                caplog.at_level(logging.ERROR):

            manager.save(model, str(dst), str(src_dir))
            path_key = os.path.abspath(str(dst))

            # Fetching result surfaces exception in the future, but shouldn't crash loop
            try:
                manager._registry[path_key][0].result(timeout=10)
            except Exception:
                pass

        # Verify background callback logged the error
        start = time.time()
        logged = False
        while time.time() - start < 5:
            if any("Lustre exploded" in r.message for r in caplog.records):
                logged = True
                break
            time.sleep(0.1)

        assert logged, "Error should be logged"

    def test_replace_failure_cleans_sibling_tmp(self, manager, tmp):
        """If os.replace fails the sibling_tmp should not be left behind."""
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"

        cancelled = threading.Event()
        fd, tmp_src = tempfile.mkstemp(dir=str(src_dir), suffix=".pt.staging")
        os.close(fd)
        torch.save(_make_model().state_dict(), tmp_src)

        with patch("torch_atomic_save.manager._same_filesystem", return_value=False), \
                patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError):
                manager._atomic_copy_and_cleanup(tmp_src, str(dst), cancelled)

        leftovers = list(dst_dir.glob("*.atomic_tmp"))
        assert leftovers == [], "sibling_tmp must be cleaned up even on replace failure"

    def test_staging_file_cleaned_even_on_copy_failure(self, manager, tmp):
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"

        cancelled = threading.Event()
        fd, tmp_src = tempfile.mkstemp(dir=str(src_dir), suffix=".pt.staging")
        os.close(fd)
        torch.save(_make_model().state_dict(), tmp_src)

        with patch("torch_atomic_save.manager._same_filesystem", return_value=False), \
                patch("torch_atomic_save.manager.shutil.copy2", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                manager._atomic_copy_and_cleanup(tmp_src, str(dst), cancelled)

        assert not os.path.exists(tmp_src), "Staging file must be cleaned up after failure"


# ===========================================================================
# 6. Multiple independent paths
# ===========================================================================

class TestMultiplePaths:

    def test_independent_paths_tracked_separately(self, manager, tmp):
        src_dir, dst_dir = tmp
        dst_a = dst_dir / "model_a.pt"
        dst_b = dst_dir / "model_b.pt"

        # Use a barrier to force the background threads to wait
        # 3 parties: Worker A, Worker B, and this Test Thread
        barrier = threading.Barrier(3)

        def blocked_save(*args, **kwargs):
            barrier.wait(timeout=5)
            # Use the real torch.save logic after the barrier
            torch.save(*args, **kwargs)

        # Patch torch.save to block the workers in the registry
        with patch("torch_atomic_save.manager.torch.save", side_effect=blocked_save):
            manager.save(_make_model(), str(dst_a), str(src_dir))
            manager.save(_make_model(), str(dst_b), str(src_dir))

            key_a = os.path.abspath(str(dst_a))
            key_b = os.path.abspath(str(dst_b))

            # 1. Verify they are both in the registry while "blocked"
            assert key_a in manager._registry
            assert key_b in manager._registry
            assert manager._registry[key_a] is not manager._registry[key_b]

            # 2. Release the workers
            barrier.wait(timeout=5)

        # 3. Now wait for completion normally
        manager.wait_for_all(timeout=10)

        assert dst_a.exists()
        assert dst_b.exists()

    def test_cancel_does_not_affect_other_paths(self, manager, tmp):
        src_dir, dst_dir = tmp
        dst_a = dst_dir / "model_a.pt"
        dst_b = dst_dir / "model_b.pt"

        # Save to B and wait
        manager.save(_make_model(), str(dst_b), str(src_dir))
        manager._registry[os.path.abspath(str(dst_b))][0].result(timeout=10)

        # Trigger a replacement save on A
        manager.save(_make_model(), str(dst_a), str(src_dir))
        manager.save(_make_model(), str(dst_a), str(src_dir))

        # Ensure B is unaffected and still exists
        assert dst_b.exists()


# ===========================================================================
# 7. Thread pool
# ===========================================================================

class TestThreadPool:

    def test_executor_is_reused_across_calls(self, manager, tmp):
        src_dir, dst_dir = tmp
        # Since the executor is now instance-bound, we verify it's the same
        # object across multiple .save calls on the same manager instance
        e1 = manager._executor
        manager.save(_make_model(), str(dst_dir / "a.pt"), str(src_dir))
        e2 = manager._executor
        assert e1 is e2, "Executor should be reused by the manager instance"

    def test_many_sequential_saves_all_land(self, manager, tmp):
        src_dir, dst_dir = tmp
        dst = dst_dir / "ckpt.pt"
        n = 10

        # Use a model with a single weight we can track
        model = _make_model()

        for i in range(1, n + 1):
            with torch.no_grad():
                model.weight.fill_(float(i))
            manager.save(model, str(dst), str(src_dir))

        # Wait for whichever future is current in the registry
        path_key = os.path.abspath(str(dst))
        manager._registry[path_key][0].result(timeout=30)

        loaded = torch.load(str(dst))
        # Check that the weight matches the final iteration
        assert torch.all(loaded['weight'] == float(n))

    def test_context_manager_usage(self, tmp):
        """__enter__ and __exit__ logic."""
        from torch_atomic_save.manager import SlurmAtomicManager

        with SlurmAtomicManager(max_workers=1) as manager:
            assert manager._executor is not None

        # Verify executor is shut down after context exit
        assert manager._executor._shutdown is True

    def test_shutdown_ignores_new_saves(self, manager, tmp, caplog):
        """Manager is shutting down warning."""
        src_dir, dst_dir = tmp
        manager.shutdown(wait=True)

        with caplog.at_level(logging.WARNING):
            manager.save(torch.nn.Linear(1, 1), str(dst_dir / "late.pt"), str(src_dir))

        assert "Manager is shutting down" in caplog.text

    def test_cleanup_registry_safety(self, manager, tmp):
        """Ensure cleanup handles missing paths gracefully."""
        # Manually call private cleanup on a non-existent path to hit the 'if current' check
        future = MagicMock()
        manager._cleanup_registry("non_existent_path", future)
        # Should not raise any errors

    def test_done_callback_race_deterministic(self, tmp):
        src_dir, dst_dir = tmp
        dst = str(dst_dir / "deterministic_race.pt")

        manager = SlurmAtomicManager(max_workers=4)
        deadlock_detected = threading.Event()

        try:
            original_submit = manager._executor.submit

            def delayed_submit(fn, *args, **kwargs):
                future = original_submit(fn, *args, **kwargs)
                # Block until the future is actually done before returning,
                # so add_done_callback is guaranteed to fire synchronously
                # on the calling thread which still holds self._lock
                future.result(timeout=5.0)
                return future

            with patch('torch.save', return_value=None), \
                    patch.object(manager, '_atomic_copy_and_cleanup', return_value=None), \
                    patch.object(manager._executor, 'submit', side_effect=delayed_submit):

                state = {k: v.cpu().clone() for k, v in _make_model().state_dict().items()}

                def do_save():
                    # This will deadlock here if the bug is present,
                    # since _atomic_save holds self._lock when delayed_submit
                    # returns a completed future, causing add_done_callback
                    # to fire synchronously and re-acquire self._lock
                    manager._atomic_save(state, dst, str(src_dir))

                t = threading.Thread(target=do_save, daemon=True)
                t.start()
                t.join(timeout=5.0)

                assert not t.is_alive(), \
                    "Deadlock: _atomic_save hung, done callback fired while lock was held"
        finally:
            # Can't call shutdown normally if deadlocked — force it
            manager._executor.shutdown(wait=False)

# ===========================================================================
# 8. Slurm Signals
# ===========================================================================

class TestSignals:
    def test_sigterm_handler_logic(self, manager):
        """Verifies handler logic without killing the process (and the coverage report)."""
        install_slurm_handler(manager)

        # Retrieve the function actually registered with the OS
        handler_func = signal.getsignal(signal.SIGTERM)

        with patch.object(manager, 'shutdown') as mock_shutdown:
            # Manually trigger the function. This is 'Logic Coverage'.
            handler_func(signal.SIGTERM, None)

            # This confirms your code DOES the right thing when the signal hits
            mock_shutdown.assert_called_once_with(wait=True)

    def test_sigterm_handler_manual_trigger(self, manager, caplog):
        """Targets lines 140-141: SIGTERM handler logging."""
        from torch_atomic_save.manager import install_slurm_handler
        import signal

        install_slurm_handler(manager)
        handler = signal.getsignal(signal.SIGTERM)

        with caplog.at_level(logging.INFO), patch.object(manager, 'shutdown') as mock_shat:
            # Simulate signal arrival
            handler(signal.SIGTERM, None)

        assert "Received SIGTERM" in caplog.text
        mock_shat.assert_called_once_with(wait=True)

