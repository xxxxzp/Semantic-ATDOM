"""Worker failures must interrupt a producer without stranding Queue feeders."""
import multiprocessing as mp
import threading
import unittest

from tools.train_progressive import DataPreProccesser


def _produce_frames(queue):
    for _ in range(32):
        queue.put(b'x' * (1024 * 1024))


def _failed_output(queue):
    queue.get(timeout=10)
    raise OSError('injected output write failure')


class WorkerFailureTests(unittest.TestCase):
    def test_preprocessor_detects_failed_output_while_trainer_is_alive(self):
        context = mp.get_context('spawn')
        frames = context.Queue(maxsize=2)
        responses = context.Queue()
        trainer = context.Process(target=_produce_frames, args=(frames,))
        output = context.Process(target=_failed_output, args=(frames,))
        trainer.start()
        output.start()
        done = threading.Event()
        errors = []
        processor = object.__new__(DataPreProccesser)
        processor.GaussianTrainer_process = trainer
        processor.DataSaver_process = output
        processor.DataPreProccesser_queue = responses

        def await_message():
            try:
                processor._wait_for_trainer_message()
            except RuntimeError as error:
                errors.append(str(error))
            finally:
                done.set()

        thread = threading.Thread(target=await_message, daemon=True)
        try:
            output.join(timeout=15)
            self.assertFalse(output.is_alive())
            self.assertTrue(trainer.is_alive())
            thread.start()
            self.assertTrue(done.wait(1), 'output failure was not detected')
            self.assertIn('output', errors[0].lower())
        finally:
            for process in (trainer, output):
                if process.is_alive():
                    process.terminate()
                process.join(timeout=3)
            if thread.ident is not None:
                thread.join(timeout=1)
            for queue in (frames, responses):
                queue.cancel_join_thread()
                queue.close()


def _drain_frames(queue):
    while queue.get() != ['stop']:
        pass


class WorkerShutdownTests(unittest.TestCase):
    def test_shutdown_detects_consumer_failure_and_cleans_blocked_producer(self):
        from semantic_atdom.utils.worker_supervision import WorkerGroup
        context = mp.get_context('spawn')
        frames = context.Queue(maxsize=2)
        control = context.Queue()
        trainer = context.Process(target=_produce_frames, args=(frames,))
        output = context.Process(target=_failed_output, args=(frames,))
        workers = WorkerGroup({'trainer': trainer, 'output': output}, timeout=10)
        trainer.start()
        output.start()
        try:
            with self.assertRaisesRegex(RuntimeError, 'output'):
                workers.stop('trainer', control)
        finally:
            workers.abort()
            for queue in (frames, control):
                queue.cancel_join_thread()
                queue.close()
        self.assertFalse(trainer.is_alive())
        self.assertFalse(output.is_alive())

    def test_successful_shutdown_drains_pending_large_frames(self):
        from semantic_atdom.utils.worker_supervision import WorkerGroup
        context = mp.get_context('spawn')
        frames = context.Queue(maxsize=2)
        output = context.Process(target=_drain_frames, args=(frames,))
        output.start()
        workers = WorkerGroup({'output': output}, timeout=15)
        try:
            frames.put(b'x' * (4 * 1024 * 1024))
            workers.stop('output', frames)
            self.assertEqual(output.exitcode, 0)
        finally:
            workers.abort()
            frames.cancel_join_thread()
            frames.close()
