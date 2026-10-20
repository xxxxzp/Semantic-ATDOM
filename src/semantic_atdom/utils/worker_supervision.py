"""Bounded shutdown for cooperating multiprocessing workers and their queues."""
import time
from queue import Full


class WorkerGroup:
    def __init__(self, workers, timeout=300.0):
        self.workers = dict(workers)
        self.expected = set()
        self.timeout = timeout

    def check(self):
        for name, process in self.workers.items():
            if process.exitcode is not None and (
                name not in self.expected or process.exitcode != 0
            ):
                raise RuntimeError(
                    f"{name} worker exited unexpectedly with code {process.exitcode}"
                )

    def stop(self, name, queue):
        """Drain this worker while continuing to check every dependent worker."""
        deadline = time.monotonic() + self.timeout
        self.check()
        self.expected.add(name)
        while True:
            self.check()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out stopping {name} worker")
            try:
                queue.put(["stop"], timeout=0.1)
                break
            except Full:
                continue
        process = self.workers[name]
        while process.is_alive():
            self.check()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out draining {name} worker")
            process.join(timeout=0.1)
        self.check()

    def abort(self):
        # Terminate *all* producers before joining: a Queue feeder may be
        # waiting for an already dead consumer even after its target returned.
        for process in self.workers.values():
            if process.is_alive():
                process.terminate()
        for process in self.workers.values():
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
