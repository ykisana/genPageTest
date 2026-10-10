"""Phase timers for batch-1 serving latency.

With sync=True every section synchronizes the device on entry/exit so asynchronous MPS work is
charged to the phase that issued it (adds a little overhead; use sync=False for end-to-end).
"""
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


class Profiler:
    def __init__(self, dev, enabled=True, sync_sections=True):
        self.dev, self.enabled, self.sync_sections = dev, enabled, sync_sections
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        if self.sync_sections:
            sync(self.dev)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_sections:
                sync(self.dev)
            self.t[name] += time.perf_counter() - t0
            self.n[name] += 1


class NullProfiler(Profiler):
    def __init__(self):
        super().__init__(torch.device("cpu"), enabled=False)


NULL = NullProfiler()
