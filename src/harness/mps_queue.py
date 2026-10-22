"""One bounded FIFO, one executor thread, no overlapping model calls."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.config import WorkerConfig
from src.harness.qwen_model import Probe, ProbeResult


@dataclass
class _Job:
    probe: Probe
    future: asyncio.Future


class MPSBatchWorker:
    def __init__(self, harness, config: WorkerConfig):
        self.harness, self.config = harness, config
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mps-consumer")
        self._task = None
        self._closed = False
        self._submit_lock = asyncio.Lock()
        self.batches = 0
        self.harness.claim(self)

    async def __aenter__(self):
        if self._task is not None or self._closed:
            raise RuntimeError("Worker cannot be started twice")
        self._task = asyncio.create_task(self._consume())
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def submit(self, probe: Probe) -> ProbeResult:
        async with self._submit_lock:
            if self._closed or self._task is None:
                raise RuntimeError("Worker is not accepting jobs")
            future = asyncio.get_running_loop().create_future()
            await self.queue.put(_Job(probe, future))
        return await future

    async def map(self, probes: list[Probe]) -> list[ProbeResult]:
        # Bound producer task creation too; callers can stream corpus chunks.
        results = []
        for start in range(0, len(probes), self.config.batch_size):
            results.extend(
                await asyncio.gather(
                    *[self.submit(p) for p in probes[start : start + self.config.batch_size]]
                )
            )
        return results

    async def close(self):
        async with self._submit_lock:
            if not self._closed:
                self._closed = True
                if self._task is not None:
                    await self.queue.put(None)
        if self._task is not None:
            await asyncio.shield(self._task)
        self.executor.shutdown(wait=True)

    async def _consume(self):
        loop = asyncio.get_running_loop()
        pending = None
        while True:
            job = pending if pending is not None else await self.queue.get()
            pending = None
            if job is None:
                self.queue.task_done()
                break
            batch = [job]
            # Let simultaneous producers enqueue before collecting this deterministic FIFO batch.
            await asyncio.sleep(0)
            limit = (
                self.config.attribution_batch_size
                if job.probe.spec.attribution_target
                else self.config.batch_size
            )
            stop = False
            while len(batch) < limit:
                try:
                    other = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if other is None:
                    self.queue.task_done()
                    stop = True
                    break
                if other.probe.spec != job.probe.spec:
                    pending = other
                    break
                batch.append(other)
            active = [item for item in batch if not item.future.cancelled()]
            try:
                if active:
                    results = await loop.run_in_executor(
                        self.executor, self.harness._execute, [j.probe for j in active], self
                    )
                    if len(results) != len(active):
                        raise RuntimeError("Harness returned incorrect result count")
                    for item, result in zip(active, results):
                        if not item.future.done():
                            item.future.set_result(result)
            except Exception as exc:
                for item in active:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()
                if active:
                    self.batches += 1
                    if self.batches % self.config.purge_every == 0:
                        # Cache purge failure must not strand queued futures.
                        try:
                            await loop.run_in_executor(self.executor, self.harness.purge)
                        except RuntimeError:
                            pass
            if stop:
                break
