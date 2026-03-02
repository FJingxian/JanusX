from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, Iterable, Iterator, TypeVar

T = TypeVar("T")


def prefetch_iter(iterable: Iterable[T], *, in_flight: int = 2) -> Iterator[T]:
    """
    Yield items from `iterable` with a small async prefetch pipeline.

    Notes
    -----
    - `in_flight=2` means "current + one prefetched item" in memory.
    - Uses a single background thread to fetch `next()` while caller computes.
    """
    depth = int(max(1, in_flight))
    if depth <= 1:
        yield from iterable
        return

    it = iter(iterable)
    sentinel = object()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="janusx-prefetch") as ex:
        queue: Deque = deque(
            [ex.submit(next, it, sentinel) for _ in range(depth)]
        )
        while len(queue) > 0:
            fut = queue.popleft()
            item = fut.result()
            if item is sentinel:
                break
            queue.append(ex.submit(next, it, sentinel))
            yield item
