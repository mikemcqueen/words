# client.py
#
# Shared async HTTP client utilities for concurrent request processing.

import asyncio
import time

import httpx

SERVER_URL = "http://localhost"


def get_num_hosts():
    """Return number of backend uvicorn hosts. May later query nginx."""
    return 1


MAX_CONCURRENT = get_num_hosts() # + 1


async def run_concurrent(items, process_fn, max_concurrent=4, timeout=30.0):
    """
    Process items with bounded concurrency, yielding results as they complete.
    At most max_concurrent requests in flight at any time.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        pending = set()
        items_iter = iter(items)
        in_flight = 0

        async def tracked(item):
            nonlocal in_flight
            in_flight += 1
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] >>> REQUEST START {item['pair']} in_flight={in_flight} pending={len(pending)}")
            if in_flight > max_concurrent:
                print(f"[{ts}] !!! BUG: in_flight ({in_flight}) > max_concurrent ({max_concurrent})")
            try:
                return await process_fn(client, item)
            finally:
                in_flight -= 1
                ts_end = time.strftime("%H:%M:%S")
                print(f"[{ts_end}] <<< REQUEST END {item['pair']} in_flight={in_flight}")

        # Start initial batch
        for _ in range(max_concurrent):
            item = next(items_iter, None)
            if item is None:
                break
            task = asyncio.create_task(tracked(item))
            pending.add(task)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                yield task.result()

            # Refill to max_concurrent
            while len(pending) < max_concurrent:
                item = next(items_iter, None)
                if item is None:
                    break
                task = asyncio.create_task(tracked(item))
                pending.add(task)
