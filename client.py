# client.py
#
# Shared async HTTP client utilities for concurrent request processing.

import asyncio
import httpx

SERVER_URL = "http://localhost"


def get_num_hosts():
    """Return number of backend uvicorn hosts. May later query nginx."""
    return 2


MAX_CONCURRENT = get_num_hosts() # + 1


async def run_concurrent(items, process_fn, max_concurrent=4, timeout=30.0):
    """
    Async generator that yields results as they complete.

    Args:
        items: Iterable of items to process
        process_fn: Async function(client, item) -> result
        max_concurrent: Max concurrent requests
        timeout: Request timeout in seconds

    Yields: Results as each request completes
    """
    limits = httpx.Limits(
        max_connections=max_concurrent,
        max_keepalive_connections=max_concurrent
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        pending = set()
        items_iter = iter(items)

        # Start initial batch
        for _ in range(max_concurrent):
            item = next(items_iter, None)
            if item is None:
                break
            task = asyncio.create_task(process_fn(client, item))
            pending.add(task)

        # As each completes, start the next
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                yield task.result()

            # Refill to max_concurrent
            while len(pending) < max_concurrent:
                item = next(items_iter, None)
                if item is None:
                    break
                task = asyncio.create_task(process_fn(client, item))
                pending.add(task)
