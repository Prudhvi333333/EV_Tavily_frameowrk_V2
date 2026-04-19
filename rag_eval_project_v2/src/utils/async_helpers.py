from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def gather_with_concurrency(limit: int, tasks: list[Awaitable[T]]) -> list[T]:
    semaphore = asyncio.Semaphore(limit)

    async def _run(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(*(_run(t) for t in tasks))

