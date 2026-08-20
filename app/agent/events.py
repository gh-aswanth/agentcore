"""Event FanOut
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator

STREAM_KEY_TEMPLATE = "run:{run_id}:events"
STREAM_TTL_SECONDS = 3600
STREAM_MAXLEN = 10_000
BEGINNING = "0"


@dataclass(frozen=True)
class RunEvent:
    id: str
    data: dict


def stream_key(run_id: str) -> str:
    return STREAM_KEY_TEMPLATE.format(run_id=run_id)


async def publish_event(redis, run_id: str, event: dict) -> str:
    """XADD + EXPIRE in one round trip. Returns the entry ID."""
    key = stream_key(run_id)
    pipe = redis.pipeline(transaction=False)
    pipe.xadd(key, {"data": json.dumps(event)}, maxlen=STREAM_MAXLEN, approximate=True)
    pipe.expire(key, STREAM_TTL_SECONDS)
    entry_id, _ = await pipe.execute()
    return entry_id


async def read_events(
    redis,
    run_id: str,
    last_id: str = BEGINNING,
    block_ms: int = 1000,
) -> AsyncIterator[RunEvent | None]:
    """Yield events from ``last_id`` onwards, forever.

    There is no replay mode and no live mode: XREAD returns any backlog
    immediately and only blocks once it reaches the tail, so there is no window
    in which an event can fall between "finished replaying" and "started
    tailing". Yields ``None`` on each block timeout as a heartbeat tick, which
    is what returns control to the caller so it can check for disconnect.
    """
    key = stream_key(run_id)
    cursor = last_id
    while True:
        response = await redis.xread({key: cursor}, count=100, block=block_ms)
        if not response:
            yield None
            continue
        for _stream, entries in response:
            for entry_id, fields in entries:
                cursor = entry_id      # XREAD is exclusive of the cursor: no dupes, no skips
                yield RunEvent(id=entry_id, data=json.loads(fields["data"]))


async def stream_exists(redis, run_id: str) -> bool:
    return bool(await redis.exists(stream_key(run_id)))
