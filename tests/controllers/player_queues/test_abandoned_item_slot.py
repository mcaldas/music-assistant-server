"""
Tests for a starting item taking a one-stream provider's slot from a source nothing reads.

A track longer than its buffer keeps its provider's stream slot while its source is parked
on the full buffer. A player that leaves such a track for another item leaves nobody to
read it, and the other item's source then waited out its whole budget behind it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import PlaybackState

import music_assistant.controllers.streams.audio as audio_mod
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.models.music_provider import MusicProvider, ProviderStreamLimitError
from tests.controllers.player_queues.test_paused_queue_slot import PCM_FORMAT, _queue_item, _Rig

if TYPE_CHECKING:
    from music_assistant_models.queue_item import QueueItem
    from music_assistant_models.streamdetails import StreamDetails

QUEUE = "balcony"


class _TwoStreamProvider(MusicProvider):
    """Streaming provider with two source slots."""

    @property
    def max_concurrent_streams(self) -> int:
        """Return two source slots."""
        return 2


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> _Rig:
    """Build the rig with every music source open to the queue's playback."""
    monkeypatch.setattr(audio_mod, "playback_sources", AsyncMock(return_value=(None, [])))
    return _Rig()


def _add_item(rig: _Rig, name: str) -> QueueItem:
    """
    Append another item of the provider, with ids of its own, to the queue.

    :param rig: The rig, with the queue added.
    :param name: What the item's ids are made from.
    """
    item = _queue_item(name)
    item.queue_id = QUEUE
    assert item.streamdetails is not None
    item.streamdetails.queue_id = QUEUE
    rig.queues._queue_data[QUEUE].items.append(item)
    return item


def _asks_for(rig: _Rig, *items: QueueItem) -> None:
    """Set the items the queue's player has an open response for."""
    rig.mass.streams.open_item_stream_ids.return_value = {item.queue_item_id for item in items}


def _details_outside_a_queue(name: str) -> StreamDetails:
    """Build stream details of the provider that no queue item carries, as an analysis reads."""
    details = _queue_item(name).streamdetails
    assert details is not None
    details.queue_id = None
    return details


def _watch_releases(rig: _Rig) -> MagicMock:
    """Return a spy on the looks for a source nothing reads, which still run."""
    spy = MagicMock(wraps=rig.queues.release_abandoned_stream_slot)
    rig.queues.release_abandoned_stream_slot = spy  # type: ignore[method-assign]
    return spy


async def test_a_start_takes_the_slot_of_an_item_the_player_left(rig: _Rig) -> None:
    """The old response ended first: the asked item's source aborts the one nothing reads."""
    left = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    left_buffer = await rig.fill(left)
    asked = _add_item(rig, "asked")
    _asks_for(rig, asked)

    buffer = await rig.audio.get_audio_buffer(asked, reason="streaming", capacity_wait_timeout=2)

    assert buffer.is_buffering
    assert left_buffer.cancelled
    # the aborted buffer stays attached, as for a start
    assert left.streamdetails is not None
    assert left.streamdetails.buffer is left_buffer
    await buffer.clear()


async def test_a_start_that_came_first_gets_the_slot_when_the_old_response_ends(rig: _Rig) -> None:
    """The asked item's request came first: it waits as before, until the old response ended."""
    left = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    left_buffer = await rig.fill(left)
    asked = _add_item(rig, "asked")
    _asks_for(rig, left, asked)
    start = asyncio.ensure_future(
        rig.audio.get_audio_buffer(asked, reason="streaming", capacity_wait_timeout=2)
    )
    async with asyncio.timeout(1):
        while not rig.tasks:
            await asyncio.sleep(0)
    # the start's own look found the old item still read
    assert not await rig.tasks[0]
    assert not start.done()
    assert left_buffer.is_buffering

    # what the handler of the old response does once it has ended
    _asks_for(rig, asked)
    assert await rig.queues.release_abandoned_stream_slot(QUEUE)

    buffer = await start
    assert buffer.is_buffering
    assert left_buffer.cancelled
    await buffer.clear()


async def test_a_start_behind_an_item_still_read_waits_as_before(rig: _Rig) -> None:
    """An item with an open response keeps its source, and a start behind it fails as it did."""
    left = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    left_buffer = await rig.fill(left)
    asked = _add_item(rig, "asked")
    _asks_for(rig, left, asked)

    with pytest.raises(ProviderStreamLimitError):
        await rig.audio.get_audio_buffer(asked, reason="streaming", capacity_wait_timeout=0.2)

    assert left_buffer.is_buffering
    await left_buffer.clear()


async def test_a_probe_gets_the_slot_too(rig: _Rig) -> None:
    """The look comes before the wait for the slot, so a start that does not wait gets it."""
    left = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    left_buffer = await rig.fill(left)
    asked = _add_item(rig, "asked")
    _asks_for(rig, asked)
    assert asked.streamdetails is not None

    buffer = await AudioBuffer.get_buffer(
        rig.mass, asked.streamdetails, wait_ready=True, source_wait_timeout=0
    )

    assert buffer.is_buffering
    assert left_buffer.cancelled
    await buffer.clear()


async def test_a_release_that_fails_leaves_the_start_waiting(rig: _Rig) -> None:
    """A source that cannot be aborted costs the start nothing: it waits for the slot as before."""
    left = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    left_buffer = await rig.fill(left)
    asked = _add_item(rig, "asked")
    _asks_for(rig, asked)
    abort = left_buffer.clear
    failing_abort = AsyncMock(side_effect=RuntimeError("teardown failed"))
    left_buffer.clear = failing_abort  # type: ignore[method-assign]

    with pytest.raises(ProviderStreamLimitError):
        await rig.audio.get_audio_buffer(asked, reason="streaming", capacity_wait_timeout=0.2)

    failing_abort.assert_awaited_once()
    await abort()


async def test_a_start_that_is_given_up_leaves_the_release_to_finish(rig: _Rig) -> None:
    """
    A start that is cancelled while the release it asked for runs ends alone.

    The release is shared by every start of the queue that finds no slot. Cancelled with
    one of them it would stop in the middle of the abort, or take the cancel for itself
    and leave the start running that was given up.
    """
    aborting, closing = asyncio.Event(), asyncio.Event()

    async def _slow_to_close(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        try:
            for _ in range(3):
                yield b"\x00" * PCM_FORMAT.pcm_sample_size
            await asyncio.Event().wait()
        finally:
            # a source takes a moment to close, and the abort waits for it
            aborting.set()
            await closing.wait()

    rig.audio._get_media_stream = _slow_to_close  # type: ignore[method-assign]
    left = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    left_buffer = await rig.fill(left)
    asked = _add_item(rig, "asked")
    _asks_for(rig, asked)
    assert asked.streamdetails is not None
    try:
        asked_buffer = await AudioBuffer.get_buffer(rig.mass, asked.streamdetails)
        # the start waits for the release, and the release for the source it aborts
        await asyncio.wait_for(aborting.wait(), 1)
        release = rig.tasks[0]

        # the player gave up on the item: its start ends at once, the release goes on
        given_up = asyncio.ensure_future(asked_buffer.clear())
        await asyncio.wait({given_up}, timeout=1)
        assert given_up.done()
        assert not release.done()
    finally:
        closing.set()

    assert await release
    assert left_buffer.cancelled
    assert rig.provider.has_available_stream_slot


async def test_a_response_that_ends_with_nothing_asked_looks_no_further(rig: _Rig) -> None:
    """With no item asked for there is no start to hand a slot to: the look ends there."""
    left = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    left_buffer = await rig.fill(left)
    _asks_for(rig)
    fade_targets = MagicMock(wraps=rig.audio.crossfade_targets)
    rig.audio.crossfade_targets = fade_targets  # type: ignore[method-assign]

    assert not await rig.queues.release_abandoned_stream_slot(QUEUE)

    fade_targets.assert_not_called()
    assert left_buffer.is_buffering
    await left_buffer.clear()


async def test_a_read_outside_the_queues_keeps_the_slot(rig: _Rig) -> None:
    """A source no queue item carries is never aborted, and its own start looks at no queue."""
    asked = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    _asks_for(rig, asked)
    loose_buffer = await AudioBuffer.get_buffer(
        rig.mass, _details_outside_a_queue("analysis"), wait_ready=True, source_wait_timeout=0
    )
    releases = _watch_releases(rig)

    with pytest.raises(ProviderStreamLimitError):
        await AudioBuffer.get_buffer(
            rig.mass, _details_outside_a_queue("preview"), wait_ready=True, source_wait_timeout=0
        )
    releases.assert_not_called()

    with pytest.raises(ProviderStreamLimitError):
        await rig.audio.get_audio_buffer(asked, reason="streaming", capacity_wait_timeout=0.2)

    assert loose_buffer.is_buffering
    await loose_buffer.clear()


async def test_nothing_is_asked_while_the_slot_is_free(rig: _Rig) -> None:
    """A start that finds a free slot takes it without a look at the queue."""
    asked = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    _asks_for(rig, asked)
    releases = _watch_releases(rig)

    buffer = await rig.audio.get_audio_buffer(asked, reason="streaming", capacity_wait_timeout=2)

    assert buffer.is_buffering
    releases.assert_not_called()
    await buffer.clear()


async def test_a_provider_with_more_than_one_stream_is_untouched(rig: _Rig) -> None:
    """Which of several slots a source holds is not known, so such a provider is left alone."""
    rig.provider = _TwoStreamProvider(MagicMock(), rig.provider.manifest, rig.provider.config)  # type: ignore[assignment]
    left = rig.add_queue(QUEUE, PlaybackState.PLAYING)
    left_buffer = await rig.fill(left)
    read = _add_item(rig, "read")
    read_buffer = await rig.fill(read)
    asked = _add_item(rig, "asked")
    _asks_for(rig, read, asked)
    releases = _watch_releases(rig)

    with pytest.raises(ProviderStreamLimitError):
        await rig.audio.get_audio_buffer(asked, reason="streaming", capacity_wait_timeout=0.2)

    releases.assert_not_called()
    assert left_buffer.is_buffering
    await left_buffer.clear()
    await read_buffer.clear()
