"""
Tests for which buffers a queue's audio cleanup is allowed to clear.

A stop tears down the audio of the session it was issued for. When playback restarts
before that teardown gets to run - only possible once the playback lock gives up on a
wedged holder - the replacement session's producers must survive it, while the stopped
session's producers still have to be killed.

An item that is taken off the queue leaves its prepared audio with it, unless a reader has
it: no cleanup of the queue reaches an item that is no longer one of its items.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import ContentType, MediaType, PlaybackState, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.streams.audio_buffer import AudioBuffer

QUEUE_ID = "q1"


def _item(item_id: str, session_id: str | None) -> QueueItem:
    """
    Build a queue item with a buffer attached, owned by the given playback session.

    :param item_id: The queue item id.
    :param session_id: Session to stamp on the stream details, or None to leave unstamped.
    """
    queue_item = QueueItem(queue_id=QUEUE_ID, queue_item_id=item_id, name=item_id, duration=180)
    audio_buffer = MagicMock(spec=AudioBuffer)
    audio_buffer.clear = AsyncMock()
    # its first audio is in
    audio_buffer.ready = asyncio.Event()
    audio_buffer.ready.set()
    queue_item.streamdetails = StreamDetails(
        provider="local--1",
        item_id=item_id,
        audio_format=AudioFormat(content_type=ContentType.MP3),
        media_type=MediaType.TRACK,
        stream_type=StreamType.HTTP,
        path=f"http://test.invalid/{item_id}.mp3",
        queue_id=QUEUE_ID,
    )
    queue_item.streamdetails.queue_session_id = session_id
    queue_item.streamdetails.buffer = audio_buffer
    return queue_item


def _controller(items: list[QueueItem], playing: str | None = None) -> PlayerQueuesController:
    """
    Build a bare controller holding one queue with the given items.

    :param items: The queue's items.
    :param playing: Session the queue is playing now, or None once the stop cleared it.
    """
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    ctrl.logger = MagicMock()
    ctrl._queue_data = {
        QUEUE_ID: PlayerQueueData(queue=MagicMock(), items=items, session_id=playing)
    }
    ctrl.mass = MagicMock()
    return ctrl


def _playing_controller(
    items: list[QueueItem], read: tuple[str, ...] = (), asked: tuple[str, ...] = ()
) -> PlayerQueuesController:
    """
    Build a bare controller whose queue plays, for taking items off it.

    :param items: The queue's items.
    :param read: The items a stream has read from in this session.
    :param asked: The items the player has a response open for.
    """
    ctrl = _controller(items, playing="sess-1")
    ctrl.signal_update = Mock()  # type: ignore[method-assign]
    ctrl.update_next_item_on_player = Mock()  # type: ignore[method-assign]
    ctrl.mass.streams.audio.read_positions = dict.fromkeys(read, 0.0)
    ctrl.mass.streams.open_item_stream_ids.return_value = set(asked)
    return ctrl


def _details(item: QueueItem) -> Any:
    """Return the item's stream details, untyped so a test can reach its buffer."""
    return cast("Any", item.streamdetails)


def _run_tasks(ctrl: PlayerQueuesController) -> list[asyncio.Task[Any]]:
    """
    Let the controller's tasks run, started at once as MusicAssistant.create_task starts them.

    :param ctrl: The controller whose tasks to run.
    :return: The tasks it creates from here on.
    """
    tasks: list[asyncio.Task[Any]] = []

    def _create_task(coro: Any, **_kwargs: Any) -> asyncio.Task[Any]:
        task = asyncio.Task(coro, loop=asyncio.get_running_loop(), eager_start=True)
        tasks.append(task)
        return task

    cast("MagicMock", ctrl.mass).create_task = Mock(side_effect=_create_task)
    return tasks


async def test_a_stop_leaves_a_newer_sessions_buffers_alone() -> None:
    """Playback that restarted before the teardown ran keeps the audio it prepared."""
    stopped = _item("stopped", "sess-1")
    replacement = _item("replacement", "sess-2")
    ctrl = _controller([stopped, replacement], playing="sess-2")

    await ctrl._cleanup_queue_audio_data(QUEUE_ID, "sess-1")

    assert stopped.streamdetails is not None
    assert stopped.streamdetails.buffer is None
    assert replacement.streamdetails is not None
    assert replacement.streamdetails.buffer is not None
    replacement.streamdetails.buffer.clear.assert_not_awaited()


async def test_a_stop_still_kills_every_buffer_of_its_own_session() -> None:
    """The stopped session's producers are what a stop exists to release."""
    playing = _item("playing", "sess-1")
    preloaded = _item("preloaded", "sess-1")
    ctrl = _controller([playing, preloaded])

    await ctrl._cleanup_queue_audio_data(QUEUE_ID, "sess-1")

    for item in (playing, preloaded):
        assert item.streamdetails is not None
        assert item.streamdetails.buffer is None


async def test_a_stop_clears_what_a_session_that_already_ended_left_behind() -> None:
    """
    Audio of a session that is no longer playing is nobody's to come back for.

    Sessions rotate without a stop - starting another item mints a new one - and a buffer
    that finished filling is left attached, so a claim from an ended session would never
    be released again if a differing stamp alone were enough to skip it.
    """
    leftover = _item("leftover", "sess-0")
    own = _item("own", "sess-1")
    ctrl = _controller([leftover, own], playing="sess-2")

    await ctrl._cleanup_queue_audio_data(QUEUE_ID, "sess-1")

    for item in (leftover, own):
        assert item.streamdetails is not None
        assert item.streamdetails.buffer is None


async def test_a_buffer_without_a_session_is_cleared_by_a_stop() -> None:
    """
    Audio that cannot be proven to belong to a later session is torn down.

    Leaving it would keep a producer alive - and its provider's stream slot with it -
    which is exactly what a stop has to prevent.
    """
    unstamped = _item("unstamped", None)
    ctrl = _controller([unstamped])

    await ctrl._cleanup_queue_audio_data(QUEUE_ID, "sess-1")

    assert unstamped.streamdetails is not None
    assert unstamped.streamdetails.buffer is None


async def test_without_a_session_every_buffer_is_cleared() -> None:
    """A clear/replace drops the items themselves, so all their audio goes with them."""
    items = [_item("a", "sess-1"), _item("b", "sess-2"), _item("c", None)]
    ctrl = _controller(items)

    await ctrl._cleanup_queue_audio_data(QUEUE_ID)

    for item in items:
        assert item.streamdetails is not None
        assert item.streamdetails.buffer is None


async def test_a_buffer_attached_while_it_is_released_is_kept() -> None:
    """
    Releasing a buffer suspends, and what a later session attaches then must survive.

    Cancelling the producer waits on the producer task, so the replacement session gets to
    run and attach its own buffer to the same stream details while that is in flight.
    """
    stopped = _item("stopped", "sess-1")
    ctrl = _controller([stopped], playing="sess-2")
    assert stopped.streamdetails is not None
    replacement_buffer = MagicMock(spec=AudioBuffer)

    async def _attach_a_replacement() -> None:
        # stands in for the new session claiming this item while the old buffer is released
        stopped.streamdetails.buffer = replacement_buffer  # type: ignore[union-attr]

    stopped.streamdetails.buffer.clear = AsyncMock(side_effect=_attach_a_replacement)

    await ctrl._cleanup_queue_audio_data(QUEUE_ID, "sess-1")

    assert stopped.streamdetails.buffer is replacement_buffer


async def test_a_session_that_starts_mid_cleanup_keeps_what_it_attaches() -> None:
    """
    The session playing now is re-read for every buffer, not decided once up front.

    Releasing a buffer suspends, so playback can start while the cleanup is part way
    through and claim items it has not reached yet.
    """
    first = _item("first", "sess-1")
    later = _item("later", "sess-1")
    ctrl = _controller([first, later])

    async def _start_a_session() -> None:
        # the queue is idle until this runs, so nothing was protected when the loop began
        ctrl._queue_data[QUEUE_ID].session_id = "sess-2"
        later.streamdetails.queue_session_id = "sess-2"  # type: ignore[union-attr]

    first.streamdetails.buffer.clear = AsyncMock(  # type: ignore[union-attr]
        side_effect=_start_a_session
    )

    await ctrl._cleanup_queue_audio_data(QUEUE_ID, "sess-1")

    assert first.streamdetails is not None
    assert first.streamdetails.buffer is None
    assert later.streamdetails is not None
    assert later.streamdetails.buffer is not None


async def test_a_pending_crossfade_handover_is_always_dropped() -> None:
    """A restarted session starts its first track from scratch, with nothing to fade from."""
    ctrl = _controller([_item("a", "sess-2")], playing="sess-2")

    await ctrl._cleanup_queue_audio_data(QUEUE_ID, "sess-1")

    clear_crossfade_handover = cast("MagicMock", ctrl.mass.streams.audio.clear_crossfade_handover)
    clear_crossfade_handover.assert_called_once_with(QUEUE_ID)


async def test_an_item_taken_off_the_queue_releases_its_prepared_audio() -> None:
    """
    A removed item's audio is let go at the removal, not at its inactivity timeout.

    The queue's cleanups walk its items, so nothing reaches the item again, and its source
    would keep its provider's stream slot until then.
    """
    playing, nxt, later = (_item(item_id, "sess-1") for item_id in ("playing", "nxt", "later"))
    ctrl = _playing_controller([playing, nxt, later], read=("playing",))
    buffer = _details(nxt).buffer

    ctrl.update_items(QUEUE_ID, [playing, later])

    assert _details(nxt).buffer is None
    buffer.clear.assert_called_once_with()
    cast("MagicMock", ctrl.mass.create_task).assert_called_once()
    for kept in (playing, later):
        assert _details(kept).buffer is not None
        _details(kept).buffer.clear.assert_not_called()


async def test_a_reorder_releases_nothing() -> None:
    """Items that only change places are all still on the queue."""
    playing, nxt, later = (_item(item_id, "sess-1") for item_id in ("playing", "nxt", "later"))
    ctrl = _playing_controller([playing, nxt, later], read=("playing",))

    ctrl.update_items(QUEUE_ID, [playing, later, nxt])

    for item in (playing, nxt, later):
        assert _details(item).buffer is not None
    cast("MagicMock", ctrl.mass.create_task).assert_not_called()


@pytest.mark.parametrize("reader", ["read", "response open"])
async def test_a_removed_item_a_reader_has_keeps_its_audio(reader: str) -> None:
    """
    An item a stream has read from, or the player has asked for, plays on from its audio.

    :param reader: Who has the removed item.
    """
    playing, nxt = _item("playing", "sess-1"), _item("nxt", "sess-1")
    ctrl = _playing_controller(
        [playing, nxt],
        read=("playing", "nxt") if reader == "read" else ("playing",),
        asked=("nxt",) if reader == "response open" else (),
    )
    queue_data = ctrl._queue_data[QUEUE_ID]
    queue_data.next_item_id_preparing = "nxt"
    buffer = _details(nxt).buffer

    ctrl.update_items(QUEUE_ID, [playing])

    assert _details(nxt).buffer is buffer
    buffer.clear.assert_not_called()
    cast("MagicMock", ctrl.mass.cancel_task).assert_not_called()
    assert queue_data.next_item_id_preparing == "nxt"
    # a response of a session the queue has moved past is one the player has left
    open_responses = cast("MagicMock", ctrl.mass.streams.open_item_stream_ids)
    open_responses.assert_called_once_with(QUEUE_ID, "sess-1")


async def test_removing_the_item_being_prepared_ends_its_preparation() -> None:
    """A preparation still waiting for a removed item's first audio ends with the item."""
    playing, nxt, later = (_item(item_id, "sess-1") for item_id in ("playing", "nxt", "later"))
    ctrl = _playing_controller([playing, nxt, later], read=("playing",))
    queue_data = ctrl._queue_data[QUEUE_ID]
    queue_data.next_item_id_preparing = "nxt"
    buffer = _details(nxt).buffer
    buffer.ready.clear()
    cancel_task = cast("MagicMock", ctrl.mass.cancel_task)

    # another item leaves: the preparation is not its own
    ctrl.update_items(QUEUE_ID, [playing, nxt])

    cancel_task.assert_not_called()
    assert queue_data.next_item_id_preparing == "nxt"
    assert _details(nxt).buffer is buffer

    ctrl.update_items(QUEUE_ID, [playing])

    cancel_task.assert_called_once_with(f"prepare_next_audio_buffer_{QUEUE_ID}")
    assert queue_data.next_item_id_preparing is None
    assert _details(nxt).buffer is None
    buffer.clear.assert_called_once_with()


async def test_a_removed_item_whose_first_audio_someone_else_awaits_is_left() -> None:
    """
    A source without its first audio that no preparation of the queue started is left alone.

    Whoever started it waits for that audio, and releasing the buffer would not wake them: a
    player that reads an item as raw PCM is in no list of open responses.
    """
    playing, nxt = _item("playing", "sess-1"), _item("nxt", "sess-1")
    ctrl = _playing_controller([playing, nxt], read=("playing",))
    buffer = _details(nxt).buffer
    buffer.ready.clear()

    ctrl.update_items(QUEUE_ID, [playing])

    assert _details(nxt).buffer is buffer
    buffer.clear.assert_not_called()


async def test_a_source_parked_on_a_full_buffer_is_closed_when_its_item_is_deleted() -> None:
    """
    A deleted track longer than its buffer gives its provider's stream slot back at once.

    Its source is parked on the full buffer with the slot held and only goes on while the
    buffer is read, and nobody reads the audio of an item that left the queue.
    """
    playing, nxt = _item("playing", "sess-1"), _item("nxt", "sess-1")
    ctrl = _playing_controller([playing, nxt], read=("playing",))
    ctrl._queue_data[QUEUE_ID].queue = PlayerQueue(
        queue_id=QUEUE_ID,
        active=True,
        display_name="Q1",
        available=True,
        items=2,
        state=PlaybackState.PLAYING,
        current_index=0,
        index_in_buffer=0,
    )
    _run_tasks(ctrl)
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_S16LE, sample_rate=8000, bit_depth=16, channels=2
    )
    slot_released = asyncio.Event()

    async def _source() -> AsyncGenerator[bytes]:
        try:
            for _ in range(10):
                yield bytes(pcm_format.pcm_sample_size)
        finally:
            # stands for the provider's stream slot, held for as long as the source lives
            slot_released.set()

    buffer = AudioBuffer(pcm_format)
    buffer.max_size_seconds = 3
    buffer.fill(_source())
    _details(nxt).buffer = buffer
    try:
        async with asyncio.timeout(1):
            while buffer.seconds_available < 3:
                await asyncio.sleep(0)
        # a source that still had room would have gone on by now
        for _ in range(5):
            await asyncio.sleep(0)
        assert buffer.seconds_available == 3
        assert buffer.is_buffering
        assert not slot_released.is_set()

        ctrl.delete_item(QUEUE_ID, "nxt")

        await asyncio.wait_for(slot_released.wait(), timeout=1)
        assert buffer.cancelled
        assert _details(nxt).buffer is None
    finally:
        await buffer.clear()


async def test_a_clear_releases_every_buffer_once() -> None:
    """
    A clear's own cleanup and the release at the swap never take the same buffer twice.

    The cleanup is a task that waits on the first buffer it releases, so the items are
    swapped out while the others are still attached.
    """
    items = [_item(item_id, "sess-1") for item_id in ("playing", "nxt", "later")]
    ctrl = _playing_controller(items, read=("playing",))
    ctrl.store_sources = Mock()  # type: ignore[method-assign]
    ctrl.is_smart_shuffle_active = Mock(return_value=False)  # type: ignore[method-assign]
    buffers = [_details(item).buffer for item in items]

    async def _wait_on_the_source() -> None:
        await asyncio.sleep(0)

    for buffer in buffers:
        buffer.clear = AsyncMock(side_effect=_wait_on_the_source)
    tasks = _run_tasks(ctrl)

    ctrl._clear(QUEUE_ID, skip_stop=True)
    await asyncio.gather(*tasks)

    for item, buffer in zip(items, buffers, strict=True):
        assert _details(item).buffer is None
        buffer.clear.assert_awaited_once_with()
