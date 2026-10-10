"""Regression tests for delayed next-track enqueueing in the player queue stream feeder."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType, PlaybackState, RepeatMode
from music_assistant_models.errors import AudioError, QueueEmpty
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.player_queues.stream_feeder import (
    PRELOAD_IDLE_ATTEMPTS,
    PRELOAD_RETRY_DELAY,
    PREPARE_AFTER_PART_DELAY,
)
from music_assistant.controllers.streams.constants import STREAM_SLOT_WAIT_TIMEOUT
from music_assistant.models.music_provider import MusicProvider, ProviderStreamLimitError

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


@pytest.mark.parametrize(
    "index_in_buffer",
    [0, 1],
    ids=["aligned-buffer-index", "dynamic-queue-reindexed"],
)
async def test_enqueue_next_item_waits_for_playing_player_update(index_in_buffer: int) -> None:
    """Enqueue the expected next item after an update, even if its buffered index is stale."""
    controller = PlayerQueuesController.__new__(PlayerQueuesController)
    controller.logger = MagicMock()

    wait_entered = asyncio.Event()
    release_wait = asyncio.Event()

    @asynccontextmanager
    async def wait_for_player_update(*_args: object, **_kwargs: object) -> AsyncIterator[None]:
        wait_entered.set()
        await release_wait.wait()
        yield

    player_state = SimpleNamespace(
        playback_state=PlaybackState.IDLE,
        active_source="q1",
    )
    player = SimpleNamespace(state=player_state)

    mass = MagicMock()
    mass.players = MagicMock()
    mass.players.wait_for_player_update = MagicMock(side_effect=wait_for_player_update)
    mass.players.get_player = MagicMock(return_value=player)
    mass.players.enqueue_next_media = AsyncMock()
    controller.mass = mass

    current_item = _make_queue_item("q1", "nerin")
    next_item = _make_queue_item("q1", "another-love")
    future_item = _make_queue_item("q1", "future-track")
    queue_items = [current_item, next_item, future_item]
    queue = PlayerQueue(
        queue_id="q1",
        active=True,
        display_name="Q1",
        available=True,
        items=len(queue_items),
        state=PlaybackState.IDLE,
        current_index=0,
        index_in_buffer=index_in_buffer,
        current_item=current_item,
    )
    controller._queue_data = {
        "q1": PlayerQueueData(
            queue=queue,
            items=queue_items,
            session_id="session-1",
        )
    }

    controller._enqueue_next_item("q1", next_item)
    enqueue_callback = mass.call_later.call_args.args[1]
    enqueue_task = asyncio.create_task(enqueue_callback(next_item))
    await asyncio.sleep(0)

    assert wait_entered.is_set()
    mass.players.enqueue_next_media.assert_not_awaited()

    player_state.playback_state = PlaybackState.PLAYING
    release_wait.set()
    await enqueue_task

    mass.players.wait_for_player_update.assert_called_once_with(
        "q1",
        attribute_name="playback_state",
        attribute_value=PlaybackState.PLAYING,
    )
    mass.players.enqueue_next_media.assert_awaited_once()
    assert mass.players.enqueue_next_media.await_args.kwargs["player_id"] == "q1"
    assert (
        mass.players.enqueue_next_media.await_args.kwargs["media"].queue_item_id
        == next_item.queue_item_id
    )
    assert controller._queue_data["q1"].next_item_id_enqueued == next_item.queue_item_id


def _make_queue_item(queue_id: str, item_id: str) -> QueueItem:
    """Build a minimal playable queue item."""
    return QueueItem(
        queue_id=queue_id,
        queue_item_id=item_id,
        name=item_id,
        duration=60,
    )


def _controller_with_next_item() -> tuple[PlayerQueuesController, SimpleNamespace, MagicMock]:
    """Build a bare controller whose streamed item is followed by an unprepared next item."""
    controller = PlayerQueuesController.__new__(PlayerQueuesController)
    controller.logger = MagicMock()
    current_item = SimpleNamespace(
        queue_item_id="current",
        media_type=MediaType.TRACK,
        streamdetails=None,
        name="Current",
        available=True,
        extra_attributes={},
    )
    next_item = SimpleNamespace(
        queue_item_id="next",
        media_type=MediaType.TRACK,
        streamdetails=SimpleNamespace(buffer=None),
        name="Next",
        available=True,
        extra_attributes={},
    )
    queue = SimpleNamespace(
        current_item=current_item,
        next_item=next_item,
        current_index=0,
        index_in_buffer=0,
        repeat_mode=RepeatMode.OFF,
        display_name="Queue",
    )
    controller.get = MagicMock(return_value=queue)  # type: ignore[method-assign]
    controller._queue_data = {
        "queue-1": cast(
            "Any",
            SimpleNamespace(
                queue=queue,
                items=[current_item, next_item],
                session_id="session-1",
                next_item_id_preparing=None,
                last_served_item_id=None,
                prepared_ahead=None,
                transitioning=False,
            ),
        )
    }

    async def _load_next(queue_id: str, item_id: str, **_kwargs: object) -> Any:
        if (item := controller.get_next_item(queue_id, item_id)) is None:
            raise QueueEmpty
        return item

    controller.load_next_queue_item = AsyncMock(side_effect=_load_next)  # type: ignore[method-assign]
    mass = MagicMock()
    controller.mass = mass
    return controller, next_item, mass


def _streamed_item(controller: PlayerQueuesController) -> SimpleNamespace:
    """Return the item whose stream precedes the next item."""
    return cast("SimpleNamespace", controller._queue_data["queue-1"].items[0])


async def test_reusing_a_warm_buffer_claims_it_for_the_current_session() -> None:
    """
    A prewarm that is already warm still becomes this session's audio.

    Without the claim the buffer keeps the session that filled it, and that session's stop
    releases audio the current one is relying on.
    """
    controller, next_item, mass = _controller_with_next_item()
    warm = MagicMock()
    warm.is_valid.return_value = True
    warm.eof = True
    warm.has_error = False
    next_item.streamdetails = SimpleNamespace(
        buffer=warm, queue_session_id="session-0", media_type=MediaType.TRACK, seek_position=0
    )
    controller._queue_data["queue-1"].last_served_item_id = "current"

    controller.prepare_next_audio_buffer("queue-1", "current")

    assert next_item.streamdetails.queue_session_id == "session-1"
    mass.create_task.assert_not_called()


async def test_prepare_next_uses_the_speculative_capacity_budget() -> None:
    """Warming the next track never waits longer for capacity than a speculative attempt may."""
    controller, next_item, mass = _controller_with_next_item()
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    mass.streams.audio.get_audio_buffer.assert_awaited_once_with(
        next_item,
        seek_position_ms=0,
        reason="prepare_next",
        capacity_wait_timeout=STREAM_SLOT_WAIT_TIMEOUT,
        allow_provider_match=False,
        stop_paused_queues=False,
    )
    assert mass.create_task.call_args.kwargs == {
        "task_id": "prepare_next_audio_buffer_queue-1",
        "task_name": "prepare_next_audio_buffer_queue-1",
        "abort_existing": True,
    }


@pytest.mark.parametrize(
    ("is_buffering", "expect_cleared"),
    [(True, True), (False, False)],
    ids=["still_filling", "completed"],
)
async def test_an_aborted_prepare_releases_its_half_filled_source(
    is_buffering: bool, expect_cleared: bool
) -> None:
    """Aborting a prewarm must free its slot instead of pinning it until the inactivity sweep."""
    controller, next_item, mass = _controller_with_next_item()
    buffer = MagicMock()
    buffer.is_buffering = is_buffering
    buffer.clear = AsyncMock()
    started = asyncio.Event()

    async def _hang(*_args: object, **_kwargs: object) -> None:
        # the producer is already running and owns a source slot at this point
        next_item.streamdetails.buffer = buffer
        started.set()
        await asyncio.Event().wait()

    mass.streams.audio.get_audio_buffer = _hang

    controller.prepare_next_audio_buffer("queue-1", "current")
    prepare_task = asyncio.create_task(mass.create_task.call_args.args[0])
    await started.wait()
    prepare_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_task

    assert buffer.clear.await_count == (1 if expect_cleared else 0)


async def test_prepare_next_gives_up_softly_on_a_capacity_failure() -> None:
    """A speculative source-capacity miss leaves the next item playable."""
    controller, next_item, mass = _controller_with_next_item()
    provider = MagicMock(spec=MusicProvider)
    provider.max_concurrent_streams = 1
    provider.name = "Limited"
    provider.instance_id = "limited--1"
    mass.streams.audio.get_audio_buffer = AsyncMock(
        side_effect=ProviderStreamLimitError(provider, STREAM_SLOT_WAIT_TIMEOUT)
    )

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    assert next_item.available


async def test_a_prepare_looks_ahead_without_writing_items_off() -> None:
    """Preparing the next item only looks ahead, and ends quietly when its provider is away."""
    controller, _next_item, mass = _controller_with_next_item()
    load = AsyncMock(side_effect=AudioError("offline"))
    controller.load_next_queue_item = load  # type: ignore[method-assign]
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    # the load is what would write the item off, and looking ahead it does not
    load.assert_awaited_once_with("queue-1", "current", speculative=True)
    mass.streams.audio.get_audio_buffer.assert_not_awaited()


async def test_prepare_next_defers_while_the_streamed_item_holds_the_only_source_slot() -> None:
    """A preload for the same single-slot source waits for the boundary, not for a timeout."""
    controller, next_item, mass = _controller_with_next_item()
    next_item.streamdetails = SimpleNamespace(buffer=None, provider="limited--1")
    playing = MagicMock()
    playing.eof = False
    _streamed_item(controller).streamdetails = SimpleNamespace(
        provider="limited--1", buffer=playing, is_realtime=True
    )
    # the audible item trails the stream and holds no source
    cast("Any", controller.get("queue-1")).current_item = SimpleNamespace(streamdetails=None)
    provider = MagicMock(spec=MusicProvider)
    provider.max_concurrent_streams = 1
    provider.has_available_stream_slot = False
    mass.get_provider = MagicMock(return_value=provider)
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    mass.streams.audio.get_audio_buffer.assert_not_awaited()


async def test_prepare_next_only_defers_for_a_realtime_source() -> None:
    """A source that fills ahead of playback frees its slot in time and gets no deferral."""
    controller, next_item, mass = _controller_with_next_item()
    next_item.streamdetails = SimpleNamespace(buffer=None, provider="limited--1")
    playing = MagicMock()
    playing.eof = False
    _streamed_item(controller).streamdetails = SimpleNamespace(
        provider="limited--1", buffer=playing, is_realtime=False
    )
    provider = MagicMock(spec=MusicProvider)
    provider.max_concurrent_streams = 1
    provider.has_available_stream_slot = False
    mass.get_provider = MagicMock(return_value=provider)
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    mass.streams.audio.get_audio_buffer.assert_awaited_once()


async def test_prepare_next_runs_once_the_streamed_item_released_the_slot() -> None:
    """The same preload goes ahead when the streamed item's source has finished."""
    controller, next_item, mass = _controller_with_next_item()
    next_item.streamdetails = SimpleNamespace(buffer=None, provider="limited--1")
    finished = MagicMock()
    finished.eof = True
    _streamed_item(controller).streamdetails = SimpleNamespace(
        provider="limited--1", buffer=finished, is_realtime=True
    )
    provider = MagicMock(spec=MusicProvider)
    provider.max_concurrent_streams = 1
    provider.has_available_stream_slot = False
    mass.get_provider = MagicMock(return_value=provider)
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    mass.streams.audio.get_audio_buffer.assert_awaited_once()


async def test_prepare_next_skips_an_item_that_left_the_queue_while_it_was_fetched() -> None:
    """
    A replace that lands while the stream details are still being fetched ends the prewarm.

    The item the prewarm was scheduled for is no longer on the queue, so warming its audio
    would decode a track nobody will play and pin a source slot on an orphaned buffer.
    """
    controller, next_item, mass = _controller_with_next_item()
    next_item.streamdetails = None

    async def _replace_queue_meanwhile(*_args: object, **_kwargs: object) -> SimpleNamespace:
        controller._queue_data["queue-1"].items.clear()
        next_item.streamdetails = SimpleNamespace(buffer=None)
        return next_item

    controller.load_next_queue_item = _replace_queue_meanwhile  # type: ignore[method-assign, assignment]
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    mass.streams.audio.get_audio_buffer.assert_not_awaited()


async def test_prepare_next_releases_a_buffer_whose_item_left_the_queue_mid_fill() -> None:
    """
    A removal that lands while the buffer fills still releases the warmed audio.

    Replace-next and delete do not cancel the prewarm, so without the check after the fill
    the finished buffer stays attached to an item no cleanup reaches any more, holding its
    audio until the inactivity sweep.
    """
    controller, next_item, mass = _controller_with_next_item()
    buffer = MagicMock()
    buffer.clear = AsyncMock()

    async def _remove_item_meanwhile(*_args: object, **_kwargs: object) -> MagicMock:
        controller._queue_data["queue-1"].items.clear()
        next_item.streamdetails.buffer = buffer
        return buffer

    mass.streams.audio.get_audio_buffer = _remove_item_meanwhile

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    buffer.clear.assert_awaited_once()
    assert next_item.streamdetails.buffer is None


async def test_prepare_next_leaves_the_buffer_of_an_item_still_on_the_queue() -> None:
    """A fill that raced nothing keeps its buffer attached for the upcoming track."""
    controller, next_item, mass = _controller_with_next_item()
    buffer = MagicMock()
    buffer.clear = AsyncMock()

    async def _fill(*_args: object, **_kwargs: object) -> MagicMock:
        next_item.streamdetails.buffer = buffer
        return buffer

    mass.streams.audio.get_audio_buffer = _fill

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    buffer.clear.assert_not_awaited()
    assert next_item.streamdetails.buffer is buffer


async def test_prepare_next_creates_no_buffer_once_the_session_ended() -> None:
    """
    A stop that lands while the next item is resolved ends the prewarm.

    The stop already released its session's audio, so a buffer warmed now would stay attached
    to a stopped queue that nothing cleans up any more.
    """
    controller, next_item, mass = _controller_with_next_item()

    async def _stop_meanwhile(*_args: object, **_kwargs: object) -> SimpleNamespace:
        controller._queue_data["queue-1"].session_id = None
        return next_item

    controller.load_next_queue_item = _stop_meanwhile  # type: ignore[method-assign, assignment]
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    mass.streams.audio.get_audio_buffer.assert_not_awaited()


async def test_prepare_next_releases_a_buffer_that_filled_after_the_session_ended() -> None:
    """
    A stop that lands while the buffer fills still releases the warmed audio.

    The stop's cleanup has already run by the time the fill finishes, so the buffer is
    detached and released here instead of holding its source on a stopped queue.
    """
    controller, next_item, mass = _controller_with_next_item()
    buffer = MagicMock()
    detached_on_clear: list[bool] = []
    buffer.clear = AsyncMock(
        side_effect=lambda: detached_on_clear.append(next_item.streamdetails.buffer is None)
    )

    async def _stop_meanwhile(*_args: object, **_kwargs: object) -> MagicMock:
        controller._queue_data["queue-1"].session_id = None
        next_item.streamdetails.buffer = buffer
        return buffer

    mass.streams.audio.get_audio_buffer = _stop_meanwhile

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    buffer.clear.assert_awaited_once()
    assert next_item.streamdetails.buffer is None
    assert detached_on_clear == [True]


async def test_prepare_next_keeps_the_buffer_when_the_session_rotated_mid_fill() -> None:
    """A skip that starts a new session while the buffer fills keeps the audio for that session."""
    controller, next_item, mass = _controller_with_next_item()
    buffer = MagicMock()
    buffer.clear = AsyncMock()

    async def _skip_meanwhile(*_args: object, **_kwargs: object) -> MagicMock:
        controller._queue_data["queue-1"].session_id = "session-2"
        next_item.streamdetails.buffer = buffer
        return buffer

    mass.streams.audio.get_audio_buffer = _skip_meanwhile

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    buffer.clear.assert_not_awaited()
    assert next_item.streamdetails.buffer is buffer


async def test_prepare_next_follows_the_streamed_item_not_the_audible_one() -> None:
    """
    The item after the streamed one is prepared while the player still plays an earlier one.

    A player that reads a whole track ahead reports an audible item that trails the stream,
    so the queue's next item is the streamed item itself.
    """
    controller, next_item, mass = _controller_with_next_item()
    audible_item = SimpleNamespace(
        queue_item_id="audible",
        media_type=MediaType.TRACK,
        streamdetails=None,
        name="Audible",
        available=True,
    )
    queue_data = controller._queue_data["queue-1"]
    streamed_item = _streamed_item(controller)
    queue_data.items.insert(0, cast("Any", audible_item))
    queue = cast("Any", queue_data.queue)
    queue.current_item = audible_item
    queue.next_item = streamed_item
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    cast("AsyncMock", controller.load_next_queue_item).assert_awaited_once_with(
        "queue-1", "current", speculative=True
    )
    mass.streams.audio.get_audio_buffer.assert_awaited_once()
    assert mass.streams.audio.get_audio_buffer.await_args.args[0] is next_item


async def test_a_prepare_aborted_while_resolving_the_item_just_stops() -> None:
    """Aborting a prewarm before its item is resolved has no source to release."""
    controller, _next_item, mass = _controller_with_next_item()
    started = asyncio.Event()

    async def _hang(*_args: object, **_kwargs: object) -> None:
        started.set()
        await asyncio.Event().wait()

    controller.load_next_queue_item = _hang  # type: ignore[method-assign, assignment]
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    prepare_task = asyncio.create_task(mass.create_task.call_args.args[0])
    await started.wait()
    prepare_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_task

    mass.streams.audio.get_audio_buffer.assert_not_awaited()


async def test_prepare_next_does_nothing_when_the_item_repeats_itself() -> None:
    """With repeat-one the streamed item follows itself, and its own audio is not prepared."""
    controller, _next_item, mass = _controller_with_next_item()
    cast("Any", controller._queue_data["queue-1"].queue).repeat_mode = RepeatMode.ONE

    controller.prepare_next_audio_buffer("queue-1", "current")

    mass.create_task.assert_not_called()


def _fully_buffered_controller(
    *, is_realtime: bool, served: str | None
) -> tuple[PlayerQueuesController, MagicMock]:
    """
    Build a controller whose streamed item has fully arrived, with a stubbed preparation.

    :param is_realtime: Whether the streamed item's source hands over its audio just-in-time.
    :param served: The item the player is fetching, or None when it fetched nothing yet.
    """
    controller, _next_item, _mass = _controller_with_next_item()
    _streamed_item(controller).streamdetails = SimpleNamespace(
        is_realtime=is_realtime, media_type=MediaType.TRACK, seek_position=0
    )
    controller._queue_data["queue-1"].last_served_item_id = served
    prepare = MagicMock()
    controller.prepare_next_audio_buffer = prepare  # type: ignore[method-assign]
    return controller, prepare


async def test_a_fully_arrived_realtime_track_prepares_its_successor() -> None:
    """A realtime track the player is fetching chains into preparing the next item."""
    controller, prepare = _fully_buffered_controller(is_realtime=True, served="current")

    controller.track_fully_buffered("queue-1", "current")

    prepare.assert_called_once_with("queue-1", "current")


async def test_a_fully_arrived_source_that_fills_ahead_prepares_nothing() -> None:
    """A source that fills ahead of playback leaves its successor to the end of its stream."""
    controller, prepare = _fully_buffered_controller(is_realtime=False, served="current")

    controller.track_fully_buffered("queue-1", "current")

    prepare.assert_not_called()


async def test_a_fully_arrived_track_the_player_has_not_fetched_prepares_nothing() -> None:
    """
    The fills do not chain ahead of what the player has fetched.

    The crossfade path raises the buffered index to the incoming track before the player
    asks for it, so that index does not count as the player fetching the track.
    """
    controller, prepare = _fully_buffered_controller(is_realtime=True, served="audible")
    queue_data = controller._queue_data["queue-1"]
    queue_data.items.insert(
        0, cast("Any", SimpleNamespace(queue_item_id="audible", available=True))
    )
    queue = cast("Any", queue_data.queue)
    queue.current_index = 0
    queue.index_in_buffer = 1

    controller.track_fully_buffered("queue-1", "current")

    prepare.assert_not_called()


async def test_a_fully_arrived_track_without_a_served_item_prepares_nothing() -> None:
    """A queue whose player fetched nothing yet gives the fills nothing to chain on."""
    controller, prepare = _fully_buffered_controller(is_realtime=True, served=None)

    controller.track_fully_buffered("queue-1", "current")

    prepare.assert_not_called()


async def test_a_repeated_prepare_for_the_same_item_joins_the_running_one() -> None:
    """Asking again for the audio of the same next item does not restart its preparation."""
    controller, _next_item, mass = _controller_with_next_item()

    controller.prepare_next_audio_buffer("queue-1", "current")
    controller.prepare_next_audio_buffer("queue-1", "current")

    assert [call.kwargs["abort_existing"] for call in mass.create_task.call_args_list] == [
        True,
        False,
    ]
    for call in mass.create_task.call_args_list:
        call.args[0].close()


async def test_a_prepare_for_another_item_replaces_the_running_one() -> None:
    """A queue that changed under a preparation gets the new next item prepared instead."""
    controller, _next_item, mass = _controller_with_next_item()
    other_item = SimpleNamespace(
        queue_item_id="other",
        media_type=MediaType.TRACK,
        streamdetails=None,
        name="Other",
        available=True,
    )

    controller.prepare_next_audio_buffer("queue-1", "current")
    # the item following the streamed one changes before the second call
    controller._queue_data["queue-1"].items.insert(1, cast("Any", other_item))
    controller.prepare_next_audio_buffer("queue-1", "current")

    assert [call.kwargs["abort_existing"] for call in mass.create_task.call_args_list] == [
        True,
        True,
    ]
    for call in mass.create_task.call_args_list:
        call.args[0].close()


async def test_a_repeated_prepare_hands_back_the_preparation_already_running(
    mass_minimal: MusicAssistant,
) -> None:
    """The second call for the same item returns the running preparation, uncancelled."""
    controller, _next_item, _mass = _controller_with_next_item()
    controller.mass = mass_minimal
    resolving = asyncio.Event()

    async def _hang(*_args: object, **_kwargs: object) -> None:
        resolving.set()
        await asyncio.Event().wait()

    controller.load_next_queue_item = _hang  # type: ignore[method-assign, assignment]

    first = controller.prepare_next_audio_buffer("queue-1", "current")
    await resolving.wait()
    second = controller.prepare_next_audio_buffer("queue-1", "current")

    assert first is not None
    assert second is first
    assert not first.cancelled()
    assert not first.done()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_a_repeated_prepare_joins_a_preparation_that_skipped_ahead() -> None:
    """A preparation that skipped an unplayable item is joined, not restarted."""
    controller, next_item, mass = _controller_with_next_item()
    later_item = SimpleNamespace(
        queue_item_id="later",
        media_type=MediaType.TRACK,
        streamdetails=None,
        name="Later",
        available=True,
        extra_attributes={},
    )
    controller._queue_data["queue-1"].items.append(cast("Any", later_item))
    filling = asyncio.Event()

    async def _skip_unplayable(*_args: object, **_kwargs: object) -> SimpleNamespace:
        next_item.available = False
        return later_item

    async def _hang(*_args: object, **_kwargs: object) -> None:
        # no buffer is attached yet while the source is being opened
        filling.set()
        await asyncio.Event().wait()

    controller.load_next_queue_item = _skip_unplayable  # type: ignore[method-assign, assignment]
    mass.streams.audio.get_audio_buffer = _hang

    controller.prepare_next_audio_buffer("queue-1", "current")
    preparation = asyncio.create_task(mass.create_task.call_args.args[0])
    await filling.wait()
    controller.prepare_next_audio_buffer("queue-1", "current")

    assert [call.kwargs["abort_existing"] for call in mass.create_task.call_args_list] == [
        True,
        False,
    ]
    mass.create_task.call_args.args[0].close()
    preparation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparation


def _controller_with_playing_item() -> tuple[PlayerQueuesController, SimpleNamespace, MagicMock]:
    """Build a controller whose streamed item is the playing one, three minutes long."""
    controller, next_item, mass = _controller_with_next_item()
    _streamed_item(controller).duration = 180
    queue = cast("Any", controller.get("queue-1"))
    queue.flow_mode = False
    queue.state = PlaybackState.PLAYING
    queue.corrected_elapsed_time = 0
    return controller, next_item, mass


async def test_a_preload_whose_provider_did_not_answer_asks_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A next item whose provider could not be asked is looked up again and handed over."""
    controller, next_item, mass = _controller_with_playing_item()
    load = AsyncMock(side_effect=[AudioError("offline"), next_item])
    controller.load_next_queue_item = load  # type: ignore[method-assign]
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    controller._preload_next_item("queue-1", "current")
    await mass.create_task.call_args.args[0]

    assert [awaited.kwargs for awaited in load.await_args_list] == [{"speculative": True}] * 2
    sleep.assert_awaited_once_with(PRELOAD_RETRY_DELAY)
    assert mass.call_later.call_args.args[2] is next_item


@pytest.mark.parametrize(
    ("duration", "last_attempt_at"),
    [(180, 150), (165, 140)],
    ids=["30_s_before_the_end", "not_sooner_than_the_first_wait"],
)
async def test_a_preload_may_step_over_once_the_playing_item_is_nearly_over(
    monkeypatch: pytest.MonkeyPatch, duration: int, last_attempt_at: int
) -> None:
    """With too little of the playing item left to ask again, the load may step over."""
    controller, next_item, mass = _controller_with_playing_item()
    _streamed_item(controller).duration = duration
    queue = cast("Any", controller.get("queue-1"))

    async def _load(_queue_id: str, _item_id: str, speculative: bool = False) -> SimpleNamespace:
        if speculative:
            raise AudioError("offline")
        return next_item

    async def _play_on(seconds: float) -> None:
        queue.corrected_elapsed_time += seconds

    load = AsyncMock(side_effect=_load)
    controller.load_next_queue_item = load  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", _play_on)

    controller._preload_next_item("queue-1", "current")
    await mass.create_task.call_args.args[0]

    # asked at 0, 10, 30, 70 and 130 s, then once more for the item to be stepped over
    assert [awaited.kwargs for awaited in load.await_args_list] == [
        *[{"speculative": True}] * 5,
        {"speculative": False},
    ]
    # five failed lookups, said once
    assert cast("MagicMock", controller.logger).warning.call_count == 1
    assert queue.corrected_elapsed_time == last_attempt_at
    assert mass.call_later.call_args.args[2] is next_item


async def test_a_preload_counts_the_seconds_a_failed_lookup_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lookup that was slow to fail leaves less of the playing item to wait in."""
    controller, next_item, mass = _controller_with_playing_item()
    queue = cast("Any", controller.get("queue-1"))

    async def _load(_queue_id: str, _item_id: str, speculative: bool = False) -> SimpleNamespace:
        if not speculative:
            return next_item
        if queue.corrected_elapsed_time == 130:
            # this one hangs for half a minute before it fails
            queue.corrected_elapsed_time += 30
        raise AudioError("offline")

    async def _play_on(seconds: float) -> None:
        queue.corrected_elapsed_time += seconds

    controller.load_next_queue_item = AsyncMock(side_effect=_load)  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", _play_on)

    controller._preload_next_item("queue-1", "current")
    await mass.create_task.call_args.args[0]

    # the attempt that may step over still comes before the item ends at 180 s
    assert queue.corrected_elapsed_time == 170
    assert mass.call_later.call_args.args[2] is next_item


async def test_a_preload_in_flow_mode_leaves_stepping_over_to_the_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flow steps over at its own boundary, so its preload only ever looks ahead."""
    controller, next_item, mass = _controller_with_playing_item()
    queue = cast("Any", controller.get("queue-1"))
    queue.flow_mode = True
    queue.corrected_elapsed_time = 170
    load = AsyncMock(side_effect=AudioError("offline"))
    controller.load_next_queue_item = load  # type: ignore[method-assign]

    async def _flow_moves_on(_seconds: float) -> None:
        if load.await_count == 3:
            queue.current_item = next_item

    monkeypatch.setattr(asyncio, "sleep", _flow_moves_on)

    controller._preload_next_item("queue-1", "current")
    await mass.create_task.call_args.args[0]

    assert [awaited.kwargs for awaited in load.await_args_list] == [{"speculative": True}] * 3


async def test_a_preload_stops_when_the_player_has_moved_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preload whose item is no longer the playing one after its wait asks nothing more."""
    controller, next_item, mass = _controller_with_playing_item()
    queue = cast("Any", controller.get("queue-1"))
    load = AsyncMock(side_effect=AudioError("offline"))
    controller.load_next_queue_item = load  # type: ignore[method-assign]

    async def _move_on(_seconds: float) -> None:
        queue.current_item = next_item

    monkeypatch.setattr(asyncio, "sleep", _move_on)

    controller._preload_next_item("queue-1", "current")
    await mass.create_task.call_args.args[0]

    load.assert_awaited_once()
    mass.call_later.assert_not_called()


async def test_a_preload_gives_up_on_a_queue_that_is_not_playing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paused item never nears its end, so the asking about its next item ends by count."""
    controller, next_item, mass = _controller_with_playing_item()
    # a pause on the device can read as idle, with the session and the item still there
    cast("Any", controller.get("queue-1")).state = PlaybackState.IDLE
    load = AsyncMock(side_effect=AudioError("offline"))
    controller.load_next_queue_item = load  # type: ignore[method-assign]

    async def _stay_paused(_seconds: float) -> None:
        assert load.await_count < 20, "the preload asks for as long as the queue is paused"

    monkeypatch.setattr(asyncio, "sleep", _stay_paused)

    controller._preload_next_item("queue-1", "current")
    await mass.create_task.call_args.args[0]

    assert [awaited.kwargs for awaited in load.await_args_list] == [
        {"speculative": True}
    ] * PRELOAD_IDLE_ATTEMPTS
    assert next_item.available
    mass.call_later.assert_not_called()


def _part(
    item_id: str, *, eof: bool = True, positions: dict[str, float] | None = None
) -> SimpleNamespace:
    """
    Build a track that plays only a part (it ends early), with its audio buffered.

    :param item_id: The queue item id.
    :param eof: Whether the part's source has delivered all of its audio.
    :param positions: The item's start and end attributes, instead of an end at 90 s.
    """
    buffer = MagicMock()
    buffer.is_valid.return_value = True
    buffer.eof = eof
    buffer.has_error = False
    return SimpleNamespace(
        queue_item_id=item_id,
        media_type=MediaType.TRACK,
        streamdetails=SimpleNamespace(
            buffer=buffer,
            media_type=MediaType.TRACK,
            is_realtime=False,
            seek_position=0,
            allow_seek=True,
            duration=200,
            queue_session_id="session-1",
        ),
        name=item_id,
        available=True,
        extra_attributes={"end_position": 90.0} if positions is None else positions,
    )


def _controller_with_parts(
    *item_ids: str, served: str | None, last_ready: bool = False
) -> tuple[PlayerQueuesController, MagicMock]:
    """
    Build a controller whose queue is a whole streamed track followed by buffered parts.

    :param item_ids: The parts that follow the streamed item "current", in queue order.
    :param served: The item the player is fetching.
    :param last_ready: Whether the last part is buffered too, instead of not prepared yet.
    """
    controller, _next_item, mass = _controller_with_next_item()
    queue_data = controller._queue_data["queue-1"]
    queue_data.items[1:] = [cast("Any", _part(item_id)) for item_id in item_ids]
    if not last_ready:
        queue_data.items[-1].streamdetails.buffer = None
    queue_data.last_served_item_id = served
    return controller, mass


def _prepared_for(mass: MagicMock) -> list[str]:
    """Return the item each created preparation was started behind, closing the coroutines."""
    started = []
    for call in mass.create_task.call_args_list:
        started.append(call.args[0].cr_frame.f_locals["queue_item_id"])
        call.args[0].close()
    return started


async def test_a_fully_arrived_part_behind_the_fetched_item_prepares_its_successor() -> None:
    """
    The item after a part is prepared as soon as the part's own audio has arrived.

    A player that fetches a track moments before it plays reads a part to its fade-out
    within a second, so preparing its successor then leaves it too little time to deliver.
    """
    controller, mass = _controller_with_parts("part", "after", served="current")

    controller.track_fully_buffered("queue-1", "part")

    assert _prepared_for(mass) == ["part"]


async def test_a_fully_arrived_part_the_player_fetches_prepares_its_successor() -> None:
    """A part the player is fetching itself chains into its successor as well."""
    controller, mass = _controller_with_parts("part", "after", served="part")

    controller.track_fully_buffered("queue-1", "part")

    assert _prepared_for(mass) == ["part"]


async def test_a_fully_arrived_part_two_items_ahead_prepares_nothing() -> None:
    """The fills behind parts stop two items ahead of what the player has fetched."""
    controller, mass = _controller_with_parts("part", "after", "last", served="current")

    controller.track_fully_buffered("queue-1", "after")

    mass.create_task.assert_not_called()


async def test_a_part_without_a_served_item_prepares_nothing() -> None:
    """A queue whose player fetched nothing yet gives a part nothing to chain on."""
    controller, mass = _controller_with_parts("part", "after", served=None)

    controller.track_fully_buffered("queue-1", "part")

    mass.create_task.assert_not_called()


async def test_a_ready_part_behind_the_streamed_item_has_its_successor_prepared() -> None:
    """When the next item is ready and plays only a part, the item after it is prepared."""
    controller, mass = _controller_with_parts("part", "after", served="current")

    assert controller.prepare_next_audio_buffer("queue-1", "current") is None

    assert _prepared_for(mass) == ["part"]


async def test_a_part_that_is_still_filling_prepares_nothing() -> None:
    """A part whose source still runs keeps its successor waiting: it may hold the only slot."""
    controller, mass = _controller_with_parts("part", "after", served="current")
    controller._queue_data["queue-1"].items[1].streamdetails.buffer.eof = False

    controller.prepare_next_audio_buffer("queue-1", "current")

    mass.create_task.assert_not_called()


async def test_a_part_that_only_starts_late_prepares_its_successor() -> None:
    """A start position alone makes an item a part."""
    controller, mass = _controller_with_parts("part", "after", served="current")
    part = controller._queue_data["queue-1"].items[1]
    part.extra_attributes = {"start_position": 30.0}

    controller.track_fully_buffered("queue-1", "part")

    assert _prepared_for(mass) == ["part"]


@pytest.mark.parametrize("media_type", [MediaType.RADIO, MediaType.PODCAST_EPISODE])
async def test_a_live_or_spoken_item_behind_a_part_is_not_prepared_early(
    media_type: MediaType,
) -> None:
    """Only a track is prepared early behind a part: a source opened this soon would sit idle."""
    controller, mass = _controller_with_parts("part", "after", served="current")
    controller._queue_data["queue-1"].items[2].media_type = media_type

    controller.track_fully_buffered("queue-1", "part")

    mass.create_task.assert_not_called()


async def test_two_ready_parts_on_repeat_do_not_prepare_each_other_for_ever() -> None:
    """Two buffered parts that follow each other (repeat all) end the chain after one step."""
    controller, mass = _controller_with_parts("part", served="current", last_ready=True)
    queue_data = controller._queue_data["queue-1"]
    queue_data.items[:] = [cast("Any", _part("a")), cast("Any", _part("b"))]
    cast("Any", queue_data.queue).repeat_mode = RepeatMode.ALL
    queue_data.last_served_item_id = "a"

    assert controller.prepare_next_audio_buffer("queue-1", "a") is None

    mass.create_task.assert_not_called()


async def test_an_item_a_fade_is_being_mixed_into_is_not_prepared_again() -> None:
    """
    The next item is left alone once the boundary into it is being mixed.

    A long track's buffer drops what was read, so it no longer looks prepared from its start,
    and preparing it again would fetch it a second time under the fade that reads it.
    """
    controller, mass = _controller_with_parts("part", "after", served="part")
    queue_data = controller._queue_data["queue-1"]
    cast("Any", queue_data.queue).index_in_buffer = 2
    queue_data.items[2].streamdetails = SimpleNamespace(buffer=MagicMock())
    queue_data.items[2].streamdetails.buffer.is_valid.return_value = False

    controller.track_fully_buffered("queue-1", "part")

    mass.create_task.assert_not_called()


async def test_the_served_item_coming_round_on_repeat_is_not_prepared_again() -> None:
    """With two items on repeat, the part's successor is the item the player is fetching."""
    controller, mass = _controller_with_parts("part", served="current", last_ready=True)
    queue_data = controller._queue_data["queue-1"]
    cast("Any", queue_data.queue).repeat_mode = RepeatMode.ALL
    read_past = MagicMock()
    read_past.is_valid.return_value = False
    _streamed_item(controller).streamdetails = SimpleNamespace(buffer=read_past)

    controller.track_fully_buffered("queue-1", "part")

    mass.create_task.assert_not_called()


async def test_ready_parts_chain_one_item_only() -> None:
    """Parts that are all ready do not walk the queue: one step past the next item."""
    controller, mass = _controller_with_parts(
        "part", "after", "last", served="current", last_ready=True
    )
    queue_data = controller._queue_data["queue-1"]
    cast("Any", queue_data.queue).repeat_mode = RepeatMode.ALL

    controller.prepare_next_audio_buffer("queue-1", "current")

    mass.create_task.assert_not_called()
    assert queue_data.items[3].streamdetails.buffer.is_valid.call_count == 0


async def test_a_queue_change_prepares_the_new_item_behind_a_buffered_part() -> None:
    """An item put behind a part that is already buffered is prepared at the change."""
    controller, mass = _controller_with_parts("part", served="current", last_ready=True)
    queue_data = controller._queue_data["queue-1"]
    queue = cast("Any", queue_data.queue)
    queue.state = PlaybackState.PLAYING
    queue.flow_mode = False
    queue_data.transitioning = False
    queue_data.next_item_id_enqueued = "part"
    queue_data.items.append(
        cast(
            "Any",
            SimpleNamespace(
                queue_item_id="new",
                media_type=MediaType.TRACK,
                streamdetails=None,
                name="New",
                available=True,
                extra_attributes={},
            ),
        )
    )

    controller.update_next_item_on_player("queue-1")

    # not at once: the client that added the item may still set its start and end
    mass.create_task.assert_not_called()
    delay, look_again, queue_id = mass.call_later.call_args.args
    assert delay == PREPARE_AFTER_PART_DELAY
    assert mass.call_later.call_args.kwargs == {"task_id": "prepare_after_parts_queue-1"}
    look_again(queue_id)
    assert _prepared_for(mass) == ["part"]


async def test_an_item_prepared_behind_a_part_is_released_when_it_leaves_the_queue() -> None:
    """The queue's cleanups walk its items, so audio prepared two ahead is let go here."""
    controller, _mass = _controller_with_parts("part", "after", served="current", last_ready=True)
    queue_data = controller._queue_data["queue-1"]
    cast("Any", queue_data.queue).state = PlaybackState.PLAYING
    queue_data.transitioning = False
    controller.track_fully_buffered("queue-1", "part")
    after = queue_data.items.pop(2)
    buffer = after.streamdetails.buffer

    controller._prepare_after_parts("queue-1")

    assert after.streamdetails.buffer is None
    buffer.clear.assert_called_once_with()
    assert queue_data.prepared_ahead is None


async def test_an_item_prepared_behind_a_part_keeps_its_audio_while_it_is_queued() -> None:
    """Looking again leaves the audio of an item that is still on the queue alone."""
    controller, _mass = _controller_with_parts("part", "after", served="current", last_ready=True)
    queue_data = controller._queue_data["queue-1"]
    cast("Any", queue_data.queue).state = PlaybackState.PLAYING
    queue_data.transitioning = False
    controller.track_fully_buffered("queue-1", "part")
    after = queue_data.items[2]

    controller._prepare_after_parts("queue-1")

    after.streamdetails.buffer.clear.assert_not_called()
    assert queue_data.prepared_ahead is after
