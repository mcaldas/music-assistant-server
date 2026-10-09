"""Tests that starting an item releases the source slots its own queue no longer needs."""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.models.music_provider import MusicProvider

QUEUE_ID = "q1"
LIMITED = "limited--1"
UNLIMITED = "local--1"


def _item(
    queue_id: str,
    item_id: str,
    provider: str | None,
    buffering: bool = True,
    ready: bool = True,
) -> QueueItem:
    """
    Build a queue item, optionally with a buffer attached to its stream details.

    :param queue_id: The queue the item belongs to.
    :param item_id: The queue item id.
    :param provider: Provider instance owning the source, or None for an item without details.
    :param buffering: Whether the attached buffer is still filling from its source.
    :param ready: Whether the attached buffer holds the audio a reader starts with.
    """
    queue_item = QueueItem(queue_id=queue_id, queue_item_id=item_id, name=item_id, duration=180)
    if provider is None:
        return queue_item
    audio_buffer = MagicMock(spec=AudioBuffer)
    audio_buffer.is_buffering = buffering
    audio_buffer.ready = asyncio.Event()
    if ready:
        audio_buffer.ready.set()
    audio_buffer.clear = AsyncMock()
    queue_item.streamdetails = StreamDetails(
        provider=provider,
        item_id=item_id,
        audio_format=AudioFormat(content_type=ContentType.MP3),
        media_type=MediaType.TRACK,
        stream_type=StreamType.HTTP,
        path=f"http://test.invalid/{item_id}.mp3",
    )
    queue_item.streamdetails.buffer = audio_buffer
    return queue_item


def _controller(*queues: tuple[str, list[QueueItem]]) -> PlayerQueuesController:
    """Build a bare controller holding the given queues and their items."""
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    ctrl.logger = MagicMock()
    ctrl._queue_data = {
        queue_id: PlayerQueueData(queue=MagicMock(), items=items, session_id=queue_id)
        for queue_id, items in queues
    }
    limited = MagicMock(spec=MusicProvider)
    limited.name = "Limited"
    limited.max_concurrent_streams = 2
    # pinned rather than left to MagicMock truthiness: this decides whether a prewarm survives
    limited.has_available_stream_slot = True
    unlimited = MagicMock(spec=MusicProvider)
    unlimited.name = "Local"
    unlimited.max_concurrent_streams = None
    unlimited.has_available_stream_slot = True
    providers = {LIMITED: limited, UNLIMITED: unlimited}
    ctrl.mass = MagicMock()
    ctrl.mass.get_provider.side_effect = lambda instance, **_kwargs: providers.get(instance)
    return ctrl


def _limited_double(ctrl: PlayerQueuesController) -> MagicMock:
    """Return the slot-limited provider double the controller resolves LIMITED to."""
    return cast("MagicMock", ctrl.mass.get_provider(LIMITED))


async def test_starting_an_item_aborts_the_other_filling_sources_of_its_queue() -> None:
    """A still-filling source of a preceding item in the same queue hands its slot over."""
    target = _item(QUEUE_ID, "target", LIMITED)
    filling = _item(QUEUE_ID, "filling", LIMITED)
    ctrl = _controller((QUEUE_ID, [filling, target]))

    await ctrl._abort_superseded_source_buffers(target)

    # the aborted buffer stays attached so the flow stream can see the abort
    assert filling.streamdetails is not None
    assert filling.streamdetails.buffer is not None
    filling.streamdetails.buffer.clear.assert_awaited_once()
    assert target.streamdetails is not None
    assert target.streamdetails.buffer is not None
    assert target.streamdetails.buffer.clear.await_count == 0


async def test_the_started_items_successor_keeps_its_prewarm() -> None:
    """A resume or seek must not cost the crossfade prewarm of the upcoming track."""
    target = _item(QUEUE_ID, "target", LIMITED)
    upcoming = _item(QUEUE_ID, "upcoming", LIMITED)
    stale = _item(QUEUE_ID, "stale", LIMITED)
    ctrl = _controller((QUEUE_ID, [target, upcoming, stale]))

    await ctrl._abort_superseded_source_buffers(target)

    assert upcoming.streamdetails is not None
    assert upcoming.streamdetails.buffer is not None
    assert upcoming.streamdetails.buffer.clear.await_count == 0
    assert stale.streamdetails is not None
    assert stale.streamdetails.buffer is not None
    stale.streamdetails.buffer.clear.assert_awaited_once()


async def test_a_saturated_provider_takes_back_the_successors_prewarm() -> None:
    """A prewarm may not sit on the last slot the item being started needs."""
    target = _item(QUEUE_ID, "target", LIMITED)
    upcoming = _item(QUEUE_ID, "upcoming", LIMITED)
    ctrl = _controller((QUEUE_ID, [target, upcoming]))
    _limited_double(ctrl).has_available_stream_slot = False

    await ctrl._abort_superseded_source_buffers(target)

    assert upcoming.streamdetails is not None
    assert upcoming.streamdetails.buffer is not None
    upcoming.streamdetails.buffer.clear.assert_awaited_once()
    assert target.streamdetails is not None
    assert target.streamdetails.buffer is not None
    assert target.streamdetails.buffer.clear.await_count == 0


async def test_freeing_a_slot_first_lets_the_successor_keep_its_prewarm() -> None:
    """The prewarm is only given up when aborting the stale sources did not free a slot."""
    stale = _item(QUEUE_ID, "stale", LIMITED)
    target = _item(QUEUE_ID, "target", LIMITED)
    upcoming = _item(QUEUE_ID, "upcoming", LIMITED)
    ctrl = _controller((QUEUE_ID, [stale, target, upcoming]))
    provider = _limited_double(ctrl)
    provider.has_available_stream_slot = False

    async def _release_slot() -> None:
        provider.has_available_stream_slot = True

    assert stale.streamdetails is not None
    assert stale.streamdetails.buffer is not None
    stale.streamdetails.buffer.clear.side_effect = _release_slot

    await ctrl._abort_superseded_source_buffers(target)

    # the successor is only re-checked after the stale aborts, so it survives
    stale.streamdetails.buffer.clear.assert_awaited_once()
    assert upcoming.streamdetails is not None
    assert upcoming.streamdetails.buffer is not None
    assert upcoming.streamdetails.buffer.clear.await_count == 0


async def test_completed_and_unlimited_sources_are_left_alone() -> None:
    """Only a source that still holds a capped provider slot is worth aborting."""
    target = _item(QUEUE_ID, "target", LIMITED)
    completed = _item(QUEUE_ID, "completed", LIMITED, buffering=False)
    unlimited = _item(QUEUE_ID, "unlimited", UNLIMITED)
    ctrl = _controller((QUEUE_ID, [completed, unlimited, target]))

    await ctrl._abort_superseded_source_buffers(target)

    for item in (completed, unlimited):
        assert item.streamdetails is not None
        assert item.streamdetails.buffer is not None
        assert item.streamdetails.buffer.clear.await_count == 0


async def test_other_queues_keep_their_sources() -> None:
    """Playback on one queue never takes a source away from another queue."""
    target = _item(QUEUE_ID, "target", LIMITED)
    other = _item("q2", "other", LIMITED)
    ctrl = _controller((QUEUE_ID, [target]), ("q2", [other]))

    await ctrl._abort_superseded_source_buffers(target)

    assert other.streamdetails is not None
    assert other.streamdetails.buffer is not None
    assert other.streamdetails.buffer.clear.await_count == 0


def _one_slot_taken(
    ctrl: PlayerQueuesController, *asked: QueueItem, fade_into: QueueItem | None = None
) -> None:
    """
    Leave the limited provider one slot, taken, and say what the queues' player has open.

    :param ctrl: The controller from ``_controller``.
    :param asked: The items the player has an open response for.
    :param fade_into: An item a fade is being mixed into, or was.
    """
    provider = _limited_double(ctrl)
    provider.max_concurrent_streams = 1
    provider.has_available_stream_slot = False
    for queue_data in ctrl._queue_data.values():
        # pinned rather than left to MagicMock truthiness: a flow queue hands nothing over
        queue_data.queue.flow_mode = False
    ctrl.mass.streams.open_item_stream_ids.return_value = {item.queue_item_id for item in asked}
    ctrl.mass.streams.audio.crossfade_targets.return_value = (
        {fade_into.queue_item_id} if fade_into else set()
    )


def _aborted(item: QueueItem) -> bool:
    """Return whether the item's source was aborted, with its buffer left attached."""
    assert item.streamdetails is not None
    assert item.streamdetails.buffer is not None
    return bool(item.streamdetails.buffer.clear.await_count)


async def test_an_unread_source_hands_its_slot_to_the_item_the_player_asks_for() -> None:
    """The player left an item whose source still fills: the item it asks for gets the slot."""
    left = _item(QUEUE_ID, "left", LIMITED)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, asked)

    assert await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert _aborted(left)
    assert not _aborted(asked)
    ctrl.logger.info.assert_called_once()
    assert {"left", "asked"} <= set(ctrl.logger.info.call_args.args)


async def test_the_item_the_player_still_reads_keeps_its_source() -> None:
    """An item with an open response is never aborted, whatever waits behind it."""
    left = _item(QUEUE_ID, "left", LIMITED)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, left, asked)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(left)
    ctrl.logger.info.assert_not_called()


async def test_the_item_a_fade_is_mixed_into_keeps_its_source() -> None:
    """The next item is read by the fade into it before its own response exists."""
    left = _item(QUEUE_ID, "left", LIMITED)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, asked, fade_into=left)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(left)


async def test_a_prewarm_gives_way_to_the_asked_item_before_it() -> None:
    """A prepared item nothing reads yet may not keep the slot from the item being asked for."""
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    prewarm = _item(QUEUE_ID, "prewarm", LIMITED)
    ctrl = _controller((QUEUE_ID, [asked, prewarm]))
    _one_slot_taken(ctrl, asked)

    assert await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert _aborted(prewarm)
    assert not _aborted(asked)


@pytest.mark.parametrize(
    ("buffering", "ready"),
    [(True, True), (False, False)],
    ids=["has-its-audio", "aborted-earlier"],
)
async def test_an_asked_item_with_its_audio_leaves_the_prewarm_alone(
    buffering: bool, ready: bool
) -> None:
    """Only an asked item whose source is filling without audio yet is waiting for a slot."""
    asked = _item(QUEUE_ID, "asked", LIMITED, buffering=buffering, ready=ready)
    prewarm = _item(QUEUE_ID, "prewarm", LIMITED)
    ctrl = _controller((QUEUE_ID, [asked, prewarm]))
    _one_slot_taken(ctrl, asked)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(prewarm)


async def test_the_prepared_item_asked_for_itself_is_left_alone() -> None:
    """The player reaching the prepared item reuses its source."""
    prewarm = _item(QUEUE_ID, "prewarm", LIMITED)
    ctrl = _controller((QUEUE_ID, [prewarm]))
    _one_slot_taken(ctrl, prewarm)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(prewarm)


async def test_another_queue_keeps_its_source() -> None:
    """A source of another queue is never aborted, read or not."""
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    other = _item("q2", "other", LIMITED)
    ctrl = _controller((QUEUE_ID, [asked]), ("q2", [other]))
    _one_slot_taken(ctrl, asked)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(other)


async def test_an_item_between_two_requests_keeps_its_source() -> None:
    """A player that drops a request and repeats it finds the source it left."""
    left = _item(QUEUE_ID, "left", LIMITED)
    # the item behind it is being prepared, and waits for the slot as it always did
    prepared = _item(QUEUE_ID, "prepared", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, prepared]))
    _one_slot_taken(ctrl)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(left)


async def test_a_free_slot_changes_nothing() -> None:
    """An asked item that waits for its audio, not for a slot, costs no other item its source."""
    left = _item(QUEUE_ID, "left", LIMITED)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, asked)
    _limited_double(ctrl).has_available_stream_slot = True

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(left)


async def test_a_provider_with_more_than_one_stream_is_left_alone() -> None:
    """With several slots a filling source says nothing about who the asked item waits for."""
    left = _item(QUEUE_ID, "left", LIMITED)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, asked)
    _limited_double(ctrl).max_concurrent_streams = 2

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(left)


async def test_a_source_on_another_provider_is_no_holder() -> None:
    """Only a source of the provider the asked item waits for can hold its slot."""
    left = _item(QUEUE_ID, "left", UNLIMITED)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, asked)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(left)


async def test_a_holder_without_its_first_audio_is_left_alone() -> None:
    """A source without its first audio can be waiting itself, so it is not taken for the holder."""
    left = _item(QUEUE_ID, "left", LIMITED, ready=False)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, asked)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(left)


async def test_a_finished_source_is_no_holder() -> None:
    """A source that delivered everything gave its slot back, and its audio is kept."""
    left = _item(QUEUE_ID, "left", LIMITED, buffering=False)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, asked)

    assert not await ctrl._release_abandoned_stream_slot(QUEUE_ID)

    assert not _aborted(left)


@pytest.mark.parametrize("case", ["flow", "stopped", "unknown"])
async def test_a_flow_queue_a_stopped_queue_or_an_unknown_queue_hands_nothing_over(
    case: str,
) -> None:
    """One flow response reads every item in turn, and a queue without a session reads none."""
    left = _item(QUEUE_ID, "left", LIMITED)
    asked = _item(QUEUE_ID, "asked", LIMITED, ready=False)
    ctrl = _controller((QUEUE_ID, [left, asked]))
    _one_slot_taken(ctrl, asked)
    queue_data = ctrl._queue_data[QUEUE_ID]
    if case == "flow":
        queue_data.queue.flow_mode = True
    elif case == "stopped":
        queue_data.session_id = None

    assert not await ctrl._release_abandoned_stream_slot("gone" if case == "unknown" else QUEUE_ID)

    assert not _aborted(left)
