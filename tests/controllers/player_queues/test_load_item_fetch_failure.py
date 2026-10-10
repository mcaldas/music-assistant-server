"""Tests for loading a queue item whose details or stream details cannot be fetched."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.errors import (
    AudioError,
    MediaNotFoundError,
    ResourceTemporarilyUnavailable,
    RetriesExhausted,
)
from music_assistant_models.media_items import ProviderMapping, Track
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues.controller import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData

QUEUE_ID = "queue-1"


def _queue_item(item_id: str = "track-1") -> QueueItem:
    """
    Build a queue item holding a track without an image.

    :param item_id: The id of the queue item and of its track.
    """
    return QueueItem(
        queue_id=QUEUE_ID,
        queue_item_id=item_id,
        name=item_id,
        duration=300,
        media_item=Track(
            item_id=item_id,
            provider="spotify--abc",
            name=item_id,
            duration=300,
            provider_mappings={
                ProviderMapping(
                    item_id=item_id,
                    provider_domain="spotify",
                    provider_instance="spotify--abc",
                )
            },
        ),
    )


def _controller(item: QueueItem, fetch_error: Exception) -> PlayerQueuesController:
    """
    Build a bare controller whose queue holds the item and whose full item fetch fails.

    :param item: The item the queue holds, not in the library.
    :param fetch_error: The error fetching the full item raises.
    """
    controller = PlayerQueuesController.__new__(PlayerQueuesController)
    controller.logger = MagicMock()
    controller._queue_data = {
        QUEUE_ID: PlayerQueueData(
            queue=PlayerQueue(
                queue_id=QUEUE_ID,
                active=True,
                display_name="Test queue",
                available=True,
                items=1,
            ),
            items=[item],
        )
    }
    mass = MagicMock()
    mass.music.get_library_item_by_prov_id = AsyncMock(return_value=None)
    mass.music.get_item_by_uri = AsyncMock(side_effect=fetch_error)
    mass.streams.audio.get_stream_details = AsyncMock(return_value=MagicMock(duration=None))
    controller.mass = mass
    return controller


@pytest.mark.parametrize(
    "fetch_error",
    [RetriesExhausted("rate limited"), ResourceTemporarilyUnavailable("unavailable")],
)
async def test_temporary_fetch_failure_plays_the_item_as_listed(fetch_error: Exception) -> None:
    """A provider that is temporarily unavailable does not stop the item from playing."""
    item = _queue_item()
    original_track = item.media_item
    controller = _controller(item, fetch_error)

    await controller._load_item(item)

    assert item.media_item is original_track
    get_stream_details = cast("AsyncMock", controller.mass.streams.audio.get_stream_details)
    get_stream_details.assert_awaited_once()


async def test_unavailable_item_still_fails_to_load() -> None:
    """An item the provider no longer has still fails to load."""
    item = _queue_item()
    controller = _controller(item, MediaNotFoundError("gone"))

    with pytest.raises(MediaNotFoundError):
        await controller._load_item(item)
    get_stream_details = cast("AsyncMock", controller.mass.streams.audio.get_stream_details)
    get_stream_details.assert_not_awaited()


def _controller_with_three_items(
    failures: dict[str, Exception],
) -> tuple[PlayerQueuesController, list[QueueItem]]:
    """
    Build a bare controller whose queue holds the items a, b and c.

    :param failures: The error fetching an item's stream details raises, by queue item id.
        An item without one loads.
    """
    items = [_queue_item(item_id) for item_id in "abc"]
    # the full details are beside the point here: every item loads as listed
    controller = _controller(items[0], RetriesExhausted("rate limited"))
    controller._queue_data[QUEUE_ID].items = items
    controller.update_items = MagicMock()  # type: ignore[method-assign]

    async def _get_stream_details(queue_item: QueueItem, **_kwargs: object) -> MagicMock:
        if error := failures.get(queue_item.queue_item_id):
            raise error
        return MagicMock(duration=None)

    controller.mass.streams.audio.get_stream_details = _get_stream_details
    return controller, items


async def test_looking_ahead_does_not_write_off_an_item_whose_provider_did_not_answer() -> None:
    """An item whose provider could not be asked stays in the queue for the next look ahead."""
    failures: dict[str, Exception] = {"b": AudioError("offline")}
    controller, (_a, b, _c) = _controller_with_three_items(failures)

    with pytest.raises(AudioError):
        await controller.load_next_queue_item(QUEUE_ID, "a", speculative=True)
    assert b.available
    # the provider answers again
    failures.clear()

    assert await controller.load_next_queue_item(QUEUE_ID, "a", speculative=True) is b


async def test_items_written_off_before_the_failure_are_still_announced() -> None:
    """An item skipped as not found is signalled even when the one after it ends the look ahead."""
    controller, (_a, b, c) = _controller_with_three_items(
        {"b": MediaNotFoundError("gone"), "c": AudioError("offline")}
    )

    with pytest.raises(AudioError):
        await controller.load_next_queue_item(QUEUE_ID, "a", speculative=True)

    assert not b.available
    assert c.available
    cast("MagicMock", controller.update_items).assert_called_once()


async def test_stepping_over_an_item_written_off_earlier_announces_nothing() -> None:
    """A look ahead that fails behind an item skipped before changed nothing to signal."""
    controller, (_a, b, _c) = _controller_with_three_items({"c": AudioError("offline")})
    b.available = False

    with pytest.raises(AudioError):
        await controller.load_next_queue_item(QUEUE_ID, "a", speculative=True)

    cast("MagicMock", controller.update_items).assert_not_called()


async def test_looking_ahead_still_skips_an_item_its_provider_does_not_have() -> None:
    """Looking ahead writes off an item no provider has, like any other caller."""
    controller, (_a, b, c) = _controller_with_three_items({"b": MediaNotFoundError("gone")})

    assert await controller.load_next_queue_item(QUEUE_ID, "a", speculative=True) is c

    assert not b.available


async def test_a_caller_the_audio_waits_on_still_steps_over_at_once() -> None:
    """A caller that needs the next item now skips one whose provider did not answer."""
    controller, (_a, b, c) = _controller_with_three_items({"b": AudioError("offline")})

    assert await controller.load_next_queue_item(QUEUE_ID, "a") is c

    assert not b.available
