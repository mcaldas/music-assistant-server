"""Tests for PlayerQueuesController.set_transition: a client's request for the next transition."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, Mock

import pytest
from music_assistant_models.enums import CrossfadeMode, MediaType, PlaybackState, RepeatMode
from music_assistant_models.errors import (
    ActionUnavailable,
    InvalidCommand,
    InvalidDataError,
    PlayerUnavailableError,
    QueueEmpty,
)
from music_assistant_models.media_items import ItemMapping, ProviderMapping, Track
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.unique_list import UniqueList

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.streams.smart_fades.planner.requested import (
    REQUEST_PREFIX,
    TransitionRequest,
)
from music_assistant.helpers.api import APICommandHandler, parse_arguments

TRACKS = ["t0", "t1", "t2", "t3", "t4"]


def _track(item_id: str) -> Track:
    """Build a playable 60s Track on the 'test' provider."""
    return Track(
        item_id=item_id,
        provider="test",
        name=f"Track {item_id}",
        duration=60,
        artists=UniqueList(
            [ItemMapping(item_id="a", provider="test", name="A", media_type=MediaType.ARTIST)]
        ),
        provider_mappings={
            ProviderMapping(item_id=item_id, provider_domain="test", provider_instance="test")
        },
    )


def _controller(
    *,
    current_index: int = 0,
    index_in_buffer: int | None = 0,
    repeat_mode: RepeatMode = RepeatMode.OFF,
    crossfade_mode: CrossfadeMode = CrossfadeMode.SMART_CROSSFADE,
) -> Any:
    """Build a bare controller holding queue "q1" loaded with TRACKS, playing one of them."""
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    ctrl.logger = Mock()
    ctrl.mass = MagicMock()
    ctrl.mass.streams.get_crossfade_mode.return_value = crossfade_mode
    ctrl.signal_update = Mock()  # type: ignore[method-assign]
    ctrl._check_player_permission = Mock()  # type: ignore[method-assign]
    queue = PlayerQueue(queue_id="q1", active=True, display_name="Q1", available=True, items=0)
    ctrl._queue_data = {"q1": PlayerQueueData(queue=queue)}
    items = [QueueItem.from_media_item("q1", _track(item_id)) for item_id in TRACKS]
    ctrl._queue_data["q1"].items = items
    queue.items = len(items)
    queue.state = PlaybackState.PLAYING
    queue.repeat_mode = repeat_mode
    queue.current_index = current_index
    queue.current_item = items[current_index]
    queue.index_in_buffer = index_in_buffer
    return ctrl


def _item(ctrl: Any, index: int) -> QueueItem:
    """Return the queue item at the given index."""
    return cast("QueueItem", ctrl._queue_data["q1"].items[index])


def _request_keys(item: QueueItem) -> dict[str, Any]:
    """Return the transition request stored on a queue item."""
    return {k: v for k, v in item.extra_attributes.items() if k.startswith(REQUEST_PREFIX)}


async def test_stores_the_request_bound_to_the_next_item() -> None:
    """The request lands on the outgoing item, made for the item that follows it."""
    ctrl = _controller()
    outgoing = _item(ctrl, 1)

    await ctrl.set_transition("q1", outgoing.queue_item_id, "blend", bars=8, exit_at=40.0)

    assert TransitionRequest.read(outgoing.extra_attributes) == TransitionRequest(
        "blend", _item(ctrl, 2).queue_item_id, 8, 40.0
    )
    assert all(not _request_keys(_item(ctrl, i)) for i in (0, 2, 3, 4))
    ctrl.signal_update.assert_called_once_with("q1")


@pytest.mark.parametrize(("style", "bars"), [("filter_sweep", 8), ("echo_out", 0)])
async def test_stores_the_effect_styles_like_the_others(style: str, bars: int) -> None:
    """A filter sweep takes a blend's bars, an echo out none; both are stored as asked."""
    ctrl = _controller()
    outgoing = _item(ctrl, 1)

    await ctrl.set_transition("q1", outgoing.queue_item_id, style, bars=bars, exit_at=40.0)

    assert TransitionRequest.read(outgoing.extra_attributes) == TransitionRequest(
        style, _item(ctrl, 2).queue_item_id, bars, 40.0
    )


async def test_a_new_request_replaces_the_old_and_auto_drops_it() -> None:
    """A later request overwrites every key of the earlier one; "auto" removes it."""
    ctrl = _controller()
    outgoing = _item(ctrl, 1)
    await ctrl.set_transition("q1", outgoing.queue_item_id, "blend", bars=8, exit_at=40.0)

    await ctrl.set_transition("q1", outgoing.queue_item_id, "cut")
    request = TransitionRequest.read(outgoing.extra_attributes)
    assert request is not None
    assert (request.style, request.bars, request.exit_at) == ("cut", 0, 0.0)

    await ctrl.set_transition("q1", outgoing.queue_item_id, "auto")
    assert not _request_keys(outgoing)
    assert ctrl.signal_update.call_count == 3
    # nothing left to drop: nothing to signal
    await ctrl.set_transition("q1", outgoing.queue_item_id, "auto")
    assert ctrl.signal_update.call_count == 3


@pytest.mark.parametrize(
    ("style", "bars", "exit_at"),
    [
        ("bogus", 0, 0.0),
        ("blend", 0, 0.0),
        ("blend", 3, 0.0),
        ("cut", 4, 0.0),
        ("filter_sweep", 0, 0.0),
        ("filter_sweep", 3, 0.0),
        ("echo_out", 2, 0.0),
        ("cut", 0, -1.0),
        ("cut", 0, 25.0),  # outside the item's last 30s (half of its 60s)
        ("cut", 0, 61.0),
        ("auto", 0, 40.0),
    ],
)
async def test_rejects_invalid_arguments(style: str, bars: int, exit_at: float) -> None:
    """Malformed requests are refused before anything is stored."""
    ctrl = _controller()
    outgoing = _item(ctrl, 1)

    with pytest.raises(InvalidDataError):
        await ctrl.set_transition("q1", outgoing.queue_item_id, style, bars=bars, exit_at=exit_at)
    assert not _request_keys(outgoing)


async def test_rejects_unknown_queues_and_items() -> None:
    """An unknown queue or queue item is refused."""
    ctrl = _controller()
    with pytest.raises(PlayerUnavailableError):
        await ctrl.set_transition("nope", "x", "cut")
    with pytest.raises(InvalidDataError):
        await ctrl.set_transition("q1", "nope", "cut")


async def test_rejects_what_cannot_crossfade() -> None:
    """No next item, a non-track neighbour, repeat one or no Smart Fades: refused."""
    ctrl = _controller()
    with pytest.raises(QueueEmpty):
        await ctrl.set_transition("q1", _item(ctrl, 4).queue_item_id, "cut")
    ctrl._queue_data["q1"].items[2] = QueueItem(
        queue_id="q1", queue_item_id="other", name="Other", duration=60
    )
    with pytest.raises(InvalidCommand):
        await ctrl.set_transition("q1", _item(ctrl, 1).queue_item_id, "cut")

    ctrl = _controller(repeat_mode=RepeatMode.ONE)
    with pytest.raises(InvalidCommand):
        await ctrl.set_transition("q1", _item(ctrl, 1).queue_item_id, "cut")

    for mode in (CrossfadeMode.STANDARD_CROSSFADE, CrossfadeMode.DISABLED):
        ctrl = _controller(crossfade_mode=mode)
        with pytest.raises(InvalidCommand):
            await ctrl.set_transition("q1", _item(ctrl, 1).queue_item_id, "cut")


@pytest.mark.parametrize("style", ["cut", "auto"])
async def test_refuses_once_the_boundary_is_fixed(style: str) -> None:
    """Once the next item is locked in, neither a request nor a cancel can change it."""
    # the next item is buffered behind the playing one
    ctrl = _controller(current_index=0, index_in_buffer=1)
    outgoing = _item(ctrl, 0)
    outgoing.extra_attributes.update(
        TransitionRequest("blend", _item(ctrl, 1).queue_item_id, 8).to_attributes()
    )
    with pytest.raises(ActionUnavailable):
        await ctrl.set_transition("q1", outgoing.queue_item_id, style)
    assert TransitionRequest.read(outgoing.extra_attributes) is not None

    # the boundary into the next item was already reported
    ctrl = _controller(current_index=2, index_in_buffer=2)
    _item(ctrl, 1).extra_attributes["transition_next_item_id"] = _item(ctrl, 2).queue_item_id
    with pytest.raises(ActionUnavailable):
        await ctrl.set_transition("q1", _item(ctrl, 1).queue_item_id, style)


async def test_accepts_requests_for_boundaries_still_to_come() -> None:
    """A wrapped buffer index or a report about another boundary does not block a request."""
    # repeat all: item 0 plays and is buffered, item 4's next wraps round to item 0
    ctrl = _controller(current_index=0, index_in_buffer=0, repeat_mode=RepeatMode.ALL)
    await ctrl.set_transition("q1", _item(ctrl, 4).queue_item_id, "cut")
    request = TransitionRequest.read(_item(ctrl, 4).extra_attributes)
    assert request is not None
    assert request.next_item_id == _item(ctrl, 0).queue_item_id

    # a report on item 2 that names another item than its next one
    _item(ctrl, 2).extra_attributes["transition_next_item_id"] = _item(ctrl, 4).queue_item_id
    await ctrl.set_transition("q1", _item(ctrl, 2).queue_item_id, "quick_fade")
    assert TransitionRequest.read(_item(ctrl, 2).extra_attributes) is not None


def test_json_numbers_reach_the_command_as_its_types() -> None:
    """An integer exit_at and a float bar count from JSON arrive as float and int."""
    handler = APICommandHandler.parse("player_queues/set_transition", _controller().set_transition)

    args = parse_arguments(
        handler.signature,
        handler.type_hints,
        {"queue_id": "q1", "queue_item_id": "x", "style": "blend", "bars": 8.0, "exit_at": 150},
    )

    assert args["exit_at"] == 150.0
    assert isinstance(args["exit_at"], float)
    assert args["bars"] == 8
    assert isinstance(args["bars"], int)


def test_an_item_that_stops_playing_loses_its_unused_request() -> None:
    """When the queue moves on, a request still on the item that was playing lapses."""
    ctrl = _controller(current_index=0)
    played, playing = _item(ctrl, 0), _item(ctrl, 1)
    played.extra_attributes.update(TransitionRequest("cut", playing.queue_item_id).to_attributes())
    played.extra_attributes["transition_next_item_id"] = playing.queue_item_id
    played.extra_attributes["transition_mode"] = "smart_crossfade"
    playing.extra_attributes.update(
        TransitionRequest("cut", _item(ctrl, 2).queue_item_id).to_attributes()
    )
    ctrl._get_output_player_ids = Mock(return_value=set())
    ctrl._handle_playback_progress_report = Mock()
    ctrl.is_smart_shuffle_active = Mock(return_value=False)
    player = MagicMock()
    player.player_id = "q1"
    player.state.playback_state = PlaybackState.PLAYING
    queue = ctrl._queue_data["q1"].queue
    ctrl._update_current_index_from_player = Mock(return_value=True)
    ctrl._update_queue_from_player(player)
    assert TransitionRequest.read(played.extra_attributes) is not None

    # the queue moves on to item 1
    queue.current_index, queue.current_item = 1, playing
    ctrl._update_queue_from_player(player)

    # its request and its report described this pass only, and the saved queue forgets them
    assert not _request_keys(played)
    assert not any(key.startswith("transition_") for key in played.extra_attributes)
    assert ctrl._queue_data["q1"].items_cache_dirty
    assert TransitionRequest.read(playing.extra_attributes) is not None


async def test_auto_drops_a_request_whose_next_item_is_gone() -> None:
    """With nothing after the item any more, "auto" still drops its request."""
    ctrl = _controller()
    last = _item(ctrl, 4)
    last.extra_attributes.update(TransitionRequest("cut", "removed").to_attributes())

    await ctrl.set_transition("q1", last.queue_item_id, "auto")

    assert not _request_keys(last)
    ctrl.signal_update.assert_called_once_with("q1")
