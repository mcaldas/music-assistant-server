"""Tests for PlayerQueuesController.set_end_position: a client ends a queue item early."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, Mock

import pytest
from music_assistant_models.enums import PlaybackState
from music_assistant_models.errors import (
    ActionUnavailable,
    InvalidCommand,
    InvalidDataError,
    PlayerUnavailableError,
)
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.playback_tracker import PlaybackTrackerMixin
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.streams.smart_fades.planner.requested import TransitionRequest
from music_assistant.helpers.api import APICommandHandler, parse_arguments
from tests.controllers.player_queues.test_play_index_elapsed import (
    QUEUE_ID,
    _controller_with_stale_queue,
)
from tests.controllers.player_queues.test_set_transition import _controller, _item

if TYPE_CHECKING:
    from music_assistant_models.player_queue import PlayerQueue


def _ctrl(**kwargs: Any) -> Any:
    """Build the set_transition test controller with an empty read-position map."""
    ctrl = _controller(**kwargs)
    ctrl.mass.streams.audio.read_positions = {}
    return ctrl


async def test_stores_and_clears_the_end() -> None:
    """The end is stored on the item, 0 clears it, and only a change is signalled."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    await ctrl.set_end_position("q1", item.queue_item_id, 40.25)
    assert item.extra_attributes["end_position"] == 40.25
    ctrl.signal_update.assert_called_once_with("q1", items_changed=True)
    await ctrl.set_end_position("q1", item.queue_item_id, 0)
    assert "end_position" not in item.extra_attributes
    assert ctrl.signal_update.call_count == 2
    # clearing nothing signals nothing
    await ctrl.set_end_position("q1", item.queue_item_id)
    assert ctrl.signal_update.call_count == 2


@pytest.mark.parametrize("position", [-1.0, 60.0, 75.0])
async def test_rejects_positions_outside_the_item(position: float) -> None:
    """An end must lie inside the item."""
    ctrl = _ctrl()
    with pytest.raises(InvalidDataError):
        await ctrl.set_end_position("q1", _item(ctrl, 1).queue_item_id, position)


async def test_rejects_unknown_queues_items_and_non_tracks() -> None:
    """Only a track with a known duration in a known queue takes an end."""
    ctrl = _ctrl()
    with pytest.raises(PlayerUnavailableError):
        await ctrl.set_end_position("nope", _item(ctrl, 1).queue_item_id, 30.0)
    with pytest.raises(InvalidDataError):
        await ctrl.set_end_position("q1", "nope", 30.0)
    _item(ctrl, 1).duration = None
    with pytest.raises(InvalidCommand):
        await ctrl.set_end_position("q1", _item(ctrl, 1).queue_item_id, 30.0)
    # a queue item without a track behind it
    ctrl._queue_data["q1"].items[2] = QueueItem(
        queue_id="q1", queue_item_id="other", name="Other", duration=60
    )
    with pytest.raises(InvalidCommand):
        await ctrl.set_end_position("q1", "other", 30.0)


async def test_refuses_an_end_its_stream_has_read_past() -> None:
    """An end at or before what the item's stream has read is refused."""
    ctrl = _ctrl()
    item = _item(ctrl, 0)
    item.extra_attributes["end_position"] = 50.0
    ctrl.mass.streams.audio.read_positions[item.queue_item_id] = 30.0
    with pytest.raises(ActionUnavailable):
        await ctrl.set_end_position("q1", item.queue_item_id, 30.0)
    # later than what was read: moved, or cleared
    await ctrl.set_end_position("q1", item.queue_item_id, 35.0)
    assert item.extra_attributes["end_position"] == 35.0
    await ctrl.set_end_position("q1", item.queue_item_id, 0)
    # once read to its end, nothing moves for this pass
    item.extra_attributes["end_position"] = 50.0
    ctrl.mass.streams.audio.read_positions[item.queue_item_id] = float("inf")
    for position in (45.0, 0.0):
        with pytest.raises(ActionUnavailable):
            await ctrl.set_end_position("q1", item.queue_item_id, position)


def test_json_numbers_reach_the_command_as_a_float() -> None:
    """A JSON integer position arrives as a float, not as None."""
    handler = APICommandHandler.parse("player_queues/set_end_position", _ctrl().set_end_position)
    args = parse_arguments(
        handler.signature,
        handler.type_hints,
        {"queue_id": "q1", "queue_item_id": "x", "position": 90},
    )
    assert args["position"] == 90.0
    assert isinstance(args["position"], float)


async def test_a_transition_exit_lies_in_the_last_seconds_before_the_end() -> None:
    """A requested exit must lie in the window before the end, not the natural end."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    item.extra_attributes["end_position"] = 40.0
    # 60 s track: without an end its window is (30, 60]; with the end at 40 it is (20, 40]
    await ctrl.set_transition("q1", item.queue_item_id, "cut", exit_at=25.0)
    with pytest.raises(InvalidDataError):
        await ctrl.set_transition("q1", item.queue_item_id, "cut", exit_at=50.0)


async def test_an_exit_sent_as_the_end_is_accepted_at_the_end() -> None:
    """The end is stored to the millisecond; the same number sent as exit_at is still at it."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    await ctrl.set_end_position("q1", item.queue_item_id, 40.0004)

    await ctrl.set_transition("q1", item.queue_item_id, "cut", exit_at=40.0004)

    assert TransitionRequest.read(item.extra_attributes) == TransitionRequest(
        "cut", ctrl.get_next_item("q1", item.queue_item_id).queue_item_id, exit_at=40.0
    )


async def test_an_end_drops_the_transition_request_made_for_the_old_end() -> None:
    """Setting, moving or clearing the end drops a pending request; a clear of nothing keeps it."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    request = TransitionRequest("cut", _item(ctrl, 2).queue_item_id, 0, 50.0)
    for position in (40.0, 30.0, 0.0):
        item.extra_attributes.update(request.to_attributes())
        await ctrl.set_end_position("q1", item.queue_item_id, position)
        assert TransitionRequest.read(item.extra_attributes) is None
    item.extra_attributes.update(request.to_attributes())
    await ctrl.set_end_position("q1", item.queue_item_id, 0)
    assert TransitionRequest.read(item.extra_attributes) == request


def test_an_item_that_stops_playing_loses_its_end() -> None:
    """The end lapses, saved, when the item stops being current; the next item keeps its own."""
    ctrl = _ctrl(current_index=0)
    played, playing = _item(ctrl, 0), _item(ctrl, 1)
    played.extra_attributes["end_position"] = 30.0
    playing.extra_attributes["end_position"] = 30.0
    ctrl.mass.streams.audio.read_positions[played.queue_item_id] = float("inf")
    ctrl._get_output_player_ids = Mock(return_value=set())
    ctrl._handle_playback_progress_report = Mock()
    ctrl.is_smart_shuffle_active = Mock(return_value=False)
    player = MagicMock()
    player.player_id = "q1"
    player.state.playback_state = PlaybackState.PLAYING
    queue = ctrl._queue_data["q1"].queue
    ctrl._update_current_index_from_player = Mock(return_value=True)
    ctrl._update_queue_from_player(player)
    assert played.extra_attributes["end_position"] == 30.0
    assert not ctrl._queue_data["q1"].items_cache_dirty
    queue.current_index, queue.current_item = 1, playing
    ctrl._update_queue_from_player(player)
    assert "end_position" not in played.extra_attributes
    # saved: a restart must not bring the end back
    assert ctrl._queue_data["q1"].items_cache_dirty
    assert played.queue_item_id not in ctrl.mass.streams.audio.read_positions
    assert playing.extra_attributes["end_position"] == 30.0


async def test_the_player_is_told_the_stream_ends_at_the_end() -> None:
    """The player is handed a stream as long as the part that plays."""
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    item = QueueItem(queue_id="q1", queue_item_id="i1", name="T", duration=300)
    item.streamdetails = StreamDetails(
        provider="test", item_id="i1", audio_format=AudioFormat(), duration=300
    )
    queue_data = PlayerQueueData(queue=MagicMock())
    queue_data.session_id = "s1"
    ctrl._queue_data = {"q1": queue_data}
    ctrl.mass = MagicMock()
    item.extra_attributes["end_position"] = 90.5
    media = await ctrl.player_media_from_queue_item(item)
    assert media.duration == 300
    assert media.stream_duration == 90
    item.streamdetails.seek_position = 30
    assert (await ctrl.player_media_from_queue_item(item)).stream_duration == 60
    # a seek past the end plays the rest: the stream is what remains of the item
    item.streamdetails.seek_position = 120
    assert (await ctrl.player_media_from_queue_item(item)).stream_duration == 180


def test_a_last_item_that_ended_at_its_end_settles_the_queue() -> None:
    """A last item that played to its end position ends the queue."""
    item = QueueItem(queue_id="q1", queue_item_id="i1", name="T", duration=240)
    item.streamdetails = StreamDetails(
        provider="test", item_id="i1", audio_format=AudioFormat(), duration=240
    )
    tracker = MagicMock()
    tracker._queue_data = {"q1": SimpleNamespace(items=[item], flow_mode_stream_log=[])}
    queue = cast(
        "PlayerQueue",
        SimpleNamespace(queue_id="q1", display_name="Q1", next_item=None, flow_mode=False),
    )
    prev_state: Any = {
        "state": PlaybackState.PLAYING,
        "current_item_id": "i1",
        "current_item": item,
        "last_playing_elapsed_time": 89,
    }
    new_state: Any = {"state": PlaybackState.IDLE}
    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)
    tracker.mass.create_task.assert_not_called()  # 89 s of 240: a stop, not the end
    for task in tracker.mass.create_task.call_args_list:
        task.args[0].close()
    item.extra_attributes["end_position"] = 90.0
    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)
    tracker.mass.create_task.assert_called_once()
    tracker.mass.create_task.call_args.args[0].close()


async def test_a_new_session_starts_the_read_marks_over() -> None:
    """A seek, a restart or a resume (play_index) streams afresh: no older read holds an end back."""
    ctrl, _queue, _signals = _controller_with_stale_queue()
    read_positions = {"old": float("inf"), "new": 12.0, "other-queue": 5.0}
    ctrl.mass.streams.audio.read_positions = read_positions
    await ctrl.play_index(QUEUE_ID, 0, seek_position=30)
    assert read_positions == {"other-queue": 5.0}
