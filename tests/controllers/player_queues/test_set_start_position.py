"""Tests for PlayerQueuesController.set_start_position: a client starts a queue item later."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import PlaybackState, RepeatMode
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
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.streams.smart_fades.planner.requested import TransitionRequest
from music_assistant.helpers.api import APICommandHandler, parse_arguments
from tests.controllers.player_queues.test_play_index_elapsed import (
    QUEUE_ID,
    _controller_with_stale_queue,
)
from tests.controllers.player_queues.test_set_end_position import _ctrl
from tests.controllers.player_queues.test_set_transition import _item, _track


def _details(seek_position: float = 0, allow_seek: bool = True) -> StreamDetails:
    """Build resolved stream details for a 60 s track."""
    return StreamDetails(
        provider="test",
        item_id="x",
        audio_format=AudioFormat(),
        duration=60,
        allow_seek=allow_seek,
        seek_position=seek_position,
    )


async def test_stores_and_clears_the_start() -> None:
    """The start is stored to the millisecond, 0 clears it, and only a change is signalled."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    await ctrl.set_start_position("q1", item.queue_item_id, 16.0049999)
    assert item.extra_attributes["start_position"] == 16.005
    ctrl.signal_update.assert_called_once_with("q1", items_changed=True)
    await ctrl.set_start_position("q1", item.queue_item_id, 0)
    assert "start_position" not in item.extra_attributes
    assert ctrl.signal_update.call_count == 2
    # clearing nothing signals nothing
    await ctrl.set_start_position("q1", item.queue_item_id)
    assert ctrl.signal_update.call_count == 2


@pytest.mark.parametrize("position", [-1.0, 60.0, 75.0])
async def test_rejects_positions_outside_the_item(position: float) -> None:
    """A start must lie inside the item."""
    ctrl = _ctrl()
    with pytest.raises(InvalidDataError):
        await ctrl.set_start_position("q1", _item(ctrl, 1).queue_item_id, position)


async def test_rejects_a_start_at_or_after_the_end() -> None:
    """With an end set, the start must lie before it."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    item.extra_attributes["end_position"] = 40.0
    for position in (40.0, 45.0):
        with pytest.raises(InvalidDataError):
            await ctrl.set_start_position("q1", item.queue_item_id, position)
    await ctrl.set_start_position("q1", item.queue_item_id, 39.5)
    assert item.extra_attributes["start_position"] == 39.5


async def test_rejects_unknown_queues_items_and_non_tracks() -> None:
    """Only a track with a known duration in a known queue takes a start."""
    ctrl = _ctrl()
    with pytest.raises(PlayerUnavailableError):
        await ctrl.set_start_position("nope", _item(ctrl, 1).queue_item_id, 30.0)
    with pytest.raises(InvalidDataError):
        await ctrl.set_start_position("q1", "nope", 30.0)
    _item(ctrl, 1).duration = None
    with pytest.raises(InvalidCommand):
        await ctrl.set_start_position("q1", _item(ctrl, 1).queue_item_id, 30.0)
    # a queue item without a track behind it
    ctrl._queue_data["q1"].items[2] = QueueItem(
        queue_id="q1", queue_item_id="other", name="Other", duration=60
    )
    with pytest.raises(InvalidCommand):
        await ctrl.set_start_position("q1", "other", 30.0)


async def test_refused_once_the_items_audio_is_read() -> None:
    """Any read of the item in this pass (a prefetch, a planned fade, a stream) fixes its start."""
    ctrl = _ctrl()
    item = _item(ctrl, 2)
    ctrl.mass.streams.audio.read_positions[item.queue_item_id] = 12.0
    for position in (30.0, 0.0):
        with pytest.raises(ActionUnavailable):
            await ctrl.set_start_position("q1", item.queue_item_id, position)
    assert "start_position" not in item.extra_attributes


async def test_accepted_again_once_a_new_session_starts() -> None:
    """play_index streams every item afresh, so a start can be set again."""
    ctrl, _queue, _signals = _controller_with_stale_queue()
    item = QueueItem.from_media_item(QUEUE_ID, _track("t1"))
    ctrl._queue_data[QUEUE_ID].items.append(item)
    ctrl.mass.streams.audio.read_positions = {item.queue_item_id: float("inf")}
    with pytest.raises(ActionUnavailable):
        await ctrl.set_start_position(QUEUE_ID, item.queue_item_id, 30.0)
    await ctrl.play_index(QUEUE_ID, 0, seek_position=10)
    await ctrl.set_start_position(QUEUE_ID, item.queue_item_id, 30.0)
    assert item.extra_attributes["start_position"] == 30.0


def test_json_numbers_reach_the_command_as_a_float() -> None:
    """A JSON integer position arrives as a float, not as None."""
    handler = APICommandHandler.parse(
        "player_queues/set_start_position", _ctrl().set_start_position
    )
    args = parse_arguments(
        handler.signature,
        handler.type_hints,
        {"queue_id": "q1", "queue_item_id": "x", "position": 30},
    )
    assert args["position"] == 30.0
    assert isinstance(args["position"], float)


async def test_details_resolved_ahead_start_at_the_start() -> None:
    """The stream an item's details describe starts at the start, or at 0 once it is cleared."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    item.streamdetails = _details()
    await ctrl.set_start_position("q1", item.queue_item_id, 37.25)
    assert item.streamdetails.seek_position == 37.25
    await ctrl.set_start_position("q1", item.queue_item_id, 0)
    assert item.streamdetails.seek_position == 0


async def test_a_stream_that_cannot_seek_keeps_playing_from_its_beginning() -> None:
    """The start is stored, but details that cannot seek still start at 0."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    item.streamdetails = _details(allow_seek=False)
    await ctrl.set_start_position("q1", item.queue_item_id, 37.25)
    assert item.extra_attributes["start_position"] == 37.25
    assert item.streamdetails.seek_position == 0


@pytest.mark.parametrize(("serves", "kept"), [(True, True), (False, False)])
async def test_prepared_audio_that_cannot_serve_the_start_is_let_go(
    serves: bool, kept: bool
) -> None:
    """Audio prepared from elsewhere is detached and released; audio that serves it is kept."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    item.streamdetails = _details()
    prepared = MagicMock()
    prepared.is_valid.return_value = serves
    prepared.clear = AsyncMock()
    item.streamdetails.buffer = prepared
    await ctrl.set_start_position("q1", item.queue_item_id, 37.25)
    prepared.is_valid.assert_called_once_with(37250)
    assert (item.streamdetails.buffer is prepared) is kept
    assert prepared.clear.await_count == (0 if kept else 1)


async def test_a_start_drops_the_transition_request_made_for_what_played() -> None:
    """Setting, moving or clearing the start drops a pending request; a clear of nothing keeps it."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    request = TransitionRequest("cut", _item(ctrl, 2).queue_item_id, 0, 50.0)
    for position in (20.0, 10.0, 0.0):
        item.extra_attributes.update(request.to_attributes())
        await ctrl.set_start_position("q1", item.queue_item_id, position)
        assert TransitionRequest.read(item.extra_attributes) is None
    item.extra_attributes.update(request.to_attributes())
    await ctrl.set_start_position("q1", item.queue_item_id, 0)
    assert TransitionRequest.read(item.extra_attributes) == request


async def test_an_end_must_lie_after_the_start() -> None:
    """An end at or before the start is refused; one after it is stored."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    item.extra_attributes["start_position"] = 20.0
    for position in (20.0, 10.0):
        with pytest.raises(InvalidDataError):
            await ctrl.set_end_position("q1", item.queue_item_id, position)
    await ctrl.set_end_position("q1", item.queue_item_id, 20.5)
    assert item.extra_attributes["end_position"] == 20.5


async def test_the_exit_window_is_half_of_what_plays() -> None:
    """From 20 s to an end at 40 s the item plays 20 s: an exit must lie in its last 10 s."""
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    item.extra_attributes["start_position"] = 20.0
    item.extra_attributes["end_position"] = 40.0
    with pytest.raises(InvalidDataError):
        await ctrl.set_transition("q1", item.queue_item_id, "cut", exit_at=25.0)
    await ctrl.set_transition("q1", item.queue_item_id, "cut", exit_at=31.0)


async def test_the_exit_window_is_the_tail_the_item_holds() -> None:
    """
    From 60.25 s to 100 s the item plays 39.75 s and holds a 19 s tail: exits lie in it.

    An exit before it (80.25 s, past half of what plays) would find no tail to cut in; an
    item playing under 6 s holds none, so it takes no exit at all.
    """
    ctrl = _ctrl()
    item = _item(ctrl, 1)
    item.extra_attributes["start_position"] = 60.25
    item.extra_attributes["end_position"] = 100.0
    for exit_at in (80.25, 80.9):
        with pytest.raises(InvalidDataError):
            await ctrl.set_transition("q1", item.queue_item_id, "cut", exit_at=exit_at)
    await ctrl.set_transition("q1", item.queue_item_id, "cut", exit_at=81.25)
    item.extra_attributes["end_position"] = 65.75
    with pytest.raises(InvalidDataError, match="too little"):
        await ctrl.set_transition("q1", item.queue_item_id, "cut", exit_at=65.75)


async def test_a_seek_clears_the_start() -> None:
    """A seek is the listener's own position: the item plays from it, and seek 0 plays 0."""
    ctrl, queue, _signals = _controller_with_stale_queue()
    item = queue.current_item
    assert item is not None
    item.extra_attributes["start_position"] = 37.25
    ctrl.mass.players.get_player.return_value = MagicMock()
    ctrl.signal_update.reset_mock()  # type: ignore[attr-defined]
    await ctrl.seek(QUEUE_ID, 0)
    assert "start_position" not in item.extra_attributes
    ctrl._load_item.assert_awaited_once()  # type: ignore[attr-defined]
    assert ctrl._load_item.await_args.kwargs["seek_position"] == 0  # type: ignore[attr-defined]
    assert queue.elapsed_time == 0
    # saved, so a restart does not bring the start back
    assert ctrl.signal_update.call_args_list[0].kwargs == {"items_changed": True}  # type: ignore[attr-defined]


async def test_play_index_plays_the_item_from_its_start() -> None:
    """A fresh play of the item loads it at its start, and elapsed starts there."""
    ctrl, queue, signals = _controller_with_stale_queue()
    new = ctrl._queue_data[QUEUE_ID].items[1]
    new.extra_attributes["start_position"] = 37.25

    await ctrl.play_index(QUEUE_ID, 1)

    assert ctrl._load_item.await_args.kwargs["seek_position"] == 37.25  # type: ignore[attr-defined]
    assert queue.elapsed_time == 37.25
    assert all(elapsed == 37.25 for item_id, elapsed in signals if item_id == "new"), signals


@pytest.mark.parametrize(
    ("repeat_mode", "loaded"), [(RepeatMode.OFF, 2), (RepeatMode.ONE, 1)], ids=["next", "repeat"]
)
async def test_the_next_item_and_a_repeat_load_at_their_start(
    repeat_mode: RepeatMode, loaded: int
) -> None:
    """The item after the current one, or the current one again, is loaded at its start."""
    ctrl = _ctrl(current_index=1, repeat_mode=repeat_mode)
    ctrl._load_item = AsyncMock()
    _item(ctrl, loaded).extra_attributes["start_position"] = 37.25

    item = await ctrl.load_next_queue_item("q1", _item(ctrl, 1).queue_item_id)

    assert item is _item(ctrl, loaded)
    ctrl._load_item.assert_awaited_once_with(item, seek_position=37.25)


async def test_the_lapse_drops_the_start_with_the_end() -> None:
    """The start lapses, saved, when the item stops being current; the next item keeps its own."""
    ctrl = _ctrl(current_index=0)
    played, playing = _item(ctrl, 0), _item(ctrl, 1)
    played.extra_attributes["start_position"] = 10.0
    playing.extra_attributes["start_position"] = 10.0
    ctrl._get_output_player_ids = Mock(return_value=set())
    ctrl._handle_playback_progress_report = Mock()
    ctrl.is_smart_shuffle_active = Mock(return_value=False)
    player = MagicMock()
    player.player_id = "q1"
    player.state.playback_state = PlaybackState.PLAYING
    queue = ctrl._queue_data["q1"].queue
    ctrl._update_current_index_from_player = Mock(return_value=True)
    ctrl._update_queue_from_player(player)
    queue.current_index, queue.current_item = 1, playing
    ctrl._update_queue_from_player(player)
    assert "start_position" not in played.extra_attributes
    assert ctrl._queue_data["q1"].items_cache_dirty
    assert playing.extra_attributes["start_position"] == 10.0


async def test_the_player_is_told_the_stream_runs_from_the_start_to_the_end() -> None:
    """The player is handed a stream as long as the part between the start and the end."""
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
    # as set_start_position leaves a track whose details were resolved ahead
    item.extra_attributes["start_position"] = 40.25
    item.streamdetails.seek_position = 40.25
    media = await ctrl.player_media_from_queue_item(item)
    assert media.duration == 300
    assert media.stream_duration == 50


async def test_prepare_next_warms_the_next_item_at_its_start() -> None:
    """The next item's audio is prepared from its start, so the fade into it can read it."""
    from tests.controllers.player_queues.test_stream_feeder import (  # noqa: PLC0415
        _controller_with_next_item,
    )

    controller, next_item, mass = _controller_with_next_item()
    next_item.extra_attributes = {"start_position": 37.25}
    next_item.streamdetails = _details(seek_position=37.25)
    # audio prepared from 0 that the producer has not brought near 37 s cannot serve it
    stale = MagicMock()
    stale.is_valid.side_effect = lambda seek_ms=0: seek_ms < 20000
    next_item.streamdetails.buffer = stale
    mass.streams.audio.get_audio_buffer = AsyncMock()

    controller.prepare_next_audio_buffer("queue-1", "current")
    await mass.create_task.call_args.args[0]

    stale.is_valid.assert_called_with(37250)
    assert mass.streams.audio.get_audio_buffer.await_args.kwargs["seek_position_ms"] == 37250
