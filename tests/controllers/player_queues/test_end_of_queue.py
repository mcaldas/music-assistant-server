"""
Tests for what a queue looks like once it played to its end.

A queue that ran out of items keeps them and parks its playback position on the last one, flagged as
ended, so clients can say it finished and pressing play starts it over. Everything with a natural end
settles that same way; only a live source, which has no end, is left exactly as it was.
"""

from __future__ import annotations

import time
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import MediaType, PlaybackState, QueueOption
from music_assistant_models.errors import PlayerUnavailableError
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import ATTR_PLAY_ACTION_IN_PROGRESS
from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.playback_tracker import PlaybackTrackerMixin
from music_assistant.controllers.player_queues.state import PlayerQueueData

if TYPE_CHECKING:
    from music_assistant.controllers.player_queues.helpers import CompareState

QUEUE_ID = "q1"


def _controller(*, player_available: bool = True) -> tuple[PlayerQueuesController, PlayerQueue]:
    """
    Build a bare controller holding a queue that just finished its last item.

    :param player_available: Whether the queue's player can be resolved, so a caller can exercise
        a playback start that fails before anything is loaded.
    """
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    queue = PlayerQueue(queue_id=QUEUE_ID, active=True, display_name="Q1", available=True, items=0)
    items = [
        QueueItem(queue_id=QUEUE_ID, queue_item_id="first", name="first", duration=100),
        QueueItem(queue_id=QUEUE_ID, queue_item_id="last", name="last", duration=100),
    ]
    queue.items = len(items)
    queue.state = PlaybackState.IDLE
    queue.current_index = 1
    queue.current_item = items[1]
    queue.elapsed_time = 99.0
    queue.elapsed_time_last_updated = time.time() - 900
    queue.resume_pos = 99
    queue.index_in_buffer = 1
    queue_data = PlayerQueueData(queue=queue)
    queue_data.items = items
    ctrl._queue_data = {QUEUE_ID: queue_data}
    ctrl.signal_update = Mock()  # type: ignore[method-assign]
    ctrl.on_player_update = Mock()  # type: ignore[method-assign]
    ctrl._check_player_permission = Mock()  # type: ignore[method-assign]
    ctrl.play_index = AsyncMock()  # type: ignore[method-assign]
    ctrl._managed_pool = MagicMock()
    ctrl.mass = MagicMock()
    ctrl.mass.create_task = Mock(side_effect=lambda coro, **_kw: coro.close())
    ctrl.mass.players.play_media = AsyncMock()
    ctrl.mass.players._handle_cmd_stop = AsyncMock()
    ctrl.mass.music.get_playback_speed = AsyncMock(return_value=1.0)
    ctrl.mass.players.get_player = Mock(
        return_value=MagicMock(state=MagicMock(playback_state=PlaybackState.IDLE))
        if player_available
        else None
    )
    ctrl.logger = MagicMock()
    return ctrl, queue


def test_mark_ended_keeps_the_items_and_parks_on_the_last_one() -> None:
    """Ending a queue leaves it replayable, with the position on the item that finished."""
    ctrl, queue = _controller()

    ctrl.mark_ended(QUEUE_ID)

    assert queue.ended is True
    # the items survive so the queue can be replayed, and there is still evidence it played
    assert queue.items == 2
    assert [x.queue_item_id for x in ctrl._queue_data[QUEUE_ID].items] == ["first", "last"]
    # parked on the last item: a null index reads as "never started", one past the end is
    # silently misread by everything that does arithmetic on it
    assert queue.current_index == 1
    assert queue.current_item is not None
    assert queue.current_item.queue_item_id == "last"
    assert queue.next_item is None
    assert queue.elapsed_time == 0
    assert queue.resume_pos == 0
    assert queue.index_in_buffer is None


def test_mark_ended_on_an_itemless_queue_clears_it() -> None:
    """With nothing to replay there is nothing to advertise as finished either."""
    ctrl, queue = _controller()
    ctrl._queue_data[QUEUE_ID].items = []
    queue.items = 0

    ctrl.mark_ended(QUEUE_ID)

    assert queue.ended is False
    assert queue.items == 0
    assert queue.current_index is None


def test_clear_still_empties_the_queue() -> None:
    """An explicit clear keeps emptying the queue, and an emptied queue is not "ended"."""
    ctrl, queue = _controller()
    queue.next_item = ctrl._queue_data[QUEUE_ID].items[0]

    ctrl.clear(QUEUE_ID)

    assert queue.items == 0
    assert ctrl._queue_data[QUEUE_ID].items == []
    assert queue.current_index is None
    assert queue.current_item is None
    assert queue.ended is False
    assert queue.next_item is None


async def test_play_on_an_ended_queue_restarts_from_the_beginning() -> None:
    """Pressing play on a queue that reached its end replays it from the first item."""
    ctrl, _queue = _controller()
    ctrl.mark_ended(QUEUE_ID)

    await ctrl.resume(QUEUE_ID)

    ctrl.play_index.assert_awaited_once()  # type: ignore[attr-defined]
    queue_id, item_id, seek_pos = ctrl.play_index.await_args.args[:3]  # type: ignore[attr-defined]
    assert queue_id == QUEUE_ID
    assert item_id == "first"
    assert seek_pos == 0


def _stub_play_index_deps(ctrl: PlayerQueuesController) -> None:
    """Stub out what the real play_index reaches for, so it can run on the bare controller."""
    ctrl._set_transitioning = Mock()  # type: ignore[method-assign]
    ctrl._load_item = AsyncMock()  # type: ignore[method-assign]
    ctrl._get_next_index = Mock(return_value=None)  # type: ignore[method-assign]
    ctrl.player_media_from_queue_item = AsyncMock()  # type: ignore[method-assign]


def test_reordering_a_finished_queue_leaves_it_finished() -> None:
    """A reorder (or a shuffle toggle, which is one) must not silently un-finish the queue."""
    ctrl, queue = _controller()
    ctrl.mark_ended(QUEUE_ID)

    ctrl.update_items(QUEUE_ID, [*reversed(ctrl._queue_data[QUEUE_ID].items)])

    # still finished, so a play press still restarts it instead of replaying its last item
    assert queue.ended is True


async def test_starting_playback_ends_the_ended_state() -> None:
    """Playing anything means the queue is running again, not sitting at its end."""
    ctrl, queue = _controller()
    ctrl.mark_ended(QUEUE_ID)
    _stub_play_index_deps(ctrl)

    # the helper mocks play_index away for the resume test; call the real one here
    await PlayerQueuesController.play_index(ctrl, QUEUE_ID, 0)

    assert queue.ended is False


async def test_a_failed_start_leaves_the_queue_finished() -> None:
    """If playback never gets going, the queue stays finished instead of losing its position."""
    ctrl, queue = _controller(player_available=False)
    ctrl.mark_ended(QUEUE_ID)
    _stub_play_index_deps(ctrl)

    with pytest.raises(PlayerUnavailableError):
        await PlayerQueuesController.play_index(ctrl, QUEUE_ID, 0)

    # still finished, so the queue can still be started over rather than replaying its last item
    assert queue.ended is True


async def test_restarting_a_finished_queue_ignores_a_left_over_resume_point() -> None:
    """A restart must start over, not seek an audiobook back to where it was left off."""
    ctrl, _queue = _controller()
    first_item = ctrl._queue_data[QUEUE_ID].items[0]
    first_item.media_item = SimpleNamespace(  # type: ignore[assignment]
        resume_position_ms=3_600_000, media_type=MediaType.AUDIOBOOK
    )
    ctrl.mark_ended(QUEUE_ID)
    _stub_play_index_deps(ctrl)

    await PlayerQueuesController.play_index(ctrl, QUEUE_ID, 0)

    assert ctrl._load_item.await_args is not None  # type: ignore[attr-defined]
    assert ctrl._load_item.await_args.kwargs["seek_position"] == 0  # type: ignore[attr-defined]


async def test_playing_an_unfinished_item_still_honours_its_resume_point() -> None:
    """The resume point still applies on a queue that did not play to its end."""
    ctrl, _queue = _controller()
    first_item = ctrl._queue_data[QUEUE_ID].items[0]
    first_item.media_item = SimpleNamespace(  # type: ignore[assignment]
        resume_position_ms=3_600_000, media_type=MediaType.AUDIOBOOK
    )
    _stub_play_index_deps(ctrl)

    await PlayerQueuesController.play_index(ctrl, QUEUE_ID, 0)

    assert ctrl._load_item.await_args is not None  # type: ignore[attr-defined]
    assert ctrl._load_item.await_args.kwargs["seek_position"] == 3599  # type: ignore[attr-defined]


# --- enqueueing onto a finished queue ---


def _enqueue_controller() -> tuple[PlayerQueuesController, PlayerQueue, dict[str, Any]]:
    """Build a finished queue whose `load` splices items the way the real one does."""
    ctrl, queue = _controller()
    ctrl.mark_ended(QUEUE_ID)
    calls: dict[str, Any] = {}

    async def _fake_load(
        queue_id: str,
        queue_items: list[QueueItem],
        insert_at_index: int = 0,
        keep_remaining: bool = True,
        keep_played: bool = True,
        shuffle: bool = False,  # noqa: ARG001
    ) -> None:
        current = ctrl._queue_data[queue_id].items
        prev_items = current[:insert_at_index] if keep_played else []
        next_items = current[insert_at_index:] if keep_remaining else []
        ctrl.update_items(queue_id, [*prev_items, *queue_items, *next_items])

    ctrl.load = AsyncMock(side_effect=_fake_load)  # type: ignore[method-assign]
    ctrl.play_index = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda _qid, index, *_a, **_kw: calls.__setitem__("played", index)
    )
    ctrl._load_pinned_first = AsyncMock()  # type: ignore[method-assign]
    return ctrl, queue, calls


def _new_item() -> list[QueueItem]:
    """Build the batch being enqueued."""
    return [QueueItem(queue_id=QUEUE_ID, queue_item_id="new", name="new", duration=100)]


async def test_playing_something_new_replaces_a_finished_queue() -> None:
    """A finished queue is done, so a play/replace request starts a fresh one."""
    for option in (QueueOption.PLAY, QueueOption.REPLACE, QueueOption.REPLACE_NEXT):
        ctrl, queue, _calls = _enqueue_controller()

        await ctrl._enqueue_with_option(QUEUE_ID, _new_item(), option)

        assert [x.queue_item_id for x in ctrl._queue_data[QUEUE_ID].items] == ["new"], option
        assert queue.ended is False, option


async def test_queueing_something_next_replaces_a_finished_queue() -> None:
    """NEXT onto a finished queue starts fresh too, and stages without playing."""
    ctrl, queue, calls = _enqueue_controller()

    await ctrl._enqueue_with_option(QUEUE_ID, _new_item(), QueueOption.NEXT)

    assert [x.queue_item_id for x in ctrl._queue_data[QUEUE_ID].items] == ["new"]
    assert queue.current_index == 0
    assert queue.ended is False
    assert "played" not in calls


async def test_adding_to_a_finished_queue_keeps_it_and_continues_from_the_end() -> None:
    """An explicit ADD keeps what played and points the queue at what was just added."""
    ctrl, queue, calls = _enqueue_controller()

    await ctrl._enqueue_with_option(QUEUE_ID, _new_item(), QueueOption.ADD)

    # the tracks that played are kept, so the queue still shows what it played
    assert [x.queue_item_id for x in ctrl._queue_data[QUEUE_ID].items] == ["first", "last", "new"]
    # ...and the position moves onto the added item, so a play press starts there
    assert queue.current_index == 2
    assert queue.current_item is not None
    assert queue.current_item.queue_item_id == "new"
    assert queue.ended is False
    # adding never starts playback by itself
    assert "played" not in calls


async def test_adding_a_batch_to_a_finished_queue_reports_what_follows() -> None:
    """Ending the queue cleared its next item; a batch of added items must refresh it."""
    ctrl, queue, _calls = _enqueue_controller()
    batch = [
        QueueItem(queue_id=QUEUE_ID, queue_item_id=x, name=x, duration=100) for x in ("new", "new2")
    ]

    await ctrl._enqueue_with_option(QUEUE_ID, batch, QueueOption.ADD)

    assert queue.current_item is not None
    assert queue.current_item.queue_item_id == "new"
    assert queue.next_item is not None
    assert queue.next_item.queue_item_id == "new2"


# --- what "finished" settles into, per media type ---


def _tracker(*items: Any) -> Any:
    """Build a playback tracker stand-in owning a queue holding the given items."""
    tracker = MagicMock()
    tracker._queue_data = {
        QUEUE_ID: SimpleNamespace(
            items=list(items),
            flow_mode_stream_log=[],
            session_id="session",
            last_served_item_id=None,
        )
    }
    return tracker


def _queue_stub() -> PlayerQueue:
    """Build a queue stand-in that has no next item left to play."""
    return cast(
        "PlayerQueue",
        SimpleNamespace(
            queue_id=QUEUE_ID,
            display_name="Q1",
            next_item=None,
            flow_mode=False,
            active=True,
            extra_attributes={},
        ),
    )


def _item(media_type: MediaType) -> Any:
    """Build a queue item stand-in of the given media type."""
    return SimpleNamespace(
        media_type=media_type, streamdetails=None, duration=3600, name=media_type.value
    )


def test_finished_music_is_marked_ended() -> None:
    """A queue that ran out of music keeps its items and says it finished."""
    item = _item(MediaType.TRACK)
    tracker = _tracker(item)

    PlaybackTrackerMixin._finish_queue(tracker, _queue_stub(), item)

    tracker.mark_ended.assert_called_once_with(QUEUE_ID)
    tracker.clear.assert_not_called()


def test_a_finished_audiobook_or_episode_is_marked_ended_too() -> None:
    """Anything with a natural end settles the same way, so the outcome stays predictable."""
    for media_type in (MediaType.AUDIOBOOK, MediaType.PODCAST_EPISODE):
        item = _item(media_type)
        tracker = _tracker(item)

        PlaybackTrackerMixin._finish_queue(tracker, _queue_stub(), item)

        tracker.mark_ended.assert_called_once_with(QUEUE_ID)
        tracker.clear.assert_not_called()


def test_a_live_source_is_left_alone_even_without_a_previous_item() -> None:
    """A radio or audio source that stopped did not run out; leave the queue for a later resume."""
    for media_type in (MediaType.RADIO, MediaType.AUDIO_SOURCE):
        tracker = _tracker(_item(media_type))

        # prev_item is None here: the media type is recovered from the queue's last item
        PlaybackTrackerMixin._finish_queue(tracker, _queue_stub(), None)

        tracker.clear.assert_not_called()
        tracker.mark_ended.assert_not_called()


def test_an_unknown_ending_item_keeps_the_items() -> None:
    """With no idea what ended, keep the queue rather than destroy it."""
    tracker = _tracker()

    PlaybackTrackerMixin._finish_queue(tracker, _queue_stub(), None)

    tracker.mark_ended.assert_called_once_with(QUEUE_ID)
    tracker.clear.assert_not_called()


# --- which stops reach the end-of-queue handling at all ---


def _stop_states(prev_item: Any) -> tuple[CompareState, CompareState]:
    """Build the state pair for a queue that went from playing the given item to idle."""
    prev_state = {
        "state": PlaybackState.PLAYING,
        "current_item_id": "i1",
        "current_item": prev_item,
        "last_playing_elapsed_time": 3600,
    }
    return cast("CompareState", prev_state), cast("CompareState", {"state": PlaybackState.IDLE})


def test_a_live_source_never_reaches_the_end_of_queue_handling() -> None:
    """Radio and audio sources short-circuit before the settle is scheduled."""
    for media_type in (MediaType.RADIO, MediaType.AUDIO_SOURCE):
        tracker = _tracker()
        prev_state, new_state = _stop_states(_item(media_type))

        PlaybackTrackerMixin._handle_end_of_queue(tracker, _queue_stub(), prev_state, new_state)

        tracker.mass.create_task.assert_not_called()


def test_a_queue_with_something_left_to_play_never_settles() -> None:
    """Repeat wraps to a next item, so a repeating queue can never reach its end."""
    tracker = _tracker()
    queue = _queue_stub()
    queue.next_item = _item(MediaType.TRACK)
    prev_state, new_state = _stop_states(_item(MediaType.TRACK))

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    tracker.mass.create_task.assert_not_called()


def test_stopping_part_way_through_a_track_does_not_settle_the_queue() -> None:
    """The near-completion heuristic is what separates "ran out" from "user stopped it"."""
    tracker = _tracker()
    prev_state, new_state = _stop_states(_item(MediaType.TRACK))
    prev_state["last_playing_elapsed_time"] = 30  # 30s into a 3600s track

    PlaybackTrackerMixin._handle_end_of_queue(tracker, _queue_stub(), prev_state, new_state)

    tracker.mass.create_task.assert_not_called()


def test_playing_a_track_to_its_end_settles_the_queue() -> None:
    """Running out of music schedules the settle (debounced, so late arrivals still win)."""
    tracker = _tracker()
    prev_state, new_state = _stop_states(_item(MediaType.TRACK))

    PlaybackTrackerMixin._handle_end_of_queue(tracker, _queue_stub(), prev_state, new_state)

    tracker.mass.create_task.assert_called_once()


def test_a_completed_sound_effect_settles_the_queue() -> None:
    """A sound effect that played to its end reaches the settle handling like any other item."""
    tracker = _tracker()
    prev_state, new_state = _stop_states(_item(MediaType.SOUND_EFFECT))

    PlaybackTrackerMixin._handle_end_of_queue(tracker, _queue_stub(), prev_state, new_state)

    tracker.mass.create_task.assert_called_once()


def _item_end(
    last_playing: int = 234, **ended_attributes: Any
) -> tuple[Any, PlayerQueue, CompareState, CompareState]:
    """
    Build a queue whose player went idle on an item that has another one queued behind it.

    :param last_playing: The last position the player reported while it played the item.
    :param ended_attributes: Extra attributes of the item that was playing.
    """
    ended = QueueItem(queue_id=QUEUE_ID, queue_item_id="ended", name="Ended", duration=236)
    ended.extra_attributes.update(ended_attributes)
    tracker = _tracker(ended)
    # bind the real check: the mocked self would silently swallow it otherwise
    tracker._publish_playback_stall = MethodType(
        PlaybackTrackerMixin._publish_playback_stall, tracker
    )
    # the player fetched the item it was on and nothing after it
    tracker._queue_data[QUEUE_ID].last_served_item_id = "ended"
    queue = _queue_stub()
    queue.next_item = QueueItem(queue_id=QUEUE_ID, queue_item_id="next", name="Next", duration=233)
    prev_state, new_state = _stop_states(ended)
    prev_state["last_playing_elapsed_time"] = last_playing
    return tracker, queue, prev_state, new_state


def _stall_warning(tracker: Any) -> str:
    """Return the one warning the tracker logged, as it reads in the log."""
    tracker.logger.warning.assert_called_once()
    message, *args = tracker.logger.warning.call_args.args
    return str(message % tuple(args))


def test_an_item_that_ended_with_a_next_one_never_fetched_is_published() -> None:
    """A player that stops at an item's end with more queued leaves that on the queue."""
    tracker, queue, prev_state, new_state = _item_end()

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    published = dict(queue.extra_attributes)
    assert isinstance(published.pop("playback_stalled_at"), float)
    assert published == {
        "playback_stalled_item_id": "ended",
        "playback_stalled_next_item_id": "next",
    }
    tracker.signal_update.assert_called_once_with(QUEUE_ID)
    warning = _stall_warning(tracker)
    assert all(name in warning for name in ("Q1", "Ended", "Next", "no fade"))
    # it is told, not acted on: nothing is started or settled
    tracker.mass.create_task.assert_not_called()


def test_an_item_that_ends_in_its_fade_counts_as_ended() -> None:
    """An item mixed into the next one is over where that mix ends, short of its duration."""
    tracker, queue, prev_state, new_state = _item_end(
        last_playing=198,
        transition_mix_end=200.0,
        transition_next_item_id="next",
        transition_mode="standard_crossfade",
    )

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    assert queue.extra_attributes["playback_stalled_next_item_id"] == "next"
    assert "already mixed" in _stall_warning(tracker)


@pytest.mark.parametrize(
    ("end_position", "stream_duration"),
    [(100.0, 180), (None, 100)],
    ids=["a_clients_end_position", "the_duration_of_its_stream"],
)
def test_an_item_without_a_fade_ends_where_its_audio_does(
    end_position: float | None, stream_duration: int
) -> None:
    """A client's end position comes before the stream's duration, and that before the listed one."""
    attributes = {} if end_position is None else {"end_position": end_position}
    tracker, queue, prev_state, new_state = _item_end(last_playing=98, **attributes)
    ended = prev_state["current_item"]
    assert ended is not None
    ended.streamdetails = StreamDetails(
        provider="test", item_id="ended", audio_format=AudioFormat(), duration=stream_duration
    )

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    assert queue.extra_attributes["playback_stalled_item_id"] == "ended"


@pytest.mark.parametrize(
    "report",
    [{"transition_mode": "disabled"}, {}],
    ids=["played_without_a_fade", "only_planned"],
)
def test_a_boundary_that_was_locked_without_a_fade_is_not_called_mixed(
    report: dict[str, str],
) -> None:
    """An item names its next one at every locked boundary, with or without a fade into it."""
    tracker, queue, prev_state, new_state = _item_end(transition_next_item_id="next", **report)

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    assert queue.extra_attributes["playback_stalled_next_item_id"] == "next"
    assert "no fade" in _stall_warning(tracker)


def test_a_stop_part_way_through_a_track_is_not_published() -> None:
    """A player that goes idle in the middle of an item was paused or stopped on the device."""
    tracker, queue, prev_state, new_state = _item_end(last_playing=30)

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    assert not queue.extra_attributes
    tracker.logger.warning.assert_not_called()
    tracker.signal_update.assert_not_called()


def test_a_next_item_the_player_already_fetched_is_not_published() -> None:
    """A player that asked for the next item's audio is changing tracks, not stuck."""
    tracker, queue, prev_state, new_state = _item_end()
    tracker._queue_data[QUEUE_ID].last_served_item_id = "next"

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    assert not queue.extra_attributes
    tracker.logger.warning.assert_not_called()


@pytest.mark.parametrize("ours", ["play_action_in_progress", "session_ended"])
def test_a_stop_music_assistant_made_is_not_published(ours: str) -> None:
    """A stop, a next or a play of our own is no stall, however close to the item's end."""
    tracker, queue, prev_state, new_state = _item_end()
    if ours == "play_action_in_progress":
        queue.extra_attributes[ATTR_PLAY_ACTION_IN_PROGRESS] = True
    else:
        # a stop that has gone through has ended the queue's session
        tracker._queue_data[QUEUE_ID].session_id = None

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    assert "playback_stalled_at" not in queue.extra_attributes
    tracker.logger.warning.assert_not_called()


def test_a_player_taken_by_another_source_is_not_published() -> None:
    """A queue that is no longer its player's source did not stall: something else plays."""
    tracker, queue, prev_state, new_state = _item_end()
    queue.active = False

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    assert not queue.extra_attributes
    tracker.logger.warning.assert_not_called()


async def test_a_start_of_our_own_that_has_not_come_up_yet_is_not_published() -> None:
    """A next in an item's last seconds is no stall while the player still starts the new item."""
    ctrl, queue = _controller()
    ctrl.on_player_update = MethodType(  # type: ignore[method-assign]
        PlayerQueuesController.on_player_update, ctrl
    )
    ctrl._update_current_index_from_player = Mock(return_value=True)  # type: ignore[method-assign]
    ctrl._get_output_player_ids = Mock(return_value=set())  # type: ignore[method-assign]
    ctrl._handle_playback_progress_report = Mock()  # type: ignore[method-assign]
    ctrl.is_smart_shuffle_active = Mock(return_value=False)  # type: ignore[method-assign]
    ctrl._get_resume_position = AsyncMock(return_value=0)  # type: ignore[method-assign]
    ctrl._load_item = AsyncMock()  # type: ignore[method-assign]
    ctrl.player_media_from_queue_item = AsyncMock()  # type: ignore[method-assign]
    player = ctrl.mass.players.get_player.return_value
    player.player_id, player.extra_data, player.state.active_source = QUEUE_ID, {}, None
    # the player is two seconds from the end of the first item, with the last one queued
    queue_data = ctrl._queue_data[QUEUE_ID]
    first, last = queue_data.items
    queue.current_index, queue.current_item, queue.next_item = 0, first, last
    queue.elapsed_time, queue.elapsed_time_last_updated = 98, time.time()
    queue_data.session_id, queue_data.last_served_item_id = "session", "first"
    player.state.playback_state = PlaybackState.PLAYING
    ctrl.on_player_update(player, {})

    # the old stream is gone and the new item has not started when the play action returns
    player.state.playback_state = PlaybackState.IDLE
    await PlayerQueuesController.play_index(ctrl, QUEUE_ID, 1)

    assert queue.current_item is last
    assert "playback_stalled_at" not in queue.extra_attributes
    ctrl.logger.warning.assert_not_called()


@pytest.mark.parametrize("was", ["paused", "a_flow"])
def test_a_paused_queue_or_a_flow_going_idle_is_not_published(was: str) -> None:
    """A pause that ran out is somebody's pause, and a flow stream restarts by itself."""
    tracker, queue, prev_state, new_state = _item_end()
    if was == "paused":
        prev_state["state"] = PlaybackState.PAUSED
    else:
        queue.flow_mode = True

    PlaybackTrackerMixin._handle_end_of_queue(tracker, queue, prev_state, new_state)

    assert not queue.extra_attributes
    tracker.logger.warning.assert_not_called()


async def test_the_published_stall_is_withdrawn_when_the_queue_plays_or_is_stopped() -> None:
    """The marker is about now: it goes when music plays again or the queue is stopped."""
    ctrl, queue = _controller()
    ctrl._update_current_index_from_player = Mock(return_value=True)  # type: ignore[method-assign]
    ctrl._get_output_player_ids = Mock(return_value=set())  # type: ignore[method-assign]
    ctrl._handle_playback_progress_report = Mock()  # type: ignore[method-assign]
    ctrl.is_smart_shuffle_active = Mock(return_value=False)  # type: ignore[method-assign]
    player = MagicMock(player_id=QUEUE_ID)
    stall: dict[str, str | float] = {
        "playback_stalled_at": 100.0,
        "playback_stalled_item_id": "first",
        "playback_stalled_next_item_id": "last",
    }

    # a queue that stays idle keeps saying why
    queue.extra_attributes.update(stall)
    player.state.playback_state = PlaybackState.IDLE
    ctrl._update_queue_from_player(player)
    assert queue.extra_attributes == stall

    player.state.playback_state = PlaybackState.PLAYING
    ctrl._update_queue_from_player(player)
    assert not queue.extra_attributes

    queue.extra_attributes.update(stall)
    await PlayerQueuesController._handle_stop(ctrl, QUEUE_ID)
    assert queue.extra_attributes == {ATTR_PLAY_ACTION_IN_PROGRESS: False}


def test_clearing_an_idle_queue_withdraws_a_published_stall() -> None:
    """The marker names two items of the queue, so it goes with them."""
    ctrl, queue = _controller()
    queue.extra_attributes.update(
        playback_stalled_at=100.0,
        playback_stalled_item_id="first",
        playback_stalled_next_item_id="last",
    )

    ctrl.clear(QUEUE_ID)

    assert not queue.extra_attributes
