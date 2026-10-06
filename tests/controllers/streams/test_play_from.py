"""Tests for an item's start position in the stream engine: real readers over real AudioBuffers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from music_assistant_models.enums import CrossfadeMode, MediaType
from music_assistant_models.errors import ActionUnavailable, QueueEmpty

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.streams.audio import (
    StreamsAudio,
    _IncomingFadePrefetcher,
    tail_hold_target,
)
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.controllers.streams.constants import BufferMode, BufferSize
from tests.controllers.streams.test_audio_buffer import (
    _make_mass_for_get_buffer,
    _make_stream_details,
)
from tests.controllers.streams.test_crossfade_same_album import (
    PROVIDER_ALBUM,
    _crossfade_allowed,
    _queue_item,
)
from tests.controllers.streams.test_crossfade_transition import TEST_PCM_FORMAT, _transition_keys
from tests.controllers.streams.test_play_until import (
    FADE,
    FRAME,
    SECOND,
    SR,
    _filled,
    _item,
    _run_flow,
    _run_single,
    _single_audio,
)

# where B is asked to start: not on a second, nor on a 100 ms step
START = 12.25
# the stand-in plan: B's head trimmed by TRIM, blended over CF
TRIM = 1.5
CF = 8.0


def _source(tag: int, first: int, last: int, gate: asyncio.Event | None = None) -> Any:
    """Yield seconds ``first`` to ``last`` of song ``tag`` (frames say their second and index)."""
    frames = np.arange(SR, dtype=np.int16)

    async def _gen() -> AsyncGenerator[bytes]:
        for sec in range(first, last):
            if gate is not None and sec == first + 20:
                # a slow source: it stalls 20 s in until let go
                await gate.wait()
            chunk = np.empty((SR, 2), dtype="<i2")
            chunk[:, 0] = tag * 1000 + sec
            chunk[:, 1] = frames
            yield chunk.tobytes()
            await asyncio.sleep(0)

    return _gen()


def _buffer_from(
    tag: int,
    first: int,
    last: int,
    size: BufferSize = BufferSize.BALANCED,
    *,
    realtime: bool = False,
    gate: asyncio.Event | None = None,
) -> AudioBuffer:
    """Return a real buffer whose producer starts at ``first`` (a seek at the source)."""
    audio_buffer = AudioBuffer(
        TEST_PCM_FORMAT, buffer_size=size, mode=BufferMode.SEEKABLE, is_realtime=realtime
    )
    audio_buffer._discarded_chunks = first
    audio_buffer._ready_at_chunk = first + 1
    audio_buffer.fill(_source(tag, first, last, gate), source_name=f"song-{tag}")
    return audio_buffer


def _started(item_id: str, tag: int, seconds: int, start: float, **kwargs: Any) -> Any:
    """Build a queue item with a client's start, its details loaded at it."""
    item = _item(item_id, tag, seconds, kwargs.pop("size", BufferSize.BALANCED))
    item.streamdetails.allow_seek = kwargs.pop("allow_seek", True)
    item.extra_attributes["start_position"] = start
    # as loading the item leaves it (a stream that cannot seek starts at 0)
    item.streamdetails.seek_position = start if item.streamdetails.allow_seek else 0
    return item


def _assert_runs(
    data: bytes | bytearray, tag: int, start: float, end: float, faded_in: bool = False
) -> None:
    """
    Assert ``data`` is song ``tag`` from ``start`` up to ``end``, every frame exactly once.

    ``faded_in``: ``data`` is a read from the item's start, which fades in over its first
    frames (END_POSITION_FADE): those are checked as the ramp of the frames they carry.
    """
    frames = np.frombuffer(bytes(data), dtype="<i2").reshape(-1, 2).astype(np.int64)
    expected = np.arange(round(start * SR), round(end * SR))
    assert len(frames) == len(expected)
    if faded_in:
        head = expected[:FADE]
        source = np.stack([tag * 1000 + head // SR, head % SR], axis=1)
        ramp = (source * (np.arange(FADE) / FADE)[:, None]).astype(np.int16)
        assert np.array_equal(frames[:FADE], ramp)
        frames, expected = frames[FADE:], expected[FADE:]
    assert (frames[:, 0] // 1000 == tag).all()
    positions = (frames[:, 0] % 1000) * SR + frames[:, 1]
    assert len(positions) == len(expected)
    assert np.array_equal(positions, expected), (positions[:3] / SR, start)


def _split(data: bytes, tag: int) -> tuple[bytes, bytes]:
    """Split the output at the first frame of song ``tag``."""
    tags = np.frombuffer(data, dtype="<i2")[0::2] // 1000
    first = int(np.argmax(tags == tag)) if (tags == tag).any() else len(tags)
    return data[: first * FRAME], data[first * FRAME :]


async def _timed_build(**kw: Any) -> SimpleNamespace:
    """Plan a fade that trims TRIM of B's head and blends over CF, ending with A's tail."""
    tail = len(kw["fade_out_data"]) / SECOND
    return SimpleNamespace(
        timing_info=SimpleNamespace(
            fadein_trimmed_duration=TRIM,
            crossfade_duration=CF,
            pre_crossfade_duration=tail - CF,
            post_crossfade_duration=0.0,
        )
    )


async def _cut_mix(
    fade: Any, *, fade_in_part: AsyncGenerator[bytes], fade_out_part: bytes, **_kw: Any
) -> AsyncGenerator[bytes]:
    """
    Stand in for the mixer, keeping its timing: A alone, then B from its trimmed head.

    The overlap carries only B's frames, so the output says which of B's frames the fade
    played; past the trim every frame of B the mix reads comes out once, as in a real mix.
    """
    timing = fade.timing_info
    yield fade_out_part[: int(timing.pre_crossfade_duration * SR) * FRAME]
    skip = int(timing.fadein_trimmed_duration * SR) * FRAME
    async for chunk in fade_in_part:
        part, skip = chunk[skip:], max(0, skip - len(chunk))
        if part:
            yield part


def _boundary_audio(monkeypatch: pytest.MonkeyPatch, nxt: Any) -> tuple[StreamsAudio, MagicMock]:
    """Build a single-item StreamsAudio planning the timed fade into ``nxt``."""
    audio, mass = _single_audio(monkeypatch, nxt)
    monkeypatch.setattr(audio.smart_fades_mixer, "build", AsyncMock(side_effect=_timed_build))
    monkeypatch.setattr(audio.smart_fades_mixer, "mix", _cut_mix)
    return audio, mass


async def _a_then_b(
    monkeypatch: pytest.MonkeyPatch,
    second: Any,
    mode: CrossfadeMode = CrossfadeMode.SMART_CROSSFADE,
    before_boundary: Any = None,
) -> tuple[bytes, bytes, StreamsAudio, Any]:
    """Stream A (100 s) into ``second`` in single-item mode, then ``second``'s own request."""
    first = _item("a", 1, 100, BufferSize.BALANCED)
    await _filled(first, second)
    audio, mass = _boundary_audio(monkeypatch, second)
    if before_boundary is not None:
        mass.player_queues.load_next_queue_item = AsyncMock(side_effect=before_boundary(audio))
    a_out = await _run_single(audio, first, mode)
    # the read mark the boundary left on B, before B's own request reads further
    marked = audio.read_positions.get(second.queue_item_id)
    mass.player_queues.load_next_queue_item = AsyncMock(side_effect=QueueEmpty)
    b_out = await _run_single(audio, second, mode)
    return a_out, b_out, audio, (first, marked)


# ---- single-item mode ----


async def test_single_mix_reads_b_from_its_start_and_b_goes_on_from_the_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A's stream ends on B's frames from its start; B's request carries on at the next frame."""
    second = _started("b", 2, 120, START)
    a_out, b_out, audio, (first, marked) = await _a_then_b(monkeypatch, second)

    a_part, blended = _split(a_out, 2)
    _assert_runs(a_part, 1, 0, 100 - CF)
    # the overlap is B's head after the trim, from exactly its start
    _assert_runs(blended, 2, START + TRIM, START + TRIM + CF)
    # then B's own request, with no frame repeated or lost
    _assert_runs(b_out, 2, START + TRIM + CF, 120)
    keys = _transition_keys(first)
    assert keys["transition_incoming_entry"] == START + TRIM
    # B's head the mix reads (45 s) is fixed from the boundary on
    assert marked == pytest.approx(START + 45)
    # elapsed in song seconds where B's own audio starts, and B's length is its own
    assert second.streamdetails.seek_position == START + TRIM + CF
    assert second.streamdetails.duration == 120
    # a speaker asking for B again gets it from that elapsed offset, on 100 ms steps as today
    again = await _run_single(audio, second, CrossfadeMode.SMART_CROSSFADE)
    _assert_runs(again, 2, 21.7, 120)


async def test_single_a_fade_capped_by_half_of_b_hands_over_on_the_frame_the_mix_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half of B from 1.345 is 19.3275 s: the fade takes 19 s, and B goes on at the next frame."""
    second = _started("b", 2, 40, 1.345)
    a_out, b_out, _audio, (_first, marked) = await _a_then_b(monkeypatch, second)
    _assert_runs(_split(a_out, 2)[1] + b_out, 2, 1.345 + TRIM, 40)
    assert marked == pytest.approx(1.345 + 19)


async def test_single_an_item_without_a_start_is_asked_again_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Without a start, a speaker asking again reads from its elapsed offset as it always did.

    24.3 s of trim and 8 s of overlap make an offset of 32.29999...: truncated to the
    millisecond, then to 100 ms steps, the read starts at 32.2 s (rounding would say 32.3).
    """
    second = _item("b", 2, 120, BufferSize.BALANCED)
    first = _item("a", 1, 100, BufferSize.BALANCED)
    await _filled(first, second)
    audio, mass = _boundary_audio(monkeypatch, second)

    async def _deep_trim(**kw: Any) -> SimpleNamespace:
        fade = await _timed_build(**kw)
        fade.timing_info.fadein_trimmed_duration = 24.3
        return fade

    monkeypatch.setattr(audio.smart_fades_mixer, "build", AsyncMock(side_effect=_deep_trim))
    await _run_single(audio, first, CrossfadeMode.SMART_CROSSFADE)
    mass.player_queues.load_next_queue_item = AsyncMock(side_effect=QueueEmpty)
    await _run_single(audio, second, CrossfadeMode.SMART_CROSSFADE)
    assert second.streamdetails.seek_position * 1000 < 32300
    again = await _run_single(audio, second, CrossfadeMode.SMART_CROSSFADE)
    _assert_runs(again, 2, 32.2, 120)


async def test_single_a_start_sent_as_a_long_json_number_keeps_the_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A start sent as 12.3449999 is stored as 12.345 and played from exactly there."""
    second = _started("b", 2, 120, 0.0)
    del second.extra_attributes["start_position"]
    second.streamdetails.seek_position = 0
    queues = MagicMock()
    queues.get_item.return_value = second
    queues.mass.streams.audio.read_positions = {}
    await PlayerQueuesController.set_start_position(queues, "queue-1", "b", 12.3449999)
    assert second.streamdetails.seek_position == 12.345

    _a_out, b_out, _audio, _ = await _a_then_b(monkeypatch, second)
    _assert_runs(b_out, 2, 12.345 + TRIM + CF, 120)


async def test_single_a_reselection_that_rounded_the_seek_keeps_the_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Details re-resolved before the boundary carry a whole-second seek: the lock restores it."""
    second = _started("b", 2, 120, START)

    def _reselect(_audio: StreamsAudio) -> Any:
        async def _load_next(*_args: Any) -> Any:
            # what a capacity reselection that rounded the seek would leave on the item
            second.streamdetails = SimpleNamespace(**vars(second.streamdetails))
            second.streamdetails.seek_position = int(START)
            return second

        return _load_next

    a_out, b_out, _audio, _ = await _a_then_b(monkeypatch, second, before_boundary=_reselect)
    _assert_runs(_split(a_out, 2)[1], 2, START + TRIM, START + TRIM + CF)
    _assert_runs(b_out, 2, START + TRIM + CF, 120)


async def test_single_a_start_moved_while_the_next_item_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A start moved during load_next's await, then the old one written back: the new one plays."""
    second = _started("b", 2, 120, START)

    def _moved(audio: StreamsAudio) -> Any:
        async def _load_next(*_args: Any) -> Any:
            queues = MagicMock()
            queues.get_item.return_value = second
            queues.mass.streams.audio = audio
            await PlayerQueuesController.set_start_position(queues, "queue-1", "b", 20.5)
            # the details resolved by the in-flight load still say the old start
            second.streamdetails.seek_position = START
            return second

        return _load_next

    a_out, b_out, _audio, _ = await _a_then_b(monkeypatch, second, before_boundary=_moved)
    _assert_runs(_split(a_out, 2)[1], 2, 20.5 + TRIM, 20.5 + TRIM + CF)
    _assert_runs(b_out, 2, 20.5 + TRIM + CF, 120)


async def test_single_a_stream_that_cannot_seek_plays_from_0_with_its_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The start is kept on the item, but a stream that cannot seek is mixed in from 0."""
    second = _started("b", 2, 120, START, allow_seek=False)
    a_out, b_out, _audio, (first, _) = await _a_then_b(monkeypatch, second)
    _assert_runs(_split(a_out, 2)[1], 2, TRIM, TRIM + CF)
    _assert_runs(b_out, 2, TRIM + CF, 120)
    assert _transition_keys(first)["transition_incoming_entry"] == TRIM


async def test_single_repeat_one_plays_the_item_again_from_its_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An item repeated into itself: the mix reads it from its start, its request goes on."""
    item = _started("a", 1, 100, START)
    await _filled(item)
    audio, mass = _boundary_audio(monkeypatch, item)
    first_pass = await _run_single(audio, item, CrossfadeMode.SMART_CROSSFADE)
    mass.player_queues.load_next_queue_item = AsyncMock(side_effect=QueueEmpty)
    second_pass = await _run_single(audio, item, CrossfadeMode.SMART_CROSSFADE)

    played, blended = first_pass[: -int(CF * SR) * FRAME], first_pass[-int(CF * SR) * FRAME :]
    _assert_runs(played, 1, START, 100 - CF, faded_in=True)
    _assert_runs(blended, 1, START + TRIM, START + TRIM + CF)
    _assert_runs(second_pass, 1, START + TRIM + CF, 100)


async def test_single_without_a_crossfade_b_starts_exactly_at_its_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    No fade: A plays out whole and B's request starts at 12.250, not at 12.200.

    B starts mid-waveform, so it fades in over its first 20 ms instead of jumping in.
    """
    second = _started("b", 2, 120, START)
    a_out, b_out, _audio, _ = await _a_then_b(monkeypatch, second, CrossfadeMode.DISABLED)
    _assert_runs(a_out, 1, 0, 100)
    _assert_runs(b_out, 2, START, 120, faded_in=True)


@pytest.mark.parametrize("case", ["slow", "realtime", "minimal", "moved"])
async def test_single_audio_not_there_at_the_start_is_a_fade_or_a_clean_cut(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """
    Whatever B's prepared audio, the boundary fades or cuts cleanly and B plays from its start.

    slow: a source stalled 30 s short of a start at 50 s; realtime: one prepared at its start;
    minimal: a 60 s buffer from 0 that can never reach 100 s; moved: a start moved to 100 s
    after B was prepared from 0, which lets that audio go.
    """
    start = {"slow": 50.0, "realtime": START, "minimal": 100.0, "moved": START}[case]
    second = _started("b", 2, 200, start)
    gate = asyncio.Event()
    if case == "slow":
        prepared = _buffer_from(2, 0, 200, gate=gate)
    elif case == "realtime":
        prepared = _buffer_from(2, int(start), 200, realtime=True)
    else:
        prepared = _buffer_from(2, 0, 200, BufferSize.MINIMAL)
    await second.streamdetails.buffer.clear()
    second.streamdetails.buffer = prepared
    first = _item("a", 1, 100, BufferSize.BALANCED)
    await _filled(first)
    while prepared.seconds_available < 20:
        await asyncio.sleep(0.01)
    audio, mass = _boundary_audio(monkeypatch, second)
    if case == "moved":
        queues = MagicMock()
        queues.get_item.return_value = second
        queues.mass.streams.audio = audio
        await PlayerQueuesController.set_start_position(queues, "queue-1", "b", 100.0)
        # audio from 0 that cannot reach 100 s is let go
        assert second.streamdetails.buffer is None
        assert prepared.cancelled
        start = 100.0

    async def _fresh_buffer(item: Any, seek_position_ms: int = 0, **_kw: Any) -> AudioBuffer:
        # what get_audio_buffer does when the buffer cannot serve the seek: a new one there
        current = item.streamdetails.buffer
        if current is None or not current.is_valid(seek_position_ms):
            item.streamdetails.buffer = _buffer_from(2, seek_position_ms // 1000, 200)
        return cast("AudioBuffer", item.streamdetails.buffer)

    monkeypatch.setattr(audio, "get_audio_buffer", AsyncMock(side_effect=_fresh_buffer))
    a_out = await _run_single(audio, first, CrossfadeMode.SMART_CROSSFADE)
    gate.set()
    mass.player_queues.load_next_queue_item = AsyncMock(side_effect=QueueEmpty)
    b_out = await _run_single(audio, second, CrossfadeMode.SMART_CROSSFADE)

    a_part, blended = _split(a_out, 2)
    if case == "realtime":
        # its audio is at its start: the fade plays
        _assert_runs(blended, 2, start + TRIM, start + TRIM + CF)
        _assert_runs(b_out, 2, start + TRIM + CF, 200)
    else:
        # no fade: A plays out whole and B starts at its start, nothing raised or lost
        assert blended == b""
        _assert_runs(a_part, 1, 0, 100)
        _assert_runs(b_out, 2, start, 200, faded_in=True)
    await cast("AudioBuffer", second.streamdetails.buffer).clear()


# ---- flow mode ----


def _flow_items() -> tuple[Any, Any]:
    first = _item("a", 1, 40, BufferSize.BALANCED)
    second = _started("b", 2, 60, START)
    return first, second


@pytest.mark.parametrize("prefetch", [True, False], ids=["take", "fresh-open"])
async def test_flow_plays_b_from_its_start(monkeypatch: pytest.MonkeyPatch, prefetch: bool) -> None:
    """Gathered alongside A's tail or opened at the boundary, B runs from its start."""
    first, second = _flow_items()
    if not prefetch:
        monkeypatch.setattr(_IncomingFadePrefetcher, "ensure_started", lambda *_a, **_kw: None)
    out, audio, _mass, _log = await _run_flow(monkeypatch, first, second)
    a_part, b_part = out[: 40 * SECOND], out[40 * SECOND :]
    _assert_runs(a_part, 1, 0, 40)
    _assert_runs(b_part, 2, START, 60, faded_in=True)
    build = cast("AsyncMock", audio.smart_fades_mixer.build)
    assert build.call_args.kwargs["fade_in_start"] == START
    assert _transition_keys(first)["transition_incoming_entry"] == START
    # elapsed in song seconds: from the start, past the standard fade's overlap
    assert second.streamdetails.seek_position == START + 8
    assert audio.read_positions["b"] == float("inf")


async def test_flow_a_failed_mix_plays_b_from_its_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mixer fails before its first chunk: A's tail plays, then B from its start."""
    first, second = _flow_items()

    def _failing(_audio: StreamsAudio) -> Any:
        async def _mix(*_args: Any, **_kw: Any) -> AsyncGenerator[bytes]:
            raise RuntimeError("mixer down")
            yield b""  # pragma: no cover

        return _mix

    out, _audio, _mass, _log = await _run_flow(monkeypatch, first, second, mix=_failing)
    a_part, b_part = out[: 40 * SECOND], out[40 * SECOND :]
    _assert_runs(a_part, 1, 0, 40)
    _assert_runs(b_part, 2, START, 60, faded_in=True)
    assert second.streamdetails.seek_position == START


async def test_flow_fixes_the_start_when_the_item_comes_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A start sent while the boundary waits for B's audio is refused; B plays from the old one."""
    first, second = _flow_items()
    monkeypatch.setattr(_IncomingFadePrefetcher, "ensure_started", lambda *_a, **_kw: None)
    refused: list[bool] = []

    async def _await_fade_source(self: StreamsAudio, *_args: Any) -> None:
        queues = MagicMock()
        queues.get_item.return_value = second
        queues.mass.streams.audio = self
        with pytest.raises(ActionUnavailable):
            await PlayerQueuesController.set_start_position(queues, "queue-1", "b", 30.0)
        refused.append(True)

    monkeypatch.setattr(StreamsAudio, "_await_fade_source", _await_fade_source)
    out, _audio, _mass, _log = await _run_flow(monkeypatch, first, second)
    assert refused == [True]
    assert second.extra_attributes["start_position"] == START
    _assert_runs(out[40 * SECOND :], 2, START, 60, faded_in=True)


# ---- hold and fade sizing ----


async def test_a_part_holds_at_most_half_of_what_it_plays() -> None:
    """From 60 s to an end at 100 s an item plays 40 s: its held tail is at most 20 s."""
    item = _started("a", 1, 200, 60.0)
    item.extra_attributes["end_position"] = 100.0
    await _filled(item)
    assert tail_hold_target(item, 45 * SECOND, TEST_PCM_FORMAT) == 20 * SECOND
    del item.extra_attributes["end_position"]
    # to its own end it plays 140 s: the full window
    assert tail_hold_target(item, 45 * SECOND, TEST_PCM_FORMAT) == 45 * SECOND
    await item.streamdetails.buffer.clear()


def test_a_fade_into_a_started_item_of_the_same_album_is_allowed() -> None:
    """An item started later is no gapless album seam, so it gets its fade."""
    current = _queue_item("track-1", PROVIDER_ALBUM)
    later = _queue_item("track-2", PROVIDER_ALBUM)
    assert not _crossfade_allowed(current, later)
    later.extra_attributes["start_position"] = 30.0
    assert _crossfade_allowed(current, later)


async def test_a_buffer_started_inside_a_track_starts_no_analysis() -> None:
    """Audio prepared from 37 s (decoded from 0) is not analysed: only a play from 0 is."""
    mass, start_analysis, scheduled = _make_mass_for_get_buffer()
    details = _make_stream_details(MediaType.TRACK, duration=180, allow_seek=True)
    audio_buffer = await AudioBuffer.get_buffer(mass, details, seek_position_ms=37000)
    assert audio_buffer.first_buffered_chunk == 0
    assert scheduled == []
    start_analysis.assert_not_called()
    await audio_buffer.clear()
