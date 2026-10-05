"""Tests for an item's end position in the stream engine: real readers over real AudioBuffers."""

from __future__ import annotations

import asyncio
import struct
from array import array
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import CrossfadeMode, MediaType
from music_assistant_models.errors import ActionUnavailable, QueueEmpty

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.streams.audio import (
    END_POSITION_FADE,
    StreamsAudio,
    _IncomingFadePrefetcher,
)
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.controllers.streams.constants import BufferMode, BufferSize
from music_assistant.controllers.streams.smart_fades.fades import StandardCrossFade
from music_assistant.models.music_provider import MusicProvider
from tests.controllers.streams.test_crossfade_transition import (
    STANDARD_CROSSFADE_DURATION,
    TEST_PCM_FORMAT,
    _flow_audio,
    _transition_keys,
)

SR = TEST_PCM_FORMAT.sample_rate
SECOND = TEST_PCM_FORMAT.pcm_sample_size
FRAME = 4
# frames an end fades out over
FADE = int(END_POSITION_FADE * SR)

MixStandIn = Callable[..., AsyncGenerator[bytes]]


def _pcm(tag: int, seconds: int) -> AsyncGenerator[bytes]:
    """Yield 1-second chunks; every frame says (tag*1000 + second, frame within second)."""
    frames = array("h", range(SR))

    async def _gen() -> AsyncGenerator[bytes]:
        for sec in range(seconds):
            chunk = array("h", [tag * 1000 + sec]) * (2 * SR)
            chunk[1::2] = frames
            yield chunk.tobytes()
            await asyncio.sleep(0)

    return _gen()


def _at(data: bytes | bytearray, index: int) -> float:
    """Return the media second (track-local) of the frame at ``index``."""
    sec, frame = struct.unpack_from("<hh", data, index * FRAME)
    return float(sec % 1000 + frame / SR)


def _assert_ends_at(data: bytes | bytearray, end: float) -> None:
    """Assert the audio runs exactly up to ``end`` and fades out over its last frames."""
    frames = len(data) // FRAME
    before = frames - FADE - 1
    assert _at(data, before) == pytest.approx(end - (FADE + 1) / SR)
    level = struct.unpack_from("<h", data, before * FRAME)[0]
    fade = [struct.unpack_from("<h", data, i * FRAME)[0] for i in range(before + 1, frames)]
    # a straight ramp down to silence, not a step
    assert fade == pytest.approx([level * (FADE - k) / FADE for k in range(FADE)], abs=1)


def _item(
    item_id: str, tag: int, seconds: int, size: BufferSize, end: float | None = None
) -> SimpleNamespace:
    """Build a queue item whose source fills a real AudioBuffer."""
    audio_buffer = AudioBuffer(TEST_PCM_FORMAT, buffer_size=size, mode=BufferMode.SEEKABLE)
    audio_buffer.fill(_pcm(tag, seconds), source_name=item_id)
    streamdetails = SimpleNamespace(
        audio_format=TEST_PCM_FORMAT,
        buffer=audio_buffer,
        fade_in=False,
        stream_error=False,
        uri=f"test://{item_id}",
        seek_position=0,
        seconds_streamed=0,
        duration=seconds,
        is_realtime=False,
        volume_normalization_mode=None,
        loudness=0.0,
        queue_id=None,
        provider="test",
        item_id=item_id,
        media_type=MediaType.TRACK,
    )
    return SimpleNamespace(
        queue_id="queue-1",
        queue_item_id=item_id,
        name=item_id,
        media_type=MediaType.TRACK,
        media_item=None,
        streamdetails=streamdetails,
        duration=seconds,
        available=True,
        extra_attributes={} if end is None else {"end_position": end},
    )


def _single_slot(mass: MagicMock) -> None:
    """Make every item's source a provider that streams one item at a time."""
    mass.get_provider.return_value = MagicMock(spec=MusicProvider, max_concurrent_streams=1)


async def _filled(*items: Any) -> None:
    """Let each source run ahead as a real one does (decoding is far faster than playback)."""
    for item in items:
        buf = item.streamdetails.buffer
        while not (buf.eof or buf.seconds_available >= buf.max_size_seconds):
            await asyncio.sleep(0.01)


async def _standard_build(**kw: Any) -> StandardCrossFade:
    """Build the real standard fade the mixer would fall back to."""
    fade = StandardCrossFade(logger=MagicMock(), crossfade_duration=STANDARD_CROSSFADE_DURATION)
    fade.build(len(kw["fade_out_data"]), kw["fade_in_bytes_len"], TEST_PCM_FORMAT)
    return fade


async def _run_flow(
    monkeypatch: pytest.MonkeyPatch,
    first: Any,
    second: Any,
    mode: CrossfadeMode = CrossfadeMode.STANDARD_CROSSFADE,
    *,
    single_slot: bool = False,
    mix: Callable[[StreamsAudio], MixStandIn] | None = None,
) -> tuple[bytes, StreamsAudio, MagicMock, list[Any]]:
    """Stream ``first`` then ``second`` as one flow stream; return what it emitted."""
    await _filled(first, second)
    audio, queue, mass = _flow_audio(
        monkeypatch, next_item=second, load_next=[second, QueueEmpty], crossfade_mode=mode
    )
    if single_slot:
        _single_slot(mass)
    monkeypatch.setattr(audio.smart_fades_mixer, "build", AsyncMock(side_effect=_standard_build))
    if mix is not None:
        monkeypatch.setattr(audio.smart_fades_mixer, "mix", mix(audio))
    monkeypatch.setattr(
        audio,
        "get_audio_buffer",
        AsyncMock(side_effect=lambda item, **_kw: item.streamdetails.buffer),
    )
    flow_log: list[Any] = []
    mass.player_queues.queue_data.return_value = SimpleNamespace(
        session_id="session-1", flow_mode_stream_log=flow_log
    )
    out = bytearray()
    async for chunk in audio.get_queue_flow_stream(
        cast("Any", queue), cast("Any", first), TEST_PCM_FORMAT, session_id="session-1"
    ):
        out.extend(chunk)
        await asyncio.sleep(0)
    return bytes(out), audio, mass, mass.player_queues.queue_data.return_value.flow_mode_stream_log


def _single_audio(
    monkeypatch: pytest.MonkeyPatch, nxt: Any, *, single_slot: bool = False
) -> tuple[StreamsAudio, MagicMock]:
    """Build a StreamsAudio wired for one item streamed with a fade into ``nxt``."""
    mass = MagicMock()
    queue = SimpleNamespace(queue_id="queue-1", display_name="Queue", index_in_buffer=0)
    mass.player_queues.get.return_value = queue
    mass.player_queues.get_next_item.return_value = nxt
    mass.player_queues.load_next_queue_item = AsyncMock(return_value=nxt)
    mass.player_queues.index_by_id.return_value = 1
    mass.player_queues.prepare_next_audio_buffer.return_value = None
    if single_slot:
        _single_slot(mass)
    audio = StreamsAudio(cast("Any", mass))
    audio.setup()
    audio.select_pcm_format = AsyncMock(return_value=TEST_PCM_FORMAT)  # type: ignore[method-assign]
    audio.crossfade_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]
    monkeypatch.setattr(audio.smart_fades_mixer, "build", AsyncMock(side_effect=_standard_build))

    async def _concat_mix(
        _fade: object, *, fade_in_part: Any, fade_out_part: bytes, **_kw: object
    ) -> AsyncGenerator[bytes]:
        yield fade_out_part
        async for chunk in fade_in_part:
            yield chunk

    monkeypatch.setattr(audio.smart_fades_mixer, "mix", _concat_mix)
    monkeypatch.setattr(
        audio,
        "get_audio_buffer",
        AsyncMock(side_effect=lambda item, **_kw: item.streamdetails.buffer),
    )
    return audio, mass


async def _run_single(
    audio: StreamsAudio, current: Any, mode: CrossfadeMode = CrossfadeMode.STANDARD_CROSSFADE
) -> bytes:
    """Stream one item in single-item mode; return what its stream emitted."""
    out = bytearray()
    async for chunk in audio.get_queue_item_stream_with_smartfade(
        cast("Any", SimpleNamespace(player_id="p1", name="P")),
        cast("Any", current),
        TEST_PCM_FORMAT,
        crossfade_mode=mode,
        standard_crossfade_duration=STANDARD_CROSSFADE_DURATION,
    ):
        out.extend(chunk)
        await asyncio.sleep(0)
    return bytes(out)


# ---- flow mode ----


async def test_flow_cut_item_ends_at_its_end_and_fades_into_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A's stream stops at its end, its tail is the window before it, B follows whole."""
    first = _item("a", 1, 300, BufferSize.BALANCED, end=70.5)
    second = _item("b", 2, 60, BufferSize.BALANCED)
    out, audio, mass, log = await _run_flow(monkeypatch, first, second)

    build = cast("AsyncMock", audio.smart_fades_mixer.build)
    tail = build.call_args.kwargs["fade_out_data"]
    assert build.call_args.kwargs["fade_out_end"] == 70.5
    assert (
        (STANDARD_CROSSFADE_DURATION - 1) * SECOND
        < len(tail)
        <= STANDARD_CROSSFADE_DURATION * SECOND
    )
    _assert_ends_at(tail, 70.5)
    # concat stand-in mixer: A up to its end, then all of B, nothing lost or doubled
    assert len(out) == int(70.5 * SECOND) + 60 * SECOND
    assert struct.unpack_from("<hh", out, int(70.5 * SR) * FRAME) == (2000, 0)
    # prepared 60 s before the end by the reader, then again at the boundary
    assert mass.player_queues.prepare_next_audio_buffer.call_count == 2
    assert log[0].seconds_streamed == pytest.approx(70.5)
    # the media length stays the item's own
    assert first.streamdetails.duration == 300
    assert first.duration == 300
    assert _transition_keys(first)["transition_mix_end"] == pytest.approx(70.5)
    assert audio.read_positions["a"] == float("inf")


async def test_flow_smart_mode_plans_on_the_cut_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smart Fades gets the end with a window of at most half of what A plays."""
    first = _item("a", 1, 300, BufferSize.BALANCED, end=70.5)
    second = _item("b", 2, 60, BufferSize.BALANCED)
    _out, audio, _mass, _log = await _run_flow(
        monkeypatch, first, second, mode=CrossfadeMode.SMART_CROSSFADE
    )
    build = cast("AsyncMock", audio.smart_fades_mixer.build)
    assert build.call_args.kwargs["mode"] == CrossfadeMode.SMART_CROSSFADE
    assert build.call_args.kwargs["fade_out_end"] == 70.5
    # the smart window is capped at half of what A plays (70.5 / 2 -> 35)
    assert 34 * SECOND < len(build.call_args.kwargs["fade_out_data"]) <= 35 * SECOND


@pytest.mark.parametrize("single_slot", [True, False])
async def test_flow_long_track_cut_releases_only_a_one_stream_source(
    monkeypatch: pytest.MonkeyPatch, single_slot: bool
) -> None:
    """A source still filling past the end is released only when its provider has one slot."""
    # a 60 s buffer never reaches EOF while the reader stops at 50 s of a 200 s track
    first = _item("a", 1, 200, BufferSize.MINIMAL, end=50.0)
    second = _item("b", 2, 30, BufferSize.BALANCED)
    a_buffer = first.streamdetails.buffer
    out, audio, _mass, _log = await _run_flow(monkeypatch, first, second, single_slot=single_slot)
    tail = cast("AsyncMock", audio.smart_fades_mixer.build).call_args.kwargs["fade_out_data"]
    _assert_ends_at(tail, 50.0)
    assert len(out) == 50 * SECOND + 30 * SECOND
    # any other provider keeps today's buffer lifetime
    assert a_buffer.is_buffering is not single_slot
    assert first.streamdetails.buffer is (None if single_slot else a_buffer)
    await a_buffer.clear()


async def test_flow_an_end_before_the_start_does_not_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seek past the end plays the rest of the item, and its duration is written back."""
    first = _item("a", 1, 40, BufferSize.BALANCED, end=10.0)
    first.streamdetails.seek_position = 20
    second = _item("b", 2, 30, BufferSize.BALANCED)
    out, audio, _mass, _log = await _run_flow(monkeypatch, first, second)
    assert len(out) == 20 * SECOND + 30 * SECOND
    assert cast("AsyncMock", audio.smart_fades_mixer.build).call_args.kwargs["fade_out_end"] is None
    assert first.extra_attributes["end_position"] == 10.0
    assert first.streamdetails.duration == 40  # written back as today


async def test_flow_without_an_end_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an end both tracks play whole and the mixer gets no end."""
    first = _item("a", 1, 40, BufferSize.BALANCED)
    second = _item("b", 2, 30, BufferSize.BALANCED)
    out, audio, _mass, _log = await _run_flow(monkeypatch, first, second)
    assert len(out) == 70 * SECOND
    assert cast("AsyncMock", audio.smart_fades_mixer.build).call_args.kwargs["fade_out_end"] is None


async def test_flow_an_end_without_a_fade_fades_out_instead_of_chopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no crossfade A still stops at its end on a short fade-out, then B starts."""
    first = _item("a", 1, 300, BufferSize.BALANCED, end=70.5)
    second = _item("b", 2, 30, BufferSize.BALANCED)
    out, _audio, _mass, _log = await _run_flow(
        monkeypatch, first, second, mode=CrossfadeMode.DISABLED
    )
    a_part = int(70.5 * SR) * FRAME
    assert len(out) == a_part + 30 * SECOND
    _assert_ends_at(out[:a_part], 70.5)
    assert struct.unpack_from("<hh", out, a_part) == (2000, 0)


async def test_flow_an_end_set_on_the_incoming_item_during_the_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """During the mix B's blended head is fixed: an end inside it is refused, a later one cuts B."""
    first = _item("a", 1, 40, BufferSize.BALANCED)
    second = _item("b", 2, 60, BufferSize.BALANCED)
    # the incoming head is then read by the boundary itself, not gathered ahead
    monkeypatch.setattr(_IncomingFadePrefetcher, "ensure_started", lambda *_a, **_kw: None)
    queues = MagicMock()
    queues.get_item.return_value = second

    def _mix(audio: StreamsAudio) -> MixStandIn:
        queues.mass.streams.audio = audio

        async def _concat_mix(
            _fade: object, *, fade_in_part: AsyncGenerator[bytes], fade_out_part: bytes, **_kw: Any
        ) -> AsyncGenerator[bytes]:
            yield fade_out_part
            async for chunk in fade_in_part:
                if "end_position" not in second.extra_attributes:
                    # mid-mix, with B's head partly read: the client sets B's end
                    with pytest.raises(ActionUnavailable):
                        await PlayerQueuesController.set_end_position(queues, "queue-1", "b", 5.0)
                    await PlayerQueuesController.set_end_position(queues, "queue-1", "b", 20.0)
                yield chunk

        return _concat_mix

    out, audio, _mass, _log = await _run_flow(monkeypatch, first, second, mix=_mix)
    assert second.extra_attributes["end_position"] == 20.0
    assert len(out) == 40 * SECOND + 20 * SECOND
    _assert_ends_at(out, 20.0)
    assert audio.read_positions["b"] == float("inf")


async def test_prefetch_gathers_half_of_the_incoming_part_and_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefetch reads at most half of what B plays, and that head can no longer change."""
    first = _item("a", 1, 120, BufferSize.BALANCED)
    second = _item("b", 2, 120, BufferSize.BALANCED, end=40.0)
    await _filled(second)
    audio, queue, _mass = _flow_audio(monkeypatch, next_item=second, load_next=[QueueEmpty])
    prefetcher = _IncomingFadePrefetcher(audio, TEST_PCM_FORMAT, "session-1")
    prefetcher.ensure_started(
        cast("Any", queue),
        cast("Any", first),
        CrossfadeMode.SMART_CROSSFADE,
        STANDARD_CROSSFADE_DURATION,
    )
    assert prefetcher._target == 20 * SECOND
    assert audio.read_positions["b"] == 20.0
    await prefetcher.close()
    await first.streamdetails.buffer.clear()


def test_incoming_window_respects_the_incoming_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fade-in window is at most half of what the incoming item plays."""
    audio, _queue, _mass = _flow_audio(monkeypatch, next_item=None, load_next=[QueueEmpty])
    details = SimpleNamespace(duration=300, seek_position=0, uri="x")
    details.buffer = MagicMock(
        has_error=False,
        eof=True,
        duration_available=300.0,
        ready=SimpleNamespace(is_set=lambda: True),
    )
    details.buffer.is_valid.return_value = True
    _mode, window = audio._select_buffered_crossfade(
        cast("Any", details),
        CrossfadeMode.SMART_CROSSFADE,
        STANDARD_CROSSFADE_DURATION,
        fade_out_seconds=45.0,
        media_end=40.0,
    )
    assert window == pytest.approx(20.0)


# ---- single-item mode ----


async def test_single_cut_item_holds_its_tail_to_the_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """A's stream carries A up to its end, its tail the window before it."""
    first = _item("a", 1, 300, BufferSize.BALANCED, end=70.5)
    second = _item("b", 2, 60, BufferSize.BALANCED)
    await _filled(first, second)
    audio, mass = _single_audio(monkeypatch, second)
    out = await _run_single(audio, first)

    build = cast("AsyncMock", audio.smart_fades_mixer.build)
    assert build.call_args.kwargs["fade_out_end"] == 70.5
    _assert_ends_at(build.call_args.kwargs["fade_out_data"], 70.5)
    # PRE + CF of the concat stand-in is A's tail
    assert len(out) == int(70.5 * SR) * FRAME
    _assert_ends_at(out, 70.5)
    assert mass.player_queues.prepare_next_audio_buffer.call_count == 2
    assert first.streamdetails.duration == 300
    assert _transition_keys(first)["transition_mix_end"] == pytest.approx(70.5)


async def test_single_minimal_buffer_releases_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-stream source still filling past the end is released at the cut, the fade intact."""
    first = _item("a", 1, 200, BufferSize.MINIMAL, end=50.0)
    second = _item("b", 2, 60, BufferSize.BALANCED)
    a_buffer = first.streamdetails.buffer
    await _filled(first, second)
    audio, _mass = _single_audio(monkeypatch, second, single_slot=True)
    out = await _run_single(audio, first)
    assert len(out) == 50 * SECOND
    tail = cast("AsyncMock", audio.smart_fades_mixer.build).call_args.kwargs["fade_out_data"]
    assert len(tail) >= (STANDARD_CROSSFADE_DURATION - 1) * SECOND
    assert not a_buffer.is_buffering
    assert first.streamdetails.buffer is None


async def test_single_boundary_holds_the_incoming_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the fade into B is fixed, B's head it blends in counts as read."""
    first = _item("a", 1, 40, BufferSize.BALANCED)
    second = _item("b", 2, 60, BufferSize.BALANCED)
    await _filled(first, second)
    audio, _mass = _single_audio(monkeypatch, second)
    marks: list[float | None] = []

    async def _build(**kw: Any) -> StandardCrossFade:
        marks.append(audio.read_positions.get("b"))
        return await _standard_build(**kw)

    monkeypatch.setattr(audio.smart_fades_mixer, "build", AsyncMock(side_effect=_build))
    await _run_single(audio, first)
    assert marks == [pytest.approx(STANDARD_CROSSFADE_DURATION)]


async def test_reader_follows_an_end_set_while_it_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """An end set mid-read is where the stream stops."""
    item = _item("a", 1, 120, BufferSize.BALANCED)
    await _filled(item)
    audio, _mass = _single_audio(monkeypatch, None)
    out = bytearray()
    async for chunk in audio.get_queue_item_stream(cast("Any", item), TEST_PCM_FORMAT):
        out.extend(chunk)
        if len(out) == 20 * SECOND:
            assert audio.read_positions["a"] == pytest.approx(20.0)
            item.extra_attributes["end_position"] = 33.25
    assert len(out) == int(33.25 * SR) * FRAME
    _assert_ends_at(out, 33.25)
    assert audio.read_positions["a"] == float("inf")


async def test_an_abandoned_read_keeps_its_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader that stops early leaves how far it read."""
    item = _item("a", 1, 120, BufferSize.BALANCED)
    await _filled(item)
    audio, _mass = _single_audio(monkeypatch, None)
    stream = audio.get_queue_item_stream(cast("Any", item), TEST_PCM_FORMAT, seek_position=10)
    read = 0
    async for chunk in stream:
        read += len(chunk)
        if read >= 5 * SECOND:
            break
    await stream.aclose()
    assert audio.read_positions["a"] == pytest.approx(15.0)
    await item.streamdetails.buffer.clear()
