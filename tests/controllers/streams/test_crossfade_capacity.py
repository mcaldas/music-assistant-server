"""Tests for crossfade degradation when incoming source capacity is unavailable."""

from __future__ import annotations

import asyncio
import logging
import re
import struct
import time
from collections.abc import AsyncGenerator, Iterable
from itertools import pairwise
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from music_assistant_models.enums import ContentType, CrossfadeMode, MediaType, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.streams.audio import (
    MIN_CROSSFADE_DURATION,
    CrossfadeHandover,
    StreamsAudio,
)
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.controllers.streams.constants import BufferSize
from music_assistant.controllers.streams.smart_fades.fades import StandardCrossFade
from music_assistant.controllers.streams.smart_fades.helpers import SMART_CROSSFADE_DURATION
from music_assistant.controllers.streams.smart_fades.planner.requested import TransitionRequest


def _audio(pcm_format: AudioFormat, seconds: float) -> bytes:
    """
    Return PCM that reads as audio rather than as an item's trailing silence.

    The holdback measures the silent run a buffer ends with, so a fixture filled
    with zeroes would stand in for a track that has already finished.
    """
    frame = struct.pack("<2h", 9000, -9000)
    size = int(pcm_format.pcm_sample_size * seconds)
    return (frame * (size // len(frame) + 1))[:size]


def _streamdetails(audio_buffer: AudioBuffer | None) -> StreamDetails:
    """Build incoming track details with an optional prepared buffer."""
    streamdetails = StreamDetails(
        provider="test--1",
        item_id="track-1",
        audio_format=AudioFormat(content_type=ContentType.FLAC),
        media_type=MediaType.TRACK,
        stream_type=StreamType.HTTP,
        path="http://test.invalid/track.flac",
        duration=180,
    )
    streamdetails.buffer = audio_buffer
    return streamdetails


def _buffer(duration_available: float, ready: bool, eof: bool = False) -> AudioBuffer:
    """Build a valid buffer with the requested resident duration."""
    audio_buffer = MagicMock(spec=AudioBuffer)
    audio_buffer.has_error = False
    audio_buffer.cancelled = False
    audio_buffer.is_valid.return_value = True
    audio_buffer.duration_available = duration_available
    audio_buffer.eof = eof
    audio_buffer.ready = asyncio.Event()
    if ready:
        audio_buffer.ready.set()
    return audio_buffer


def _delivered_buffer(seconds: float = 16.0) -> SimpleNamespace:
    """
    Build the outgoing track's buffer, with its source done delivering.

    :param seconds: The audio it holds, from the track's beginning.
    """
    return SimpleNamespace(
        eof=True,
        cancelled=False,
        has_error=False,
        max_size_seconds=300,
        first_buffered_chunk=0,
        duration_available=seconds,
    )


def test_ready_incoming_buffer_keeps_smart_crossfade() -> None:
    """A tail that carries the full smart window keeps the requested Smart Fade."""
    audio = StreamsAudio(MagicMock())

    mode, duration = audio._select_buffered_crossfade(
        _streamdetails(_buffer(SMART_CROSSFADE_DURATION, ready=True)),
        CrossfadeMode.SMART_CROSSFADE,
        standard_crossfade_duration=8,
        fade_out_seconds=SMART_CROSSFADE_DURATION,
    )

    assert mode == CrossfadeMode.SMART_CROSSFADE
    assert duration == SMART_CROSSFADE_DURATION


def test_a_partly_resident_incoming_buffer_keeps_the_full_window() -> None:
    """The incoming side streams in while the blend plays, so residency does not cap it."""
    audio = StreamsAudio(MagicMock())

    mode, duration = audio._select_buffered_crossfade(
        _streamdetails(_buffer(2, ready=True)),
        CrossfadeMode.SMART_CROSSFADE,
        standard_crossfade_duration=8,
        fade_out_seconds=SMART_CROSSFADE_DURATION,
    )

    assert mode == CrossfadeMode.SMART_CROSSFADE
    assert duration == SMART_CROSSFADE_DURATION


@pytest.mark.parametrize(
    "audio_buffer",
    [
        None,
        _buffer(30, ready=False),
    ],
)
def test_unprepared_incoming_buffer_disables_crossfade(
    audio_buffer: AudioBuffer | None,
) -> None:
    """An incoming source that is not delivering yet falls back to playback without a fade."""
    audio = StreamsAudio(MagicMock())

    mode, duration = audio._select_buffered_crossfade(
        _streamdetails(audio_buffer),
        CrossfadeMode.SMART_CROSSFADE,
        standard_crossfade_duration=8,
        fade_out_seconds=SMART_CROSSFADE_DURATION,
    )

    assert mode == CrossfadeMode.DISABLED
    assert duration == 0


def test_a_tail_below_the_minimum_disables_the_crossfade() -> None:
    """Too short a held tail is played out instead of blended."""
    audio = StreamsAudio(MagicMock())

    mode, duration = audio._select_buffered_crossfade(
        _streamdetails(_buffer(30, ready=True)),
        CrossfadeMode.SMART_CROSSFADE,
        standard_crossfade_duration=8,
        fade_out_seconds=MIN_CROSSFADE_DURATION - 0.5,
    )

    assert mode == CrossfadeMode.DISABLED
    assert duration == 0


async def test_unprepared_next_track_flushes_outgoing_tail_without_opening_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing incoming PCM emits the complete outgoing track without a blocking fade fetch."""
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_S16LE,
        sample_rate=8000,
        bit_depth=16,
        channels=2,
    )
    current_details = SimpleNamespace(
        duration=16,
        seek_position=0,
        seconds_streamed=0,
        uri="test://current",
        buffer=_delivered_buffer(),
        is_realtime=False,
    )
    next_details = SimpleNamespace(
        audio_format=pcm_format,
        buffer=None,
        duration=16,
        seek_position=0,
        uri="test://next",
        is_realtime=False,
    )
    current_item = SimpleNamespace(
        queue_id="queue-1",
        queue_item_id="current",
        name="Current",
        streamdetails=current_details,
        extra_attributes={},
    )
    next_item = SimpleNamespace(
        queue_id="queue-1",
        queue_item_id="next",
        name="Next",
        streamdetails=next_details,
        extra_attributes={},
        available=True,
    )
    queue = SimpleNamespace(
        queue_id="queue-1",
        display_name="Queue",
        index_in_buffer=0,
    )
    player = SimpleNamespace(player_id="player-1", name="Player")
    mass = MagicMock()
    mass.player_queues.get.return_value = queue
    mass.player_queues.load_next_queue_item = AsyncMock(return_value=next_item)
    mass.player_queues.index_by_id.return_value = 1
    # nothing was left to prepare, so the boundary has nothing to wait for
    mass.player_queues.prepare_next_audio_buffer.return_value = None
    audio = StreamsAudio(cast("Any", mass))
    audio.setup()
    audio.select_pcm_format = AsyncMock(return_value=pcm_format)  # type: ignore[method-assign]
    audio.crossfade_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]
    build = AsyncMock()
    monkeypatch.setattr(audio.smart_fades_mixer, "build", build)

    async def _current_stream(
        queue_item: object,
        *_args: object,
        **_kwargs: object,
    ) -> AsyncGenerator[bytes]:
        if queue_item is not current_item:
            pytest.fail("The incoming source was opened during crossfade fallback")
        yield _audio(pcm_format, 8)
        yield _audio(pcm_format, 8)

    monkeypatch.setattr(audio, "get_queue_item_stream", _current_stream)
    stream = audio.get_queue_item_stream_with_smartfade(
        cast("Any", player),
        cast("Any", current_item),
        pcm_format,
        crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE,
        standard_crossfade_duration=8,
    )

    started = time.monotonic()
    output = b"".join([chunk async for chunk in stream])

    assert time.monotonic() - started < 0.5
    assert len(output) == pcm_format.pcm_sample_size * 16
    assert next_item.available
    build.assert_not_awaited()
    # the missing incoming audio is (re)requested relative to the outgoing item
    mass.player_queues.prepare_next_audio_buffer.assert_called_once_with("queue-1", "current")
    # the next item was locked and signalled, then the boundary reported as no fade
    assert queue.index_in_buffer == 1
    assert mass.player_queues.signal_update.call_args_list == [call("queue-1"), call("queue-1")]
    assert current_item.extra_attributes["transition_mode"] == "disabled"
    assert current_item.extra_attributes["transition_next_item_id"] == "next"


@pytest.mark.parametrize("playback_speed", [0.5, 2.0])
async def test_crossfade_reads_its_window_past_the_resident_buffer(
    monkeypatch: pytest.MonkeyPatch,
    playback_speed: float,
) -> None:
    """The blend consumes its whole window as it arrives, and hands on the media time used."""
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_S16LE,
        sample_rate=8000,
        bit_depth=16,
        channels=2,
    )
    resident_media_duration = 2.0
    current_details = SimpleNamespace(
        duration=16,
        seek_position=0,
        seconds_streamed=0,
        uri="test://current",
        buffer=_delivered_buffer(),
        is_realtime=False,
    )
    next_details = SimpleNamespace(
        audio_format=pcm_format,
        buffer=_buffer(resident_media_duration, ready=True),
        duration=16,
        seek_position=0,
        uri="test://next",
        volume_normalization_mode=None,
        is_realtime=False,
    )
    current_item = SimpleNamespace(
        queue_id="queue-1",
        queue_item_id="current",
        name="Current",
        streamdetails=current_details,
        extra_attributes={},
    )
    next_item = SimpleNamespace(
        queue_id="queue-1",
        queue_item_id="next",
        name="Next",
        streamdetails=next_details,
        extra_attributes={"playback_speed": playback_speed},
        available=True,
    )
    queue = SimpleNamespace(
        queue_id="queue-1",
        display_name="Queue",
        index_in_buffer=0,
    )
    player = SimpleNamespace(player_id="player-1", name="Player")
    mass = MagicMock()
    mass.player_queues.get.return_value = queue
    mass.player_queues.load_next_queue_item = AsyncMock(return_value=next_item)
    mass.player_queues.index_by_id.return_value = 1
    audio = StreamsAudio(cast("Any", mass))
    audio.setup()
    audio.select_pcm_format = AsyncMock(return_value=pcm_format)  # type: ignore[method-assign]
    audio.crossfade_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]
    crossfade_duration = 8
    smart_fade = SimpleNamespace(
        timing_info=SimpleNamespace(
            pre_crossfade_duration=0,
            post_crossfade_duration=0,
            crossfade_duration=crossfade_duration,
            fadein_trimmed_duration=0,
        )
    )
    build = AsyncMock(return_value=smart_fade)
    monkeypatch.setattr(audio.smart_fades_mixer, "build", build)
    # a client asked for a cut into the next item before the boundary was planned
    current_item.extra_attributes.update(TransitionRequest("cut", "next").to_attributes())

    async def _mix(
        _smart_fade: object,
        *,
        fade_in_part: AsyncGenerator[bytes],
        **_kwargs: object,
    ) -> AsyncGenerator[bytes]:
        async for chunk in fade_in_part:
            yield chunk

    monkeypatch.setattr(audio.smart_fades_mixer, "mix", _mix)
    incoming_seconds_read = 0

    async def _item_stream(
        queue_item: object,
        *_args: object,
        **_kwargs: object,
    ) -> AsyncGenerator[bytes]:
        nonlocal incoming_seconds_read
        if queue_item is current_item:
            yield _audio(pcm_format, 8)
            yield _audio(pcm_format, 8)
            return
        # the incoming source keeps delivering beyond what was resident at the boundary
        for _ in range(20):
            incoming_seconds_read += 1
            yield _audio(pcm_format, 1)

    monkeypatch.setattr(audio, "get_queue_item_stream", MagicMock(side_effect=_item_stream))
    stream = audio.get_queue_item_stream_with_smartfade(
        cast("Any", player),
        cast("Any", current_item),
        pcm_format,
        crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE,
        standard_crossfade_duration=crossfade_duration,
    )

    _ = [chunk async for chunk in stream]

    assert incoming_seconds_read > resident_media_duration
    crossfade_data = audio._crossfade_handover["queue-1"]
    assert crossfade_data.queue_item_id == "next"
    # the window is stream time, so fast playback reaches the incoming track's
    # half-duration cap sooner: at 2x an 8s overlap would eat this whole track
    expected_window = min(crossfade_duration, next_details.duration / playback_speed / 2)
    # the next track resumes at the media time the blend already played
    assert crossfade_data.fade_in_media_duration == pytest.approx(expected_window * playback_speed)
    assert crossfade_data.fade_in_media_duration <= next_details.duration / 2
    # the outgoing item's own request names no point to keep; the fade says where the next
    # item's own request resumes, so its reads leave that audio in the buffer
    assert [
        item_call.kwargs.get("keep_from")
        for item_call in cast("MagicMock", audio.get_queue_item_stream).call_args_list
    ] == [None, pytest.approx(expected_window * playback_speed)]
    # the request reached the mixer and was used up by the boundary's report; the fade
    # built here is no smart one, so nothing planned it
    assert build.await_args is not None
    assert build.await_args.kwargs["request"] == TransitionRequest("cut", "next")
    assert TransitionRequest.read(current_item.extra_attributes) is None
    assert current_item.extra_attributes["transition_request"] == "ignored"
    # the boundary was published with the mode that was built
    assert current_item.extra_attributes["transition_mode"] == "standard_crossfade"
    assert current_item.extra_attributes["transition_next_item_id"] == "next"
    assert current_item.extra_attributes["transition_overlap"] == crossfade_duration


_SINGLE_PCM = AudioFormat(
    content_type=ContentType.PCM_S16LE,
    sample_rate=8000,
    bit_depth=16,
    channels=2,
)


def _single_boundary(
    chunk_seconds: list[float], duration: int = 16
) -> tuple[StreamsAudio, SimpleNamespace]:
    """
    Wire a single-item stream of an outgoing track into a next one of 16 s.

    :param chunk_seconds: The outgoing track's source chunks, in seconds; none when the
        test streams its own (``_play_out``).
    :param duration: The outgoing track's length in seconds.
    """
    current_item = SimpleNamespace(
        queue_id="queue-1",
        queue_item_id="current",
        name="Current",
        streamdetails=SimpleNamespace(
            duration=duration,
            seek_position=0,
            seconds_streamed=0,
            uri="test://current",
            # the source delivered what its reader will get
            buffer=_delivered_buffer(sum(chunk_seconds) or duration),
            is_realtime=False,
            allow_seek=True,
        ),
        extra_attributes={},
    )
    next_item = SimpleNamespace(
        queue_id="queue-1",
        queue_item_id="next",
        name="Next",
        streamdetails=SimpleNamespace(
            audio_format=_SINGLE_PCM,
            buffer=_buffer(16, ready=True),
            duration=16,
            seek_position=0,
            uri="test://next",
            volume_normalization_mode=None,
            is_realtime=False,
        ),
        extra_attributes={},
        available=True,
    )
    mass = MagicMock()
    mass.player_queues.get.return_value = SimpleNamespace(
        queue_id="queue-1", display_name="Queue", index_in_buffer=0
    )
    mass.player_queues.load_next_queue_item = AsyncMock(return_value=next_item)
    mass.player_queues.index_by_id.return_value = 1
    audio = StreamsAudio(cast("Any", mass))
    audio.setup()
    audio.select_pcm_format = AsyncMock(return_value=_SINGLE_PCM)  # type: ignore[method-assign]
    audio.crossfade_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]

    async def _item_stream(
        queue_item: object, *_args: object, **_kwargs: object
    ) -> AsyncGenerator[bytes]:
        for seconds in chunk_seconds if queue_item is current_item else []:
            yield _audio(_SINGLE_PCM, seconds)

    audio.get_queue_item_stream = _item_stream  # type: ignore[method-assign]
    return audio, current_item


def _plain_mix(audio: StreamsAudio) -> AsyncMock:
    """
    Stand in for the mixer: the mix is the outgoing tail as it is, without the next item.

    :param audio: The StreamsAudio whose mixer is replaced.
    :return: The mock that was handed the tail (``fade_out_data``).
    """
    build = AsyncMock(
        return_value=SimpleNamespace(
            timing_info=SimpleNamespace(
                pre_crossfade_duration=0,
                post_crossfade_duration=0,
                crossfade_duration=8,
                fadein_trimmed_duration=0,
            )
        )
    )

    async def _mix(*_args: object, **kwargs: Any) -> AsyncGenerator[bytes]:
        yield kwargs["fade_out_part"]

    audio.smart_fades_mixer.build = build  # type: ignore[method-assign]
    audio.smart_fades_mixer.mix = _mix  # type: ignore[method-assign]
    return build


def _single_stream(audio: StreamsAudio, current_item: SimpleNamespace) -> AsyncGenerator[bytes]:
    """Open the outgoing track's single-item stream with an 8 s standard crossfade."""
    return audio.get_queue_item_stream_with_smartfade(
        cast("Any", SimpleNamespace(player_id="player-1", name="Player")),
        cast("Any", current_item),
        _SINGLE_PCM,
        crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE,
        standard_crossfade_duration=8,
    )


async def test_a_failed_mix_is_reported_as_no_fade() -> None:
    """A single-item mix that fails before any audio plays the tail as is and reports it."""
    audio, current_item = _single_boundary([1.0] * 16)
    audio.smart_fades_mixer.build = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            timing_info=SimpleNamespace(
                pre_crossfade_duration=0,
                post_crossfade_duration=0,
                crossfade_duration=8,
                fadein_trimmed_duration=0,
            )
        )
    )

    async def _failing_mix(*_args: object, **_kwargs: object) -> AsyncGenerator[bytes]:
        msg = "mixer failed"
        raise RuntimeError(msg)
        yield b""  # type: ignore[unreachable]

    audio.smart_fades_mixer.mix = _failing_mix  # type: ignore[method-assign]

    emitted = sum([len(chunk) async for chunk in _single_stream(audio, current_item)])

    # the whole outgoing track still plays, and nobody is told a fade is coming
    assert emitted == _SINGLE_PCM.pcm_sample_size * 16
    assert current_item.extra_attributes["transition_mode"] == "disabled"
    assert current_item.extra_attributes["transition_next_item_id"] == "next"
    assert "transition_overlap" not in current_item.extra_attributes


async def test_single_item_places_the_reported_fade_on_the_song() -> None:
    """The reported mix ends where the song's audio ends, not at its whole-second duration."""
    # 15.5 s of audio, while the item's duration says 16
    audio, current_item = _single_boundary([1.0] * 15 + [0.5])

    async def _build(**kwargs: Any) -> StandardCrossFade:
        fade = StandardCrossFade(logger=MagicMock(), crossfade_duration=8)
        fade.build(len(kwargs["fade_out_data"]), kwargs["fade_in_bytes_len"], _SINGLE_PCM)
        return fade

    async def _mix(*_args: object, **kwargs: Any) -> AsyncGenerator[bytes]:
        yield kwargs["fade_out_part"]

    audio.smart_fades_mixer.build = AsyncMock(side_effect=_build)  # type: ignore[method-assign]
    audio.smart_fades_mixer.mix = _mix  # type: ignore[method-assign]

    _ = [chunk async for chunk in _single_stream(audio, current_item)]

    attributes = current_item.extra_attributes
    assert attributes["transition_mode"] == "standard_crossfade"
    assert attributes["transition_mix_end"] == pytest.approx(15.5, abs=0.01)
    assert attributes["transition_mix_start"] == pytest.approx(
        15.5 - attributes["transition_overlap"], abs=0.01
    )


async def test_the_time_without_output_at_a_seam_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stream says at debug level how long it sent nothing behind a fade and for its tail."""
    audio, current_item = _single_boundary([1.0] * 16)
    _plain_mix(audio)

    async def _rest_of_the_mix() -> AsyncGenerator[bytes]:
        yield _audio(_SINGLE_PCM, 1)

    # the item is faded into: its request first plays out the mix the item before it left
    audio._crossfade_handover["queue-1"] = CrossfadeHandover(
        stream=_rest_of_the_mix(),
        fade_in_media_duration=0.0,
        pcm_format=_SINGLE_PCM,
        queue_item_id="current",
    )

    with caplog.at_level(logging.DEBUG, logger="music_assistant.streams.audio"):
        _ = [chunk async for chunk in _single_stream(audio, current_item)]

    text = "\n".join(record.getMessage() for record in caplog.records)
    seam = r"First audio of Current after its fade came \d+\.\d\ds after the fade ended"
    assert len(re.findall(seam, text)) == 1
    hold = r"Held back 8\.0s of Current for its fade: \d+\.\d\ds without output"
    assert len(re.findall(hold, text)) == 1


async def test_a_fade_into_a_track_longer_than_its_buffer_leaves_its_resume_point() -> None:
    """The fade's reads free a full buffer up to where the item's own request resumes."""
    # a 7.5 s tail, so the fade ends inside the next item's eighth second
    audio, current_item = _single_boundary([4.0, 3.5])

    async def _source() -> AsyncGenerator[bytes]:
        for _ in range(100):
            yield _audio(_SINGLE_PCM, 1)

    # the next item is longer than its buffer: the source is parked on a full one
    incoming = AudioBuffer(_SINGLE_PCM, buffer_size=BufferSize.MINIMAL)
    incoming.fill(_source(), source_name="next")

    async def _refilled() -> None:
        async with asyncio.timeout(5):
            while incoming.size_seconds < incoming.max_size_seconds:
                await asyncio.sleep(0)

    await _refilled()
    next_details = SimpleNamespace(
        audio_format=_SINGLE_PCM,
        buffer=incoming,
        fade_in=False,
        stream_error=False,
        uri="test://next",
        seek_position=0,
        seconds_streamed=0,
        duration=200,
        is_realtime=False,
        # nothing to filter, so the buffer is read as it is
        volume_normalization_mode=None,
        loudness=0.0,
        queue_id=None,
        provider="test",
        item_id="next",
        media_type=MediaType.TRACK,
    )
    next_item = SimpleNamespace(
        queue_id="queue-1",
        queue_item_id="next",
        name="Next",
        media_type=MediaType.TRACK,
        media_item=None,
        streamdetails=next_details,
        duration=200,
        available=True,
        extra_attributes={},
    )
    mass = cast("MagicMock", audio.mass)
    mass.player_queues.load_next_queue_item = AsyncMock(return_value=next_item)
    outgoing_stream = audio.get_queue_item_stream

    def _item_stream(queue_item: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[bytes]:
        if queue_item is current_item:
            return outgoing_stream(queue_item, *args, **kwargs)
        # the next item is read by the real reader, from its real buffer
        return StreamsAudio.get_queue_item_stream(audio, queue_item, *args, **kwargs)

    audio.get_queue_item_stream = _item_stream  # type: ignore[method-assign]
    audio.smart_fades_mixer.build = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            timing_info=SimpleNamespace(
                pre_crossfade_duration=0,
                post_crossfade_duration=0,
                crossfade_duration=7.5,
                fadein_trimmed_duration=0,
            )
        )
    )

    async def _mix(
        _smart_fade: object, *, fade_in_part: AsyncGenerator[bytes], **_kwargs: object
    ) -> AsyncGenerator[bytes]:
        async for chunk in fade_in_part:
            # a mix is taken at the player's pace: the source fills the space a read made
            await _refilled()
            yield chunk

    audio.smart_fades_mixer.mix = _mix  # type: ignore[method-assign]

    _ = [chunk async for chunk in _single_stream(audio, current_item)]

    handover = audio._crossfade_handover["queue-1"]
    assert handover.fade_in_media_duration == 7.5
    # the fade made room up to the second the next item's own request resumes in, and
    # left that one: the request finds its audio and the source is not fetched again
    assert incoming.first_buffered_chunk == 7
    assert incoming.is_valid(7500)
    assert next_details.buffer is incoming
    assert not incoming.cancelled
    await handover.close()
    await incoming.clear()


def test_crossfade_targets_names_the_pending_and_the_published_item() -> None:
    """An item is named from the start of the fade into it until its own request takes over."""
    audio = StreamsAudio(MagicMock())
    audio._crossfade_pending["queue-1"] = ("pending", asyncio.Event())
    audio._crossfade_handover["queue-1"] = CrossfadeHandover(
        stream=None,
        fade_in_media_duration=0.0,
        pcm_format=_SINGLE_PCM,
        queue_item_id="published",
    )

    assert audio.crossfade_targets("queue-1") == {"pending", "published"}
    assert audio.crossfade_targets("queue-2") == set()

    audio.clear_crossfade_handover("queue-1")

    assert audio.crossfade_targets("queue-1") == set()


def _ramp(seconds: int, chunk_seconds: float = 1.0) -> list[bytes]:
    """
    Return a track's audio in chunks that are told apart by their bytes.

    :param seconds: The length of the audio.
    :param chunk_seconds: The length of each chunk.
    """
    size = int(_SINGLE_PCM.pcm_sample_size * chunk_seconds)
    return [bytes([index % 250 + 1]) * size for index in range(round(seconds / chunk_seconds))]


async def _play_out(
    audio: StreamsAudio, current_item: SimpleNamespace, source: Iterable[bytes]
) -> tuple[list[int], bytes, bytes]:
    """
    Stream the outgoing track from its source chunks, up to and with a plain mix.

    :param audio: The StreamsAudio from ``_single_boundary``.
    :param current_item: The outgoing track.
    :param source: The chunks its reader delivers; a generator can act between them.
    :return: For each slice handed on before the mix how many seconds were read by then,
        those slices joined, and the tail the mix was handed.
    """
    seconds_read = 0.0

    async def _item_stream(
        queue_item: object, *_args: object, **_kwargs: object
    ) -> AsyncGenerator[bytes]:
        nonlocal seconds_read
        for chunk in source if queue_item is current_item else []:
            seconds_read += len(chunk) / _SINGLE_PCM.pcm_sample_size
            yield chunk

    audio.get_queue_item_stream = _item_stream  # type: ignore[method-assign]
    build = _plain_mix(audio)
    read_at: list[float] = []
    pieces: list[bytes] = []
    async for piece in _single_stream(audio, current_item):
        read_at.append(seconds_read)
        pieces.append(piece)
    assert build.await_args is not None
    tail = build.await_args.kwargs["fade_out_data"]
    # the plain mix is the tail, handed on in one piece after the track's own slices
    assert pieces[-1] == tail
    return read_at[:-1], b"".join(pieces[:-1]), tail


def _assert_flows(read_at: list[float], chunk_seconds: float = 1.0) -> None:
    """
    Assert that no more than two seconds were read for each second that was handed on.

    :param read_at: The seconds read when each one-second slice was handed on.
    :param chunk_seconds: The length of the source's chunks: a slice leaves with the
        chunk that completes it.
    """
    assert read_at
    read_for_a_slice = max(later - earlier for earlier, later in pairwise([0.0, *read_at]))
    assert read_for_a_slice < 2 + chunk_seconds


@pytest.mark.parametrize(
    ("seconds", "starts_at", "attributes", "source_seconds", "chunk_seconds", "tail_seconds"),
    [
        # a long track, faded out over the whole window
        (200, 0, {}, 200, 1.0, 8),
        # the same in pieces that are no whole seconds, as a filter hands them on
        (200, 0, {}, 200, 0.4, 8),
        # less than twice the window is left of it when its stream starts
        (16, 4, {}, 12, 1.0, 8),
        # an end position 8 s after its stream starts: half of the 12 s it plays is held
        (200, 4, {"end_position": 12.0}, 8, 1.0, 6),
        # a part, read from its start: it holds half of the 16 s it plays
        (200, 30, {"start_position": 30.0, "end_position": 46.0}, 16, 1.0, 8),
        # played at double speed: 24 s of the track are 12 s of the stream
        (64, 40, {"playback_speed": 2.0}, 12, 1.0, 8),
    ],
    ids=["long", "pieces", "short", "end", "part", "fast"],
)
async def test_a_complete_item_keeps_audio_flowing_while_its_tail_is_collected(
    seconds: int,
    starts_at: int,
    attributes: dict[str, float],
    source_seconds: int,
    chunk_seconds: float,
    tail_seconds: int,
) -> None:
    """A complete source no longer has its whole window read before its audio goes on."""
    audio, current_item = _single_boundary([], duration=seconds)
    current_item.streamdetails.seek_position = starts_at
    current_item.extra_attributes.update(attributes)
    source = _ramp(source_seconds, chunk_seconds)

    read_at, handed_on, tail = await _play_out(audio, current_item, source)

    # audio leaves from the first seconds on, a second for every two that are read
    _assert_flows(read_at, chunk_seconds)
    # and at the end the mix gets the tail it got before, of the same audio in the same order
    assert len(tail) == tail_seconds * _SINGLE_PCM.pcm_sample_size
    assert handed_on + tail == b"".join(source)


@pytest.mark.parametrize("part", [False, True], ids=["track", "part"])
async def test_an_item_with_no_more_than_its_window_left_is_held_as_before(part: bool) -> None:
    """What is left of an item when its stream starts is no more than its tail: all is held."""
    if part:
        # a part of 16 s that was faded into over its first 8 s
        audio, current_item = _single_boundary([], duration=200)
        current_item.streamdetails.seek_position = 30.0
        current_item.extra_attributes.update(start_position=30.0, end_position=46.0)
        audio._crossfade_handover["queue-1"] = CrossfadeHandover(
            stream=None,
            fade_in_media_duration=38.0,
            pcm_format=_SINGLE_PCM,
            queue_item_id="current",
            fade_in_start=30.0,
        )
    else:
        audio, current_item = _single_boundary([])
        current_item.streamdetails.seek_position = 8
    source = _ramp(8)

    read_at, handed_on, tail = await _play_out(audio, current_item, source)

    assert read_at == []
    assert handed_on == b""
    assert tail == b"".join(source)


async def test_a_source_that_completes_while_its_item_plays_does_not_stop_the_stream() -> None:
    """The tail of a source that completes mid-stream is collected while audio keeps leaving."""
    audio, current_item = _single_boundary([], duration=200)
    audio_buffer = current_item.streamdetails.buffer
    audio_buffer.eof = False
    ramp = _ramp(200)

    def _source() -> Iterable[bytes]:
        for second, chunk in enumerate(ramp):
            if second == 30:
                audio_buffer.eof = True
            yield chunk

    read_at, handed_on, tail = await _play_out(audio, current_item, _source())

    # while the source was filling every second went straight on
    assert read_at[:30] == list(range(1, 31))
    # and from the moment it was complete never more than two were read for one handed on
    _assert_flows(read_at)
    assert len(tail) == 8 * _SINGLE_PCM.pcm_sample_size
    assert handed_on + tail == b"".join(ramp)


async def test_an_end_moved_close_while_the_tail_builds_up_holds_what_is_in_hand() -> None:
    """Once no more than the window is left nothing more leaves, whatever was built up."""
    audio, current_item = _single_boundary([], duration=200)
    audio_buffer = current_item.streamdetails.buffer
    audio_buffer.eof = False
    ramp = _ramp(200)

    def _source() -> Iterable[bytes]:
        for second, chunk in enumerate(ramp):
            if second == 30:
                audio_buffer.eof = True
            if second == 40:
                # 40 s are read; the client moves the end to 2 s from there
                current_item.extra_attributes["end_position"] = 42.0
            if second >= current_item.extra_attributes.get("end_position", 200):
                return
            yield chunk

    read_at, handed_on, tail = await _play_out(audio, current_item, _source())

    # 10 s into building up the 8 s window: 5 s are held, 2 s are still to come
    assert read_at[-1] == 40
    assert len(tail) == 7 * _SINGLE_PCM.pcm_sample_size
    assert handed_on + tail == b"".join(ramp[:42])


async def test_the_logged_time_without_output_is_the_last_stretch_the_tail_is_read_in(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    The line times the reading that sends nothing: from the last slice handed on to the tail.

    While the window builds up a slice leaves for every two seconds read. Once no more than
    the window is left of the item all of it is held, and that is the stretch a player can
    give up in.
    """
    # 12 s are left of it: slices leave up to the 8th second read, the last 4 are all tail
    audio, current_item = _single_boundary([], duration=16)
    current_item.streamdetails.seek_position = 4

    def _source() -> Iterable[bytes]:
        for second, chunk in enumerate(_ramp(12)):
            # a slow filter: 0.4 s for the seconds read while audio leaves, 0.4 s for the rest
            time.sleep(0.05 if second < 8 else 0.1)
            yield chunk

    with caplog.at_level(logging.DEBUG, logger="music_assistant.streams.audio"):
        read_at, _handed_on, _tail = await _play_out(audio, current_item, _source())

    assert read_at == [2, 4, 6, 8]
    text = "\n".join(record.getMessage() for record in caplog.records)
    held = re.findall(r"Held back 8\.0s of Current for its fade: (\d+\.\d\d)s without output", text)
    assert len(held) == 1
    assert 0.39 <= float(held[0]) < 0.6
