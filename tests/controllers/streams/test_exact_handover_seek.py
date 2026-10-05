"""Tests for an exact item-stream seek: a crossfade handover resumes the next item on its exact sample."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from music_assistant_models.enums import ContentType, MediaType, VolumeNormalizationMode
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

import music_assistant.controllers.streams.audio as audio_mod
from music_assistant.controllers.streams.audio import StreamsAudio

PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE,
    codec_type=ContentType.PCM_S16LE,
    sample_rate=44100,
    bit_depth=16,
    channels=1,
)


class _CountingBuffer:
    """AudioBuffer test double: each sample holds its own index; seeks as the real buffer does."""

    has_error = False
    pcm_format = PCM_FORMAT

    @classmethod
    async def get_buffer(cls, **_kwargs: Any) -> _CountingBuffer:
        return cls()

    async def get_stream(
        self,
        output_format: AudioFormat,
        seek_position_ms: int = 0,
        filter_params: list[str] | None = None,
        exact_seek: bool = False,
    ) -> AsyncGenerator[bytes]:
        del output_format, filter_params
        if not exact_seek:
            seek_position_ms = seek_position_ms // 100 * 100
        # AudioBuffer.get: whole 1 s chunks, then the millisecond's samples, rounded down
        rate = PCM_FORMAT.sample_rate
        first = seek_position_ms // 1000 * rate + rate * (seek_position_ms % 1000) // 1000
        samples = np.arange(first, first + 3 * rate, dtype=np.int64) % 32768
        data = samples.astype("<i2").tobytes()
        for start in range(0, len(data), 1000):
            yield data[start : start + 1000]


@pytest.fixture
def audio(monkeypatch: pytest.MonkeyPatch) -> StreamsAudio:
    """Build a StreamsAudio whose buffer is the counting double."""
    monkeypatch.setattr(audio_mod, "AudioBuffer", _CountingBuffer)
    monkeypatch.setattr(
        audio_mod, "get_normalization_mode", lambda *_a, **_k: VolumeNormalizationMode.DISABLED
    )
    controller = StreamsAudio(MagicMock())
    mass = cast("MagicMock", controller.mass)
    mass.streams.audio_analysis.get_audio_analysis = AsyncMock(return_value=None)
    mass.config.get_player_config = AsyncMock(return_value=MagicMock())
    mass.player_queues.queue_data_or_none.return_value = None
    return controller


def _queue_item() -> MagicMock:
    streamdetails = StreamDetails(
        provider="test_provider",
        item_id="track_b",
        audio_format=PCM_FORMAT,
        media_type=MediaType.TRACK,
    )
    streamdetails.queue_id = "queue_1"
    queue_item = MagicMock()
    queue_item.media_type = MediaType.TRACK
    queue_item.streamdetails = streamdetails
    queue_item.name = "Track B"
    return queue_item


async def _first_sample(audio: StreamsAudio, seek_position: float, exact_seek: bool) -> int:
    stream = audio.get_queue_item_stream(
        _queue_item(), PCM_FORMAT, seek_position=seek_position, exact_seek=exact_seek
    )
    async for chunk in stream:
        await stream.aclose()
        return int(np.frombuffer(chunk[:2], dtype="<i2")[0])
    raise AssertionError("no audio")


@pytest.mark.asyncio
@pytest.mark.parametrize("seek_position", [7.869387755, 12.3456789, 3.0])
async def test_an_exact_seek_resumes_on_the_exact_sample(
    audio: StreamsAudio, seek_position: float
) -> None:
    """A handover between milliseconds resumes on its own sample, not up to 1 ms earlier."""
    first = await _first_sample(audio, seek_position, exact_seek=True)

    assert first == round(seek_position * PCM_FORMAT.sample_rate) % 32768


@pytest.mark.asyncio
async def test_a_user_seek_keeps_its_100ms_steps(audio: StreamsAudio) -> None:
    """A user seek (not exact) is served from its 100 ms step, as before."""
    first = await _first_sample(audio, 12.3456789, exact_seek=False)

    assert first == round(12.3 * PCM_FORMAT.sample_rate) % 32768
