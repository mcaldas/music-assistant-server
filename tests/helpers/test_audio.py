"""Tests for music_assistant.helpers.audio."""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.helpers import audio as audio_helper
from music_assistant.helpers.audio import (
    build_concat_filelist,
    calculate_content_length,
    fade_out_pcm,
    get_output_format_key,
    parse_loudnorm,
    realtime_pcm_pacer,
    resolve_output_player_ids,
)
from music_assistant.helpers.ffmpeg import DEFAULT_MP3_BIT_RATE


def test_resolve_output_player_ids_resolves_parents_and_duplicates() -> None:
    """Output destinations use visible protocol parents without duplicates."""
    mass = MagicMock()
    players = {
        "leader": SimpleNamespace(protocol_parent_id=None),
        "protocol-child": SimpleNamespace(protocol_parent_id="child"),
        "child": SimpleNamespace(protocol_parent_id=None),
    }
    mass.players.get_player.side_effect = players.get

    result = resolve_output_player_ids(
        mass,
        ("leader", "leader", "protocol-child", "child", "missing", "missing"),
    )

    assert result == {"leader", "child", "missing"}


def test_mp3_content_length_uses_encoder_bitrate() -> None:
    """MP3 size estimation uses the bitrate configured for FFmpeg encoding."""
    seconds = 2
    assert calculate_content_length(
        AudioFormat(content_type=ContentType.MP3),
        seconds,
    ) == int(((DEFAULT_MP3_BIT_RATE * 1000) / 8) * seconds)


def test_build_concat_filelist_plain_paths() -> None:
    """Paths without special characters are wrapped verbatim, one per line."""
    result = build_concat_filelist(["/music/a.mp3", "/music/b.mp3"])
    assert result == "file '/music/a.mp3'\nfile '/music/b.mp3'\n"


def test_build_concat_filelist_escapes_apostrophes() -> None:
    r"""
    A single quote in the path is escaped as '\'' for the concat demuxer.

    Regression test for multipart playback failing on paths such as
    "Amelia Bedelia's", where the demuxer truncated the path at the apostrophe.
    """
    path = "/audiobooks/Herman Parish - Young Amelia Bedelia's Audio Collection/01.mp3"
    result = build_concat_filelist([path])
    assert (
        result
        == "file '/audiobooks/Herman Parish - Young Amelia Bedelia'\\''s Audio Collection/01.mp3'\n"
    )
    # The original apostrophe must survive once the escaping is unwrapped.
    assert path in result.replace("'\\''", "'")


def test_build_concat_filelist_escapes_multiple_apostrophes() -> None:
    """Every apostrophe in a path is escaped, not just the first."""
    result = build_concat_filelist(["/x/it's a, b's & c's.mp3"])
    assert result == "file '/x/it'\\''s a, b'\\''s & c'\\''s.mp3'\n"


# verbatim ffmpeg 7.1 output: the report is a block below the marker line, not inline
FFMPEG_LOUDNORM_OUTPUT = b"""[out#0/null @ 0x93b] Output stream
[Parsed_loudnorm_0 @ 0x93b41d440] \n{
\t"input_i" : "-17.86",
\t"input_tp" : "-1.89",
\t"input_lra" : "0.40",
\t"input_thresh" : "-27.86",
\t"output_i" : "-23.92",
\t"normalization_type" : "dynamic",
\t"target_offset" : "-0.08"
}
size=N/A time=00:00:03.20 bitrate=N/A speed=  86x
"""


def test_parse_loudnorm_reads_the_measurement_ffmpeg_actually_prints() -> None:
    """The integrated loudness is read from loudnorm's own JSON report block."""
    assert parse_loudnorm(FFMPEG_LOUDNORM_OUTPUT) == -17.86


def test_parse_loudnorm_accepts_a_decoded_string() -> None:
    """Callers that already decoded the output get the same measurement."""
    assert parse_loudnorm(FFMPEG_LOUDNORM_OUTPUT.decode()) == -17.86


def test_parse_loudnorm_without_a_report_returns_none() -> None:
    """Output from a run that never reached the filter carries no measurement."""
    assert parse_loudnorm(b"ffmpeg: Invalid data found when processing input") is None


def test_parse_loudnorm_with_a_truncated_report_returns_none() -> None:
    """A report cut off mid-object is not mistaken for a measurement."""
    assert parse_loudnorm(b'[Parsed_loudnorm_0 @ 0x1] \n{\n\t"input_i" : "-17.8') is None


def test_parse_loudnorm_treats_digital_silence_as_no_measurement() -> None:
    """A silent clip reports -inf, which is the absence of a level, not a level."""
    silent = FFMPEG_LOUDNORM_OUTPUT.replace(b'"-17.86"', b'"-inf"')
    assert parse_loudnorm(silent) is None


def test_parse_loudnorm_reads_a_report_from_further_down_the_filter_chain() -> None:
    """The marker carries the filter's position, which is not zero behind another filter."""
    chained = FFMPEG_LOUDNORM_OUTPUT.replace(b"[Parsed_loudnorm_0 @", b"[Parsed_loudnorm_1 @")
    assert parse_loudnorm(chained) == -17.86


@pytest.mark.asyncio
async def test_realtime_pcm_pacer_grants_bounded_initial_burst() -> None:
    """The pacer lets a bounded head start through unpaced, then enforces realtime."""
    # tiny format keeps the test fast: 16000 B/s, so 1s of audio is 16000 bytes
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_S16LE,
        sample_rate=8000,
        bit_depth=16,
        channels=1,
    )

    async def _instant_producer() -> AsyncGenerator[bytes]:
        for _ in range(10):
            yield b"\x00" * 1600  # 0.1s of audio per chunk, produced instantly

    loop = asyncio.get_running_loop()
    start = loop.time()
    async for _ in realtime_pcm_pacer(_instant_producer(), pcm_format):
        pass
    elapsed = loop.time() - start

    # 1.0s of audio with a 0.5s burst allowance should take ~0.5s: clearly less
    # than realtime (burst granted) but still paced (not instant). Bounds are
    # deliberately wide to stay robust on loaded CI runners.
    assert 0.3 < elapsed < 0.9


def test_the_content_length_key_moves_with_the_encoder_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A measured content length must not outlive the encoder settings it was measured under.

    The cached value is a byte count announced to the player as Content-Length, so an
    entry from an older encoder leaves it waiting out a body that already ended.
    """
    fmt = AudioFormat(content_type=ContentType.FLAC, sample_rate=44100, bit_depth=16, channels=2)
    before = get_output_format_key(fmt)
    monkeypatch.setattr(audio_helper, "OUTPUT_ENCODING_REVISION", 99)

    assert get_output_format_key(fmt) != before


@pytest.mark.parametrize(
    ("content_type", "bit_depth", "level"),
    [
        (ContentType.PCM_S16LE, 16, -16384),
        (ContentType.PCM_S24LE, 24, -(1 << 22)),
        (ContentType.PCM_S32LE, 32, -(1 << 30)),
        (ContentType.PCM_F32LE, 32, -0.5),
        (ContentType.PCM_F64LE, 64, -0.5),
    ],
)
def test_fade_out_pcm_ramps_every_sample_type_to_silence(
    content_type: ContentType, bit_depth: int, level: float
) -> None:
    """A fade spread over two chunks is the same straight ramp to silence as over one."""
    pcm_format = AudioFormat(
        content_type=content_type, bit_depth=bit_depth, sample_rate=1000, channels=2
    )
    width = bit_depth // 8
    frame = 2 * width
    floating = isinstance(level, float)
    float_code = "<f" if width == 4 else "<d"

    def _sample(data: bytes, index: int) -> float:
        raw = data[index * width : (index + 1) * width]
        if floating:
            return float(struct.unpack(float_code, raw)[0])
        return int.from_bytes(raw, "little", signed=True)

    one = (
        struct.pack(float_code, level)
        if floating
        else int(level).to_bytes(width, "little", signed=True)
    )
    audio = one * 200  # 100 frames
    fade = 20 * frame
    whole = fade_out_pcm(audio, pcm_format, len(audio), fade)
    head, tail = audio[: 90 * frame], audio[90 * frame :]
    split = fade_out_pcm(head, pcm_format, len(audio), fade)
    assert split + fade_out_pcm(tail, pcm_format, len(tail), fade) == whole
    assert whole[: 80 * frame] == audio[: 80 * frame]
    assert [_sample(whole, 2 * index + 1) for index in range(80, 100)] == pytest.approx(
        [level * (100 - index) / 20 for index in range(80, 100)], abs=1e-6 if floating else 1
    )


def test_fade_out_pcm_fades_the_whole_frames_of_a_chunk_that_splits_frames() -> None:
    """24-bit audio from ffmpeg comes in chunks that start and stop inside a frame."""
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_S24LE, bit_depth=24, sample_rate=1000, channels=2
    )
    frame = 6
    audio = (1000).to_bytes(3, "little", signed=True) * 200  # 100 frames
    fade = 20 * frame
    whole = fade_out_pcm(audio, pcm_format, len(audio), fade)
    # from 2 bytes into frame 85 to 4 bytes into frame 95
    start, stop = 85 * frame + 2, 95 * frame + 4
    chunk = fade_out_pcm(audio[start:stop], pcm_format, len(audio) - start, fade)
    # the frames it holds only part of are left as they are, every whole one is faded
    assert chunk[: frame - 2] == audio[start : 86 * frame]
    assert chunk[frame - 2 : -4] == whole[86 * frame : 95 * frame]
    assert chunk[-4:] == audio[95 * frame : stop]
