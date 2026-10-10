"""Tests for marking a library mapping unavailable when its provider no longer finds the item."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from music_assistant_models.enums import MediaType, StreamType
from music_assistant_models.errors import (
    AudioError,
    InvalidDataError,
    MediaNotFoundError,
    ProviderUnavailableError,
)
from music_assistant_models.media_items import ProviderMapping, Track
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.streams.audio import StreamsAudio
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

INSTANCE = "tidal--abc"
SIBLING_INSTANCE = "tidal--xyz"


def _mapping(item_id: str, instance: str = INSTANCE) -> ProviderMapping:
    """Build a mapping of the given item on the given provider instance."""
    return ProviderMapping(item_id=item_id, provider_domain="tidal", provider_instance=instance)


def _provider(instance: str, error: Exception | None = None) -> MagicMock:
    """Build a provider instance that raises the given error, or resolves streamdetails."""

    async def _get_stream_details(item_id: str, media_type: MediaType) -> StreamDetails:
        if error:
            raise error
        return StreamDetails(
            provider=instance,
            item_id=item_id,
            audio_format=MagicMock(),
            media_type=media_type,
            stream_type=StreamType.HTTP,
            path="http://localhost/track.flac",
        )

    provider = MagicMock(spec=MusicProvider)
    provider.instance_id = instance
    provider.get_stream_details = _get_stream_details
    return provider


def _audio() -> tuple[StreamsAudio, MagicMock, list[Coroutine[Any, Any, None]]]:
    """Build a StreamsAudio, handing back the mark mock and the background tasks scheduled."""
    mass = MagicMock()
    mark = AsyncMock()
    mass.music.mark_provider_mapping_unavailable = mark
    scheduled: list[Coroutine[Any, Any, None]] = []
    mass.create_task.side_effect = lambda coro, **_kwargs: scheduled.append(coro)
    return StreamsAudio(mass), mark, scheduled


async def test_a_dead_mapping_is_marked_and_the_next_one_plays() -> None:
    """A mapping its provider does not find is marked; the next candidate still serves."""
    dead, alive = _mapping("dead"), _mapping("alive")
    track = Track(item_id="42", provider="library", name="15 Step", provider_mappings={dead, alive})
    audio, mark, scheduled = _audio()

    streamdetails = await audio._request_streamdetails(
        [
            (dead, _provider(INSTANCE, MediaNotFoundError("gone"))),
            (alive, _provider(INSTANCE)),
        ],
        track,
    )

    assert streamdetails is not None
    assert streamdetails.item_id == "alive"
    assert len(scheduled) == 1
    await scheduled[0]
    mark.assert_awaited_once_with(track, dead)


async def test_a_sibling_instance_miss_does_not_mark_the_mapping() -> None:
    """Another account of the same service lacking the item says nothing about the mapping."""
    mapping = _mapping("item")
    track = Track(item_id="42", provider="library", name="15 Step", provider_mappings={mapping})
    audio, _, scheduled = _audio()

    await audio._request_streamdetails(
        [(mapping, _provider(SIBLING_INSTANCE, MediaNotFoundError("gone")))], track
    )

    assert scheduled == []


async def test_a_transient_error_does_not_mark_the_mapping() -> None:
    """A provider that is merely unreachable leaves the mapping available."""
    mapping = _mapping("item")
    track = Track(item_id="42", provider="library", name="15 Step", provider_mappings={mapping})
    audio, _, scheduled = _audio()

    with pytest.raises(AudioError):
        await audio._request_streamdetails(
            [(mapping, _provider(INSTANCE, ProviderUnavailableError("offline")))], track
        )

    assert scheduled == []


@pytest.mark.parametrize(
    "error",
    [
        InvalidDataError("Error while fetching playlist http://localhost/track.m3u8"),
        ProviderUnavailableError("offline"),
        aiohttp.ClientConnectionError(),
        TimeoutError(),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_a_provider_that_could_not_be_asked_is_not_reported_as_not_found(
    error: Exception,
) -> None:
    """A provider that failed to answer raises, instead of reading as an item nobody has."""
    mapping = _mapping("item")
    track = Track(item_id="42", provider="library", name="15 Step", provider_mappings={mapping})
    audio, _, _ = _audio()

    with pytest.raises(AudioError) as raised:
        await audio._request_streamdetails([(mapping, _provider(INSTANCE, error))], track)

    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    ("error", "traced"),
    [(TimeoutError(), True), (InvalidDataError("no playlist"), False)],
    ids=["not_raised_by_the_provider", "raised_by_the_provider"],
)
async def test_a_failure_the_provider_did_not_raise_itself_is_logged_with_its_trace(
    error: Exception, traced: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """An error that is not one of our own is logged with where it came from."""
    mapping = _mapping("item")
    track = Track(item_id="42", provider="library", name="15 Step", provider_mappings={mapping})
    audio, _, _ = _audio()

    with caplog.at_level(logging.WARNING), pytest.raises(AudioError):
        await audio._request_streamdetails([(mapping, _provider(INSTANCE, error))], track)

    (record,) = caplog.records
    assert (record.exc_info is not None) is traced
    if traced:
        assert "TimeoutError" in record.getMessage()


@pytest.mark.parametrize("unreachable_first", [False, True], ids=["after", "before"])
async def test_a_not_found_beside_an_unreachable_candidate_is_not_none(
    unreachable_first: bool,
) -> None:
    """One provider lacking the item does not speak for another that could not be asked."""
    gone, unreachable = _mapping("gone"), _mapping("unreachable")
    track = Track(
        item_id="42", provider="library", name="15 Step", provider_mappings={gone, unreachable}
    )
    audio, _, _ = _audio()
    candidates = [
        (gone, _provider(SIBLING_INSTANCE, MediaNotFoundError("gone"))),
        (unreachable, _provider(INSTANCE, InvalidDataError("no playlist"))),
    ]
    if unreachable_first:
        candidates.reverse()

    with pytest.raises(AudioError):
        await audio._request_streamdetails(candidates, track)


async def test_not_found_on_every_candidate_still_returns_none() -> None:
    """An item none of its providers has is still answered with no stream details."""
    first, second = _mapping("first"), _mapping("second")
    track = Track(
        item_id="42", provider="library", name="15 Step", provider_mappings={first, second}
    )
    audio, _, _ = _audio()

    streamdetails = await audio._request_streamdetails(
        [
            (first, _provider(SIBLING_INSTANCE, MediaNotFoundError("gone"))),
            (second, _provider(SIBLING_INSTANCE, MediaNotFoundError("gone"))),
        ],
        track,
    )

    assert streamdetails is None
