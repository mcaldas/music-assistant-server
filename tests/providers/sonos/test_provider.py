"""Tests for Sonos provider discovery."""

import asyncio
import ipaddress
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientConnectionError
from zeroconf import ServiceStateChange

from music_assistant.mass import MusicAssistant
from music_assistant.providers.sonos.player import SonosPlayer
from music_assistant.providers.sonos.provider import SonosPlayerProvider

PLAYER_ID = "sonos_player"
SERVICE_NAME = "Toilet._sonos._tcp.local."
ADDRESS = "192.168.1.10"
DISCOVERY_INFO = {"device": {"id": PLAYER_ID, "name": "Toilet", "capabilities": ["PLAYBACK"]}}
PROVIDER = "music_assistant.providers.sonos.provider"


def _bind_provider(mass: MusicAssistant | MagicMock) -> SonosPlayerProvider:
    """Create a Sonos provider bound to the given MusicAssistant."""
    provider = SonosPlayerProvider.__new__(SonosPlayerProvider)
    provider.mass = mass
    provider.logger = logging.getLogger("test.sonos.discovery")
    provider._ignored_disabled_players = set()
    provider._pending_setup_tasks = set()
    provider._pending_refresh_tasks = set()
    provider._setups_in_progress = set()
    provider._setup_failures = {}
    provider._unloaded = False
    return provider


def _make_provider(enabled: bool = False) -> tuple[SonosPlayerProvider, MagicMock]:
    """Create a Sonos provider with mocked discovery dependencies."""
    mass = MagicMock()
    mass.config.get_raw_player_config_value.return_value = enabled
    mass.players.get_player.return_value = None
    return _bind_provider(mass), mass


def _bind_discovering_provider(mass: MusicAssistant) -> SonosPlayerProvider:
    """Create a Sonos provider that discovers players on the real timer machinery."""
    mass.config = MagicMock()
    mass.config.get_raw_player_config_value.return_value = True
    mass.players = MagicMock()
    mass.players.get_player.return_value = None
    mass.streams = MagicMock()
    return _bind_provider(mass)


def _make_discovery_info(player_id: str = PLAYER_ID) -> MagicMock:
    """Create minimal Sonos mDNS discovery information."""
    info = MagicMock()
    info.decoded_properties = {"uuid": player_id}
    info.ip_addresses_by_version.return_value = [ipaddress.ip_address(ADDRESS)]
    return info


def _bind_setup_provider(mass: MusicAssistant) -> tuple[SonosPlayerProvider, dict[str, Any]]:
    """Create a discovering provider whose registered players are kept in the returned dict."""
    provider = _bind_discovering_provider(mass)
    provider.config = MagicMock()
    provider.config.get_value.return_value = [ADDRESS]
    mass._http_session_no_ssl = MagicMock()
    registered: dict[str, Any] = {}
    cast("MagicMock", mass.players).get_player.side_effect = registered.get
    return provider, registered


def _fake_player_class(
    registered: dict[str, Any],
    setup_error: Exception | None = None,
    gate: asyncio.Event | None = None,
    connects: bool = True,
) -> type[Any]:
    """Create a stand-in for SonosPlayer that connects, then registers or fails as told."""

    class FakeSonosPlayer:
        created: ClassVar[list[Any]] = []

        def __init__(self, _prov: object, player_id: str, discovery_info: object) -> None:
            self.player_id = player_id
            self.device_info = MagicMock()
            self._listen_task: object | None = None
            self.unloaded = False
            self.created.append(self)

        async def setup(self) -> None:
            if connects:
                self._listen_task = object()
            if gate is not None:
                await gate.wait()
            if setup_error is not None:
                raise setup_error
            registered[self.player_id] = self

        async def on_unload(self) -> None:
            self.unloaded = True

    return FakeSonosPlayer


@contextmanager
def _discovering(fake: type[Any], lookup: Callable[..., Any] | None = None) -> Iterator[None]:
    """Let the provider create the given player class from the given discovery lookup."""
    with (
        patch(f"{PROVIDER}.SonosPlayer", fake),
        patch(f"{PROVIDER}.get_discovery_info", lookup or AsyncMock(return_value=DISCOVERY_INFO)),
    ):
        yield


async def _wait_until(condition: Callable[[], object]) -> None:
    """Wait for the condition to hold, failing the test when it never does."""
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.01)


def _unreachable() -> ClientConnectionError:
    """Create the error of a speaker that does not answer."""
    return ClientConnectionError("no route to host")


@pytest.mark.asyncio
async def test_disabled_discovery_is_not_scheduled_repeatedly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test repeated announcements for a disabled player are ignored once."""
    provider, mass = _make_provider()
    info = _make_discovery_info()

    with caplog.at_level(logging.DEBUG, logger=provider.logger.name):
        await provider.on_mdns_service_state_change(SERVICE_NAME, ServiceStateChange.Added, info)
        await provider.on_mdns_service_state_change(SERVICE_NAME, ServiceStateChange.Updated, info)

    mass.call_later.assert_not_called()
    ignored_records = [
        record for record in caplog.records if "in discovery as it is disabled" in record.message
    ]
    assert len(ignored_records) == 1


@pytest.mark.asyncio
async def test_disabled_discovery_logs_again_after_reenable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a new disabled period produces a new discovery diagnostic."""
    provider, mass = _make_provider()
    info = _make_discovery_info()

    with caplog.at_level(logging.DEBUG, logger=provider.logger.name):
        await provider.on_mdns_service_state_change(SERVICE_NAME, ServiceStateChange.Added, info)
        mass.config.get_raw_player_config_value.return_value = True
        await provider.on_mdns_service_state_change(SERVICE_NAME, ServiceStateChange.Updated, info)
        mass.config.get_raw_player_config_value.return_value = False
        await provider.on_mdns_service_state_change(SERVICE_NAME, ServiceStateChange.Updated, info)

    mass.call_later.assert_called_once()
    ignored_records = [
        record for record in caplog.records if "in discovery as it is disabled" in record.message
    ]
    assert len(ignored_records) == 2


@pytest.mark.asyncio
async def test_unload_cancels_a_pending_player_setup(timer_mass: MusicAssistant) -> None:
    """Test a discovered player that is still waiting to be set up is dropped on unload."""
    provider = _bind_discovering_provider(timer_mass)

    await provider.on_mdns_service_state_change(
        SERVICE_NAME, ServiceStateChange.Added, _make_discovery_info()
    )
    handle = timer_mass._tracked_timers[f"setup_sonos_{PLAYER_ID}"]

    await provider.unload(is_removed=True)

    assert handle.cancelled()
    assert provider._pending_setup_tasks == set()


@pytest.mark.asyncio
async def test_unload_cancels_a_player_setup_that_already_started(
    timer_mass: MusicAssistant,
) -> None:
    """Test a player setup that is already running is aborted when the provider unloads."""
    provider = _bind_discovering_provider(timer_mass)
    setup_started = asyncio.Event()

    async def _setup_player(*_args: object) -> None:
        setup_started.set()
        await asyncio.sleep(5)

    provider._setup_player = _setup_player  # type: ignore[assignment]
    await provider.on_mdns_service_state_change(
        SERVICE_NAME, ServiceStateChange.Added, _make_discovery_info()
    )
    task_id = f"setup_sonos_{PLAYER_ID}"
    # let the armed timer fire now instead of waiting out the discovery debounce
    timer_mass.call_later(0, provider._setup_player, task_id=task_id)
    await setup_started.wait()
    task = timer_mass._tracked_tasks[task_id]

    await provider.unload(is_removed=True)

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_an_announcement_arriving_after_the_unload_is_ignored(
    timer_mass: MusicAssistant,
) -> None:
    """Test a discovery callback that lands after the unload does not arm a new setup."""
    provider = _bind_discovering_provider(timer_mass)

    await provider.unload(is_removed=True)
    await provider.on_mdns_service_state_change(
        SERVICE_NAME, ServiceStateChange.Added, _make_discovery_info()
    )

    assert timer_mass._tracked_timers == {}
    assert provider._pending_setup_tasks == set()


@pytest.mark.asyncio
async def test_unload_sweeps_every_discovered_player(timer_mass: MusicAssistant) -> None:
    """Test a burst of discovered players leaves no setup behind after unload."""
    provider = _bind_discovering_provider(timer_mass)

    for index in range(3):
        await provider.on_mdns_service_state_change(
            f"Speaker{index}._sonos._tcp.local.",
            ServiceStateChange.Added,
            _make_discovery_info(f"sonos_player_{index}"),
        )
    assert len(timer_mass._tracked_timers) == 3

    await provider.unload(is_removed=True)

    assert timer_mass._tracked_timers == {}
    assert provider._pending_setup_tasks == set()


@pytest.mark.asyncio
async def test_a_withdrawn_announcement_is_handed_to_the_player() -> None:
    """Test a removed Sonos announcement lets the player decide whether it went to sleep."""
    provider, mass = _make_provider()
    player = MagicMock(spec=SonosPlayer)
    goodbye = object()
    player.on_mdns_goodbye = MagicMock(return_value=goodbye)
    mass.players.get_player.return_value = player

    await provider.on_mdns_service_state_change(
        "RINCON_C438750D189C01400@Move 2._sonos._tcp.local.", ServiceStateChange.Removed, None
    )

    mass.players.get_player.assert_called_once_with("RINCON_C438750D189C01400")
    mass.create_task.assert_called_once_with(goodbye)


@pytest.mark.asyncio
async def test_a_manual_address_does_not_set_up_a_registered_player_again(
    timer_mass: MusicAssistant,
) -> None:
    """Test a speaker that mdns already set up is left alone when its address is listed too."""
    provider, registered = _bind_setup_provider(timer_mass)
    cast("MagicMock", provider.config).get_value.return_value = [ADDRESS, ADDRESS]
    fake = _fake_player_class(registered)

    with _discovering(fake):
        await provider._setup_player(PLAYER_ID, SERVICE_NAME, _make_discovery_info())
        await provider.loaded_in_mass()

    assert len(fake.created) == 1
    assert registered[PLAYER_ID] is fake.created[0]


@pytest.mark.asyncio
async def test_a_speaker_found_both_ways_at_once_is_set_up_once(
    timer_mass: MusicAssistant,
) -> None:
    """Test a manual address and an announcement racing for one speaker set it up once."""
    provider, registered = _bind_setup_provider(timer_mass)
    gate = asyncio.Event()
    fake = _fake_player_class(registered, gate=gate)

    with _discovering(fake):
        listed = asyncio.create_task(provider.loaded_in_mass())
        announced = asyncio.create_task(
            provider._setup_player(PLAYER_ID, SERVICE_NAME, _make_discovery_info())
        )
        await asyncio.sleep(0.01)
        gate.set()
        await asyncio.gather(listed, announced)

    assert len(fake.created) == 1
    assert registered[PLAYER_ID] is fake.created[0]
    assert not fake.created[0].unloaded


@pytest.mark.asyncio
async def test_an_announcement_does_not_abort_a_setup_in_progress(
    timer_mass: MusicAssistant,
) -> None:
    """Test an announcement for a speaker that is being set up arms no second setup."""
    provider, registered = _bind_setup_provider(timer_mass)
    gate = asyncio.Event()
    fake = _fake_player_class(registered, gate=gate)
    info = _make_discovery_info()
    task_id = f"setup_sonos_{PLAYER_ID}"

    with _discovering(fake):
        timer_mass.call_later(
            0, provider._setup_player, PLAYER_ID, SERVICE_NAME, info, task_id=task_id
        )
        await _wait_until(lambda: fake.created)
        task = timer_mass._tracked_tasks[task_id]
        await provider.on_mdns_service_state_change(SERVICE_NAME, ServiceStateChange.Updated, info)
        assert task_id not in timer_mass._tracked_timers
        gate.set()
        await task

    assert len(fake.created) == 1
    assert registered[PLAYER_ID] is fake.created[0]


@pytest.mark.asyncio
async def test_a_setup_withdraws_the_timer_armed_during_its_lookup(
    timer_mass: MusicAssistant,
) -> None:
    """Test an announcement that arrives while the speaker is looked up cannot abort its setup."""
    provider, registered = _bind_setup_provider(timer_mass)
    started = asyncio.Event()
    release = asyncio.Event()
    gate = asyncio.Event()
    fake = _fake_player_class(registered, gate=gate)
    info = _make_discovery_info()
    task_id = f"setup_sonos_{PLAYER_ID}"

    async def _lookup(*_args: object) -> dict[str, Any]:
        started.set()
        await release.wait()
        return DISCOVERY_INFO

    with _discovering(fake, _lookup):
        timer_mass.call_later(
            0, provider._setup_player, PLAYER_ID, SERVICE_NAME, info, task_id=task_id
        )
        await started.wait()
        task = timer_mass._tracked_tasks[task_id]
        await provider.on_mdns_service_state_change(SERVICE_NAME, ServiceStateChange.Updated, info)
        assert task_id in timer_mass._tracked_timers
        release.set()
        await _wait_until(lambda: fake.created)
        assert task_id not in timer_mass._tracked_timers
        gate.set()
        await task

    assert registered[PLAYER_ID] is fake.created[0]


@pytest.mark.asyncio
async def test_a_failed_setup_is_retried_until_it_succeeds(timer_mass: MusicAssistant) -> None:
    """Test a speaker that does not answer at first is set up once it does."""
    provider, registered = _bind_setup_provider(timer_mass)
    fake = _fake_player_class(registered)
    lookup = AsyncMock(side_effect=[_unreachable(), DISCOVERY_INFO])

    with _discovering(fake, lookup), patch(f"{PROVIDER}.SETUP_RETRY_DELAYS", (0.01,)):
        await provider._setup_player(PLAYER_ID, SERVICE_NAME, _make_discovery_info())
        await _wait_until(lambda: PLAYER_ID in registered)

    assert len(fake.created) == 1
    assert provider._setup_failures == {}


@pytest.mark.asyncio
async def test_a_streak_of_failures_is_logged_once_and_ends(
    timer_mass: MusicAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """Test a speaker that never answers is warned about once and given up on in the end."""
    provider, registered = _bind_setup_provider(timer_mass)
    fake = _fake_player_class(registered)
    lookup = AsyncMock(side_effect=_unreachable())

    def _warnings() -> list[logging.LogRecord]:
        return [
            record
            for record in caplog.records
            if record.name == provider.logger.name and record.levelno >= logging.WARNING
        ]

    with (
        caplog.at_level(logging.DEBUG, logger=provider.logger.name),
        _discovering(fake, lookup),
        patch(f"{PROVIDER}.SETUP_RETRY_DELAYS", (0.01, 0.01)),
    ):
        await provider._setup_player(PLAYER_ID, SERVICE_NAME, _make_discovery_info())
        assert len(_warnings()) == 1
        assert SERVICE_NAME in _warnings()[0].message
        assert "no route to host" in _warnings()[0].message
        await _wait_until(lambda: lookup.await_count == 3)
        await asyncio.sleep(0.05)

    assert lookup.await_count == 3
    assert len(_warnings()) == 1
    assert timer_mass._tracked_timers == {}


@pytest.mark.asyncio
async def test_an_unreachable_manual_address_is_retried(timer_mass: MusicAssistant) -> None:
    """Test a manual IP address that does not answer when the provider loads is tried again."""
    provider, registered = _bind_setup_provider(timer_mass)
    fake = _fake_player_class(registered)
    lookup = AsyncMock(side_effect=[_unreachable(), DISCOVERY_INFO])

    with _discovering(fake, lookup), patch(f"{PROVIDER}.SETUP_RETRY_DELAYS", (0.01,)):
        await provider.loaded_in_mass()
        await _wait_until(lambda: PLAYER_ID in registered)

    assert len(fake.created) == 1


@pytest.mark.asyncio
async def test_unload_cancels_a_pending_retry(timer_mass: MusicAssistant) -> None:
    """Test a setup that waits for its retry is dropped on unload."""
    provider, registered = _bind_setup_provider(timer_mass)
    fake = _fake_player_class(registered)
    lookup = AsyncMock(side_effect=_unreachable())

    with _discovering(fake, lookup), patch(f"{PROVIDER}.SETUP_RETRY_DELAYS", (60,)):
        await provider.loaded_in_mass()
        handle = timer_mass._tracked_timers[f"setup_sonos_{ADDRESS}"]
        await provider.unload(is_removed=True)

    assert handle.cancelled()


@pytest.mark.asyncio
async def test_a_goodbye_ends_the_retries(timer_mass: MusicAssistant) -> None:
    """Test a speaker that withdraws its announcement is no longer tried."""
    provider, registered = _bind_setup_provider(timer_mass)
    fake = _fake_player_class(registered)
    lookup = AsyncMock(side_effect=_unreachable())
    task_id = f"setup_sonos_{PLAYER_ID}"

    with _discovering(fake, lookup), patch(f"{PROVIDER}.SETUP_RETRY_DELAYS", (60,)):
        await provider._setup_player(PLAYER_ID, SERVICE_NAME, _make_discovery_info())
        assert task_id in timer_mass._tracked_timers
        await provider.on_mdns_service_state_change(
            f"{PLAYER_ID}@{SERVICE_NAME}", ServiceStateChange.Removed, None
        )

    assert task_id not in timer_mass._tracked_timers
    assert provider._setup_failures == {}


@pytest.mark.asyncio
async def test_a_player_that_connected_but_was_not_registered_is_unloaded(
    timer_mass: MusicAssistant,
) -> None:
    """Test a setup that fails after connecting does not leave its connection behind."""
    provider, registered = _bind_setup_provider(timer_mass)
    fake = _fake_player_class(registered, setup_error=RuntimeError("boom"))

    with _discovering(fake), patch(f"{PROVIDER}.SETUP_RETRY_DELAYS", ()):
        await provider._setup_player(PLAYER_ID, SERVICE_NAME, _make_discovery_info())

    assert fake.created[0].unloaded
    assert provider._setups_in_progress == set()


@pytest.mark.asyncio
async def test_a_player_that_never_connected_is_not_unloaded(timer_mass: MusicAssistant) -> None:
    """Test a setup that fails to connect has nothing to release."""
    provider, registered = _bind_setup_provider(timer_mass)
    fake = _fake_player_class(registered, setup_error=RuntimeError("boom"), connects=False)

    with _discovering(fake), patch(f"{PROVIDER}.SETUP_RETRY_DELAYS", ()):
        await provider._setup_player(PLAYER_ID, SERVICE_NAME, _make_discovery_info())

    assert not fake.created[0].unloaded


@pytest.mark.parametrize(
    ("enabled", "capabilities"),
    [(False, ["PLAYBACK"]), (True, [])],
    ids=["disabled", "satellite"],
)
@pytest.mark.asyncio
async def test_a_manual_address_skips_a_disabled_player_and_a_satellite(
    timer_mass: MusicAssistant, enabled: bool, capabilities: list[str]
) -> None:
    """Test a manual IP address is ignored for the same reasons as an announcement."""
    provider, registered = _bind_setup_provider(timer_mass)
    cast("MagicMock", timer_mass.config).get_raw_player_config_value.return_value = enabled
    fake = _fake_player_class(registered)
    lookup = AsyncMock(
        return_value={"device": {**DISCOVERY_INFO["device"], "capabilities": capabilities}}
    )

    with _discovering(fake, lookup):
        await provider.loaded_in_mass()

    assert fake.created == []
    assert timer_mass._tracked_timers == {}


@pytest.mark.asyncio
async def test_an_announcement_without_an_address_is_asked_again(
    timer_mass: MusicAssistant,
) -> None:
    """Test an announcement that came without its address records is resolved once more."""
    provider, registered = _bind_setup_provider(timer_mass)
    timer_mass.discovery = MagicMock()
    fake = _fake_player_class(registered)
    info = _make_discovery_info()
    info.ip_addresses_by_version.return_value = []

    async def _request(*_args: object) -> bool:
        info.ip_addresses_by_version.return_value = [ipaddress.ip_address(ADDRESS)]
        return True

    info.async_request = AsyncMock(side_effect=_request)

    with _discovering(fake):
        await provider._setup_player(PLAYER_ID, SERVICE_NAME, info)

    assert registered[PLAYER_ID] is fake.created[0]
