"""Unit tests for the Apple Music setup flow."""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import ClientError
from music_assistant_models.enums import FlowStepType

from music_assistant.models.setup_flow import SetupFlowContext, SetupFlowError, SetupSession
from music_assistant.providers.apple_music import setup_flow as apple_flow
from music_assistant.providers.apple_music.constants import (
    CONF_MUSIC_APP_TOKEN,
    CONF_MUSIC_USER_MANUAL_TOKEN,
    CONF_MUSIC_USER_TOKEN,
    CONF_OWN_APP_TOKEN,
    CONF_USE_OWN_APP_TOKEN,
)
from music_assistant.providers.apple_music.setup_flow import _app_token_accepted


class _FakeRequestCtx:
    """Minimal async context manager mimicking aiohttp's request context."""

    def __init__(self, status: int) -> None:
        self._response = MagicMock(status=status)

    async def __aenter__(self) -> MagicMock:
        return self._response

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _make_mass(status: int) -> MagicMock:
    mass = MagicMock()
    mass.http_session.get = MagicMock(return_value=_FakeRequestCtx(status))
    return mass


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, True),
        (401, False),
        (403, False),
        # a throttled or broken /v1/test says nothing about the token itself
        (429, None),
        (500, None),
        (503, None),
    ],
)
@pytest.mark.asyncio
async def test_app_token_accepted_per_status(status: int, expected: bool | None) -> None:
    """Only an explicit rejection by Apple marks the developer token as invalid."""
    assert await _app_token_accepted(_make_mass(status), "app-token") is expected


@pytest.mark.asyncio
async def test_app_token_accepted_empty_token_is_rejected() -> None:
    """An empty (e.g. not provisioned) token is rejected without calling the API."""
    mass = _make_mass(200)
    assert await _app_token_accepted(mass, "") is False
    mass.http_session.get.assert_not_called()


@pytest.mark.asyncio
async def test_app_token_accepted_network_error_is_inconclusive() -> None:
    """An unreachable API does not make a valid token look invalid."""
    mass = MagicMock()
    mass.http_session.get = MagicMock(side_effect=ClientError("boom"))
    assert await _app_token_accepted(mass, "app-token") is None


# --- run_setup: the user's own developer token ---------------------------------------------

_USER_TOKEN = "A" * 40 + "=="


def _flow_session(
    kind: str = "setup",
    setup_data: dict[str, Any] | None = None,
    fail_finish: int = 0,
) -> tuple[SetupSession, dict[str, Any]]:
    """Build a SetupSession that records what the flow finishes with (failing the first N)."""
    finished: dict[str, Any] = {}
    failures = {"left": fail_finish}

    async def finish_handler(_session: SetupSession, values: dict[str, Any]) -> dict[str, str]:
        if failures["left"]:
            failures["left"] -= 1
            raise SetupFlowError("provider failed to load")
        finished.update(values)
        return {"instance_id": "apple_music--test"}

    context = SetupFlowContext(
        kind=kind,  # type: ignore[arg-type]
        reason="user",  # type: ignore[arg-type]
        domain="apple_music",
        instance_id="apple_music--test" if kind == "reconfigure" else None,
        setup_data=setup_data or {},
    )
    return SetupSession(MagicMock(), "flow-test", context, finish_handler), finished


async def _wait_for(predicate: Any, timeout: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def _run(
    session: SetupSession,
    submits: list[dict[str, Any]],
    accepted: dict[str, bool | None],
) -> tuple[list[tuple[str, bool]], list[Any], list[dict[str, str]]]:
    """
    Drive run_setup, submitting each form shown with the next values.

    Returns the MusicKit sign-ins (developer token, prefill), the form steps shown and their
    errors as shown (a later accepted submit clears a step's errors in place).
    """
    signins: list[tuple[str, bool]] = []
    forms: list[Any] = []
    errors: list[dict[str, str]] = []

    async def fake_accepted(_mass: Any, token: str) -> bool | None:
        return accepted.get(token, False)

    async def fake_musickit(_session: Any, app_token: str, prefill: bool = True) -> dict[str, str]:
        signins.append((app_token, prefill))
        return {"music-user-token": _USER_TOKEN, "music-user-token-timestamp": "1700000000"}

    def new_form() -> Any:
        step = session.current_step
        if step and step.type == FlowStepType.FORM and (not forms or step is not forms[-1]):
            return step
        return None

    with (
        patch.object(apple_flow, "MUSIC_APP_TOKEN", "bundled"),
        patch.object(apple_flow, "_app_token_accepted", fake_accepted),
        patch.object(apple_flow, "_musickit_authenticate", fake_musickit),
    ):
        task = asyncio.create_task(apple_flow.run_setup(session))
        for values in submits:
            forms.append(await _wait_for(new_form))
            errors.append(dict(forms[-1].errors))
            session.handle_submit(values)
        await _wait_for(lambda: session.finished)
        await task
    return signins, forms, errors


def _keys(step: Any) -> list[str]:
    return [e.key for e in step.entries]


async def test_default_flow_keeps_the_bundled_token() -> None:
    """Without the advanced option nothing about the developer token is stored."""
    session, finished = _flow_session()
    signins, forms, _errors = await _run(session, [{}], {"bundled": True})
    assert CONF_MUSIC_APP_TOKEN not in finished
    assert signins == [("bundled", True)]
    token_field = next(e for e in forms[0].entries if e.key == CONF_OWN_APP_TOKEN)
    assert token_field.depends_on == CONF_USE_OWN_APP_TOKEN


async def test_own_token_is_stored_and_used_for_the_sign_in() -> None:
    """An accepted own token replaces the bundled one, and the user signs in with it afresh."""
    session, finished = _flow_session()
    signins, _, _ = await _run(
        session,
        [{CONF_USE_OWN_APP_TOKEN: True, CONF_OWN_APP_TOKEN: " mine "}],
        {"bundled": True, "mine": True},
    )
    assert finished[CONF_MUSIC_APP_TOKEN] == "mine"
    assert CONF_OWN_APP_TOKEN not in finished
    assert signins == [("mine", False)]


@pytest.mark.parametrize("answer", [False, None])
async def test_new_own_token_must_be_confirmed(answer: bool | None) -> None:
    """A new token Apple rejects, or can't confirm (throttled), re-renders the form with an error."""
    session, finished = _flow_session()
    signins, _forms, errors = await _run(
        session,
        [
            {CONF_USE_OWN_APP_TOKEN: True, CONF_OWN_APP_TOKEN: "bad"},
            {CONF_USE_OWN_APP_TOKEN: True, CONF_OWN_APP_TOKEN: "mine"},
        ],
        {"bundled": True, "bad": answer, "mine": True},
    )
    assert errors[1] == {CONF_OWN_APP_TOKEN: "invalid_value"}
    assert finished[CONF_MUSIC_APP_TOKEN] == "mine"
    assert signins == [("mine", False)]


async def test_switch_on_without_any_token_asks_for_one() -> None:
    """Switching on with an empty field and nothing stored is an error, not the bundled token."""
    session, finished = _flow_session()
    _, _forms, errors = await _run(
        session,
        [{CONF_USE_OWN_APP_TOKEN: True}, {CONF_USE_OWN_APP_TOKEN: False}],
        {"bundled": True},
    )
    assert errors[1] == {CONF_OWN_APP_TOKEN: "invalid_value"}
    assert CONF_MUSIC_APP_TOKEN not in finished


async def test_reconfigure_keeps_the_stored_own_token() -> None:
    """
    Keep the stored own token when the field is left empty on a reconfigure.

    A secure value never reaches the form, so the switch carries it: kept even while Apple's
    check is throttled, and the stored user token is offered to the sign-in again.
    """
    stored = {CONF_MUSIC_APP_TOKEN: "mine", CONF_MUSIC_USER_TOKEN: _USER_TOKEN}
    session, finished = _flow_session("reconfigure", stored)
    signins, forms, _errors = await _run(
        session, [{CONF_USE_OWN_APP_TOKEN: True}], {"bundled": True, "mine": None}
    )
    switch = next(e for e in forms[0].entries if e.key == CONF_USE_OWN_APP_TOKEN)
    assert switch.value is True
    assert finished[CONF_MUSIC_APP_TOKEN] == "mine"
    assert signins == [("mine", True)]


async def test_switching_off_goes_back_to_the_bundled_token() -> None:
    """Switching the option off clears the stored token; the old user token is not reused."""
    stored = {CONF_MUSIC_APP_TOKEN: "mine", CONF_MUSIC_USER_TOKEN: _USER_TOKEN}
    session, finished = _flow_session("reconfigure", stored)
    signins, _, _ = await _run(session, [{CONF_USE_OWN_APP_TOKEN: False}], {"bundled": True})
    assert finished[CONF_MUSIC_APP_TOKEN] == ""
    assert signins == [("bundled", False)]


async def test_a_failed_load_keeps_the_switch_and_the_accepted_token() -> None:
    """When the provider fails to load, the form comes back with the switch on and the token kept."""
    session, finished = _flow_session(fail_finish=1)
    signins, forms, _errors = await _run(
        session,
        [
            {CONF_USE_OWN_APP_TOKEN: True, CONF_OWN_APP_TOKEN: "mine"},
            {CONF_USE_OWN_APP_TOKEN: True},
        ],
        {"bundled": True, "mine": True},
    )
    switch = next(e for e in forms[1].entries if e.key == CONF_USE_OWN_APP_TOKEN)
    assert switch.value is True
    assert finished[CONF_MUSIC_APP_TOKEN] == "mine"
    assert signins == [("mine", False), ("mine", False)]


async def test_a_new_sign_in_clears_an_old_manual_user_token() -> None:
    """A manual user token from an earlier run would win over the new sign-in: it is cleared."""
    stored = {CONF_MUSIC_USER_MANUAL_TOKEN: _USER_TOKEN}
    session, finished = _flow_session("reconfigure", stored)
    await _run(
        session,
        [{CONF_USE_OWN_APP_TOKEN: True, CONF_OWN_APP_TOKEN: "mine"}],
        {"bundled": True, "mine": True},
    )
    assert finished[CONF_MUSIC_USER_MANUAL_TOKEN] == ""
    assert finished[CONF_MUSIC_USER_TOKEN] == _USER_TOKEN


async def test_bundled_token_rejected_asks_once_and_hides_the_switch() -> None:
    """With the bundled token unusable the token typed in its own step is used; no switch shown."""
    session, finished = _flow_session()
    signins, forms, _errors = await _run(
        session, [{CONF_MUSIC_APP_TOKEN: "typed"}, {}], {"bundled": False, "typed": True}
    )
    assert forms[0].step_id == "app_token"
    assert CONF_USE_OWN_APP_TOKEN not in _keys(forms[1])
    assert CONF_OWN_APP_TOKEN not in _keys(forms[1])
    assert finished[CONF_MUSIC_APP_TOKEN] == "typed"
    assert signins == [("typed", False)]


async def test_bundled_token_rejected_uses_the_stored_one_without_asking() -> None:
    """A stored token that still works is kept when the bundled one fails: nothing to retype."""
    stored = {CONF_MUSIC_APP_TOKEN: "mine", CONF_MUSIC_USER_TOKEN: _USER_TOKEN}
    session, finished = _flow_session("reconfigure", stored)
    signins, forms, _errors = await _run(session, [{}], {"bundled": False, "mine": True})
    assert forms[0].step_id == "user"
    assert finished[CONF_MUSIC_APP_TOKEN] == "mine"
    assert signins == [("mine", True)]


async def test_bundled_token_rejected_the_stored_one_can_still_be_replaced() -> None:
    """With the bundled token unusable the stored token is in use, and a typed one replaces it."""
    stored = {CONF_MUSIC_APP_TOKEN: "mine", CONF_MUSIC_USER_TOKEN: _USER_TOKEN}
    session, finished = _flow_session("reconfigure", stored)
    signins, forms, _ = await _run(
        session, [{CONF_OWN_APP_TOKEN: "newer"}], {"bundled": False, "mine": True, "newer": True}
    )
    assert CONF_USE_OWN_APP_TOKEN not in _keys(forms[0])
    token_field = next(e for e in forms[0].entries if e.key == CONF_OWN_APP_TOKEN)
    assert token_field.depends_on is None
    assert finished[CONF_MUSIC_APP_TOKEN] == "newer"
    assert signins == [("newer", False)]


@pytest.mark.parametrize(("prefill", "offered"), [(True, True), (False, False)])
async def test_sign_in_page_offers_the_stored_user_token_only_for_the_same_app_token(
    prefill: bool, offered: bool
) -> None:
    """The MusicKit page gets the stored user token only when asked to prefill it."""
    routes: dict[str, Any] = {}
    mass = MagicMock()
    mass.version = "2.11.0"
    mass.webserver.base_url = "http://ma.local:8095"
    mass.webserver.register_dynamic_route = lambda path, handler: (
        routes.__setitem__(path, handler) or (lambda: None)
    )
    context = SetupFlowContext(
        kind="reconfigure",  # type: ignore[arg-type]
        reason="user",  # type: ignore[arg-type]
        domain="apple_music",
        instance_id="apple_music--test",
        setup_data={CONF_MUSIC_USER_TOKEN: _USER_TOKEN},
    )
    session = SetupSession(mass, "flow-test", context, MagicMock())
    glue: dict[str, str] = {}

    async def fake_external(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        script = next(h for p, h in routes.items() if p.endswith("index.js"))
        glue["js"] = (await script(MagicMock())).text
        return {}

    with patch.object(session, "external", fake_external):
        await apple_flow._musickit_authenticate(session, "mine", prefill=prefill)
    assert (_USER_TOKEN in glue["js"]) is offered
