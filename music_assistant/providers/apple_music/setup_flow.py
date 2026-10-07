"""Setup flow for the Apple Music provider."""

from __future__ import annotations

import json
import pathlib
import re
from typing import TYPE_CHECKING

from aiohttp import ClientError, ClientTimeout, web
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from music_assistant.constants import CONF_ENTRY_UNOFFICIAL_PROVIDER
from music_assistant.models.setup_flow import AbortFlow, SetupFlowError

from .constants import (
    CONF_MUSIC_APP_TOKEN,
    CONF_MUSIC_USER_MANUAL_TOKEN,
    CONF_MUSIC_USER_TOKEN,
    CONF_MUSIC_USER_TOKEN_TIMESTAMP,
    CONF_OWN_APP_TOKEN,
    CONF_USE_OWN_APP_TOKEN,
    MUSIC_APP_TOKEN,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType

    from music_assistant import MusicAssistant
    from music_assistant.models.setup_flow import SetupSession

# the MusicKit page self-closes and posts its (possibly empty) token this many seconds
# before the server-side external-step deadline, so the flow always resumes with params
MUSICKIT_FLOW_TIMEOUT = 600


async def run_setup(session: SetupSession) -> None:
    """
    Run the Apple Music setup flow.

    Resolves the Apple Music developer (app) token - the bundled one, the user's own when they
    opt in (an advanced option), or a manual one when the bundle is empty/expired - then obtains
    a music user token via the MusicKit browser sign-in (or an advanced manual override) and
    persists them as setup data.

    :param session: The setup session driving the flow.
    """
    mass = session.mass
    setup_data = session.context.setup_data
    collected: dict[str, ConfigValueType] = {}
    # the developer token the stored user token was issued with (the user's own, or one asked
    # for when the bundled token failed); empty: the bundled one
    stored_app_token = str(setup_data.get(CONF_MUSIC_APP_TOKEN) or "")
    app_token = MUSIC_APP_TOKEN
    asked = False
    # a throttled /v1/test says nothing about validity, so only an explicit rejection counts
    if await _app_token_accepted(mass, app_token) is False:
        # the bundle ships an empty/expired token (e.g. on development builds): a stored token
        # that still works is kept (and may be replaced below), else the user is asked for one
        if stored_app_token and await _app_token_accepted(mass, stored_app_token) is not False:
            app_token = stored_app_token
        else:
            app_token, asked = await _ask_app_token(session), True
        collected[CONF_MUSIC_APP_TOKEN] = app_token

    errors: dict[str, str | SetupFlowError] | None = None
    # the user's own token, kept across a re-rendered form (a secure value never reaches the
    # form, so the switch says whether one is kept and an empty field keeps it)
    own_on, own_kept = bool(stored_app_token), stored_app_token
    while True:
        bundled_ok = CONF_MUSIC_APP_TOKEN not in collected
        entries = _user_step_entries(own_on, bundled_ok, show_token=bundled_ok or not asked)
        user_values = await session.form(entries, step_id="user", errors=errors)
        attempt = dict(collected)
        typed = str(user_values.get(CONF_OWN_APP_TOKEN) or "").strip()
        if bundled_ok:
            own_on = bool(user_values.get(CONF_USE_OWN_APP_TOKEN))
        else:
            # the bundled token is unusable and a token is in use: a typed one replaces a stored
            # one; one just asked for stays as it is
            own_on, own_kept = bool(typed) and not asked, str(collected[CONF_MUSIC_APP_TOKEN])
        if own_on:
            own_token = typed or own_kept
            # a new token must be positively accepted; a kept one only must not be rejected
            accepted = await _app_token_accepted(mass, own_token) if own_token else False
            if accepted is False or (accepted is None and own_token != own_kept):
                errors = {CONF_OWN_APP_TOKEN: "invalid_value"}
                continue
            own_kept = app_token = attempt[CONF_MUSIC_APP_TOKEN] = own_token
            if not bundled_ok:
                collected[CONF_MUSIC_APP_TOKEN] = own_token
        elif bundled_ok:
            app_token = MUSIC_APP_TOKEN
            if stored_app_token:
                attempt[CONF_MUSIC_APP_TOKEN] = ""
        manual_token = str(user_values.get(CONF_MUSIC_USER_MANUAL_TOKEN) or "").strip()
        if manual_token:
            # advanced escape hatch: a manual user token skips the browser sign-in
            # (e.g. child accounts, where MusicKit authorize() is unavailable)
            attempt[CONF_MUSIC_USER_MANUAL_TOKEN] = manual_token
        else:
            # a user token belongs to the developer token it was issued with: offer the stored
            # one again only when that developer token is still the one in use
            same_app_token = app_token == (stored_app_token or MUSIC_APP_TOKEN)
            attempt.update(await _sign_in(session, app_token, prefill=same_app_token))
            if setup_data.get(CONF_MUSIC_USER_MANUAL_TOKEN):
                # a manual token from an earlier run is read before the signed-in one, and may
                # belong to another developer token: this sign-in replaces it
                attempt[CONF_MUSIC_USER_MANUAL_TOKEN] = ""
        try:
            await session.finish(attempt)
            return
        except SetupFlowError as err:
            errors = {"base": err}


def _user_step_entries(own_on: bool, bundled_ok: bool, show_token: bool) -> list[ConfigEntry]:
    """
    Return the fields of the sign-in ("user") step.

    :param own_on: Whether the user's own developer token is switched on.
    :param bundled_ok: Whether the bundled developer token works (then the switch is offered).
    :param show_token: Whether to offer the own developer token field.
    """
    entries = [CONF_ENTRY_UNOFFICIAL_PROVIDER]
    if bundled_ok:
        # the bundled token works, but every installation shares it, and with it Apple's
        # per-account rate limit: advanced users may bring their own instead
        entries.append(
            ConfigEntry(
                key=CONF_USE_OWN_APP_TOKEN,
                type=ConfigEntryType.BOOLEAN,
                default_value=False,
                required=False,
                advanced=True,
                value=own_on,
            )
        )
    if show_token:
        # with the bundled token unusable a stored token is in use: it may still be replaced
        entries.append(
            ConfigEntry(
                key=CONF_OWN_APP_TOKEN,
                type=ConfigEntryType.SECURE_STRING,
                required=False,
                advanced=True,
                depends_on=CONF_USE_OWN_APP_TOKEN if bundled_ok else None,
                help_link="https://www.music-assistant.io/music-providers/apple-music/",
            )
        )
    entries.append(
        ConfigEntry(
            key=CONF_MUSIC_USER_MANUAL_TOKEN,
            type=ConfigEntryType.SECURE_STRING,
            required=False,
            advanced=True,
            help_link="https://www.music-assistant.io/music-providers/apple-music/",
        )
    )
    return entries


async def _sign_in(
    session: SetupSession, app_token: str, prefill: bool
) -> dict[str, ConfigValueType]:
    """
    Sign in with MusicKit in the browser and return the music user token to store.

    :param session: The setup session driving the flow.
    :param app_token: The developer token to sign in with.
    :param prefill: Offer the stored user token to the page (only valid for the same app token).
    """
    params = await _musickit_authenticate(session, app_token, prefill=prefill)
    token = params.get("music-user-token")
    if not token:
        # the page closed or timed out without returning a token
        raise AbortFlow("auth_cancelled")
    # the callback stringifies posted values, so coerce the timestamp back to an
    # int; CONF_MUSIC_USER_TOKEN_TIMESTAMP is stored as INTEGER, not encrypted
    return {
        CONF_MUSIC_USER_TOKEN: token,
        CONF_MUSIC_USER_TOKEN_TIMESTAMP: int(params.get("music-user-token-timestamp", 0)),
    }


async def _ask_app_token(session: SetupSession) -> str:
    """
    Ask for a developer token until Apple accepts one (the bundled one is unusable).

    :param session: The setup session driving the flow.
    """
    errors: dict[str, str | SetupFlowError] | None = None
    while True:
        values = await session.form(
            [
                ConfigEntry(
                    key=CONF_MUSIC_APP_TOKEN,
                    type=ConfigEntryType.SECURE_STRING,
                    required=True,
                )
            ],
            step_id="app_token",
            errors=errors,
        )
        app_token = str(values.get(CONF_MUSIC_APP_TOKEN) or "")
        # a typed token must be positively accepted, not merely un-rejected
        if await _app_token_accepted(session.mass, app_token):
            return app_token
        errors = {CONF_MUSIC_APP_TOKEN: "invalid_value"}


async def _app_token_accepted(mass: MusicAssistant, app_token: str) -> bool | None:
    """
    Return whether the API accepted the given Apple Music developer (app) token.

    True when accepted, False when rejected, None when inconclusive (throttled/unreachable).

    :param mass: The MusicAssistant instance.
    :param app_token: The developer (app) token to validate.
    """
    if not app_token:
        return False
    try:
        async with mass.http_session.get(
            "https://api.music.apple.com/v1/test",
            headers={"Authorization": f"Bearer {app_token}"},
            ssl=True,
            timeout=ClientTimeout(total=10),
        ) as response:
            if response.status == 200:
                return True
            return False if response.status in (401, 403) else None
    except ClientError, TimeoutError:
        return None


def _validate_user_token(token: ConfigValueType) -> bool:
    """
    Return whether the given value looks like a (base64) Apple Music user token.

    :param token: The candidate music user token to check.
    """
    if not isinstance(token, str):
        return False
    return bool(re.findall(r"[a-zA-Z0-9=/+]{32,}==$", token))


async def _musickit_authenticate(
    session: SetupSession, app_token: str, prefill: bool = True
) -> dict[str, str]:
    """
    Serve the MusicKit JS sign-in page and return the params posted back to the flow.

    Registers the (flow-scoped) page/style/glue routes, sends the user to the page via an
    external step and always unregisters the routes afterwards. The page must stay
    MA-hosted: MusicKit's authorize() popup and postMessage need a real HTTP origin that
    can reach the local http callback.

    :param session: The setup session driving the flow.
    :param app_token: The (validated) developer token the MusicKit page configures with.
    :param prefill: Offer the stored user token to the page (only valid for the same app token).
    """
    mass = session.mass
    asset_dir = pathlib.Path(__file__).parent.joinpath("musickit_auth")
    stored = session.context.setup_data if prefill else {}
    prefill_token = stored.get(CONF_MUSIC_USER_TOKEN)
    user_token = prefill_token if _validate_user_token(prefill_token) else ""
    user_token_timestamp = (stored.get(CONF_MUSIC_USER_TOKEN_TIMESTAMP) or 0) if user_token else 0

    async def serve_mk_auth_page(request: web.Request) -> web.FileResponse:  # noqa: ARG001
        return web.FileResponse(
            asset_dir.joinpath("musickit_wrapper.html"),
            headers={"content-type": "text/html"},
        )

    async def serve_mk_auth_css(request: web.Request) -> web.FileResponse:  # noqa: ARG001
        return web.FileResponse(
            asset_dir.joinpath("musickit_wrapper.css"),
            headers={"content-type": "text/css"},
        )

    def _js_str(value: object) -> str:
        # json.dumps yields a quoted, escaped JS string literal; escaping "<" also
        # neutralizes a "</script>" inside user-supplied values (manual tokens)
        return json.dumps(str(value)).replace("<", "\\u003c")

    async def serve_mk_glue(request: web.Request) -> web.Response:  # noqa: ARG001
        glue = f"""
        const return_url={_js_str(session.callback_url)};
        const app_token={_js_str(app_token)};
        const callback_method='POST';
        const user_token={_js_str(user_token)};
        const user_token_timestamp={_js_str(user_token_timestamp)};
        const flow_timeout={max(MUSICKIT_FLOW_TIMEOUT - 10, 60)};
        const mass_version={_js_str(mass.version)};
        """
        return web.Response(body=glue, headers={"content-type": "text/javascript"})

    base_path = f"/apple_music_auth/{session.flow_id}/"
    unregister = [
        mass.webserver.register_dynamic_route(f"{base_path}index.html", serve_mk_auth_page),
        mass.webserver.register_dynamic_route(f"{base_path}index.css", serve_mk_auth_css),
        mass.webserver.register_dynamic_route(f"{base_path}index.js", serve_mk_glue),
    ]
    try:
        return await session.external(
            f"{mass.webserver.base_url}{base_path}index.html",
            step_id="auth",
            expires_in=MUSICKIT_FLOW_TIMEOUT,
        )
    finally:
        for remove in unregister:
            remove()
