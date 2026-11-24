"""Per-launch access token for the desktop build.

The desktop launcher binds Streamlit to 127.0.0.1 on a random port, but any
local process (or a web page that guesses the port) could still talk to it.
The launcher therefore generates a random token, passes it to the server via
``BLOTGRID_LAUNCH_TOKEN`` and opens the window at ``/?token=<token>``. A
session that cannot present the token never runs the analysis script.

When the variable is unset (``streamlit run app.py`` from a checkout) the
guard is a no-op, so development workflows are unchanged.
"""
from __future__ import annotations

import hmac
import os
from collections.abc import Mapping, MutableMapping
from typing import Any

ENV_VAR = "BLOTGRID_LAUNCH_TOKEN"
_SESSION_KEY = "_blotgrid_launch_authorized"


def expected_token(environ: Mapping[str, str] | None = None) -> str:
    return (environ if environ is not None else os.environ).get(ENV_VAR, "")


def is_authorized(
    session_state: MutableMapping[str, Any],
    query_params: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return True when this browser session may run the app.

    Authorization is remembered in the session so later reruns do not depend
    on the query string surviving navigation.
    """
    token = expected_token(environ)
    if not token:
        return True
    if session_state.get(_SESSION_KEY) is True:
        return True
    supplied = query_params.get("token", "")
    if isinstance(supplied, (list, tuple)):
        supplied = supplied[0] if supplied else ""
    if isinstance(supplied, str) and supplied and hmac.compare_digest(
        supplied.encode("utf-8"), token.encode("utf-8")
    ):
        session_state[_SESSION_KEY] = True
        return True
    return False


def enforce() -> None:
    """Stop the Streamlit script unless the session is authorized."""
    import streamlit as st

    if is_authorized(st.session_state, st.query_params):
        if expected_token() and "token" in st.query_params:
            # Keep the secret out of the visible URL and the browser history.
            del st.query_params["token"]
        return
    st.error(
        "This BlotGrid window is not authorized. Open BlotGrid from its desktop "
        "launcher instead of typing the local address into a browser."
    )
    st.stop()
