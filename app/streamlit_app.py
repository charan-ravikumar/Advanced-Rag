"""Entry point of the local app:  streamlit run app/streamlit_app.py

Order of a run: (1) put the project root on ``sys.path``; (2) prepare the environment (``.env`` allowlist and ``st.secrets`` allowlist into ``os.environ``; only this file and
``app/bootstrap.py`` do that); (3) the access-code gate (if ``APP_ACCESS_CODE`` is set); (4) the once-per-process resources and their SELF-CHECK: a failed check shows an error
screen, never a stack trace; (5) the four tabs. Nothing here runs at import except Streamlit's own page: the modules it imports do nothing at import.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import bootstrap, ui_about, ui_chat, ui_evaluation, ui_observability  # noqa: E402

TITLE = "Advanced RAG over Qasper papers"


def render_gate(gate, state: dict) -> None:
    st.title(TITLE)
    st.caption("This demo is protected by an access code.")
    wait = gate.seconds_locked(state)
    with st.form("access_gate"):
        code = st.text_input("Access code", type="password", key="access_code_input", disabled=wait > 0)
        submitted = st.form_submit_button("Enter", disabled=wait > 0)
    if wait > 0:
        st.warning(f"Too many wrong codes. Try again in {int(wait) + 1} seconds.")
    elif submitted:
        ok, message = gate.attempt(state, code)
        if ok or gate.seconds_locked(state) > 0:
            st.rerun()  # redraw: either the app, or the locked form (disabled input + countdown)
        else:
            st.error(message)


def render_error_screen(res) -> None:
    st.title(TITLE)
    st.error("The app cannot start because a required check failed. Nothing was served.")
    for c in res.selfcheck.checks:
        icon = "✅" if c.ok else ("❌" if c.fatal else "⚠️")
        st.markdown(f"{icon} **{c.name}**: {c.detail}")
    st.caption("Fix the failed check and press the button. Keys are checked for presence only; their values are never shown.")
    st.button("Check again")


def _safe(render, res) -> None:
    """A tab that hits an unexpected error shows its class name, not a stack trace, and does not take the other tabs down."""
    try:
        render(res)
    except Exception as exc:  # noqa: BLE001  (st.rerun and st.stop raise BaseException subclasses and pass through)
        st.error(f"This part of the page hit an unexpected error ({type(exc).__name__}). The other tabs still work.")


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    bootstrap.prepare_environment()
    settings = bootstrap.get_settings()
    gate = bootstrap.make_gate(settings)
    state = st.session_state.setdefault("__access__", gate.new_state())
    if not gate.is_open(state):
        render_gate(gate, state)
        st.stop()
    try:
        res = bootstrap.get_resources()
    except Exception as exc:  # noqa: BLE001: build_resources does not raise; this is the last resort and still shows no stack trace
        st.title(TITLE)
        st.error(f"The app could not start ({type(exc).__name__}).")
        st.stop()
    if not res.ready:
        render_error_screen(res)
        st.stop()
    st.title(TITLE)
    if not gate.required:
        st.caption("Demo mode: no access code is set, so anyone who can reach this page can use it.")
    chat, observability, evaluation, about = st.tabs(["Chat", "Observability", "Evaluation", "About"])
    with chat:
        _safe(ui_chat.render_chat, res)
    with observability:
        _safe(ui_observability.render_observability, res)
    with evaluation:
        _safe(ui_evaluation.render_evaluation, res)
    with about:
        _safe(ui_about.render_about, res)


main()
