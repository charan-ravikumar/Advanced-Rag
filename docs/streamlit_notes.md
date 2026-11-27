# Streamlit notes (Phase 5: the local app)

Everything here was checked against the installed package (`inspect` on signatures and docstrings) and by small probes with `AppTest`; the doc pages
(https://docs.streamlit.io/develop/api-reference/) were read for behaviour that is not visible in a signature. No performance number appears here.

## Version

- **streamlit 1.67.0**: the latest release on the configured pip index (`pip index versions streamlit` lists 1.67.0 first) and the one already installed in the main venv.
  Pinned in `docs/env_baseline.txt` (it was there) and now in `requirements.txt`, together with the parts of its dependency closure that were missing from that list
  (altair, narwhals, pydeck, python-multipart, tzdata, watchdog; `exceptiongroup` is not needed on Python 3.13).
- The app runs in the MAIN venv (`%LOCALAPPDATA%\advrag\venv`). RAGAS (`venv_eval`) is not involved.

## APIs the app uses (signatures as of 1.67.0)

| API | Facts that shaped the code |
|---|---|
| `st.write_stream(stream, *, cursor=None)` | Takes a generator or any iterable of strings; returns the full text (a string) when the stream ends. Renders in the current container, so the app streams into an `st.empty()` placeholder inside the assistant message and replaces it with the final render. |
| `st.status(label, *, expanded=False, state="running")` | A container with `.update(label=, state=, expanded=)`; used as the "thinking" indicator until the first token. |
| `st.cache_resource(ttl=, max_entries=, show_spinner=, validate=, on_release=, scope="global", ...)` | One object shared by ALL sessions and reruns of the process, NOT copied: mutations hit the cache. Used once for the bootstrap (store, FTS, embedder, retriever, service, guards). Because every session thread shares it, the retriever is wrapped in a lock and the FTS connection is created with `check_same_thread=False` (SQLite objects are otherwise bound to the creating thread). |
| `st.cache_data(ttl=, max_entries=, show_spinner=, persist=, scope="global", ...)` | Returns a COPY of a serialisable value per call; arguments are hashed, an argument whose name starts with an underscore is NOT hashed (used for the reader object). The Observability snapshot is cached here with `ttl` from `config.yaml`. |
| `st.secrets` | Reads `.streamlit/secrets.toml`. **If no secrets file exists, ANY access raises** (`StreamlitSecretNotFoundError`, a `FileNotFoundError`), also `.get()`: every read is wrapped. Root-level secrets are also exported as environment variables by Streamlit itself; the app additionally copies an explicit allowlist (never overriding a variable that is already set). |
| `st.feedback("thumbs", key=, on_change=, args=, disabled=)` | Returns `None` (nothing selected), `0` (thumbs down) or `1` (thumbs up). `on_change` is a callback run before the rerun; the app sends the score from the callback. |
| `st.graphviz_chart(dot_string, *, width=, height=)` | Accepts a plain DOT string (no `graphviz` Python package or system binary needed: it is rendered in the browser). |
| `st.chat_input(placeholder, *, max_chars=, disabled=, key=)` | Returns the submitted text only in the run in which it was submitted, so the app stores the finished turn in `st.session_state` and calls `st.rerun()` to render it from the stored result (a later click on a thumb reruns the script and must not lose the answer). |
| `st.altair_chart(chart, *, width="stretch", theme="streamlit")` and `st.dataframe(..., width="stretch")` | `use_container_width` is deprecated in favour of `width="stretch"`. |
| `st.tabs`, `st.expander`, `st.toggle`, `st.selectbox`, `st.text_input(type="password")` | As expected; every widget needs a `key` when its value is read from `session_state`. |

## Testing with `streamlit.testing.v1.AppTest`

- `AppTest.from_file(path, default_timeout=...)` runs the script in-process; **the default timeout of 3 s is too short for a cold start** (the probe timed out): the tests use 30 s.
- `from_function` cannot be used from stdin or a REPL (it needs source lines); `from_string` and `from_file` work.
- Supported accessors used by the tests: `at.chat_input[0].set_value(...)`, `at.feedback[0].set_value(1)`, `at.selectbox`, `at.toggle`, `at.text_input`, `at.tabs`, `at.markdown`, `at.caption`, `at.warning`, `at.error`, `at.info`, `at.expander`, `at.exception`, `at.session_state`, `at.secrets` (assignable). `st.write_stream`, `st.cache_resource`, `st.status` and `st.graphviz_chart` run under `AppTest`.
- Process-wide caches (`cache_resource`, `cache_data`) survive between `AppTest` instances in one pytest process, so a fixture clears both before every test.
- Dependency injection in tests: the entry point asks `app.bootstrap.get_resources()`, which returns an object stored under a session-state key if a test put one there (no real index, no network); in production that key is never set.

## Things that bit others (kept here)

- A cached resource created in one session's thread and used from another raises for SQLite and similar thread-bound objects: lock and `check_same_thread=False`.
- `st.secrets` raises when no secrets file exists; guard it.
- A script rerun happens on every widget interaction, so anything with a cost (a model call) must be triggered by the submission only, and results kept in `session_state`.
- Importing `streamlit_app.py` (the legacy root file) pulls in the deprecated `rag.retrieval`; the new entry point is `app/streamlit_app.py` and imports nothing legacy.
