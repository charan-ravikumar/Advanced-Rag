"""The Chat tab. ``res.service`` is an adapter over the HTTP API (``app.remote.RemoteService``, streaming ``POST /v1/ask``): the events it yields are the same dicts
``RagService.answer_stream`` yields, so this module neither knows nor cares whether the service is in-process or remote.

Layers, so the logic can be tested without a browser:
* PURE helpers: ``normalise_markers`` (``【S1】`` and ``［S1］`` become ``[S1]``), ``MarkerNormaliser`` (the same for a token stream: a marker split across deltas is held back until it
  is complete), ``style_markers`` (markers in the SHOWN text come from the PARSED result: a valid marker is bold, an invalid one is dropped and reported quietly),
  ``classify`` (which of the states a finished ``AnswerResult`` is in), ``escape_md``.
* ``submit``: guards, the daily budget, streaming into the page, then the finished turn is stored in ``st.session_state`` and the page reruns to render it from the stored result
  (a later click on a thumb reruns the script and must not lose or repeat the answer).
* ``render_turn`` and friends: states, source cards, feedback, details.

Every question is independent: no conversation memory. Numbers shown here (the reranker's cost, how much weaker corpus-wide search is) are read from ``app_data/results.json``
through ``app.artifacts``; the numbers of the current answer (latencies, tokens) are measured for that answer.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import streamlit as st

from app.artifacts import cross_paper_facts, fmt_ms, rerank_facts
from app.guards import SessionGuard, check_question
from rag_core.errors import InvalidQuestion
from rag_core.schemas import AnswerResult, Source

SESSION_GUARD_KEY = "__session_guard__"
TURN_KEY = "turn"
SINGLE_LABEL = "A single paper"
_OPEN = "[\u3010\uff3b"
_CLOSE = "]\u3011\uff3d"
_MARKER = re.compile(rf"[{re.escape(_OPEN)}]\s*(S\d+(?:\s*,\s*S\d+)*)\s*[{re.escape(_CLOSE)}]")
_ASCII_MARKER = re.compile(r"\[(S\d+(?:, S\d+)*)\]")
MAX_HELD = 16  # the longest an unfinished marker can be before it is not a marker (``[S12, S13]`` is 10 characters)


# ---- pure helpers ---------------------------------------------------------------------------------------------------------------------

def escape_md(text: str | None) -> str:
    """``$`` would start LaTeX in Streamlit's markdown: escape it (papers and answers contain dollar signs and formulas)."""
    return (text or "").replace("$", "\\$")


def normalise_markers(text: str) -> str:
    """``【S1】``, ``［S1］`` and ``[ S1 ]`` become ``[S1]``; ``【S1, S2】`` becomes ``[S1, S2]``."""
    return _MARKER.sub(lambda m: "[" + re.sub(r"\s*,\s*", ", ", m.group(1)) + "]", text or "")


class MarkerNormaliser:
    """``normalise_markers`` for a stream of deltas. Text is emitted as soon as it cannot be part of a marker; an unfinished marker (an opening bracket with no closing one yet,
    at most ``MAX_HELD`` characters) is held back until it is complete. ``flush`` emits what is left."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        cut = max((self._buf.rfind(c) for c in "[" + _OPEN), default=-1)
        if cut >= 0 and not any(c in self._buf[cut:] for c in "]" + _CLOSE) and len(self._buf) - cut <= MAX_HELD:
            emit, self._buf = self._buf[:cut], self._buf[cut:]
        else:
            emit, self._buf = self._buf, ""
        return normalise_markers(emit)

    def flush(self) -> str:
        out, self._buf = normalise_markers(self._buf), ""
        return out


def style_markers(text: str, invalid: Iterable[str] = ()) -> str:
    """On NORMALISED text: a valid marker is shown in bold (``**[S1]**``); invalid labels (named by the parsed result) are dropped from the text."""
    bad = set(invalid)

    def repl(m: re.Match) -> str:
        keep = [label.strip() for label in m.group(1).split(",") if label.strip() not in bad]
        if not keep:
            return ""
        before = m.string[m.start() - 1] if m.start() else ""
        after = m.string[m.end()] if m.end() < len(m.string) else ""
        # a marker touching another emphasis run ("**ROUGE-L**[S1]") would make "****" and break both runs: keep them apart
        lead = " " if before in "*_" and before else ""
        trail = " " if after in "*_" and after else ""
        return f"{lead}**[{', '.join(keep)}]**{trail}"

    return _ASCII_MARKER.sub(repl, text)


def answer_markdown(result: AnswerResult) -> str:
    """The answer as shown: markers normalised and styled from the PARSED result, ``$`` escaped."""
    return style_markers(escape_md(normalise_markers(result.answer)), result.invalid_citations)


def classify(result: AnswerResult) -> str:
    """One of ``direct`` (small talk), ``budget`` (daily budget reached: passages only), ``degraded`` (no answer could be generated), ``partial`` (the stream broke), ``abstained``, ``answer``."""
    if result.mode == "direct":
        return "direct"
    codes = {e.code for e in result.errors}
    if "daily_budget_reached" in codes:
        return "budget"
    if result.degraded:
        return "degraded"
    if result.partial:
        return "partial"
    if result.abstained:
        return "abstained"
    return "answer"


@dataclass
class Turn:
    """One finished question. Stored in ``st.session_state`` so reruns (a click on a thumb) render it again without calling anything."""

    id: str
    question: str
    scope: str  # "paper" | "all"
    paper_id: str | None
    rerank: bool
    kind: str = "answer"  # "invalid" | "refused" | classify(result)
    message: str = ""  # invalid / refused: what to tell the user
    result: AnswerResult | None = None


@dataclass
class Capture:
    sources: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    result: AnswerResult | None = None
    got_token: bool = False


def token_stream(events: Iterable[dict], capture: Capture, on_first_token=None) -> Iterator[str]:
    """Text deltas for ``st.write_stream`` from ``RagService.answer_stream`` events; sources, errors and the final result go into ``capture``."""
    norm = MarkerNormaliser()
    for event in events:
        kind = event["type"]
        if kind == "sources":
            capture.sources = list(event["sources"])
        elif kind == "token":
            if not capture.got_token:
                capture.got_token = True
                if on_first_token:
                    on_first_token()
            out = escape_md(norm.feed(event["text"]))
            if out:
                yield out
        elif kind == "error":
            capture.errors.append(event["error"])
        elif kind == "done":
            capture.result = event["result"]
    tail = escape_md(norm.flush())
    if tail:
        yield tail


# ---- submit -----------------------------------------------------------------------------------------------------------------------------

def session_guard(res) -> SessionGuard:
    guard = st.session_state.get(SESSION_GUARD_KEY)
    if guard is None:
        s = res.settings
        guard = SessionGuard(max_requests=s.rate_limit_requests, window_s=s.rate_limit_window_seconds, session_cap=s.session_request_cap)
        st.session_state[SESSION_GUARD_KEY] = guard
    return guard


def _used_no_llm(result: AnswerResult) -> bool:
    """True if the request made no LLM call (cache hit, small talk, a failure before the model): its unit of the daily budget is given back."""
    return result.cached or result.provider in (None, "rules")


def submit(res, question: str, scope: str, paper_id: str | None, rerank: bool) -> None:
    """Handle one submitted question end to end; the outcome is stored as the current turn (``st.session_state[TURN_KEY]``)."""
    service = res.service
    turn = Turn(id=uuid.uuid4().hex[:12], question=question, scope=scope, paper_id=paper_id, rerank=rerank)
    st.session_state[TURN_KEY] = turn
    try:
        cleaned = check_question(service, question)
    except InvalidQuestion as exc:
        turn.kind, turn.message = "invalid", exc.message
        return
    guard = session_guard(res)
    decision = guard.check()
    if not decision.allowed:
        turn.kind, turn.message = "refused", decision.message
        return
    guard.record()
    doc_ids = [paper_id] if scope == "paper" and paper_id else None
    if not res.daily.try_acquire():  # the process-wide daily cap: no LLM call, the retrieved passages only
        turn.result = service.passages_only(cleaned, doc_ids, rerank)
        turn.kind = classify(turn.result)
        return
    capture = Capture()
    with st.chat_message("user"):
        st.markdown(escape_md(question))
    with st.chat_message("assistant"):
        status = st.status("Searching the papers…", expanded=False)
        slot = st.empty()

        def first_token() -> None:
            status.update(label="Answering…", state="running")

        with slot.container():
            st.write_stream(token_stream(service.answer_stream(cleaned, doc_ids=doc_ids, rerank=rerank, session_id=st.session_state.get("session_id")),
                                         capture, first_token))
        status.update(label="Done", state="complete")
    result = capture.result
    if result is None:  # the stream ended without a final result (should not happen): report it, never crash
        res.daily.release()
        turn.kind, turn.message = "refused", "The answer ended unexpectedly. Please try again."
        return
    if _used_no_llm(result):
        res.daily.release()
    turn.result, turn.kind = result, classify(result)
    try:
        service.flush_traces()
    except Exception:  # noqa: BLE001: tracing must never break the page
        pass


# ---- rendering ---------------------------------------------------------------------------------------------------------------------------

def _source_cards(sources: list[Source], titles: dict[str, str], cited: Iterable[str] = ()) -> None:
    cited_set = set(cited)
    for s in sources:
        label = s.label or f"S{s.rank}"
        with st.container(border=True):
            badge = "  ·  :green[cited in the answer]" if label in cited_set else ""
            st.markdown(f"**[{label}]**  {escape_md(titles.get(s.doc_id) or s.doc_id)}  ·  *{escape_md(s.section or 'section unknown')}*  ·  rank {s.rank}{badge}")
            with st.expander("Passage", expanded=False):
                st.markdown(escape_md(s.text))


def _details(res, result: AnswerResult) -> None:
    u, lat = result.usage, result.latency_ms
    dash = lambda v: "—" if v is None else v  # noqa: E731
    ms = lambda v: "—" if v is None else fmt_ms(v)  # noqa: E731
    prompt = result.prompt
    with st.expander("Details of this answer", expanded=False):
        st.markdown("\n".join([
            f"- **Provider / model:** {dash(result.provider)} / {dash(result.model)}",
            f"- **Reasoning effort:** {dash(result.reasoning_effort)}",
            f"- **Prompt:** {'—' if prompt is None else f'{prompt.name} · version {prompt.version} · source {prompt.source}'}",
            f"- **Trace id:** {dash(result.trace_id) if result.trace_id else 'not traced'}",
            f"- **Latency (measured for this answer):** retrieval {ms(lat.retrieve)} · first token {ms(lat.first_token)} · total {ms(lat.total)}",
            f"- **Tokens:** prompt {dash(u.prompt_tokens)} · completion {dash(u.completion_tokens)} (of which reasoning {dash(u.reasoning_tokens)})",
            f"- **Served from the answer cache:** {'yes' if result.cached else 'no'}",
            f"- **Retrieved / sent to the model:** {len(result.retrieved_sources) or len(result.used_sources)} / {len(result.used_sources)} passages"]))


def _feedback(res, turn: Turn) -> None:
    result = turn.result
    request_id = result.request_id if result else None
    trace_id = result.trace_id if result else None
    fb_key, comment_key, sent_key = f"fb_{turn.id}", f"fbc_{turn.id}", f"fb_sent_{turn.id}"
    available = bool(request_id) and res.record_feedback is not None

    def send() -> None:
        value = st.session_state.get(fb_key)
        if value is None or not available:
            return
        outcome = res.record_feedback(request_id, trace_id, int(value), st.session_state.get(comment_key) or None)
        st.session_state[sent_key] = "forwarded" if outcome.forwarded else "saved" if outcome.saved else "failed"

    st.text_input("Optional comment (links, e-mail addresses and long numbers are masked before it is saved or sent)", key=comment_key, max_chars=300, disabled=not available)
    st.feedback("thumbs", key=fb_key, on_change=send, disabled=not available)
    state = st.session_state.get(sent_key)
    if state == "forwarded":
        st.caption("Thanks, your feedback was recorded.")
    elif state == "saved":
        st.caption("Thanks, your feedback was saved." + ("" if trace_id else " This answer was not traced, so it stays on this server."))
    elif state == "failed":
        st.caption("Your feedback could not be recorded.")


def render_turn(res, turn: Turn) -> None:
    titles = res.artifacts.titles
    with st.chat_message("user"):
        st.markdown(escape_md(turn.question))
    with st.chat_message("assistant"):
        if turn.kind == "invalid":
            st.warning(turn.message)
            return
        if turn.kind == "refused":
            st.info(turn.message)
            return
        result = turn.result
        kind = turn.kind
        if kind == "direct":
            st.markdown(escape_md(result.answer))
            return
        if kind in ("budget", "degraded"):
            st.warning(result.answer)
            if result.retrieved_sources:
                st.caption("Passages that matched your question (no answer was generated):")
                _source_cards(result.retrieved_sources, titles)
            _details(res, result)
            return
        if kind == "abstained":
            st.info("The retrieved passages do not contain enough information to answer this question, so no answer was given instead of a guess. "
                    "The passages that were retrieved are shown below; they may still help.")
            _source_cards(result.retrieved_sources or result.used_sources, titles)
        else:
            if result.cached:
                st.caption("Served from the answer cache (no model call).")
            st.markdown(answer_markdown(result))
            if kind == "partial":
                st.warning("The connection to the model broke while the answer was being written, so this answer is incomplete.")
            if result.citations:
                st.caption("Citations in brackets match the source cards below. A citation shows which passage the model pointed at; it does not prove that the passage supports the claim.")
            if result.invalid_citations:
                st.caption(f"Some citation markers ({', '.join(result.invalid_citations)}) did not match any source and were left out.")
            _source_cards(result.used_sources, titles, cited=result.citations)
        _feedback(res, turn)
        _details(res, result)


def _pick_example(example: dict) -> None:
    st.session_state["scope_label"] = SINGLE_LABEL
    st.session_state["paper_id"] = example["paper_id"]
    st.session_state["pending_question"] = example["question"]


def render_sidebar(res) -> bool:
    """The reranker toggle (OFF by default) with its measured cost, and the live budget. Returns whether reranking is on."""
    facts = rerank_facts(res.artifacts.results)
    with st.sidebar:
        st.subheader("Settings")
        st.toggle("Rerank the top passages", value=False, key="rerank", disabled=not res.reranker_available,
                  help="Re-scores the 20 best passages with a cross-encoder before choosing the ones the model sees.")
        if not res.reranker_available:
            st.caption("The reranker model is not in the local cache, so this option is unavailable.")
        elif facts:
            gain_note = (" (the difference is larger than chance by the pre-registered test)" if facts["holm_significant"] else " (not distinguishable from chance)")
            st.caption(f"Measured on the {facts['scope']['label']} split ({facts['scope']['n_questions']} questions): nDCG@10 {facts['ndcg_without']:.3f} without, "
                       f"{facts['ndcg_with']:.3f} with{gain_note}. It adds about {fmt_ms(facts['rerank_p50_ms'])} (median) per question on this machine's CPU.")
        else:
            st.caption("The latency cost of reranking is not available (no evaluation artifact).")
        st.caption(f"Passages sent to the model per question: {res.service.cfg.top_k} (a design default, not a tuned value).")
        guard = st.session_state.get(SESSION_GUARD_KEY)
        remaining_session = res.settings.session_request_cap if guard is None else guard.remaining
        st.caption(f"Questions left in this session: {remaining_session}. Answers left today (all visitors): {res.daily.remaining}.")
    return bool(st.session_state.get("rerank", False)) and res.reranker_available


def render_chat(res) -> None:
    arts = res.artifacts
    rerank = render_sidebar(res)
    n_papers = len(arts.papers)
    st.caption("Each question is answered on its own: there is no conversation memory. Ask about public research (Qasper NLP papers); do not enter private data.")
    all_label = f"All {n_papers} papers" if n_papers else "All papers"
    scope_label = st.radio("Search scope", [SINGLE_LABEL, all_label], key="scope_label", horizontal=True)
    scope = "paper" if scope_label == SINGLE_LABEL else "all"
    paper_id = None
    if scope == "paper":
        ids = sorted(arts.titles, key=lambda i: arts.titles[i].lower())
        if ids:
            if st.session_state.get("paper_id") not in ids:
                st.session_state["paper_id"] = ids[0]
            paper_id = st.selectbox("Paper", ids, key="paper_id", format_func=lambda i: arts.titles[i])
        else:
            st.warning("The list of papers is not available (app_data/papers.json is missing), so a single paper cannot be chosen. Search all papers instead.")
    else:
        facts = cross_paper_facts(arts.results)
        if facts:
            sc = facts["scope"]
            st.caption(f"Searching all papers is measurably weaker than searching one: on the {sc['label']} split ({sc['n_questions']} questions), the retrieval put a relevant passage "
                       f"in the top 5 for {facts['corpus_hit5']:.0%} of the questions when all papers were searched, against {facts['paper_hit5']:.0%} when the paper was given "
                       f"(the right paper was among the top 5 results for {facts['corpus_doc_hit5']:.0%}). See the Evaluation tab.")
        else:
            st.caption("Searching all papers is weaker than searching one; the measurement is not available (no evaluation artifact).")
    if arts.examples:
        st.caption("Examples (public questions from the dev set; they were chosen because the system answered them with citations, so they are not representative):")
        cols = st.columns(len(arts.examples))
        for i, (col, ex) in enumerate(zip(cols, arts.examples)):
            col.button(ex["question"], key=f"example_{i}", on_click=_pick_example, args=(ex,), help=f"Paper: {ex.get('paper_title', '')}", width="stretch")
    st.session_state.setdefault("session_id", uuid.uuid4().hex)

    typed = st.chat_input("Ask a question about the papers", max_chars=res.service.cfg.max_question_chars)
    question = typed or st.session_state.pop("pending_question", None)
    if question:
        submit(res, question, scope, paper_id, rerank)
        st.rerun()
    turn = st.session_state.get(TURN_KEY)
    if turn is not None:
        render_turn(res, turn)
