"""The About tab: architecture (a Graphviz diagram), the key decisions INCLUDING the negative results, limitations, privacy, attribution and the versions of what is running.
Numbers in the decisions are filled from ``app_data/results.json`` at run time (none is typed here); with no artifact the sentences simply say less."""

from __future__ import annotations

import streamlit as st

ARCHITECTURE = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11]; edge [fontname="Helvetica", fontsize=9];
  subgraph cluster_offline { label="offline (run by scripts)"; style=dashed; color="#999999"; fontsize=10;
    ingest [label="Ingest\\n(Qasper papers)"]; chunk [label="Chunk\\n(offsets kept)"]; evalrun [label="Evaluation runs\\n(retrieval, generation, RAGAS)", fillcolor="#f6f0e4"];
    ingest -> chunk; }
  subgraph cluster_index { label="local index"; style=dashed; color="#999999"; fontsize=10;
    vec [label="Chroma\\n(vectors, MiniLM)"]; fts [label="SQLite FTS5\\n(lexical BM25)"]; }
  chunk -> vec; chunk -> fts;
  q [label="Question\\n(validated, guarded)", fillcolor="#e6f4ea"];
  retrieve [label="Retrieve\\n(vector + lexical)"]; fuse [label="Fuse\\n(RRF)"]; rerank [label="Rerank\\n(optional)", style="rounded,dashed,filled"]; ctx [label="Context\\n(top passages, labelled S1..)"];
  llm [label="Gemini\\n(language model)", fillcolor="#fde8e8"]; ans [label="Validated answer\\n(structured claims,\\nquotes checked,\\nabstain / degrade)", fillcolor="#e6f4ea"];
  q -> retrieve; vec -> retrieve; fts -> retrieve; retrieve -> fuse -> rerank -> ctx -> llm -> ans;
  lf [label="Langfuse\\n(OpenTelemetry traces,\\nprompts, feedback)", fillcolor="#fff7d6"]; art [label="app_data/results.json\\n(Evaluation tab)", fillcolor="#f6f0e4"];
  ans -> lf [style=dotted, label="side channel"]; lf -> ctx [style=dotted, label="prompt text", constraint=false];
  evalrun -> art [style=dotted, label="saved runs"];
}
"""

LIMITATIONS = (
    "Text evidence only: tables and figures are not indexed, so questions whose answer is in a table are not served well.",
    "English NLP papers from one dataset (Qasper). The system has not been tried on other domains or languages.",
    "Free-tier latency and limits: the language model is a free-tier API, so answers can be slow and a daily budget applies (see the sidebar).",
    "Searching all papers is much weaker than searching one paper (see the Evaluation tab).",
    "A single-user demo: one process, one machine, in-memory answer cache and rate limits that reset on restart; it runs as one process and is not built for heavy load.",
    "The model can still be wrong or follow an instruction hidden in a passage; delimiters and a system rule reduce that risk, they do not remove it.",
)

PRIVACY = (
    "Your question and the retrieved public paper passages are sent to the language-model provider (Gemini) to generate the answer.",
    "Traces are sent to Langfuse in **masked** form: a shortened question and answer with links, e-mail addresses and long numbers removed, and no passage text. "
    "Optional feedback comments are masked the same way.",
    "**Do not enter private or confidential data.** This is a demo over public research papers.",
)


def decisions(results: dict | None) -> list[str]:
    """The key decisions, with numbers read from the artifact. Negative results are stated plainly."""
    out = []
    test = (results or {}).get("retrieval_test")
    cmp_ = {c["pair"]: c for c in (test or {}).get("comparisons", [])}
    ci = lambda c: f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"  # noqa: E731
    out.append("**Hybrid retrieval (vector + lexical, fused by reciprocal rank fusion) with recursive chunks was frozen before the test split was run**, chosen as the simplest "
               "configuration among those not distinguishable from the best on the development split.")
    if "A vs D" in cmp_:
        c = cmp_["A vs D"]
        out.append(f"**Vector-only search was slightly worse than the frozen hybrid:** nDCG@10 difference {c['diff']:+.3f}, 95% interval {ci(c)}; "
                   f"{'significant' if c['holm_significant'] else 'not significant'} after the pre-registered Holm correction. This is a small effect.")
    if "F vs D" in cmp_:
        c = cmp_["F vs D"]
        out.append(f"**Negative result: semantic chunking did not beat recursive chunking.** Difference {c['diff']:+.3f}, 95% interval {ci(c)}, "
                   f"{'significant' if c['holm_significant'] else 'not significant'}; the simpler recursive chunker was kept.")
    if "D vs G" in cmp_:
        c = cmp_["D vs G"]
        out.append(f"**The reranker helps but is off by default:** nDCG@10 {-c['diff']:+.3f} with a 95% interval of [{-c['ci_high']:+.3f}, {-c['ci_low']:+.3f}] "
                   f"({'significant' if c['holm_significant'] else 'not significant'}), at a large latency cost on this machine's CPU (see the sidebar and the Evaluation tab).")
    out.append("**A lexical index in SQLite FTS5 instead of an in-memory index**, so the index persists, is verified against the vector store at start-up and survives restarts.")
    out.append("**Answers are validated, not trusted:** the model must fill a JSON schema (claims, each with the passages it relies on and a word-for-word quote from each); "
               "the code checks every cited passage against the ones that were sent and looks every quote up in its passage; the model may abstain; if the "
               "provider fails the app shows the retrieved passages instead of an error. A quote that is found shows the words are there, not that the passage supports the claim.")
    out.append("**Prompts live in Langfuse with a validated local fallback**, and every answer records which prompt version produced it.")
    if (results or {}).get("generation"):
        out.append("**Generation results are exploratory:** a small sample on the development split with no baseline and an uncalibrated judge; they are shown for transparency, not as a claim.")
    return out


def _fingerprint() -> str | None:
    try:
        from utils.fingerprint import code_fingerprint

        return code_fingerprint()
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner=False)
def code_fingerprint_of_this_app() -> str | None:
    """sha256 over every .py and .yaml file of the project (a live measurement; see ``utils/fingerprint.py``)."""
    return _fingerprint()


def render_system_check(res) -> None:
    st.markdown("#### System check")
    for c in res.selfcheck.checks:
        icon = "✅" if c.ok else ("❌" if c.fatal else "⚠️")
        st.markdown(f"{icon} **{c.name}**: {c.detail}")
    st.caption("Keys are checked for presence only; their values are never read into the page.")


def render_versions(res) -> None:
    v = res.version_info
    st.markdown("#### What is running")
    if not v:
        st.caption("The index is not open, so there is no version information.")
        return
    rows = [f"- **Index:** {v['chunk_count']} chunks · version `{v['index_version']}` · ingest manifest `{v.get('manifest')}`"
            + (f" (sha256 `{v['manifest_sha256'][:16]}`)" if v.get("manifest_sha256") else ""),
            f"- **Embedding model:** {v['embedder']['name']} @ `{v['embedder']['revision'][:12]}`",
            f"- **Reranker (optional):** {v['reranker']['name']} @ `{v['reranker']['revision'][:12]}` ({'cached' if v['reranker']['cached'] else 'not cached'})",
            f"- **Generator:** {v['generator']['model']}" + (f" at reasoning effort {v['generator']['reasoning_effort']}" if v["generator"].get("reasoning_effort") else ""),
            f"- **Prompt version:** `{v['prompt_version']}` (prompt mode: {v['prompt_mode']})",
            f"- **Retrieval:** {v['retrieval']['mode']} / {v['retrieval']['fusion']} (k {v['retrieval']['rrf_k']}, weights {v['retrieval']['vector_weight']:g} / {v['retrieval']['lexical_weight']:g}), "
            f"{v['retrieval']['top_k']} passages to the model",
            f"- **Streamlit:** {st.__version__}"]
    fp = code_fingerprint_of_this_app()
    if fp:
        rows.append(f"- **Code fingerprint of this running app:** `{fp[:16]}`")
    st.markdown("\n".join(rows))


def render_about(res) -> None:
    arts = res.artifacts
    st.subheader("About")
    st.markdown("This app answers questions about NLP research papers from the Qasper dataset. It retrieves passages with a hybrid search over a local index and asks a language "
                "model to answer **only from them**, with citations. It is a local demo built to be inspected: the Evaluation tab shows what was measured and, as importantly, what was not.")
    st.markdown("#### Architecture")
    st.graphviz_chart(ARCHITECTURE, width="stretch")
    st.markdown("#### Key decisions")
    for item in decisions(arts.results):
        st.markdown(f"- {item}")
    st.markdown("#### Limitations")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
    st.markdown("#### Privacy")
    for item in PRIVACY:
        st.markdown(f"- {item}")
    st.markdown("#### Data attribution")
    src = arts.source or {}
    if src.get("license"):
        st.markdown(f"The papers and questions come from the **Qasper** dataset ({src['license']}). [{src.get('link', '')}]({src.get('link', '')})")
    else:
        st.markdown("The papers and questions come from the **Qasper** dataset (Dasigi et al.). The license information is not available (no papers artifact).")
    render_versions(res)
    render_system_check(res)
