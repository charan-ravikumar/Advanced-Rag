# ============================================
# ADVANCED RAG — STREAMLIT UI
# ============================================

import time
import streamlit as st

# ============================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================

st.set_page_config(
    page_title="Advanced RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown(
    """
<style>
/* ─── Global ─────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1f2637;
}

/* ─── Typography ──────────────────────────── */
h1 { font-size: 1.5rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h3 { font-size: 0.9rem !important; font-weight: 600 !important; color: #8b92a5 !important;
     text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }

/* ─── Chip row ────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
    letter-spacing: 0.3px;
}
.chip-on  { background: #1a3a5c; color: #60b4ff; border: 1px solid #1e4f80; }
.chip-off { background: #1e1e2e; color: #555a6e; border: 1px solid #2a2d3e; }
.chip-hybrid { background: #1a3a2c; color: #4cde96; border: 1px solid #1e5040; }
.chip-vector { background: #2a2240; color: #b97bff; border: 1px solid #3d2e60; }
.chip-bm25   { background: #3a2c14; color: #ffb84d; border: 1px solid #5a4020; }

/* ─── Source card ─────────────────────────── */
.src-card {
    background: #161b27; border: 1px solid #1f2637; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 8px;
}
.src-header { display: flex; justify-content: space-between; align-items: center;
              margin-bottom: 6px; }
.src-title { font-size: 12px; font-weight: 600; color: #c0c8e0; }
.src-score { font-size: 11px; color: #4cde96; font-variant-numeric: tabular-nums; }
.src-meta { font-size: 11px; color: #555a6e; margin-bottom: 6px; }
.src-body { font-size: 12px; color: #8b92a5; line-height: 1.6;
            max-height: 120px; overflow-y: auto;
            border-top: 1px solid #1f2637; padding-top: 8px; margin-top: 4px; }

/* ─── Latency row ─────────────────────────── */
.lat-row { display: flex; gap: 8px; flex-wrap: wrap; }
.lat-badge {
    background: #161b27; border: 1px solid #1f2637; border-radius: 6px;
    padding: 4px 10px; font-size: 11px; color: #8b92a5;
}
.lat-badge b { color: #c0c8e0; }

/* ─── Divider ─────────────────────────────── */
.thin-hr { border: none; border-top: 1px solid #1f2637; margin: 10px 0; }

/* ─── Sidebar labels ──────────────────────── */
label[data-testid="stWidgetLabel"] > div {
    font-size: 12px !important; color: #8b92a5 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================
# LAZY IMPORT RAG PIPELINE
# ============================================

@st.cache_resource(show_spinner="Initialising RAG pipeline...")
def load_pipeline():
    from rag.retrieval import (
        retrieve_and_build_context,
        generate_answer_stream,
        generate_direct_answer_stream,
        classify_query,
        reranker,
    )
    return retrieve_and_build_context, generate_answer_stream, generate_direct_answer_stream, classify_query, reranker


# ============================================
# SESSION STATE
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("## ◈ Advanced RAG")
    st.markdown("<hr class='thin-hr'>", unsafe_allow_html=True)

    st.markdown("### Retrieval")
    retrieval_mode = st.radio(
        "Mode",
        options=["hybrid", "vector", "bm25"],
        format_func=lambda x: {"hybrid": "Hybrid (Vector + BM25)",
                                "vector": "Vector Only",
                                "bm25":   "BM25 Only"}[x],
        index=0,
        label_visibility="collapsed",
    )

    col_rrf, col_rerank = st.columns(2)
    with col_rrf:
        use_rrf = st.toggle(
            "RRF Fusion",
            value=True,
            disabled=(retrieval_mode != "hybrid"),
            help="Reciprocal Rank Fusion merges vector and BM25 rankings. Active in Hybrid mode only.",
        )
    with col_rerank:
        use_reranker = st.toggle(
            "Reranker",
            value=False,
            help="Cross-encoder reranker. Slow on CPU; enable only with GPU.",
        )

    st.markdown("<hr class='thin-hr'>", unsafe_allow_html=True)

    st.markdown("### Parameters")
    top_k = st.slider("Top-K results", 1, 20, 5)
    candidate_k = st.slider("Candidate pool", 5, 50, 20)

    with st.expander("Advanced thresholds & weights"):
        vector_threshold = st.slider(
            "Vector similarity threshold", 0.0, 1.0, 0.35, 0.01,
            disabled=(retrieval_mode == "bm25"),
        )
        bm25_threshold = st.slider(
            "BM25 score threshold", 0.0, 1.0, 0.15, 0.01,
            disabled=(retrieval_mode == "vector"),
        )
        vector_weight = st.slider(
            "Vector weight (RRF)", 0.0, 2.0, 1.0, 0.05,
            disabled=(retrieval_mode != "hybrid"),
        )
        bm25_weight = st.slider(
            "BM25 weight (RRF)", 0.0, 2.0, 0.7, 0.05,
            disabled=(retrieval_mode != "hybrid"),
        )

    st.markdown("<hr class='thin-hr'>", unsafe_allow_html=True)

    st.markdown("### Metadata Filters")
    st.caption("Constrain ChromaDB retrieval by metadata fields.")

    chunking_filter = st.selectbox(
        "Chunking strategy",
        ["any", "semantic", "recursive"],
        format_func=lambda x: "Any" if x == "any" else x.capitalize(),
    )
    source_filter = st.text_input(
        "Source document (exact match)",
        placeholder="e.g. paper_001",
    )

    st.markdown("<hr class='thin-hr'>", unsafe_allow_html=True)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================
# METADATA FILTER BUILDER
# ============================================

def build_metadata_filter(chunking, source):
    conditions = []
    if chunking != "any":
        conditions.append({"chunking_strategy": {"$eq": chunking}})
    if source.strip():
        conditions.append({"source": {"$eq": source.strip()}})
    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ============================================
# HEADER + STATUS CHIPS
# ============================================

st.markdown("# ◈ Advanced RAG Assistant")

mode_cls  = {"hybrid": "chip-hybrid", "vector": "chip-vector", "bm25": "chip-bm25"}[retrieval_mode]
mode_icon = {"hybrid": "⚡", "vector": "●", "bm25": "◎"}[retrieval_mode]
rrf_active = use_rrf and retrieval_mode == "hybrid"
rrf_cls    = "chip-on" if rrf_active else "chip-off"
rerank_cls = "chip-on" if use_reranker else "chip-off"

filter_chips = ""
if chunking_filter != "any":
    filter_chips += f'<span class="chip chip-on">chunk:{chunking_filter}</span> '
if source_filter.strip():
    filter_chips += f'<span class="chip chip-on">src:{source_filter.strip()}</span> '

st.markdown(
    f"""<div class="chip-row">
  <span class="chip {mode_cls}">{mode_icon} {retrieval_mode.upper()}</span>
  <span class="chip {rrf_cls}">RRF {"ON" if rrf_active else "OFF"}</span>
  <span class="chip {rerank_cls}">Reranker {"ON" if use_reranker else "OFF"}</span>
  <span class="chip chip-off">top-{top_k} / pool-{candidate_k}</span>
  {filter_chips}
</div>""",
    unsafe_allow_html=True,
)

st.markdown("")


# ============================================
# RENDER HELPERS
# ============================================

def render_sources(sources):
    if not sources:
        st.caption("No sources returned.")
        return
    for src in sources:
        score_str   = f"{src.rerank_score:.4f}"
        title_str   = src.section_title or "—"
        source_str  = src.source or "unknown"
        snippet     = src.content[:420].replace("\n", " ")
        if len(src.content) > 420:
            snippet += " …"
        st.markdown(
            f"""<div class="src-card">
  <div class="src-header">
    <span class="src-title">{source_str}</span>
    <span class="src-score">{score_str}</span>
  </div>
  <div class="src-meta">§ {title_str}</div>
  <div class="src-body">{snippet}</div>
</div>""",
            unsafe_allow_html=True,
        )


def render_latency(latency_ms: dict):
    labels = {
        "vector_search_ms": "Vector",
        "bm25_search_ms":   "BM25",
        "fusion_ms":        "Fusion",
        "reranker_ms":      "Reranker",
        "total_retrieval_ms": "Total",
    }
    badges = []
    for key, label in labels.items():
        val = latency_ms.get(key, 0.0)
        if val > 0 or key == "total_retrieval_ms":
            badges.append(
                f'<span class="lat-badge"><b>{label}</b>&nbsp;{val:.0f}&thinsp;ms</span>'
            )
    st.markdown(
        f'<div class="lat-row">{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )


# ============================================
# CHAT HISTORY
# ============================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            sources  = msg.get("sources", [])
            latency  = msg.get("latency_ms", {})
            if sources:
                with st.expander(f"📄 {len(sources)} source(s)"):
                    render_sources(sources)
            if latency:
                with st.expander("⏱ Latency"):
                    render_latency(latency)


# ============================================
# SAMPLE QUESTIONS
# ============================================

SAMPLE_QUESTIONS = [
    "Which review dataset do they use?",
    "Is there any ethical consideration in the research?",
    "Which language is divided into six dialects in the task mentioned in the paper?",
    "How is state to learn and complete tasks represented via natural language?",
    "What out of domain datasets authors used for coarse-tuning stage?",
]

st.markdown("**Try a sample question:**")
cols = st.columns(len(SAMPLE_QUESTIONS))
for col, sample in zip(cols, SAMPLE_QUESTIONS):
    with col:
        if st.button(sample, use_container_width=True, key=f"sample_{sample[:20]}"):
            st.session_state.pending_query = sample
            st.rerun()

st.markdown("")

# ============================================
# QUERY INPUT
# ============================================

_prefill = st.session_state.pop("pending_query", "") if st.session_state.get("pending_query") else ""
query = st.chat_input("Ask a question about your documents…", key="chat_input")
# Use sample button value if set
if not query and _prefill:
    query = _prefill

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Load pipeline (cached after first call)
    retrieve_fn, stream_fn, direct_stream_fn, classify_fn, reranker_obj = load_pipeline()

    if use_reranker and reranker_obj is None:
        st.warning(
            "Reranker model was not loaded at startup (reranker_enabled is false in "
            "config.yaml). Running without reranker.",
            icon="⚠️",
        )

    # ── Query routing ─────────────────────────
    with st.status("Routing query…", expanded=False) as route_status:
        route = classify_fn(query)
        route_status.update(
            label=f"Route: {'💬 conversational' if route == 'direct' else '🔍 retrieval'}",
            state="complete",
        )

    if route == "direct":
        # ── Direct conversational answer ───────
        with st.chat_message("assistant"):
            full_answer = st.write_stream(direct_stream_fn(query=query))
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_answer,
        })

    else:
        # ── RAG: retrieve then generate ────────
        with st.status("Retrieving context…", expanded=False) as status:
            t0 = time.time()
            try:
                meta_filter = build_metadata_filter(chunking_filter, source_filter)
                result = retrieve_fn(
                    query=query,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    vector_threshold=vector_threshold,
                    bm25_threshold=bm25_threshold,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    retrieval_mode_override=retrieval_mode,
                    use_rrf_override=rrf_active,
                    reranker_override=use_reranker,
                    metadata_filter=meta_filter,
                )
                elapsed = (time.time() - t0) * 1000
                n_src = len(result.get("sources", []))
                status.update(
                    label=f"Retrieved {n_src} chunk(s) in {elapsed:.0f} ms",
                    state="complete",
                )
            except Exception as exc:
                status.update(label="Retrieval failed", state="error")
                st.error(f"Retrieval error: {exc}")
                st.stop()

        with st.chat_message("assistant"):
            full_answer = st.write_stream(
                stream_fn(query=query, context=result["context"])
            )

            sources = result.get("sources", [])
            latency = result.get("latency_ms", {})

            with st.expander(f"📄 {len(sources)} source(s)"):
                render_sources(sources)

            with st.expander("⏱ Latency"):
                render_latency(latency)

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    full_answer,
            "sources":    sources,
            "latency_ms": latency,
        })
