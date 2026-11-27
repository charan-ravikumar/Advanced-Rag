"""Measurements: (6) content loss of the v1 vs current semantic chunker, (7) recursive chunk size cap.

    python scripts/measure_chunking.py [--papers 20]

Uses the real embedder (local HuggingFace cache only). The v1 chunker is loaded read-only from
the sibling ``*__v1_snapshot`` folder and run unmodified (only the model it would load at import
is replaced by our embedder object). Results go to runs/measurements/chunk_coverage_<UTC>/results.json
together with the code fingerprint; nothing is quoted from the console.
"""

import argparse
import importlib.util
import json
import logging
import os
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
logging.disable(logging.CRITICAL)

from config import cfg  # noqa: E402
from rag.chunk_adapter import _section_ranges, adapt_document, locate, qasper_to_document  # noqa: E402
from rag.chunkers import recursive_chunk, recursive_splitter  # noqa: E402
from rag.embedders import SentenceTransformerEmbedder  # noqa: E402
from utils.fingerprint import code_fingerprint  # noqa: E402


def _uncovered_fraction(doc, covered_spans):
    """Fraction of non-whitespace characters of all sections not inside any span, plus examples."""
    covered = bytearray(len(doc.content))
    for s, e in covered_spans:
        covered[s:e] = b"\x01" * (e - s)
    total = missing = 0
    examples = []
    for start, end, title in _section_ranges(doc):
        i = start
        while i < end:
            if doc.content[i].isspace():
                i += 1
                continue
            total += 1
            if not covered[i]:
                j = i
                while j < end and not covered[j]:
                    j += 1
                lost = sum(not ch.isspace() for ch in doc.content[i:j])
                missing += lost
                total += lost - 1
                if len(examples) < 3:
                    examples.append({"section": title, "snippet": doc.content[i:j][:80]})
                i = j
            else:
                i += 1
    return total, missing, examples


def _load_v1_semantic(embedder):
    import sentence_transformers

    snap = ROOT.parent / (ROOT.name + "__v1_snapshot") / "rag" / "chunkers.py"
    if not snap.exists():
        raise SystemExit(f"v1 snapshot not found at {snap}")

    class Shim:
        def __init__(self, *a, **k):
            pass

        def encode(self, *a, **k):
            return embedder.encode(*a, **k)

    original = sentence_transformers.SentenceTransformer
    sentence_transformers.SentenceTransformer = Shim
    try:
        spec = importlib.util.spec_from_file_location("v1_chunkers", snap)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sentence_transformers.SentenceTransformer = original
    return module.semantic_chunk, str(snap)


def measure_coverage(corpus, ids, embedder):
    v1_semantic, snap_path = _load_v1_semantic(embedder)
    tot = {"v1": [0, 0], "current": [0, 0]}
    unlocated_v1 = chunks_v1 = chunks_cur = 0
    papers_with_loss = {"v1": 0, "current": 0}
    examples = {"v1": [], "current": []}
    for pid in ids:
        doc = qasper_to_document(pid, corpus[pid])
        spans, cursor = [], 0
        for c in v1_semantic(doc):
            chunks_v1 += 1
            s, e, _ = locate(c.content, doc.content, cursor)
            if s < 0:
                unlocated_v1 += 1
                continue
            cursor = s + 1
            spans.append((s, e))
        t, m, ex = _uncovered_fraction(doc, spans)
        tot["v1"][0] += t
        tot["v1"][1] += m
        papers_with_loss["v1"] += m > 0
        examples["v1"] += [{"paper": pid, **x} for x in ex][: 3 - len(examples["v1"])]

        res = adapt_document(doc, pid, "semantic", embedder=embedder)
        chunks_cur += len(res.chunks)
        spans = [(o.doc_char_start, o.doc_char_end) for o in res.offset_info.values() if o.doc_char_start >= 0]
        t, m, ex = _uncovered_fraction(doc, spans)
        tot["current"][0] += t
        tot["current"][1] += m
        papers_with_loss["current"] += m > 0
        examples["current"] += [{"paper": pid, **x} for x in ex][: 3 - len(examples["current"])]
    return {
        "v1_snapshot_chunker": snap_path,
        "papers": len(ids),
        "v1": {"chunks": chunks_v1, "chunks_not_located": unlocated_v1, "nonwhitespace_chars": tot["v1"][0],
               "uncovered_chars": tot["v1"][1], "uncovered_fraction": tot["v1"][1] / tot["v1"][0],
               "papers_with_any_loss": papers_with_loss["v1"], "examples": examples["v1"]},
        "current": {"chunks": chunks_cur, "nonwhitespace_chars": tot["current"][0],
                    "uncovered_chars": tot["current"][1], "uncovered_fraction": tot["current"][1] / tot["current"][0],
                    "papers_with_any_loss": papers_with_loss["current"], "examples": examples["current"]},
        "note": "v1 chunks that could not be located are excluded from coverage, so the v1 loss is an upper bound by that many chunks",
    }


def measure_recursive_cap(corpus, ids, embedder):
    size, minc = cfg.chunking.recursive_chunk_size, cfg.chunking.min_chunk_chars
    raw_lens, final_lens, merged_sections = [], [], 0
    for pid in ids:
        doc = qasper_to_document(pid, corpus[pid])
        for sec in doc.sections:
            raw = recursive_splitter.split_text(sec.content)
            raw_lens += [len(r) for r in raw]
        final = [len(c.content) for c in recursive_chunk(doc, embedder=embedder)]
        final_lens += final
    over = [n for n in final_lens if n > size]
    return {
        "papers": len(ids),
        "recursive_chunk_size": size,
        "min_chunk_chars": minc,
        "max_chunk_chars_hard_cap": cfg.chunking.max_chunk_chars,
        "theoretical_effective_cap_chars": size + minc - 1,
        "raw_splitter_chunks": {"n": len(raw_lens), "max_len": max(raw_lens), "over_chunk_size": sum(n > size for n in raw_lens)},
        "final_chunks": {"n": len(final_lens), "max_len": max(final_lens), "median": st.median(final_lens),
                         "over_chunk_size": len(over), "share_over_chunk_size": len(over) / len(final_lens)},
        "max_len_within_theoretical_cap": max(final_lens) <= size + minc - 1,
        "conclusion_check": "chunks longer than recursive_chunk_size can only come from the within-section merge of a tail shorter than min_chunk_chars"
        if max(raw_lens) <= size else "raw splitter exceeded chunk_size: investigate",
    }


def measure_token_counters(corpus, ids, real):
    """Real-tokenizer size of chunks built with (a) the plain regex counter, (b) the estimator,
    (c) the real tokenizer itself. Shows why a cheap counter is not a safe cap."""
    from rag.embedders import HashEmbedder
    from rag.tokens import regex_token_count

    class RegexCounted(HashEmbedder):
        token_counter = "regex"

        def count_tokens(self, text):
            return regex_token_count(text)

    makers = {"regex_counter": RegexCounted(), "estimate_counter": HashEmbedder(), "real_tokenizer": real}
    limit = min(cfg.chunking.max_tokens, real.max_seq_length)
    out = {"window": limit}
    for label, emb in makers.items():
        out[label] = {}
        for strategy in ("recursive", "layout", "semantic"):
            counts = []
            for pid in ids:
                doc = qasper_to_document(pid, corpus[pid])
                counts += [real.count_tokens(c.text) for c in adapt_document(doc, pid, strategy, embedder=emb).chunks]
            out[label][strategy] = {"chunks": len(counts), "max_real_tokens": max(counts),
                                    "over_window": sum(n > limit for n in counts)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=20)
    args = ap.parse_args()
    corpus = json.loads((ROOT / "eval/datasets/qasper_dev/corpus.json").read_text(encoding="utf-8"))
    ids = list(corpus)[: args.papers]
    embedder = SentenceTransformerEmbedder.from_config(cfg)
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_fingerprint": code_fingerprint(),
        "model": embedder.name,
        "revision": embedder.revision,
        "paper_ids": ids,
        "token_counters_first_20": measure_token_counters(corpus, ids, embedder),
        "coverage_v1_vs_current_semantic": measure_coverage(corpus, ids, embedder),
        "recursive_size_cap_first_20": measure_recursive_cap(corpus, ids, embedder),
        "recursive_size_cap_all_papers": measure_recursive_cap(corpus, list(corpus), embedder),
    }
    d = ROOT / "runs" / "measurements" / ("chunk_coverage_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(d / "results.json")


if __name__ == "__main__":
    main()
