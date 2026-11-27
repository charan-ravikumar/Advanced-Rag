"""Oracle tests for the evaluation runner, and an end-to-end run on a 5-document fixture (hash embedder,
tmp Chroma, tmp FTS). No network, no model."""

import json
import random
from pathlib import Path

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings

from config import cfg
from db.store import build_store
from eval import run_retrieval_eval as rre
from eval.determinism import compare_runs
from eval.gold_spans import gold_spans_for_question
from rag.embedders import HashEmbedder
from rag.ingestion import ingest_corpus
from rag.lexical import SqliteFtsIndex
from rag.retriever import Hit, LegHit, Retriever, RetrievalResult
from rag.schemas_ingest import Chunk

EMB = HashEmbedder()


# ====================================================================================================
# oracle: a fake retriever with known behaviour, through the REAL evaluate_arm
# ====================================================================================================

def _chunk(doc, idx, start, end):
    return Chunk.create(doc_id=doc, section="Methods", chunk_idx=idx, char_start=start, char_end=end,
                        strategy="recursive", text=f"text {doc} {idx}", doc_version="v")


def _hit(c: Chunk, rank: int) -> Hit:
    return Hit(c.chunk_id, c.doc_id, c.text, 1.0 / rank, rank, {"vector": LegHit(rank, 1.0 / rank)},
               section=c.section, char_start=c.char_start, char_end=c.char_end)


class FakeRetriever:
    """Returns hits chosen by ``policy(query, doc_ids)``; records every call."""

    def __init__(self, policy):
        self.policy, self.calls = policy, []

    def retrieve(self, query, **kw):
        self.calls.append({"query": query, **kw})
        chunks = self.policy(query, kw.get("doc_ids"))
        hits = [_hit(c, i) for i, c in enumerate(chunks[: kw.get("top_k", 10)], 1)]
        return RetrievalResult(query, kw.get("mode", "vector"), kw.get("fusion"), hits,
                               {"total": 0.5, "embed": 0.1, "vector": 0.2, "lexical": 0.0, "fuse": 0.01, "hydrate": 0.02, "rerank": 0.0},
                               len(hits))


DOCS = ["pa", "pb", "pc"]
INDEX = {d: [_chunk(d, i, i * 500, (i + 1) * 500) for i in range(4)] for d in DOCS}
GOLD_CHUNK = {d: INDEX[d][1] for d in DOCS}  # chunk 1 covers [500, 1000); the evidence span is [520, 700)


def _question(doc):
    span = {"doc_id": doc, "section": "Methods", "section_idx": 1, "char_start": 520, "char_end": 700, "evidence_idx": 0, "confidence": "exact"}
    return {"id": f"{doc}:q", "paper_id": doc, "question": f"question about {doc}", "answerable": True,
            "gold_evidence": ["some evidence text"], "gold_spans": [span]}


QUESTIONS = [_question(d) for d in DOCS]
ARM = rre.ARMS["A"]


def _run(policy, *, eval_mode="corpus", warmup=0, seed=0):
    fake = FakeRetriever(policy)
    rows = rre.evaluate_arm(ARM, fake, INDEX, QUESTIONS, eval_mode=eval_mode, warmup=warmup, seed=seed)
    return fake, rows


def _doc_of(query):
    return query.split()[-1]


def test_oracle_gold_chunk_first_gives_perfect_scores():
    _, rows = _run(lambda q, d: [GOLD_CHUNK[_doc_of(q)]] + [c for c in INDEX[_doc_of(q)] if c is not GOLD_CHUNK[_doc_of(q)]])
    assert len(rows) == 3
    for r in rows:
        m = r["metrics"]
        assert m["HitRate@1"] == 1.0 and m["MRR"] == 1.0 and m["nDCG@10"] == 1.0
        assert m["DocHit@1"] == 1.0 and m["EvidenceRecall@1"] == 1.0 and m["reachable"] is True
        assert r["retrieved"][0]["relevant"] is True


def test_oracle_only_other_documents_gives_zero():
    def other_docs(q, doc_ids):
        me = _doc_of(q)
        return [c for d in DOCS if d != me for c in INDEX[d]]

    _, rows = _run(other_docs)
    for r in rows:
        m = r["metrics"]
        assert (m["HitRate@1"], m["HitRate@5"], m["HitRate@10"], m["MRR"], m["nDCG@10"]) == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert m["DocHit@10"] == 0.0 and m["EvidenceRecall@10"] == 0.0
        assert not any(h["relevant"] for h in r["retrieved"])


def test_right_document_wrong_passage_scores_doc_hit_but_not_hit_rate():
    _, rows = _run(lambda q, d: [INDEX[_doc_of(q)][3], INDEX[_doc_of(q)][0]])
    for r in rows:
        assert r["metrics"]["DocHit@1"] == 1.0 and r["metrics"]["HitRate@10"] == 0.0


def test_gold_chunk_at_rank_three_gives_reciprocal_rank_one_third():
    _, rows = _run(lambda q, d: [INDEX[_doc_of(q)][3], INDEX[_doc_of(q)][0], GOLD_CHUNK[_doc_of(q)]])
    for r in rows:
        assert r["metrics"]["MRR"] == pytest.approx(1 / 3)
        assert (r["metrics"]["HitRate@1"], r["metrics"]["HitRate@5"]) == (0.0, 1.0)


def test_paper_mode_passes_the_gold_paper_filter_and_corpus_mode_passes_none():
    fake, _ = _run(lambda q, d: [], eval_mode="paper")
    assert sorted(c["doc_ids"][0] for c in fake.calls) == DOCS and all(len(c["doc_ids"]) == 1 for c in fake.calls)
    fake, _ = _run(lambda q, d: [], eval_mode="corpus")
    assert all(c["doc_ids"] is None for c in fake.calls)


def test_warmup_requests_are_extra_calls_and_rows_are_unaffected():
    fake0, rows0 = _run(lambda q, d: [GOLD_CHUNK[_doc_of(q)]], warmup=0)
    fake2, rows2 = _run(lambda q, d: [GOLD_CHUNK[_doc_of(q)]], warmup=2)
    assert len(fake0.calls) == 3 and len(fake2.calls) == 5
    assert rows0 == rows2  # warm-up is discarded; nothing about it leaks into the rows


def test_question_order_seed_does_not_change_the_rows():
    _, a = _run(lambda q, d: [GOLD_CHUNK[_doc_of(q)]], seed=0)
    _, b = _run(lambda q, d: [GOLD_CHUNK[_doc_of(q)]], seed=99)
    assert a == b


def test_arm_settings_reach_the_retriever_call():
    fake = FakeRetriever(lambda q, d: [])
    for name in "ABCDEF":
        rre.evaluate_arm(rre.ARMS[name], fake, INDEX, QUESTIONS[:1], eval_mode="corpus", warmup=0, seed=0)
    modes = [(c["mode"], c.get("fusion"), c.get("vector_weight"), c.get("lexical_weight"), c.get("rrf_k")) for c in fake.calls]
    assert modes == [("vector", None, None, None, None), ("lexical", None, None, None, None), ("hybrid", "naive", None, None, None),
                     ("hybrid", "rrf", 1.0, 0.7, 60), ("vector", None, None, None, None), ("hybrid", "rrf", 1.0, 0.7, 60)]
    assert all(c["top_k"] == 10 and c["candidate_k"] == 20 and "rerank" not in c for c in fake.calls)


def test_corpus_titled_mode_names_the_paper_in_the_query_but_passes_no_filter():
    titles = {d: f"Title of {d}" for d in DOCS}
    fake = FakeRetriever(lambda q, d: [])
    rre.evaluate_arm(ARM, fake, INDEX, QUESTIONS, eval_mode="corpus_titled", warmup=1, seed=0, titles=titles)
    assert len(fake.calls) == 4  # 3 questions + 1 warm-up
    assert all(c["doc_ids"] is None for c in fake.calls)
    assert sorted(c["query"] for c in fake.calls[1:]) == sorted(f'In the paper "Title of {d}": question about {d}' for d in DOCS)
    plain = FakeRetriever(lambda q, d: [])
    rre.evaluate_arm(ARM, plain, INDEX, QUESTIONS, eval_mode="corpus", warmup=0, seed=0)
    assert all(c["query"].startswith("question about") for c in plain.calls)  # other modes never touch the query


def test_the_new_arms_reach_the_retriever_with_their_settings():
    fake = FakeRetriever(lambda q, d: [])
    for name in ("G", "R20_05", "R60_10"):
        rre.evaluate_arm(rre.ARMS[name], fake, INDEX, QUESTIONS[:1], eval_mode="paper", warmup=0, seed=0)
    g, r20, r60 = fake.calls
    assert g["rerank"] is True and g["rerank_k"] == 20 and g["top_k"] == 10 and g["candidate_k"] == 20
    assert (r20["rrf_k"], r20["vector_weight"], r20["lexical_weight"]) == (20, 1.0, 0.5) and "rerank" not in r20
    assert (r60["rrf_k"], r60["lexical_weight"]) == (60, 1.0)
    grid = {(a.rrf_k, a.lexical_weight) for a in rre.ARMS.values() if a.fusion == "rrf" and a.strategy == "recursive"
            and a.lexical == "fts" and not a.rerank}
    assert grid == {(k, w) for k in (20, 60) for w in (0.5, 0.7, 1.0)}  # the pre-registered 2 x 3 grid, current default included


def test_arm_registry_holds_the_preregistered_stage_2a_fifteen_and_the_stage_2b_arms():
    assert rre.ARM_ALIASES["all"] == ["A", "B", "C", "D", "E", "F", "G", "H", "R20_05", "R20_07", "R20_10", "R60_05", "R60_10", "B2", "D2"]
    assert list(rre.ARMS)[15:] == ["GA", "GD", "RT1", "RT3", "RT5", "GRT3"]
    assert rre.ARM_ALIASES["2b_paper"] == ["A", "D", "GA", "GD"] and rre.ARM_ALIASES["2b_corpus"] == ["D", "RT1", "RT3", "RT5", "GRT3"]
    assert rre.select_arms("core") == [rre.ARMS[n] for n in "ABCDEF"]
    assert rre.arm_config(rre.select_arms("core"))["rerank"] is False and rre.arm_config(rre.select_arms("G"))["rerank"] is True


def test_a_rerank_arm_without_a_reranker_is_refused():
    with pytest.raises(ValueError, match="needs a reranker"):
        rre.retriever_for(rre.ARMS["G"], object(), None)


def test_unchanged_arms_use_the_base_retriever_and_the_variants_share_its_store_and_index(world):
    base = world["comps"]["recursive"]["retriever"]
    assert rre.retriever_for(rre.ARMS["D"], base) is base
    b2 = rre.retriever_for(rre.ARMS["B2"], base)
    assert b2 is not base and b2.store is base.store and b2.embedder is base.embedder and b2.lexical.inner is base.lexical and b2.reranker is None
    g = rre.retriever_for(rre.ARMS["G"], base, reranker=lambda q, t: [0.0] * len(t))
    assert g.lexical is base.lexical and g.reranker is not None


def test_end_to_end_all_fifteen_arms_with_a_fake_reranker_and_the_titled_mode(world, tmp_path):
    seen = []

    def reranker(query, texts):  # prefers the longest text: deterministic, and different from the fused order
        seen.append(len(texts))
        return [float(len(t)) for t in texts]

    assert _cli(world, tmp_path / "runs", arm="all", mode="paper", reranker=reranker) == 0
    assert _cli(world, tmp_path / "runs", arm="core", mode="corpus_titled") == 0
    paper = next(p for p in (tmp_path / "runs").iterdir() if p.name.endswith("_paper"))
    titled = next(p for p in (tmp_path / "runs").iterdir() if p.name.endswith("_corpus_titled"))
    summary = json.loads((paper / "summary.json").read_text(encoding="utf-8"))
    assert list(summary["arms"]) == rre.ARM_ALIASES["all"] and seen and max(seen) <= 20
    assert summary["arms"]["G"]["retrieval_latency_ms"]["rerank"]["p50"] > 0.0  # the reranker stage is timed separately
    assert summary["arms"]["D"]["retrieval_latency_ms"]["rerank"]["p50"] == 0.0
    assert all(len(json.loads(l)["retrieved"]) <= 10 for l in (paper / "per_question.jsonl").read_text(encoding="utf-8").splitlines())
    cfg_ = json.loads((titled / "config.json").read_text(encoding="utf-8"))
    assert cfg_["setting"] == "EXTENSION: corpus-wide, title-qualified (user names the paper)" and "In the paper" in cfg_["query_template"]
    assert json.loads((titled / "summary.json").read_text(encoding="utf-8"))["primary_setting"] is False
    assert rre.verify_checksums(paper) and rre.verify_checksums(titled)


def test_the_preregistration_hash_is_recorded_and_tampering_is_refused(world, tmp_path):
    prereg, recorded = tmp_path / "PRE.md", tmp_path / "PRE.sha256"
    prereg.write_bytes(b"# plan\nA vs B\n")
    import hashlib

    original = hashlib.sha256(b"# plan\r\nA vs B\r\n").hexdigest()  # the canonical (CRLF) form is what is pinned
    recorded.write_text(original + "  PRE.md\n", encoding="utf-8")
    assert _cli(world, tmp_path / "runs", arm="A", mode="paper", prereg=(prereg, recorded)) == 0
    (run,) = list((tmp_path / "runs").iterdir())
    info = json.loads((run / "config.json").read_text(encoding="utf-8"))["preregistration"]
    assert info["sha256_original_text"] == original == info["recorded_sha256"] and info["has_deviations_section"] is False
    # appending a Deviations section is allowed and leaves the original hash unchanged, whatever the line endings
    prereg.write_bytes(prereg.read_bytes() + b"\n## Deviations\n2026-11-21: arm X could not run.\n")
    info2 = rre.preregistration_info(prereg, recorded)
    assert info2["sha256_original_text"] == original and info2["sha256_file"] != original and info2["has_deviations_section"] is True
    prereg.write_bytes(prereg.read_bytes().replace(b"\n", b"\r\n"))
    assert rre.preregistration_info(prereg, recorded)["sha256_original_text"] == original
    # rewriting the original text is refused
    prereg.write_bytes(b"# plan\nA vs C\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        _cli(world, tmp_path / "runs2", arm="A", mode="paper", prereg=(prereg, recorded))


def test_routed_arms_are_refused_in_paper_mode_and_by_evaluate_arm():
    with pytest.raises(ValueError, match="not defined in paper mode"):
        rre.evaluate_arm(rre.ARMS["RT3"], FakeRetriever(lambda q, d: []), INDEX, QUESTIONS[:1], eval_mode="paper", warmup=0, seed=0, router=object())
    with pytest.raises(ValueError, match="no router"):
        rre.evaluate_arm(rre.ARMS["RT3"], FakeRetriever(lambda q, d: []), INDEX, QUESTIONS[:1], eval_mode="corpus", warmup=0, seed=0)


def test_component_key_separates_the_gemini_store_from_the_minilm_store():
    assert [rre.component_key(rre.ARMS[n]) for n in ("D", "GA", "GD", "RT3", "GRT3", "F")] == \
        ["recursive", "recursive@gemini", "recursive@gemini", "recursive", "recursive@gemini", "semantic"]


def test_end_to_end_routed_arms_in_corpus_mode_restrict_stage_two_and_record_the_route(world, tmp_path, monkeypatch):
    seen = []
    orig = EMB.encode
    monkeypatch.setattr(EMB, "encode", lambda texts, *a, **k: (seen.append(texts), orig(texts, *a, **k))[1])
    assert _cli(world, tmp_path / "runs", arm="D,RT1,RT3,RT5", mode="corpus") == 0
    (run,) = list((tmp_path / "runs").iterdir())
    rows = [json.loads(l) for l in (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["arm"].startswith("RT"):
            m = int(r["arm"][2:])
            assert len(r["routed_doc_ids"]) == m and {h["doc_id"] for h in r["retrieved"]} <= set(r["routed_doc_ids"])
            assert r["router_hit"] == (r["paper_id"] in r["routed_doc_ids"])
        else:
            assert "routed_doc_ids" not in r
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["arms"]["RT3"]["retrieval_latency_ms"]["route"]["p50"] > 0 and summary["arms"]["D"]["retrieval_latency_ms"]["route"]["p50"] == 0
    cfg_ = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert cfg_["routers"]["minilm"]["papers_indexed"] == 5 and cfg_["routers"]["minilm"]["text"] == "title + abstract of each paper only"
    papers = [t for call in seen if isinstance(call, list) for t in call if t.startswith("Paper ")]  # the router's one build call
    assert len(papers) == 5 and all("number" not in t.split("\n\n")[0] for t in papers)
    # the router's ranking is one list, so routing more papers can only add the gold paper to the routed set
    hit = {a: np.mean([r["router_hit"] for r in rows if r["arm"] == a]) for a in ("RT1", "RT3", "RT5")}
    assert hit["RT1"] <= hit["RT3"] <= hit["RT5"]


def test_routed_arms_cannot_run_in_paper_mode_through_the_cli(world, tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        _cli(world, tmp_path / "runs", arm="RT3", mode="paper")
    assert not (tmp_path / "runs").exists()


def test_gemini_arms_use_their_own_components_and_the_prefetch_separates_network_latency(world, tmp_path):
    class FakeGemini(HashEmbedder):
        name, revision = "fake-gemini", "1/dims64"

        def __init__(self):
            super().__init__()
            self.queries, self.stat_resets = [], 0

        def encode_query(self, text):
            self.queries.append(text)
            return super().encode(text)

        def reset_stats(self):
            self.stat_resets += 1

        def stats(self):
            return {"network_latency_ms": {"n": len(self.queries), "p50": 1.0}, "n_queries_seen": len(self.queries)}

        def describe(self):
            return {"class": "FakeGemini"}

    fake = FakeGemini()
    base = world["comps"]["recursive"]
    comps = {**world["comps"], "recursive@gemini": {**base, "retriever": Retriever(base["retriever"].store, fake, base["retriever"].lexical, cfg=cfg)}}
    argv = ["--split", "dev", "--arm", "A,GA,GD", "--mode", "paper", "--out", str(tmp_path / "runs"), "--warmup", "1", "--prefetch-queries",
            "--golden", str(world["golden"]), "--split-file", str(world["split"]), "--corpus", str(world["corpus"])]
    assert rre.main(argv, components=comps, embedder=EMB) == 0
    (run,) = list((tmp_path / "runs").iterdir())
    cfg_ = json.loads((run / "config.json").read_text(encoding="utf-8"))
    n_q = cfg_["scored_questions"]
    assert cfg_["gemini"]["embedder"] == {"class": "FakeGemini"} and cfg_["gemini"]["query_prefetch_network_latency"]["queries"]["n_queries_seen"] == n_q
    assert len(fake.queries) == n_q + 2 * (n_q + 1)  # prefetch once, then GA and GD each embed every question plus one warm-up
    assert fake.stat_resets >= 2 and set(cfg_["stores"]) == {"recursive", "recursive@gemini"}
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert list(summary["arms"]) == ["A", "GA", "GD"]


def test_the_stage_2b_preregistration_is_enforced_when_passed(world, tmp_path):
    from pathlib import Path

    pre, h = rre.ROOT / "eval" / "PREREGISTRATION_2b.md", rre.ROOT / "eval" / "PREREGISTRATION_2b.sha256"
    assert _cli(world, tmp_path / "runs", arm="D", mode="corpus", prereg=(pre, h)) == 0
    (run,) = list((tmp_path / "runs").iterdir())
    info = json.loads((run / "config.json").read_text(encoding="utf-8"))["preregistration"]
    assert info["sha256_original_text"] == info["recorded_sha256"] and Path(info["path"]).name == "PREREGISTRATION_2b.md"


def test_unreachable_question_is_flagged_not_dropped():
    q = dict(_question("pa"), gold_spans=[{"doc_id": "pa", "section": "Methods", "section_idx": 1, "char_start": 5000, "char_end": 5100,
                                            "evidence_idx": 0, "confidence": "exact"}])
    rows = rre.evaluate_arm(ARM, FakeRetriever(lambda qq, d: INDEX["pa"]), INDEX, [q], eval_mode="corpus", warmup=0, seed=0)
    assert len(rows) == 1 and rows[0]["metrics"]["reachable"] is False and rows[0]["metrics"]["nDCG@10"] == 0.0


# ====================================================================================================
# end to end on a 5-document fixture
# ====================================================================================================

TOPICS = ["alpha model training dataset", "network latency server cache", "protein cell enzyme sequence",
          "river delta soil water", "market price trade supply"]


def _sentences(topic, n, tag):
    return " ".join(f"{topic} {tag} number {i} appears in this sentence." for i in range(n))


def _paper(i):
    t = TOPICS[i]
    return {"title": f"Paper {i}", "abstract": _sentences(t, 6, f"abs{i}"),
            "full_text": [{"section_name": "Introduction", "paragraphs": [_sentences(t, 14, f"intro{i}")]},
                          {"section_name": "Methods", "paragraphs": [_sentences(t, 30, f"meth{i}")]},
                          {"section_name": "Results", "paragraphs": [_sentences(t, 12, f"res{i}")]}]}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval_world")
    corpus = {f"1234.0000{i}": _paper(i) for i in range(5)}
    client = chromadb.PersistentClient(path=str(tmp / "chroma"), settings=Settings(anonymized_telemetry=False))
    comps, lexes = {}, []
    for strategy in ("recursive", "semantic"):
        store = build_store(cfg, strategy, client=client)
        lex = SqliteFtsIndex(tmp / f"lexical_{strategy}.sqlite3")
        lexes.append(lex)
        ingest_corpus(corpus, strategy, store, EMB, cfg, lexical=lex, log=lambda *_: None)
        index = {}
        for c in store.get_all():
            index.setdefault(c.doc_id, []).append(c)
        comps[strategy] = {"retriever": Retriever(store, EMB, lex, cfg=cfg), "chunk_index": index,
                           "info": {"vector_collection": store.name, "vector_count": store.count(), "lexical_path": str(lex.path),
                                    "lexical_count": lex.count(), "ingest_signature": store.get_signature()}}

    questions = []
    for i, pid in enumerate(sorted(corpus)):
        for j, (sec_idx, label) in enumerate([(1, "methods"), (2, "results")]):
            ev = corpus[pid]["full_text"][sec_idx]["paragraphs"][0]
            res = gold_spans_for_question(pid, corpus[pid], [ev])
            questions.append({"id": f"{pid}:{label}", "paper_id": pid, "question": ev[:120], "answerable": True,
                              "gold_answer": "x", "gold_answers": ["x"], "gold_evidence": [ev],
                              "references": [{"answer": "x", "type": "extractive", "mixed_types": False, "evidence": [ev],
                                              "n_evidence_not_retrievable": 0}],
                              "gold_spans": [s.to_dict() for s in res["spans"]], "evidence_location": res["per_evidence"]})
        questions.append({"id": f"{pid}:unans", "paper_id": pid, "question": "nothing here", "answerable": False,
                          "gold_answer": "I could not find that information in the provided documents.", "gold_answers": [],
                          "gold_evidence": [], "references": [{"answer": None, "type": "unanswerable", "mixed_types": False,
                                                               "evidence": [], "n_evidence_not_retrievable": 0}],
                          "gold_spans": [], "evidence_location": []})
    golden = tmp / "golden.json"
    golden.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    corpus_file = tmp / "corpus.json"
    corpus_file.write_text(json.dumps(corpus), encoding="utf-8")
    dev_papers = sorted(corpus)[:3]
    split = tmp / "split.json"
    split.write_text(json.dumps({
        "golden_set_sha256": rre._sha256(golden),
        "dev": sorted(q["id"] for q in questions if q["paper_id"] in dev_papers),
        "test": sorted(q["id"] for q in questions if q["paper_id"] not in dev_papers)}), encoding="utf-8")
    yield {"tmp": tmp, "comps": comps, "golden": golden, "split": split, "corpus": corpus_file, "questions": questions, "dev_papers": dev_papers}
    for lex in lexes:
        lex.close()


def _cli(world, out, *extra, split="dev", arm="core", mode="corpus", ledger=None, freeze=None, reranker=None, prereg=None):
    argv = ["--split", split, "--arm", arm, "--mode", mode, "--out", str(out), "--warmup", "2", "--seed", "3",
            "--golden", str(world["golden"]), "--split-file", str(world["split"]), "--corpus", str(world["corpus"]),
            "--freeze-file", str(freeze or world["tmp"] / "nonexistent_freeze.json"),
            "--ledger-file", str(ledger or world["tmp"] / "ledger.jsonl"), *extra]
    if prereg:
        argv += ["--prereg", str(prereg[0]), "--prereg-hash-file", str(prereg[1])]
    return rre.main(argv, components=world["comps"], embedder=EMB, reranker=reranker)


def test_run_writes_the_folder_with_all_files_and_checksums(world, tmp_path):
    assert _cli(world, tmp_path / "runs") == 0
    (run,) = list((tmp_path / "runs").iterdir())
    assert sorted(p.name for p in run.iterdir()) == ["SHA256SUMS", "config.json", "per_question.jsonl", "summary.json"]
    parts = run.name.split("_")
    assert parts[0].endswith("Z") and len(parts[1]) == 8 and parts[2:] == ["dev", "corpus"]
    assert rre.verify_checksums(run) is True

    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["code_fingerprint"].startswith(parts[1]) and len(config["code_fingerprint"]) == 64
    assert config["rerank"] is False and config["top_k"] == 10 and config["candidate_k"] == 20 and config["seed"] == 3
    assert set(config["arms"]) == set("ABCDEF") and config["arms"]["D"]["vector_weight"] == 1.0 and config["arms"]["D"]["lexical_weight"] == 0.7
    assert config["embedder"]["name"] == EMB.name and config["golden_set"]["sha256"] == rre._sha256(world["golden"])
    assert config["hardware"]["cpu_count_logical"] and "numpy" in config["libraries"] and config["scored_questions"] == 6
    assert config["unanswerable_questions_not_scored"] == 3 and "NOT end-to-end" in config["latency_label"]

    rows = [json.loads(l) for l in (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 * 6  # 6 answerable dev questions x 6 arms; unanswerable are not scored
    assert not any(r["question_id"].endswith("unans") for r in rows)
    r0 = rows[0]
    assert {"question_id", "arm", "retrieved", "metrics", "retrieval_latency_ms"} <= set(r0)
    assert {"rank", "chunk_id", "legs", "relevant", "old_relevant"} <= set(r0["retrieved"][0])
    assert {"HitRate@5", "MRR", "nDCG@10", "DocHit@10", "EvidenceRecall@10", "old_HitRate@5", "old_MRR"} <= set(r0["metrics"])
    assert set(r0["retrieval_latency_ms"]) >= {"embed", "vector", "lexical", "fuse", "rerank", "total"}

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["warmup_requests_discarded_per_arm"] == 2 and "retrieval_latency_ms" in summary["latency_label"]
    for arm, s in summary["arms"].items():
        lat = s["retrieval_latency_ms"]["total"]
        assert set(lat) == {"p50", "p95", "max", "n_measured"} and lat["n_measured"] == 6 and lat["p50"] <= lat["p95"] <= lat["max"]
        assert s["n_questions"] == 6 and 0.0 <= s["means"]["HitRate@10"] <= 1.0


def test_evidence_f1_is_computed_from_the_annotator_references_and_the_corpus(world, tmp_path):
    _cli(world, tmp_path / "runs", mode="paper")
    (run,) = list((tmp_path / "runs").iterdir())
    rows = [json.loads(l) for l in (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        f1s = [r["metrics"][f"EvidenceF1@{k}"] for k in (1, 5, 10)]
        assert all(v is not None and 0.0 <= v <= 1.0 for v in f1s)
    assert any(r["metrics"]["EvidenceF1@10"] > 0.0 for r in rows)  # the gold paragraph is in the retrieved set for some questions
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert all(0.0 <= s["means"]["EvidenceF1@5"] <= 1.0 for s in summary["arms"].values())
    cfg_ = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert "max over annotators" in cfg_["evidence_f1"] and cfg_["corpus"]["sha256"]


def test_paper_mode_is_the_primary_setting_and_corpus_mode_an_extension(world, tmp_path):
    _cli(world, tmp_path / "runs", mode="paper")
    _cli(world, tmp_path / "runs", mode="corpus")
    paper, corpus = sorted((tmp_path / "runs").iterdir(), key=lambda p: p.name.endswith("_corpus"))
    sp = json.loads((paper / "summary.json").read_text(encoding="utf-8"))
    sc = json.loads((corpus / "summary.json").read_text(encoding="utf-8"))
    assert sp["primary_setting"] is True and sp["setting"].startswith("PRIMARY") and "single-document" in sp["setting"]
    assert sc["primary_setting"] is False and sc["setting"].startswith("EXTENSION") and "corpus-wide" in sc["setting"]
    assert json.loads((paper / "config.json").read_text(encoding="utf-8"))["setting"] == sp["setting"]


def test_the_default_mode_is_paper(world, tmp_path):
    argv = ["--split", "dev", "--arm", "A", "--out", str(tmp_path / "runs"), "--warmup", "1",
            "--golden", str(world["golden"]), "--split-file", str(world["split"]), "--corpus", str(world["corpus"])]
    assert rre.main(argv, components=world["comps"], embedder=EMB) == 0
    (run,) = list((tmp_path / "runs").iterdir())
    assert run.name.endswith("_dev_paper")


def test_compare_prints_the_setting_label_of_the_run(world, tmp_path, capsys):
    from eval import compare as cmp

    _cli(world, tmp_path / "runs", mode="paper")
    _cli(world, tmp_path / "runs", mode="corpus")
    for run in sorted((tmp_path / "runs").iterdir()):
        cmp.main(["--run", str(run), "--pair", "A:D", "--metrics", "HitRate@5"])
        out = capsys.readouterr().out
        want = "PRIMARY" if run.name.endswith("_paper") else "EXTENSION"
        assert f"SETTING: {want}" in out


def test_tampering_with_a_file_breaks_the_checksums(world, tmp_path):
    _cli(world, tmp_path / "runs")
    (run,) = list((tmp_path / "runs").iterdir())
    assert rre.verify_checksums(run)
    (run / "summary.json").write_text("{}", encoding="utf-8")
    assert rre.verify_checksums(run) is False


def test_paper_mode_restricts_every_hit_to_the_gold_paper(world, tmp_path):
    _cli(world, tmp_path / "runs", mode="paper")
    (run,) = list((tmp_path / "runs").iterdir())
    assert run.name.endswith("_dev_paper")
    for line in (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        assert r["retrieved"] and {h["doc_id"] for h in r["retrieved"]} == {r["paper_id"]}


def test_two_runs_are_identical_except_timing(world, tmp_path):
    _cli(world, tmp_path / "runs")
    _cli(world, tmp_path / "runs")
    a, b = sorted((tmp_path / "runs").iterdir())
    res = compare_runs(a, b)
    assert res["identical_except_timing"] is True, res["differences"]
    assert res["rows_compared"] == 36 and res["metrics_identical"] is True and res["rows_with_different_retrieved_lists"] == 0


def test_a_changed_retrieved_list_with_equal_metrics_is_reported_separately(world, tmp_path):
    """The real vector leg (Chroma HNSW) is not bit-reproducible across processes: a deep-rank swap can change a
    retrieved list without changing any metric. The checker must say exactly that."""
    _cli(world, tmp_path / "runs")
    _cli(world, tmp_path / "runs")
    a, b = sorted((tmp_path / "runs").iterdir())
    rows = (b / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
    r = json.loads(rows[0])
    r["retrieved"][-1]["chunk_id"] = "swapped-at-a-deep-rank"
    rows[0] = json.dumps(r)
    (b / "per_question.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = compare_runs(a, b)
    assert res["identical_except_timing"] is False and res["metrics_identical"] is True
    assert res["rows_with_different_retrieved_lists"] == 1 and res["rows_with_different_metrics"] == 0


def test_a_changed_result_is_detected_by_the_determinism_check(world, tmp_path):
    _cli(world, tmp_path / "runs")
    _cli(world, tmp_path / "runs")
    a, b = sorted((tmp_path / "runs").iterdir())
    rows = (b / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
    r = json.loads(rows[0])
    r["metrics"]["MRR"] = 0.123
    rows[0] = json.dumps(r)
    (b / "per_question.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert compare_runs(a, b)["identical_except_timing"] is False


def test_single_arm_selection(world, tmp_path):
    _cli(world, tmp_path / "runs", arm="D,E")
    (run,) = list((tmp_path / "runs").iterdir())
    assert {json.loads(l)["arm"] for l in (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines()} == {"D", "E"}


def test_unknown_arm_is_refused(world, tmp_path):
    with pytest.raises(SystemExit, match="unknown arm"):
        _cli(world, tmp_path / "runs", arm="Z")


def test_split_file_for_a_different_golden_set_is_refused(world, tmp_path):
    bad = tmp_path / "split.json"
    bad.write_text(json.dumps({"golden_set_sha256": "0" * 64, "dev": [], "test": []}), encoding="utf-8")
    w = dict(world, split=bad)
    with pytest.raises(SystemExit, match="different golden set"):
        _cli(w, tmp_path / "runs")


# ---- test-split guard and ledger ----------------------------------------------------------------------

def _freeze(path: Path, arms="core", **override):
    cfg_ = rre.arm_config(rre.select_arms(arms))
    cfg_.update(override)
    path.write_text(json.dumps(cfg_), encoding="utf-8")
    return path


def test_test_split_is_refused_without_the_flag_and_nothing_is_written(world, tmp_path):
    ledger, freeze = tmp_path / "ledger.jsonl", _freeze(tmp_path / "freeze.json")
    with pytest.raises(rre.GuardRefused, match="i-understand-this-is-final"):
        _cli(world, tmp_path / "runs", split="test", ledger=ledger, freeze=freeze)
    assert not ledger.exists() and not (tmp_path / "runs").exists()


def test_test_split_is_refused_without_a_freeze_file(world, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(rre.GuardRefused, match="no frozen configuration"):
        _cli(world, tmp_path / "runs", "--i-understand-this-is-final", split="test", ledger=ledger)
    assert not ledger.exists()


def test_test_split_is_refused_when_an_arm_differs_from_the_freeze(world, tmp_path):
    freeze = _freeze(tmp_path / "freeze.json")
    data = json.loads(freeze.read_text(encoding="utf-8"))
    data["arms"]["D"]["lexical_weight"] = 0.9  # someone tuned the weights after the freeze
    freeze.write_text(json.dumps(data), encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(rre.GuardRefused, match="arm D differs"):
        _cli(world, tmp_path / "runs", "--i-understand-this-is-final", split="test", ledger=ledger, freeze=freeze)
    assert not ledger.exists()


def test_test_split_is_refused_when_top_k_differs_from_the_freeze(world, tmp_path):
    freeze = _freeze(tmp_path / "freeze.json", top_k=5)
    with pytest.raises(rre.GuardRefused, match="top_k differs"):
        _cli(world, tmp_path / "runs", "--i-understand-this-is-final", split="test", freeze=freeze)


def test_test_split_runs_with_flag_and_matching_freeze_and_appends_the_ledger(world, tmp_path):
    ledger, freeze = tmp_path / "ledger.jsonl", _freeze(tmp_path / "freeze.json")
    assert _cli(world, tmp_path / "runs", "--i-understand-this-is-final", split="test", ledger=ledger, freeze=freeze) == 0
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["split"] == "test" and entry["arms"] == list("ABCDEF") and len(entry["code_fingerprint"]) == 64 and entry["utc"]
    (run,) = list((tmp_path / "runs").iterdir())
    assert "_test_" in run.name
    cfg_ = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert cfg_["scored_questions"] == 4  # the two test papers' answerable questions
    # a second evaluation appends a second line: the ledger is how the number of touches is proved
    _cli(world, tmp_path / "runs", "--i-understand-this-is-final", split="test", ledger=ledger, freeze=freeze, arm="A", mode="paper")
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2


def test_dev_runs_never_touch_the_ledger(world, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _cli(world, tmp_path / "runs", ledger=ledger)
    assert not ledger.exists()
