"""Pure and support-level tests of the app: chat helpers, artifacts, bootstrap (with fake factories), threading, and the artifact builder. No browser, no network, no real index."""

import json
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.bootstrap as bs
from app.artifacts import (
    cross_paper_facts,
    fmt_ms,
    load_artifacts,
    rerank_facts,
    scope_caption,
    structure_problems,
)
from app.guards import AppConfig
from app.ui_chat import (
    Capture,
    MarkerNormaliser,
    answer_markdown,
    classify,
    escape_md,
    normalise_markers,
    style_markers,
    token_stream,
)
from eval import app_results as ar
from rag_core.schemas import AnswerResult, ErrorInfo
from tests.unit.app_fakes import ROOT, real_artifacts
from tests.unit.rag_core_fakes import FakeRetriever

# ---- marker normalisation ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("LDA\u3010S1\u3011.", "LDA[S1]."), ("a\uff3bS2\uff3d b", "a[S2] b"), ("x [ S1 ] y", "x [S1] y"), ("\u3010S1, S2\u3011", "[S1, S2]"), ("\u3010S1 ,S3\u3011", "[S1, S3]"),
    ("[S1][S2]", "[S1][S2]"), ("\u3010S1\u3011\u3010S9\u3011", "[S1][S9]"), ("plain text", "plain text"), ("", ""), (None, ""),
    ("\u3010S\u3011 [s1] [S 1] [S1", "\u3010S\u3011 [s1] [S 1] [S1"),  # malformed forms are left alone
])
def test_normalise_markers(text, expected):
    assert normalise_markers(text) == expected


SAMPLES = ["The answer\u3010S1\u3011 is here [S2, S3] and \uff3bS4\uff3d done.", "a [S1] b [", "[[S1]]", "\u3010S1\u3011\u3010S2\u3011\u3010S3\u3011", "no markers at all", "[S12, S13] x"]


@pytest.mark.parametrize("text", SAMPLES)
def test_the_streaming_normaliser_gives_the_same_text_as_normalising_the_whole_answer_for_every_way_of_splitting_the_stream(text):
    want = normalise_markers(text)
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            n, out = MarkerNormaliser(), []
            for piece in (text[:i], text[i:j], text[j:]):
                out.append(n.feed(piece))
            out.append(n.flush())
            assert "".join(out) == want, (text, i, j)


def test_the_streaming_normaliser_holds_back_an_unfinished_marker_and_never_holds_ordinary_text_for_long():
    n = MarkerNormaliser()
    assert n.feed("Answer \u3010S") == "Answer " and n.feed("1") == "" and n.feed("\u3011 done") == "[S1] done"
    assert n.feed("see [") == "see " and n.flush() == "["  # an opening bracket that never closes is released at the end
    n = MarkerNormaliser()
    long_tail = "[" + "x" * 40  # not a marker: released once it is longer than any marker can be
    assert n.feed(long_tail) == long_tail


def test_markers_in_the_shown_text_follow_the_parsed_result_valid_bold_invalid_dropped():
    assert style_markers("A [S1] B [S9] C [S1, S9] D [S9, S8]", invalid=["S9"]) == "A **[S1]** B  C **[S1]** D **[S8]**"
    assert style_markers("no markers", ["S1"]) == "no markers"
    result = AnswerResult(answer="Attention\u3010S1\u3011 and cost $5 \u3010S9\u3011.", citations=["S1"], invalid_citations=["S9"])
    assert answer_markdown(result) == "Attention**[S1]** and cost \\$5 ."
    assert escape_md("$x$ and $y$") == "\\$x\\$ and \\$y\\$" and escape_md(None) == ""


def test_a_marker_touching_bold_text_does_not_make_a_run_of_asterisks_that_breaks_the_markdown():
    # found in a real browser: the model wrote "**ROUGE-L**[S1]." and the bolded marker turned it into "****[S1]**." (shown literally)
    shown = style_markers("and **ROUGE-L**[S1]. Also [S2]**next** and _it_[S1]_x_", invalid=[])
    assert "****" not in shown and "__" not in shown
    md = pytest.importorskip("markdown_it").MarkdownIt()
    html = md.render(shown)
    assert "**" not in html and "<strong>ROUGE-L</strong>" in html and "<strong>[S1]</strong>" in html and "<strong>[S2]</strong>" in html


# ---- classification -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected", [
    (dict(mode="direct"), "direct"), (dict(degraded=True, errors=[ErrorInfo(code="daily_budget_reached", message="m")]), "budget"),
    (dict(degraded=True, errors=[ErrorInfo(code="all_providers_failed", message="m")]), "degraded"), (dict(degraded=True), "degraded"), (dict(partial=True), "partial"),
    (dict(abstained=True), "abstained"), (dict(), "answer"), (dict(cached=True), "answer")])
def test_classify(kwargs, expected):
    assert classify(AnswerResult(**kwargs)) == expected


def test_token_stream_yields_normalised_escaped_text_and_collects_sources_errors_and_the_result():
    result = AnswerResult(answer="x")
    events = [{"type": "sources", "sources": ["s1"]}, {"type": "token", "text": "Costs $5 \u3010S"}, {"type": "token", "text": "1\u3011 total"}, {"type": "error", "error": "e"},
              {"type": "done", "result": result}]
    capture, first = Capture(), []
    text = "".join(token_stream(events, capture, lambda: first.append(1)))
    assert text == "Costs \\$5 [S1] total" and capture.sources == ["s1"] and capture.errors == ["e"] and capture.result is result and first == [1] and capture.got_token
    assert list(token_stream([{"type": "done", "result": result}], Capture())) == []  # an abstention streams nothing


# ---- artifacts --------------------------------------------------------------------------------------------------------------------------

def test_missing_and_broken_artifact_files_are_problems_not_exceptions(tmp_path):
    a = load_artifacts("nope/results.json", "nope/papers.json", tmp_path)
    assert a.results is None and a.papers == [] and len(a.problems) == 2 and "build_app_artifacts" in a.problems[0] and not a.has_results
    (tmp_path / "r.json").write_text("{ not json", encoding="utf-8")
    (tmp_path / "p.json").write_text("[1, 2]", encoding="utf-8")
    b = load_artifacts("r.json", "p.json", tmp_path)
    assert b.results is None and b.papers == [] and any("could not be read" in p for p in b.problems)


def test_a_section_without_a_valid_scope_is_dropped_with_a_problem():
    good = {"scope": {"label": "TEST", "run_folder": "runs/x", "code_fingerprint": "a" * 64}, "arms": []}
    results = {"schema": 1, "retrieval_test": good, "generation": {"arms": [{"mean": 0.5}]}, "retrieval_dev": {"scope": {"label": "WRONG", "run_folder": "r", "code_fingerprint": "f"}},
               "claims_log": [], "chunker_findings": [], "ragas": None}
    problems = structure_problems(results)
    assert len(problems) == 2 and any("'generation'" in p for p in problems) and any("'retrieval_dev'" in p for p in problems)


def test_load_artifacts_drops_unsourced_sections_and_reads_papers_and_examples(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"retrieval_test": {"scope": {"label": "TEST", "run_folder": "runs/x", "code_fingerprint": "a" * 64}},
                                                 "generation": {"short_mode": {"AnswerF1": {"mean": 0.9}}}}), encoding="utf-8")
    (tmp_path / "p.json").write_text(json.dumps({"papers": [{"id": "p1", "title": "T1"}], "examples": {"items": [{"question": "q?", "paper_id": "p1"}]},
                                                 "source": {"license": "L"}}), encoding="utf-8")
    a = load_artifacts("r.json", "p.json", tmp_path)
    assert set(a.results) == {"retrieval_test"} and any("generation" in p for p in a.problems)
    assert a.title("p1") == "T1" and a.title("zz") == "zz" and a.examples[0]["question"] == "q?" and a.source == {"license": "L"}


def test_facts_quoted_by_other_tabs_equal_the_artifact_and_carry_their_scope():
    arts = real_artifacts()
    results = arts.results
    test = results["retrieval_test"]
    d, g = (next(a for a in test["arms"] if a["arm"] == n) for n in "DG")
    facts = rerank_facts(results)
    assert facts["scope"]["label"] == "TEST" and facts["rerank_p50_ms"] == g["latency"]["rerank_p50_ms"] and facts["ndcg_with"] == g["metrics"]["nDCG@10"]["mean"]
    assert facts["ndcg_without"] == d["metrics"]["nDCG@10"]["mean"] and facts["holm_significant"] is True
    cross = cross_paper_facts(results)
    entries = {(e["setting"].split(" ")[0], e["arm"]): e["metrics"] for e in results["setting_comparison"]["entries"]}
    assert cross["scope"]["label"] == "DEV" and cross["corpus_hit5"] == entries[("corpus-wide", "D")]["HitRate@5"]["mean"] and cross["paper_hit5"] == entries[("paper", "D")]["HitRate@5"]["mean"]
    assert cross["corpus_hit5"] < cross["paper_hit5"]
    assert rerank_facts(None) is None and rerank_facts({}) is None and cross_paper_facts(None) is None


def test_formatting_helpers():
    assert fmt_ms(59.7) == "60 ms" and fmt_ms(1242) == "1.2 s" and fmt_ms(999.4) == "999 ms"
    scope = {"label": "TEST", "split": "test", "setting": "S", "n_questions": 292, "n_papers": 124, "run_folder": "runs/e/x", "code_fingerprint": "ab" * 32}
    assert scope_caption(scope) == "TEST · test split · S · 292 questions, 124 papers · run `runs/e/x` · fingerprint `abababababab`"


# ---- environment ----------------------------------------------------------------------------------------------------------------------

def test_copy_secrets_copies_only_the_allowlist_never_overrides_and_returns_names_only():
    env = {"GROQ_API_KEY": "already-set"}
    secrets = {"GROQ_API_KEY": "from-secrets", "LANGFUSE_PUBLIC_KEY": " pk-lf-1 ", "APP_ACCESS_CODE": "code", "SOMETHING_ELSE": "x", "LANGFUSE_SECRET_KEY": "", "LANGFUSE_BASE_URL": 5}
    names = bs.copy_secrets_to_env(secrets, env)
    assert names == ["LANGFUSE_PUBLIC_KEY", "APP_ACCESS_CODE"] and env == {"GROQ_API_KEY": "already-set", "LANGFUSE_PUBLIC_KEY": "pk-lf-1", "APP_ACCESS_CODE": "code"}


def test_copy_secrets_survives_a_missing_secrets_file():
    class NoFile(dict):
        def __contains__(self, key):
            raise FileNotFoundError("no secrets.toml")

    env = {}
    assert bs.copy_secrets_to_env(NoFile(), env) == [] and env == {} and bs.copy_secrets_to_env(None, env) == []


def test_prepare_environment_calls_load_env_once_and_importing_the_module_never_does(monkeypatch):
    calls = []
    monkeypatch.setattr(bs, "load_env", lambda *a, **k: calls.append(1) or [])
    assert calls == []
    bs.prepare_environment()
    assert calls == [1]


def test_make_gate_reads_the_environment_at_call_time_and_uses_the_configured_limits():
    s = AppConfig(access_code_attempts=2, access_code_lockout_seconds=5)
    assert not bs.make_gate(s, {}).required and bs.make_gate(s, {"APP_ACCESS_CODE": "x"}).required and bs.make_gate(s, {"APP_ACCESS_CODE": "x"}).max_attempts == 2


# ---- manifest -----------------------------------------------------------------------------------------------------------------------

def write_manifest(root: Path, name: str, location: str, count: int, strategy="recursive"):
    d = root / "runs" / "ingest" / f"{strategy}_{name}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"store": {"location": location}, "verification": {"store_count": count}}), encoding="utf-8")
    return d / "manifest.json"


def test_find_manifest_returns_the_newest_manifest_written_for_this_index_directory(tmp_path):
    index = tmp_path / "chroma_v2"
    write_manifest(tmp_path, "20261121T080000", str(tmp_path / "other_store"), 1)
    write_manifest(tmp_path, "20261121T090000", str(index), 2)
    newest = write_manifest(tmp_path, "20261121T110000", str(index).upper() if os.name == "nt" else str(index), 3)
    write_manifest(tmp_path, "20261121T120000", str(index), 4, strategy="semantic")
    path, manifest = bs.find_manifest(tmp_path / "runs" / "ingest", "recursive", index)
    assert path == newest and manifest["verification"]["store_count"] == 3
    assert bs.find_manifest(tmp_path / "runs" / "ingest", "recursive", tmp_path / "unknown") is None and bs.find_manifest(tmp_path / "missing", "recursive", index) is None
    (tmp_path / "runs" / "ingest" / "recursive_20261122T000000").mkdir()
    (tmp_path / "runs" / "ingest" / "recursive_20261122T000000" / "manifest.json").write_text("{ broken", encoding="utf-8")
    assert bs.find_manifest(tmp_path / "runs" / "ingest", "recursive", index)[0] == newest  # a broken manifest is skipped, not fatal


# ---- build_resources with fake factories ----------------------------------------------------------------------------------------------

class FakeStore:
    def __init__(self, n=8, signature="sig" * 8):
        self.n, self.signature = n, signature

    def count(self):
        return self.n

    def get_signature(self):
        return self.signature


class FakeLexical:
    def __init__(self, n=8):
        self.n = n

    def count(self):
        return self.n


def factories(store_n=8, lex_n=8, **over):
    from rag_core.gateway import LLMGateway
    from tests.unit.rag_core_fakes import ScriptedProvider, ok

    base = dict(open_store=lambda cfg, strategy, path: FakeStore(store_n), open_lexical=lambda cfg, strategy, path: FakeLexical(lex_n),
                make_embedder=lambda cfg: SimpleNamespace(name="fake", revision="r"), make_reranker=lambda cfg: (lambda q, t: [0.0] * len(t)),
                make_retriever=lambda store, embedder, lexical, reranker, cfg: FakeRetriever(), make_gateway=lambda rag_cfg: LLMGateway([ScriptedProvider("groq", "m", [ok("hi")])]),
                fts5_available=lambda: True, model_cached=lambda name, revision: True)
    base.update(over)
    return bs.Factories(**base)


@pytest.fixture
def index_root(tmp_path):
    (tmp_path / "chroma_v2").mkdir()
    write_manifest(tmp_path, "20261121T114640", str(tmp_path / "chroma_v2"), 8)
    return tmp_path


def build(index_root, *, env=None, fac=None):
    from config import cfg

    environment = {"CHROMA_LOCAL_PATH": str(index_root / "chroma_v2"), "GEMINI_API_KEY": "g", "LANGFUSE_PUBLIC_KEY": "p", "LANGFUSE_SECRET_KEY": "s"} if env is None else env
    return bs.build_resources(cfg, environment, factories=fac or factories(), root=index_root)


def failed(res):
    return [c.name for c in res.selfcheck.failures]


def test_a_healthy_index_builds_a_ready_service_with_every_check_passing_and_version_info(index_root):
    res = build(index_root)
    assert res.ready and res.selfcheck.ok and failed(res) == [] and res.service is not None and res.daily.cap == 80 and res.reader is not None
    names = [c.name for c in res.selfcheck.checks]
    assert {"Index directory", "FTS5 (SQLite full-text search)", "Vector store and lexical index open", "Chunk count equals the lexical index", "Chunk count equals the ingest manifest",
            "Embedding model in the local cache", "Embedder warm-up (one dummy query)", "Gemini API key present", "Langfuse keys present"} <= set(names)
    v = res.version_info
    assert v["chunk_count"] == 8 and v["manifest"] == "recursive_20261121T114640" and len(v["manifest_sha256"]) == 64 and v["embedder"]["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert v["prompt_version"] and v["generator"]["model"] == "gemini-3.5-flash-lite" and v["retrieval"]["top_k"] == 5 and res.warmup_ms is not None and res.reranker_available
    assert res.gemini_key_present and res.langfuse_keys_present and res.tracing_on  # keys present and tracing enabled in config.yaml: the (lazy) Langfuse tracer is wired
    assert res.version_info["index_version"].startswith("8:")


def test_the_served_index_version_follows_a_reingest_without_a_restart(index_root, monkeypatch):
    from config import cfg

    monkeypatch.setattr(cfg.app, "index_version_ttl_seconds", 0, raising=False)  # re-read on every request: no real waiting in the test
    store = FakeStore(8)
    res = build(index_root, fac=factories(open_store=lambda c, s, p: store))
    before = res.service.index_version
    assert before.startswith("8:")
    store.n = 11  # a re-ingest added chunks while the app was running
    write_manifest(index_root, "20261122T090000", str(index_root / "chroma_v2"), 11)
    after = res.service.index_version
    assert after.startswith("11:") and after != before

@pytest.mark.parametrize("env_over,fac_over,expect", [
    ({"CHROMA_LOCAL_PATH": None}, {}, "Index directory"),
    ({"CHROMA_LOCAL_PATH": "Z:/definitely/not/here"}, {}, "Index directory"),
    ({}, {"fts5_available": lambda: False}, "FTS5 (SQLite full-text search)"),
    ({}, {"open_store": lambda cfg, s, p: (_ for _ in ()).throw(RuntimeError("chroma exploded"))}, "Vector store and lexical index open"),
    ({}, {"open_lexical": lambda cfg, s, p: (_ for _ in ()).throw(RuntimeError("no fts"))}, "Vector store and lexical index open"),
    ({}, {"model_cached": lambda n, r: False}, "Embedding model in the local cache"),
    ({}, {"make_retriever": lambda s, e, lex, r, c: SimpleNamespace(retrieve=lambda *a, **k: (_ for _ in ()).throw(OSError("model files missing")), embedder=None)},
     "Embedder warm-up (one dummy query)"),
])
def test_every_failed_fatal_check_is_reported_by_name_without_raising_and_without_a_service(index_root, env_over, fac_over, expect):
    env = {"CHROMA_LOCAL_PATH": str(index_root / "chroma_v2"), "GEMINI_API_KEY": "g"}
    env.update({k: v for k, v in env_over.items() if v is not None})
    for k, v in env_over.items():
        if v is None:
            env.pop(k)
    res = build(index_root, env=env, fac=factories(**fac_over))
    assert not res.ready and expect in failed(res) and res.service is None
    assert all(isinstance(c.detail, str) and c.detail for c in res.selfcheck.checks)


def test_a_chunk_count_that_differs_from_the_lexical_index_or_the_manifest_is_a_failed_check(index_root):
    res = build(index_root, fac=factories(store_n=8, lex_n=7))
    assert "Chunk count equals the lexical index" in failed(res) and "vector store 8 vs lexical 7" in " ".join(c.detail for c in res.selfcheck.failures)
    res = build(index_root, fac=factories(store_n=9, lex_n=9))
    assert "Chunk count equals the ingest manifest" in failed(res) and "the store has 9 chunks but the manifest says 8" in " ".join(c.detail for c in res.selfcheck.failures)
    shutil.rmtree(index_root / "runs")
    res = build(index_root)
    assert "Chunk count equals the ingest manifest" in failed(res) and "no ingest manifest" in " ".join(c.detail for c in res.selfcheck.failures)


def test_missing_keys_are_warnings_not_failures_and_values_never_appear_in_any_check(index_root):
    secret = "gsk_SUPERSECRETVALUE0123456789"
    healthy = build(index_root, env={"CHROMA_LOCAL_PATH": str(index_root / "chroma_v2"), "GEMINI_API_KEY": secret, "LANGFUSE_SECRET_KEY": secret + "x", "LANGFUSE_PUBLIC_KEY": "pk-" + secret})
    assert secret not in json.dumps([c.__dict__ for c in healthy.selfcheck.checks]) and healthy.gemini_key_present
    res = build(index_root, env={"CHROMA_LOCAL_PATH": str(index_root / "chroma_v2")})
    assert res.ready and not res.gemini_key_present and not res.langfuse_keys_present and {c.name for c in res.selfcheck.warnings} == {"Gemini API key present", "Langfuse keys present"}
    assert res.tracing_on is False


def test_the_index_directory_comes_from_advrag_index_dir_before_chroma_local_path(index_root):
    other = index_root / "other"
    other.mkdir()
    seen = []
    fac = factories(open_store=lambda cfg, s, p: seen.append(p) or FakeStore(8))
    build(index_root, env={"ADVRAG_INDEX_DIR": str(index_root / "chroma_v2"), "CHROMA_LOCAL_PATH": str(other)}, fac=fac)
    assert seen == [str(index_root / "chroma_v2")]


def test_the_reranker_is_only_wired_when_its_model_is_cached(index_root):
    res = build(index_root, fac=factories(model_cached=lambda name, revision: name.startswith("sentence-transformers")))
    assert not res.reranker_available and res.ready  # the cross-encoder is not cached: a feature is off, the app still works
    assert build(index_root).reranker_available


def test_build_resources_never_raises_even_if_the_artifacts_cannot_be_loaded(index_root, monkeypatch):
    monkeypatch.setattr(bs, "load_artifacts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = build(index_root)
    assert res.ready and res.artifacts.problems == ["artifacts could not be loaded (RuntimeError)"]


# ---- threads --------------------------------------------------------------------------------------------------------------------------

def test_the_locked_retriever_serialises_calls_from_several_threads_and_passes_attributes_through():
    active, peak, lock = [0], [0], threading.Lock()

    class Slow:
        embedder = "the-embedder"

        def retrieve(self, *a, **k):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return "hits"

    locked = bs.LockedRetriever(Slow())
    threads = [threading.Thread(target=locked.retrieve, args=("q",)) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 1 and locked.embedder == "the-embedder" and locked.retrieve("q") == "hits"


def test_an_fts_index_can_be_used_from_another_thread_only_when_asked_to(tmp_path):
    from rag.lexical import SqliteFtsIndex

    def count_in_thread(index):
        out = []
        t = threading.Thread(target=lambda: out.append(_try(index)))
        t.start()
        t.join()
        return out[0]

    def _try(index):
        try:
            return index.count()
        except Exception as exc:  # noqa: BLE001
            return type(exc).__name__

    assert count_in_thread(SqliteFtsIndex(tmp_path / "a.sqlite3")) == "ProgrammingError"  # the default is unchanged
    assert count_in_thread(SqliteFtsIndex(tmp_path / "b.sqlite3", check_same_thread=False)) == 0


# ---- the artifact builder --------------------------------------------------------------------------------------------------------------

def test_the_committed_artifacts_equal_a_fresh_build_from_the_run_folders():
    results, papers = ar.build(ROOT)
    have = json.loads((ROOT / "app_data" / "results.json").read_text(encoding="utf-8"))
    assert ar.strip_volatile(have) == json.loads(json.dumps(ar.strip_volatile(results), ensure_ascii=False)), "app_data/results.json is stale: run scripts/build_app_artifacts.py"
    assert json.loads((ROOT / "app_data" / "papers.json").read_text(encoding="utf-8")) == json.loads(json.dumps(papers, ensure_ascii=False))


def test_every_section_of_the_built_results_has_an_existing_run_folder_and_a_matching_fingerprint():
    results = json.loads((ROOT / "app_data" / "results.json").read_text(encoding="utf-8"))
    assert ar.validate(results, ROOT) == []
    for key in ("retrieval_test", "retrieval_dev", "setting_comparison", "generation"):
        scope = results[key]["scope"]
        assert scope["label"] in ar.SCOPE_LABELS and (ROOT / scope["run_folder"]).is_dir() and len(scope["code_fingerprint"]) == 64 and scope["n_questions"] > 0
    assert results["retrieval_test"]["scope"]["label"] == "TEST" and results["generation"]["scope"]["label"] == "EXPLORATORY" and results["retrieval_dev"]["scope"]["label"] == "DEV"


def test_the_builder_fails_when_a_section_lacks_a_source_run_folder_a_scope_or_a_fingerprint():
    results = json.loads((ROOT / "app_data" / "results.json").read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(results))
    del broken["retrieval_test"]["scope"]["run_folder"]
    assert any("no existing source run folder" in p for p in ar.validate(broken, ROOT))
    broken = json.loads(json.dumps(results))
    broken["generation"]["scope"]["run_folder"] = "runs/gen/does_not_exist"
    assert any("generation.scope" in p for p in ar.validate(broken, ROOT))
    broken = json.loads(json.dumps(results))
    del broken["retrieval_dev"]["scope"]
    assert any("no scope" in p for p in ar.validate(broken, ROOT))
    broken = json.loads(json.dumps(results))
    broken["retrieval_test"]["scope"]["code_fingerprint"] = "0" * 64
    assert any("does not match" in p for p in ar.validate(broken, ROOT))
    broken = json.loads(json.dumps(results))
    broken["generation"]["scope"]["label"] = "FINAL"
    assert any("label" in p for p in ar.validate(broken, ROOT))
    broken = json.loads(json.dumps(results))
    broken["claims_log"][0]["run_folders"] = []
    assert any("no source run folder" in p for p in ar.validate(broken, ROOT))
    broken = json.loads(json.dumps(results))
    broken["brand_new_section"] = {"x": 1.5}
    assert any("brand_new_section" in p and "no scope" in p for p in ar.validate(broken, ROOT))


def test_the_claims_log_crosscheck_catches_a_test_number_that_differs_from_the_published_one():
    results = json.loads((ROOT / "app_data" / "results.json").read_text(encoding="utf-8"))
    rows = ar.parse_claims_log((ROOT / "docs" / "CLAIMS_LOG.md").read_text(encoding="utf-8"))
    assert ar.crosscheck_claims(results["retrieval_test"], rows) == []
    checked = [r for r in rows if r["metric"].startswith(("TEST frozen configuration D", "TEST arm"))]
    assert len(checked) >= 6  # the log states several of the TEST means, so the check is not vacuous
    tampered = json.loads(json.dumps(results["retrieval_test"]))
    tampered["arms"][1]["metrics"]["nDCG@10"]["mean"] += 0.01
    assert any("D nDCG@10" in p for p in ar.crosscheck_claims(tampered, rows))


def test_the_builder_refuses_a_run_folder_that_was_modified_after_the_run(tmp_path):
    run = tmp_path / "runs" / "eval" / "r1"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "SHA256SUMS").write_text(f"{'0' * 64}  config.json\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match SHA256SUMS"):
        ar.load_run(tmp_path, "runs/eval/r1")
    with pytest.raises(SystemExit, match="not found"):
        ar.load_run(tmp_path, "runs/eval/missing")


def test_claims_log_parsing_extracts_metric_value_folders_and_fingerprint():
    text = ("| Metric | Value | Run folder | Code fingerprint | Scope note |\n|---|---|---|---|---|\n"
            f"| **TEST** thing: nDCG@10 | 0.456 | `runs/eval/x` (`runs/analysis/y.txt`) | `{'a' * 64}` | a | b note |\n| not a row |\n")
    (row,) = ar.parse_claims_log(text)
    assert row["metric"] == "TEST thing: nDCG@10" and row["value"] == "0.456" and row["run_folders"] == ["runs/eval/x", "runs/analysis/y.txt"] and row["code_fingerprint"] == "a" * 64
    assert row["scope_note"] == "a|b note"


def test_the_examples_are_public_dev_questions_one_per_paper_chosen_by_a_stated_rule():
    papers = json.loads((ROOT / "app_data" / "papers.json").read_text(encoding="utf-8"))
    ex = papers["examples"]
    assert len(ex["items"]) == ar.EXAMPLES and len({i["paper_id"] for i in ex["items"]}) == ar.EXAMPLES and "sha256" in ex["rule"] and "not representative" in ex["rule"]
    golden = json.loads((ROOT / "eval" / "datasets" / "qasper_dev" / "golden_set_v2.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "eval" / "datasets" / "qasper_dev" / "split.json").read_text(encoding="utf-8"))
    dev_ids = set(split["dev"])
    by_id = {q["id"]: q for q in golden["questions"]}
    for item in ex["items"]:
        assert item["question_id"] in dev_ids and by_id[item["question_id"]]["question"] == item["question"]  # taken from the DEV split, never from the test split
    assert len(papers["papers"]) == 200 and papers["source"]["license"].startswith("Qasper, CC BY")
