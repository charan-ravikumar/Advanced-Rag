"""SqliteFtsIndex tests: real SQLite FTS5 in a tmp dir. No network."""

import pytest

from config import cfg
from rag.lexical import SqliteFtsIndex, escape_query, lexical_index_path
from rag.schemas_ingest import Chunk


def chunk(doc_id: str, idx: int, text: str) -> Chunk:
    return Chunk.create(doc_id=doc_id, section="S", chunk_idx=idx, char_start=0, char_end=len(text),
                        strategy="recursive", text=text, doc_version="v1")


@pytest.fixture
def index(tmp_path):
    ix = SqliteFtsIndex(tmp_path / "lex.sqlite3")
    yield ix
    ix.close()


@pytest.fixture
def populated(index):
    index.upsert([
        chunk("d1", 0, "the neural model was trained on a large corpus"),
        chunk("d1", 1, "attention mechanism for sequence translation"),
        chunk("d2", 0, "protein folding and enzyme kinetics"),
        chunk("d2", 1, "model model model repeated model words"),
        chunk("d3", 0, "naïve café résumé"),
    ])
    return index


def ids(results):
    return [cid for cid, _ in results]


def test_basic_search_returns_best_first_with_positive_higher_is_better_scores(populated):
    res = populated.search("model", k=5)
    assert res and all(score > 0 for _, score in res)
    assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)
    top = next(c for c in populated._conn.execute("SELECT chunk_id, doc_id FROM chunk_map") if c[0] == res[0][0])
    assert top[1] == "d2"  # the chunk that repeats "model" ranks above the one that mentions it once


def test_porter_stemming_matches_inflections(populated):
    assert ids(populated.search("training", k=5)) == ids(populated.search("trained", k=5)) != []


def test_unicode_diacritics_are_folded(populated):
    assert populated.search("naive cafe resume", k=5)


# ---- escaping: user text can never raise or act as an FTS5 operator ----------------------------

HAZARDS = [
    '"unbalanced', "'; DROP TABLE chunks;--", "out-of-domain", "doc_id:d1", "text:model", "NEAR(a b)",
    "model AND", "NOT", "OR", "AND OR NOT", "*", "^model", "(model", "model)", "model*", 'say "hi" now',
    "a:b:c", "-model", "+model", "{x}", "\\", "%_", "数据集 模型", "Ünïcödé", "\n\t", "",
]


@pytest.mark.parametrize("query", HAZARDS)
def test_hazardous_queries_never_raise(populated, query):
    result = populated.search(query, k=5)
    assert isinstance(result, list)
    assert populated.count() == 5  # nothing was modified or dropped


def test_empty_and_whitespace_queries_return_nothing(populated):
    assert populated.search("", k=5) == []
    assert populated.search("   \n", k=5) == []
    assert populated.search("!!! ???", k=5) == []


def test_operators_are_plain_words_not_operators(populated):
    # if NOT were an operator, "model NOT corpus" would EXCLUDE the d1 chunk that contains "corpus"
    assert populated.search("model NOT corpus", k=10)
    with_not = set(ids(populated.search("model NOT corpus", k=10)))
    without = set(ids(populated.search("model corpus", k=10)))
    assert with_not >= without
    # if AND were an operator this would require "zzzzqq" to be present and return nothing
    assert populated.search("model AND zzzzqq", k=10) != []


def test_column_filter_syntax_is_not_interpreted(populated):
    assert set(ids(populated.search("doc_id:d2 model", k=10))) == set(ids(populated.search("doc id d2 model", k=10)))


def test_escape_query_shape():
    assert escape_query("") is None and escape_query("!!!") is None
    assert escape_query('He said "hi"') == '"he" OR "said" OR "hi"'
    assert escape_query("a a A") == '"a"'  # deduplicated, lower-cased
    assert len(escape_query(" ".join(f"w{i}" for i in range(500))).split(" OR ")) == 64


# ---- filter before limit ----------------------------------------------------------------------

def test_doc_filter_applies_before_the_limit(index):
    index.upsert([chunk("A", i, "model " * 20 + f"filler{i}") for i in range(10)])  # strong matches, other doc
    index.upsert([chunk("B", 0, "a single weak mention of the model here among many other words " * 2)])
    res = index.search("model", k=1, doc_ids=["B"])
    assert len(res) == 1  # a filter applied AFTER LIMIT 1 would have returned nothing
    assert index.chunk_ids() >= set(ids(res))
    assert {r[0] for r in index._conn.execute("SELECT chunk_id FROM chunk_map WHERE doc_id = 'B'")} == set(ids(res))


def test_doc_filter_restricts_results(populated):
    for allowed in (["d1"], ["d2"], ["d1", "d3"]):
        for cid in ids(populated.search("model words attention protein naive", k=10, doc_ids=allowed)):
            assert populated._conn.execute("SELECT doc_id FROM chunk_map WHERE chunk_id=?", (cid,)).fetchone()[0] in allowed
    assert populated.search("model", k=5, doc_ids=[]) == []
    assert populated.search("model", k=5, doc_ids=["nope"]) == []


# ---- writes -----------------------------------------------------------------------------------

def test_upsert_is_idempotent(index):
    c = chunk("d", 0, "original words here")
    index.upsert([c])
    index.upsert([c])
    assert index.count() == index.fts_row_count() == 1


def test_new_text_is_a_new_chunk_id_so_replacing_a_doc_goes_through_delete_by_doc(index):
    index.upsert([chunk("d", 0, "original words here")])
    index.upsert([chunk("d", 0, "completely different")])  # different text => different chunk_id
    assert index.count() == 2
    index.delete_by_doc("d")
    index.upsert([chunk("d", 0, "completely different")])
    assert index.count() == 1 and index.search("original", k=5) == [] and index.search("different", k=5)


def test_same_chunk_id_with_new_text_overwrites(index):
    c = chunk("d", 0, "alpha beta")
    index.upsert([c])
    object.__setattr__(c, "text", "gamma delta")  # same chunk_id, new text
    index.upsert([c])
    assert index.count() == index.fts_row_count() == 1
    assert index.search("alpha", k=5) == [] and index.search("gamma", k=5)


def test_delete_by_doc_removes_only_that_doc(populated):
    before = populated.count()
    removed = populated.delete_by_doc("d1")
    assert removed == 2 and populated.count() == before - 2 == populated.fts_row_count()
    assert populated.doc_ids() == {"d2", "d3"}
    assert populated.search("attention", k=5) == []
    assert populated.delete_by_doc("d1") == 0


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "persist.sqlite3"
    a = SqliteFtsIndex(path)
    a.upsert([chunk("d", 0, "durable text here")])
    a.close()
    b = SqliteFtsIndex(path)
    assert b.count() == 1 and b.search("durable", k=3)
    b.close()


def test_ties_are_broken_by_chunk_id(index):
    same = [chunk("d", i, "identical text for every chunk") for i in range(6)]
    index.upsert(same)
    res = ids(index.search("identical", k=6))
    assert res == sorted(res)


def test_k_zero_returns_nothing(populated):
    assert populated.search("model", k=0) == []


# ---- paths ------------------------------------------------------------------------------------

def test_index_path_is_a_sibling_of_the_chroma_dir_per_strategy(tmp_path, monkeypatch):
    monkeypatch.delenv("CHROMA_LOCAL_PATH", raising=False)
    p = lexical_index_path(cfg, "recursive", chroma_path=tmp_path / "advrag" / "chroma")
    assert p == tmp_path / "advrag" / "lexical_recursive.sqlite3"
    assert lexical_index_path(cfg, "layout", chroma_path=tmp_path / "advrag" / "chroma").name == "lexical_layout.sqlite3"


def test_default_index_path_is_outside_onedrive(monkeypatch):
    monkeypatch.delenv("CHROMA_LOCAL_PATH", raising=False)
    p = lexical_index_path(cfg, "recursive")
    assert p.parent.name == "advrag" and p.name == "lexical_recursive.sqlite3"
    assert not any("onedrive" in part.lower() for part in p.parts)


def test_env_override_moves_the_index_next_to_the_chroma_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_LOCAL_PATH", str(tmp_path / "x" / "chroma"))
    assert lexical_index_path(cfg, "semantic") == tmp_path / "x" / "lexical_semantic.sqlite3"


def test_a_second_chroma_dir_name_gets_its_own_lexical_files(tmp_path, monkeypatch):
    """chroma_v2 beside chroma must never share (or overwrite) the old store's lexical files."""
    monkeypatch.delenv("CHROMA_LOCAL_PATH", raising=False)
    old = lexical_index_path(cfg, "recursive", chroma_path=tmp_path / "advrag" / "chroma")
    new = lexical_index_path(cfg, "recursive", chroma_path=tmp_path / "advrag" / "chroma_v2")
    assert old != new and new == tmp_path / "advrag" / "lexical_chroma_v2_recursive.sqlite3"
    monkeypatch.setenv("CHROMA_LOCAL_PATH", str(tmp_path / "advrag" / "chroma_v2"))
    assert lexical_index_path(cfg, "semantic") == tmp_path / "advrag" / "lexical_chroma_v2_semantic.sqlite3"


def test_onedrive_paths_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="OneDrive"):
        lexical_index_path(cfg, "recursive", chroma_path=tmp_path / "OneDrive - Corp" / "chroma")
    with pytest.raises(RuntimeError, match="OneDrive"):
        SqliteFtsIndex(tmp_path / "OneDrive - Corp" / "lex.sqlite3")
    assert not (tmp_path / "OneDrive - Corp").exists()
