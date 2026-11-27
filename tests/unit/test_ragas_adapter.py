"""eval/ragas_adapter.py against the REAL ragas 0.4.4 collections metrics, with a fake judge and a fake embedder (no network, no keys).

Runs only where ragas is installed (the evaluation venv); in the main venv the whole module is skipped:

    & "$env:LOCALAPPDATA\\advrag\\venv_eval\\Scripts\\python.exe" -m pytest tests/unit/test_ragas_adapter.py -p no:cacheprovider -q -o addopts=""
"""

import json
import math

import pytest

pytest.importorskip("ragas", reason="ragas is installed only in the evaluation venv (venv_eval)")

from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402
from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402

from eval.disk_cache import DiskCache  # noqa: E402
from eval.judge_core import CachingJudge, JudgeSettings, JudgeStats, RawJudgeResult  # noqa: E402
from eval.ragas_adapter import RagasScorer, make_minilm_embedding, make_ragas_llm, package_version  # noqa: E402

SETTINGS = JudgeSettings(model="gemini-2.5-flash", temperature=0.0, thinking_budget=1024, max_output_tokens=4096)


class FakeEmbedder:
    """encode_query like the local sentence embedder: a deterministic vector from character counts."""

    def encode_query(self, text):
        return [float(text.lower().count(c)) for c in "aeiou"] + [float(len(text) % 7)]


class FakeRaw:
    def __init__(self, verdicts=(1, 0), noncommittal=0):
        self.verdicts, self.noncommittal, self.calls = verdicts, noncommittal, []

    async def __call__(self, prompt, response_model):
        name = response_model.__name__
        self.calls.append(name)
        if name == "StatementGeneratorOutput":
            body = {"statements": [f"statement {i}" for i in range(len(self.verdicts))]}
        elif name == "NLIStatementOutput":
            body = {"statements": [{"statement": f"statement {i}", "reason": "because", "verdict": v} for i, v in enumerate(self.verdicts)]}
        elif name == "AnswerRelevanceOutput":
            body = {"question": "What does the attention mechanism compute?", "noncommittal": self.noncommittal}
        else:
            raise AssertionError(f"unexpected response model {name}")
        return RawJudgeResult(text=json.dumps(body), prompt_tokens=100, output_tokens=10, thinking_tokens=5, finish_reason="STOP")


def scorer(tmp_path, raw, stats=None):
    stats = stats or JudgeStats()
    return RagasScorer(raw, DiskCache(tmp_path / "jc"), SETTINGS, stats, make_minilm_embedding(FakeEmbedder()), version=package_version("ragas"), sleep=lambda s: None), stats


def test_the_adapter_objects_are_what_the_collections_metrics_require(tmp_path):
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    s, _ = scorer(tmp_path, FakeRaw())
    assert isinstance(s.faithfulness, Faithfulness) and isinstance(s.relevancy, AnswerRelevancy)
    judge = CachingJudge(FakeRaw(), DiskCache(tmp_path / "x"), SETTINGS, "m", "v", JudgeStats())
    assert isinstance(make_ragas_llm(judge), InstructorBaseRagasLLM) and isinstance(make_minilm_embedding(FakeEmbedder()), BaseRagasEmbedding)
    assert package_version("ragas") == "0.4.4" and package_version("not-a-package-xyz") is None
    s.close()


def test_faithfulness_is_the_share_of_supported_statements_and_relevancy_is_the_cosine_to_the_generated_question(tmp_path):
    raw = FakeRaw(verdicts=(1, 0))
    s, stats = scorer(tmp_path, raw)
    question = "What does the attention mechanism compute?"
    out = s.score(question, "Attention computes a weighted sum of values.", ["Passage one.", "Passage two."])
    assert out["Faithfulness"] == pytest.approx(0.5)
    e = FakeEmbedder()
    a, b = e.encode_query(question), e.encode_query("What does the attention mechanism compute?")
    assert out["AnswerRelevancy"] == pytest.approx(sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))))
    assert raw.calls == ["StatementGeneratorOutput", "NLIStatementOutput", "AnswerRelevanceOutput"]  # 2 faithfulness calls + 1 relevancy call (strictness 1)
    usage = stats.as_dict()
    assert usage["faithfulness"]["calls"] == 2 and usage["answer_relevancy"]["calls"] == 1 and usage["faithfulness"]["thinking_tokens"] == 10
    s.close()


def test_every_judge_call_is_cached_so_scoring_the_same_answer_again_costs_nothing_and_each_metric_has_its_own_entries(tmp_path):
    raw = FakeRaw()
    s, stats = scorer(tmp_path, raw)
    first = s.score("Q?", "An answer.", ["ctx"])
    assert len(raw.calls) == 3 and len(DiskCache(tmp_path / "jc")) == 3
    assert s.score("Q?", "An answer.", ["ctx"]) == first and len(raw.calls) == 3  # all three came from the disk cache
    s2, stats2 = scorer(tmp_path, raw)  # a new session
    assert s2.score("Q?", "An answer.", ["ctx"]) == first and len(raw.calls) == 3 and stats2.total_calls == 0 and stats2.usage("faithfulness").cache_hits == 2
    s.close()
    s2.close()


def test_a_noncommittal_answer_scores_zero_relevancy_as_ragas_defines_it(tmp_path):
    s, _ = scorer(tmp_path, FakeRaw(noncommittal=1))
    assert s.score("Q?", "I do not know.", ["ctx"])["AnswerRelevancy"] == 0.0
    s.close()


def test_the_local_embedding_wrapper_returns_plain_float_lists_sync_and_async():
    import asyncio

    emb = make_minilm_embedding(FakeEmbedder())
    assert emb.embed_text("Hello") == FakeEmbedder().encode_query("Hello") and all(isinstance(x, float) for x in emb.embed_text("Hello"))
    assert asyncio.run(emb.aembed_text("Hello")) == emb.embed_text("Hello")
