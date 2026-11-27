"""config.yaml is strict (N-14): an unknown key is an error, every retrieval knob has ONE home (``pipeline``), and the real file loads cleanly."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import config as config_module
from app.guards import AppConfig
from rag.retriever import Retriever
from rag_core.config import PromptsConfig, RagConfig, TracingConfig

ROOT = Path(__file__).resolve().parents[2]
RAW = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
RETRIEVAL_KNOBS = ("top_k", "candidate_k", "rerank_k", "rrf_k", "vector_weight")


def write(tmp_path, raw) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# ---- the real file ------------------------------------------------------------------------------------------------------------

def test_the_real_config_yaml_loads_cleanly_through_the_strict_loader_and_every_section_loader():
    cfg = config_module.load_config()
    assert cfg.pipeline.top_k == 5 and cfg.chunking.max_chunk_chars == 2000
    RagConfig.from_config(cfg), PromptsConfig.from_config(cfg), TracingConfig.from_config(cfg), AppConfig.from_config(cfg)  # none raises


def test_the_chunking_section_is_loaded_exactly_as_written():
    cfg = config_module.load_config()
    assert vars(cfg.chunking) == RAW["chunking"]  # it feeds the ingest signature: never reshaped


# ---- unknown keys are errors --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("section", ["pipeline", "embedding", "embedding_gemini", "store", "chunking", "generation", "prompts", "tracing", "app"])
def test_an_unknown_key_in_any_section_of_the_file_is_an_error_that_names_the_section_and_the_key(tmp_path, section):
    raw = copy.deepcopy(RAW)
    raw[section]["tpo_k"] = 3  # a typo
    cfg_loader = lambda: config_module.load_config(write(tmp_path, raw))  # noqa: E731
    if section in ("generation", "prompts", "tracing", "app"):  # the sections owned by a dataclass are checked by its from_config
        cfg = cfg_loader()
        loader = {"generation": RagConfig, "prompts": PromptsConfig, "tracing": TracingConfig, "app": AppConfig}[section]
        with pytest.raises(ValueError, match=rf"{section}.*tpo_k|tpo_k.*{section}"):
            loader.from_config(cfg)
    else:
        with pytest.raises(ValueError, match=rf"{section}.*tpo_k|tpo_k.*{section}"):
            cfg_loader()


def test_an_unknown_top_level_section_is_an_error(tmp_path):
    raw = copy.deepcopy(RAW)
    raw["evall"] = {"x": 1}
    with pytest.raises(ValueError, match="evall"):
        config_module.load_config(write(tmp_path, raw))


def test_a_section_that_is_not_a_mapping_is_an_error(tmp_path):
    raw = copy.deepcopy(RAW)
    raw["pipeline"] = ["top_k"]
    with pytest.raises(ValueError, match="pipeline"):
        config_module.load_config(write(tmp_path, raw))


@pytest.mark.parametrize("section,key", [("pipeline", "vector_threshold"), ("pipeline", "bm25_threshold"), ("pipeline", "chunking"), ("embedding", "cache_enabled"),
                                         ("embedding", "cache_dir"), ("embedding", "lru_cache_size"), ("store", "tiktoken_cache_dir")])
def test_keys_that_nothing_reads_were_removed_and_are_now_rejected(tmp_path, section, key):
    assert key not in RAW[section]
    raw = copy.deepcopy(RAW)
    raw[section][key] = 1
    with pytest.raises(ValueError, match=key):
        config_module.load_config(write(tmp_path, raw))


def test_the_unused_eval_section_is_gone():
    assert "eval" not in RAW


def test_overlay_rejects_unknown_keys_of_a_section_and_names_it():
    with pytest.raises(ValueError, match="generation.*not_a_setting"):
        RagConfig.from_config(SimpleNamespace(generation=SimpleNamespace(max_output_tokens=7, not_a_setting=1)))
    with pytest.raises(ValueError, match="app.*warmup_quary"):
        AppConfig.from_config(SimpleNamespace(app=SimpleNamespace(warmup_quary="x")))
    assert RagConfig.from_config(SimpleNamespace()).as_dict() == RagConfig().as_dict()  # a missing section keeps the defaults


# ---- one source per knob ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("knob", ["top_k", "candidate_k", "rerank_k", "rrf_k", "vector_weight", "lexical_weight", "retrieval_mode", "fusion"])
def test_a_retrieval_knob_under_generation_is_rejected_because_pipeline_owns_it(knob):
    with pytest.raises(ValueError, match="pipeline"):
        RagConfig.from_config(SimpleNamespace(generation=SimpleNamespace(**{knob: 3})))


def test_the_answer_path_reads_every_retrieval_knob_from_the_pipeline_section():
    cfg = SimpleNamespace(pipeline=SimpleNamespace(retrieval="vector", fusion="none", top_k=7, candidate_k=31, rerank_k=17, rrf_k=11, vector_weight=2.5, bm25_weight=0.3))
    rc = RagConfig.from_config(cfg)
    assert (rc.retrieval_mode, rc.fusion, rc.top_k, rc.candidate_k, rc.rerank_k, rc.rrf_k, rc.vector_weight, rc.lexical_weight) == ("vector", "none", 7, 31, 17, 11, 2.5, 0.3)


def test_the_retriever_defaults_the_answer_path_and_the_code_defaults_agree_for_every_knob():
    cfg = config_module.load_config()
    rc, defaults = RagConfig.from_config(cfg), RagConfig()
    retr = Retriever(None, None, None, None, cfg)._defaults
    for knob in RETRIEVAL_KNOBS:
        assert retr[knob] == getattr(rc, knob) == getattr(defaults, knob), knob
    assert retr["lexical_weight"] == rc.lexical_weight == defaults.lexical_weight == cfg.pipeline.bm25_weight
    assert (rc.retrieval_mode, rc.fusion) == (defaults.retrieval_mode, defaults.fusion) == (cfg.pipeline.retrieval, cfg.pipeline.fusion) == ("hybrid", "rrf")
    assert rc.as_dict() == defaults.as_dict()


def test_rerank_k_is_twenty_the_value_the_service_and_the_reranked_arms_use():
    cfg = config_module.load_config()
    assert cfg.pipeline.rerank_k == 20 and "PLACEHOLDER" not in (ROOT / "config.yaml").read_text(encoding="utf-8")


def test_the_stale_reranker_latency_comment_was_replaced_by_the_measured_figure():
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "11.7" not in text and "1.19" in text and "20261122T144233Z_3afc204d_test_paper" in text
