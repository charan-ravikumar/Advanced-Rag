"""
serving.py
==========
Advanced RAG Serving Layer
--------------------------
A production-grade online serving module for a Retrieval-Augmented Generation
(RAG) system built with:
  • FastAPI        – async HTTP API
  • ChromaDB       – persisted vector store
  • BM25           – sparse keyword retrieval  (rank-bm25)
  • RRF            – Reciprocal Rank Fusion for hybrid ranking
  • Cross-Encoder  – reranking  (sentence-transformers)
  • Gemini API     – free-tier LLM for query expansion & generation
  • RAGAS          – optional answer quality evaluation

Architecture
------------
  QueryExpander  ──►  HybridRetriever  ──►  RRFCombiner
                             │
                        Reranker  ──►  ContextBuilder  ──►  LLMGenerator
                                                                  │
                                                          RAGPipeline (orchestrator)
                                                                  │
                                                          FastAPI endpoints

Running the server
------------------
  1. Install dependencies:
       pip install fastapi uvicorn chromadb google-generativeai \
                   sentence-transformers rank-bm25 ragas langchain \
                   langchain-google-genai

  2. Set your Gemini API key (or edit config.py):
       export GEMINI_API_KEY="your_key_here"

  3. Ensure ChromaDB is populated (run your ingestion pipeline first).

  4. Start the server:
       uvicorn serving:app --host 0.0.0.0 --port 8000 --reload

Example curl request
--------------------
  curl -X POST http://localhost:8000/query \\
       -H "Content-Type: application/json" \\
       -d '{"query": "What is retrieval-augmented generation?"}'

Streamlit compatibility
-----------------------
  The RAGPipeline.run() method returns a plain Python dict that can be
  consumed directly by a Streamlit frontend without any modification.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import asyncio
import logging
import time
import unicodedata
import re
import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

import chromadb
from chromadb.config import Settings as ChromaSettings

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Optional RAGAS imports (guarded so the server still starts if RAGAS is absent)
# ---------------------------------------------------------------------------
try:
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local config
# ---------------------------------------------------------------------------
import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.serving")


# ============================================================================
# Pydantic models – request / response contracts
# ============================================================================

class QueryRequest(BaseModel):
    """Incoming POST /query payload."""
    query: str

    @validator("query")
    def query_must_not_be_empty(cls, v: str) -> str:  # noqa: N805
        if not v or not v.strip():
            raise ValueError("query must be a non-empty string")
        return v.strip()


class QueryResponse(BaseModel):
    """Response returned by POST /query.
    Designed to be consumed directly by a Streamlit frontend.
    """
    answer: str
    sources: List[str]
    latency: Dict[str, float]          # retrieval_s, generation_s, total_s
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None  # DEBUG_MODE only
    evaluation: Optional[Dict[str, float]] = None            # RAGAS metrics


class HealthResponse(BaseModel):
    status: str
    chroma_collection: str
    reranker_loaded: bool
    cache_enabled: bool
    ragas_enabled: bool
    debug_mode: bool


# ============================================================================
# In-memory LRU Cache
# ============================================================================

class LRUCache:
    """Thread-safe (GIL-protected) LRU cache backed by an OrderedDict."""

    def __init__(self, max_size: int = 256) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        k = self._key(query)
        if k not in self._cache:
            return None
        self._cache.move_to_end(k)
        logger.debug("Cache HIT for query hash %s", k[:8])
        return self._cache[k]

    def set(self, query: str, value: Any) -> None:
        k = self._key(query)
        self._cache[k] = value
        self._cache.move_to_end(k)
        if len(self._cache) > self._max_size:
            evicted = self._cache.popitem(last=False)
            logger.debug("Cache EVICT key %s", evicted[0][:8])

    def clear(self) -> None:
        self._cache.clear()


# ============================================================================
# QueryExpander
# ============================================================================

class QueryExpander:
    """Uses Gemini to generate semantically diverse query variations.

    If the Gemini call fails for any reason the original query is returned
    as a single-element list so the pipeline never stalls.
    """

    def __init__(self) -> None:
        self._model_name = config.GEMINI_EXPANSION_MODEL
        self._num_expansions = config.NUM_QUERY_EXPANSIONS

    def _build_prompt(self, query: str) -> str:
        return (
            f"You are a query expansion assistant. "
            f"Given the following user query, generate {self._num_expansions} "
            f"semantically diverse reformulations that preserve the original intent "
            f"but use different vocabulary, phrasing, or perspective. "
            f"Output ONLY a numbered list (1. ... 2. ... etc.), no extra text.\n\n"
            f"Original query: {query}"0
        )

    def expand(self, query: str) -> List[str]:
        """Return [original_query] + expanded variations."""
        variations: List[str] = [query]
        try:
            model = genai.GenerativeModel(self._model_name)
            resp = model.generate_content(
                self._build_prompt(query),
                generation_config=GenerationConfig(
                    temperature=.7,
                    max_output_tokens=512,
                ),
            )
            raw = resp.text.strip()
            parsed = re.findall(r"^\d+\.\s*(.+)$", raw, re.MULTILINE)
            if parsed:
                # Deduplicate while preserving order; always keep original first
                seen = {query.lower()}
                for v in parsed:
                    v = v.strip()
                    if v.lower() not in seen:
                        variations.append(v)
                        seen.add(v.lower())
            logger.info(
                "Query expansion: 1 original + %d variations", len(variations) - 1
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query expansion failed (%s) – using original query", exc)
        return variations


# ============================================================================
# HybridRetriever
# ============================================================================

class HybridRetriever:
    """Wraps ChromaDB (vector) and BM25 (keyword) retrieval.

    Results are returned as lists of dicts with keys:
        doc_id   – ChromaDB document ID
        text     – chunk text
        metadata – original metadata dict
        score    – raw similarity / BM25 score
    """

    def __init__(self, collection: chromadb.Collection) -> None:
        self._collection = collection
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_ids: List[str] = []
        self._corpus_texts: List[str] = []
        self._corpus_metadata: List[Dict[str, Any]] = []
        self._build_bm25_index()

    # ------------------------------------------------------------------
    # BM25 index construction
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """Fetch all documents from ChromaDB and build an in-memory BM25 index."""
        try:
            total = self._collection.count()
            if total == 0:
                logger.warning("ChromaDB collection is empty – BM25 index not built")
                return

            # Fetch in batches to avoid memory spikes on large collections
            batch_size = 1000
            offset = 0
            while offset < total:
                result = self._collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                for doc_id, text, meta in zip(
                    result["ids"],
                    result["documents"],
                    result["metadatas"],
                ):
                    self._corpus_ids.append(doc_id)
                    self._corpus_texts.append(text)
                    self._corpus_metadata.append(meta or {})
                offset += batch_size

            tokenised = [t.lower().split() for t in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenised)
            logger.info("BM25 index built over %d documents", len(self._corpus_texts))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to build BM25 index: %s", exc)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query ChromaDB with dense embeddings."""
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
            hits = []
            for doc_id, text, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB returns L2 distance; convert to a pseudo-similarity
                score = 1.0 / (1.0 + dist)
                hits.append(
                    {
                        "doc_id": doc_id,
                        "text": text,
                        "metadata": meta or {},
                        "score": score,
                    }
                )
            logger.debug("Vector search returned %d hits for query: %.60s", len(hits), query)
            return hits
        except Exception as exc:  # noqa: BLE001
            logger.error("Vector search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # BM25 keyword search
    # ------------------------------------------------------------------

    def bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Sparse BM25 retrieval."""
        if self._bm25 is None or not self._corpus_texts:
            logger.warning("BM25 index unavailable – skipping keyword search")
            return []
        try:
            tokenised_query = query.lower().split()
            scores = self._bm25.get_scores(tokenised_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            hits = []
            for idx in top_indices:
                if scores[idx] > 0:
                    hits.append(
                        {
                            "doc_id": self._corpus_ids[idx],
                            "text": self._corpus_texts[idx],
                            "metadata": self._corpus_metadata[idx],
                            "score": float(scores[idx]),
                        }
                    )
            logger.debug("BM25 search returned %d hits for query: %.60s", len(hits), query)
            return hits
        except Exception as exc:  # noqa: BLE001
            logger.error("BM25 search failed: %s", exc)
            return []


# ============================================================================
# RRFCombiner – Reciprocal Rank Fusion
# ============================================================================

class RRFCombiner:
    """Merges multiple ranked lists using Reciprocal Rank Fusion.

    Formula:  RRF_score(doc) = Σ  1 / (k + rank_i)
    where rank_i is the 1-based rank of the document in result list i.
    """

    def __init__(self, k: int = 60) -> None:
        self._k = k

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        ranked_lists : list of ranked document lists (each already sorted best-first).

        Returns
        -------
        Merged list sorted by descending RRF score, with rrf_score injected.
        """
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc["doc_id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self._k + rank)
                if doc_id not in doc_store:
                    doc_store[doc_id] = doc

        fused = sorted(doc_store.values(), key=lambda d: rrf_scores[d["doc_id"]], reverse=True)
        for doc in fused:
            doc["rrf_score"] = rrf_scores[doc["doc_id"]]

        logger.debug(
            "RRF fused %d lists → %d unique docs",
            len(ranked_lists),
            len(fused),
        )
        return fused


# ============================================================================
# Reranker
# ============================================================================

class Reranker:
    """Cross-encoder reranker (ms-marco-MiniLM-L-6-v2).

    Downloads the model from HuggingFace on first use (~85 MB, free).
    """

    def __init__(self) -> None:
        self._model: Optional[CrossEncoder] = None
        self._model_name = config.RERANKER_MODEL
        self._load()

    def _load(self) -> None:
        try:
            self._model = CrossEncoder(self._model_name, max_length=512)
            logger.info("Cross-encoder reranker loaded: %s", self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load reranker (%s): %s", self._model_name, exc)

    def rerank(
        self, query: str, docs: List[Dict[str, Any]], top_n: int
    ) -> List[Dict[str, Any]]:
        """Score each (query, doc_text) pair and return top_n sorted best-first."""
        if not docs:
            return []
        if self._model is None:
            logger.warning("Reranker not loaded – returning docs in original order")
            return docs[:top_n]
        try:
            pairs = [(query, doc["text"]) for doc in docs]
            scores = self._model.predict(pairs)
            for doc, score in zip(docs, scores):
                doc["rerank_score"] = float(score)
            reranked = sorted(docs, key=lambda d: d.get("rerank_score", 0.0), reverse=True)
            logger.debug("Reranked %d → kept top %d", len(reranked), top_n)
            return reranked[:top_n]
        except Exception as exc:  # noqa: BLE001
            logger.error("Reranking failed: %s", exc)
            return docs[:top_n]


# ============================================================================
# ContextBuilder
# ============================================================================

class ContextBuilder:
    """Assembles the context string injected into the LLM prompt.

    Responsibilities:
    - Token-aware truncation (approximated by character count for speed).
    - Filtering chunks below a minimum relevance score.
    - Collecting unique source metadata.
    """

    def __init__(
        self,
        max_context_chars: int = config.MAX_CONTEXT_CHARS,
        min_relevance_score: float = config.MIN_RELEVANCE_SCORE,
    ) -> None:
        self._max_chars = max_context_chars
        self._min_score = min_relevance_score

    def build(
        self, docs: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        Returns
        -------
        context_text : str   – formatted context string for the prompt
        sources      : list  – deduplicated source identifiers
        """
        context_parts: List[str] = []
        sources: List[str] = []
        total_chars = 0

        for idx, doc in enumerate(docs, start=1):
            score = doc.get("rerank_score", doc.get("rrf_score", 0.0))
            if score < self._min_score:
                logger.debug("Skipping chunk %d (score %.4f < threshold)", idx, score)
                continue

            text = doc["text"].strip()
            meta = doc.get("metadata", {})
            source = meta.get("source", meta.get("document_id", f"chunk_{idx}"))

            chunk_header = f"[Chunk {idx} | Source: {source}]"
            chunk_str = f"{chunk_header}\n{text}\n"

            if total_chars + len(chunk_str) > self._max_chars:
                # Partial inclusion of the last chunk to fill remaining space
                remaining = self._max_chars - total_chars
                if remaining > len(chunk_header) + 50:
                    context_parts.append(chunk_str[:remaining] + "…")
                    if source not in sources:
                        sources.append(source)
                break

            context_parts.append(chunk_str)
            total_chars += len(chunk_str)
            if source not in sources:
                sources.append(source)

        context_text = "\n".join(context_parts)
        logger.info(
            "Context built: %d chunks, %d chars, %d sources",
            len(context_parts),
            total_chars,
            len(sources),
        )
        return context_text, sources


# ============================================================================
# LLMGenerator (Gemini)
# ============================================================================

class LLMGenerator:
    """Wraps the Gemini generative API for grounded answer generation."""

    _SYSTEM_PROMPT = (
        "You are a precise, helpful assistant. "
        "Answer the user's question using ONLY the information provided in the context below. "
        "If the context does not contain enough information to answer, say: "
        "'I could not find a sufficient answer in the provided documents.' "
        "Do not use any external knowledge or make assumptions beyond the context."
    )

    def __init__(self) -> None:
        self._model_name = config.GEMINI_MODEL
        self._gen_config = GenerationConfig(
            temperature=config.GENERATION_TEMPERATURE,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )

    def generate(self, query: str, context: str) -> str:
        """Call Gemini and return the grounded answer text."""
        prompt = (
            f"{self._SYSTEM_PROMPT}\n\n"
            f"=== CONTEXT ===\n{context}\n\n"
            f"=== QUESTION ===\n{query}\n\n"
            f"=== ANSWER ==="
        )
        try:
            model = genai.GenerativeModel(self._model_name)
            response = model.generate_content(prompt, generation_config=self._gen_config)
            answer = response.text.strip()
            logger.debug("LLM generated answer (%d chars)", len(answer))
            return answer
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini generation failed: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc


# ============================================================================
# RAGAS Evaluator (optional)
# ============================================================================

class RAGASEvaluator:
    """Runs RAGAS evaluation metrics on a completed RAG response.

    Only active when config.ENABLE_RAGAS = True and ragas is installed.
    Metrics computed: context_precision, context_recall, faithfulness, answer_relevancy.
    """

    def __init__(self) -> None:
        if not _RAGAS_AVAILABLE:
            logger.warning(
                "RAGAS is not installed. "
                "pip install ragas langchain langchain-google-genai"
            )

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> Optional[Dict[str, float]]:
        if not _RAGAS_AVAILABLE or not config.ENABLE_RAGAS:
            return None
        try:
            dataset = HFDataset.from_dict(
                {
                    "question": [query],
                    "answer": [answer],
                    "contexts": [contexts],
                    "ground_truth": [ground_truth or answer],
                }
            )
            llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GEMINI_API_KEY,
                temperature=0,
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.GEMINI_API_KEY,
            )
            result = ragas_evaluate(
                dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ],
                llm=llm,
                embeddings=embeddings,
            )
            metrics = {
                "context_precision": float(result["context_precision"]),
                "context_recall": float(result["context_recall"]),
                "faithfulness": float(result["faithfulness"]),
                "answer_relevancy": float(result["answer_relevancy"]),
            }
            logger.info("RAGAS metrics: %s", metrics)
            return metrics
        except Exception as exc:  # noqa: BLE001
            logger.error("RAGAS evaluation failed: %s", exc)
            return None


# ============================================================================
# RAGPipeline – the main orchestrator
# ============================================================================

class RAGPipeline:
    """Orchestrates the end-to-end RAG pipeline.

    The run() method returns a plain dict that can be consumed by:
    - The FastAPI layer (serving.py)
    - A Streamlit frontend directly (future compatibility)
    """

    def __init__(self) -> None:
        logger.info("Initialising RAG pipeline …")

        # --- Configure Gemini ---
        genai.configure(api_key=config.GEMINI_API_KEY)

        # --- ChromaDB ---
        try:
            chroma_client = chromadb.PersistentClient(
                path=config.CHROMA_DB_PATH,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = chroma_client.get_or_create_collection(
                name=config.COLLECTION_NAME
            )
            logger.info(
                "ChromaDB collection '%s' loaded (%d docs)",
                config.COLLECTION_NAME,
                self._collection.count(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ChromaDB initialisation failed: %s", exc)
            raise

        # --- Sub-components ---
        self._query_expander = QueryExpander()
        self._retriever = HybridRetriever(self._collection)
        self._rrf = RRFCombiner(k=config.RRF_K)
        self._reranker = Reranker()
        self._context_builder = ContextBuilder()
        self._generator = LLMGenerator()
        self._ragas = RAGASEvaluator()
        self._cache = LRUCache(max_size=config.CACHE_MAX_SIZE)

        logger.info("RAG pipeline ready.")

    # ------------------------------------------------------------------
    # Query normalisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(query: str) -> str:
        """Unicode NFC + collapse whitespace."""
        q = unicodedata.normalize("NFC", query)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    # ------------------------------------------------------------------
    # Core pipeline execution
    # ------------------------------------------------------------------

    def run(self, raw_query: str) -> Dict[str, Any]:
        """Execute the full RAG pipeline and return a response dict.

        The returned dict matches the QueryResponse schema and can be used
        directly by a Streamlit frontend without any additional transformation.
        """
        total_start = time.perf_counter()

        # ---- 0. Cache lookup ----
        if config.ENABLE_CACHE:
            cached = self._cache.get(raw_query)
            if cached is not None:
                cached["latency"]["cache_hit"] = True
                return cached

        # ---- 1. Query normalisation ----
        query = self._normalise(raw_query)
        logger.info("Received query: %s", query)

        # ---- 2. Query expansion ----
        expanded_queries = self._query_expander.expand(query)
        logger.info("Expanded queries: %s", expanded_queries)

        # ---- 3. Hybrid retrieval per query ----
        retrieval_start = time.perf_counter()
        all_ranked_lists: List[List[Dict[str, Any]]] = []

        for eq in expanded_queries:
            vec_hits = self._retriever.vector_search(eq, config.TOP_K)
            bm25_hits = self._retriever.bm25_search(eq, config.TOP_K)
            # Each retrieval method contributes its own ranked list to RRF
            if vec_hits:
                all_ranked_lists.append(vec_hits)
            if bm25_hits:
                all_ranked_lists.append(bm25_hits)

        # ---- 4. RRF fusion across all lists ----
        fused_docs = self._rrf.fuse(all_ranked_lists)
        logger.info(
            "RRF fused → %d unique docs | top RRF scores: %s",
            len(fused_docs),
            [round(d.get("rrf_score", 0), 4) for d in fused_docs[:5]],
        )

        retrieval_time = time.perf_counter() - retrieval_start

        # ---- 5. Reranking ----
        reranked_docs = self._reranker.rerank(query, fused_docs, config.RERANK_TOP_N)
        logger.info(
            "Reranked top-%d chunks | scores: %s",
            config.RERANK_TOP_N,
            [round(d.get("rerank_score", 0), 4) for d in reranked_docs],
        )

        # ---- 6. Context construction ----
        context_text, sources = self._context_builder.build(reranked_docs)
        if not context_text:
            logger.warning("Context is empty – query may be out of domain")

        # ---- 7. LLM generation ----
        generation_start = time.perf_counter()
        answer = self._generator.generate(query, context_text)
        generation_time = time.perf_counter() - generation_start

        total_time = time.perf_counter() - total_start

        # ---- 8. Optional RAGAS evaluation ----
        evaluation: Optional[Dict[str, float]] = None
        if config.ENABLE_RAGAS:
            contexts_for_eval = [d["text"] for d in reranked_docs]
            evaluation = self._ragas.evaluate(query, answer, contexts_for_eval)

        # ---- 9. Assemble response ----
        response: Dict[str, Any] = {
            "answer": answer,
            "sources": sources,
            "latency": {
                "retrieval_s": round(retrieval_time, 4),
                "generation_s": round(generation_time, 4),
                "total_s": round(total_time, 4),
                "cache_hit": False,
            },
            "retrieved_chunks": (
                [
                    {
                        "doc_id": d["doc_id"],
                        "text": d["text"][:300] + ("…" if len(d["text"]) > 300 else ""),
                        "metadata": d.get("metadata", {}),
                        "rrf_score": round(d.get("rrf_score", 0.0), 6),
                        "rerank_score": round(d.get("rerank_score", 0.0), 6),
                    }
                    for d in reranked_docs
                ]
                if config.DEBUG_MODE
                else None
            ),
            "evaluation": evaluation,
            # Extra fields for Streamlit frontend convenience
            "_meta": {
                "expanded_queries": expanded_queries,
                "num_fused_docs": len(fused_docs),
                "context_chars": len(context_text),
            },
        }

        # ---- 10. Cache the result ----
        if config.ENABLE_CACHE:
            self._cache.set(raw_query, response)

        return response


# ============================================================================
# FastAPI application
# ============================================================================

app = FastAPI(
    title="Advanced RAG API",
    description=(
        "Production-grade Retrieval-Augmented Generation API. "
        "Uses ChromaDB + BM25 hybrid retrieval, RRF fusion, "
        "cross-encoder reranking, and Gemini LLM generation."
    ),
    version="1.0.0",
)

# Allow all origins – tighten for production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Application lifecycle ----------

_pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise the RAG pipeline once at server startup."""
    global _pipeline  # noqa: PLW0603
    try:
        config.validate()
        _pipeline = RAGPipeline()
        logger.info("FastAPI server started successfully.")
    except Exception as exc:
        logger.critical("Startup failed: %s", exc)
        raise


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse, summary="System health check")
async def health() -> HealthResponse:
    """Returns the operational status of all pipeline components."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return HealthResponse(
        status="ok",
        chroma_collection=config.COLLECTION_NAME,
        reranker_loaded=_pipeline._reranker._model is not None,
        cache_enabled=config.ENABLE_CACHE,
        ragas_enabled=config.ENABLE_RAGAS and _RAGAS_AVAILABLE,
        debug_mode=config.DEBUG_MODE,
    )


@app.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Execute the full RAG pipeline for a user query.

    - Expands the query into multiple variations.
    - Retrieves relevant chunks via vector search + BM25.
    - Fuses results with Reciprocal Rank Fusion.
    - Reranks with a cross-encoder.
    - Generates a grounded answer with Gemini.
    - Optionally evaluates quality with RAGAS.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    try:
        # Run the synchronous pipeline in a thread pool to avoid blocking the event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None, _pipeline.run, request.query
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error processing query")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency=result["latency"],
        retrieved_chunks=result.get("retrieved_chunks"),
        evaluation=result.get("evaluation"),
    )


# ============================================================================
# Entrypoint
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "serving:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )