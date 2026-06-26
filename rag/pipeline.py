"""High-level RAGPipeline class that ties retrieval and generation together."""

from typing import Iterator

from config import cfg
from observability.logger_config import get_logger
from rag.retrieval import (
    classify_query,
    generate_answer_stream,
    generate_direct_answer_stream,
    retrieve_and_build_context,
)

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrate the full retrieval-augmented generation pipeline."""

    def __init__(self, config=None):
        """Initialise with an optional config override (defaults to the global cfg)."""
        self.config = config or cfg
        logger.info("RAGPipeline initialised.")

    def run(self, query: str) -> dict:
        """
        Execute the pipeline for *query* and return a result dict.

        Returns
        -------
        dict with keys:
            route   : "direct" | "rag"
            answer  : full answer string (generator consumed internally)
            context : retrieved context string (empty for direct answers)
            sources : list of source dicts (empty for direct answers)
            latency_ms : retrieval latency in milliseconds (0 for direct answers)
        """
        route = classify_query(query)
        logger.info(f"Pipeline route: {route}")

        if route == "direct":
            answer = "".join(generate_direct_answer_stream(query))
            return {
                "route": "direct",
                "answer": answer,
                "context": "",
                "sources": [],
                "latency_ms": 0,
            }

        retrieval_result = retrieve_and_build_context(query)
        context = retrieval_result["context"]
        sources = retrieval_result["sources"]
        latency_ms = retrieval_result["latency_ms"]

        answer = "".join(generate_answer_stream(query, context))

        return {
            "route": "rag",
            "answer": answer,
            "context": context,
            "sources": sources,
            "latency_ms": latency_ms,
        }

    def stream(self, query: str) -> Iterator[str]:
        """
        Stream answer tokens for *query*.

        Yields individual text tokens. Retrieval happens synchronously
        before the first token is yielded.
        """
        route = classify_query(query)
        logger.info(f"Pipeline stream route: {route}")

        if route == "direct":
            yield from generate_direct_answer_stream(query)
            return

        retrieval_result = retrieve_and_build_context(query)
        context = retrieval_result["context"]

        yield from generate_answer_stream(query, context)
