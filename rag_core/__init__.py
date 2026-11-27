"""rag_core: the generation path (retrieve -> context -> prompt -> LLM gateway -> validated, typed answer).

Import-safe: importing any module of this package creates no client, loads no model, opens no file and makes no network call.
Provider SDKs (google-genai) are imported lazily, when a provider is first used.

This package deliberately does NOT import ``rag.retrieval`` (deprecated, not import-safe) nor any Stage 2b module
(``rag.router_docs``, ``rag.gemini_embedder``): ``tests/unit/test_rag_core_imports.py`` enforces both.
"""
