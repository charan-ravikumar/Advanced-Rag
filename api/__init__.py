"""A REST + server-sent-events API over the RAG service (FastAPI) and its Python client.

* ``api.main``: the application factory (``create_app``) and ``app`` for ``uvicorn api.main:app``.
* ``api.routes``: the endpoints. ``api.schemas``: every request, response and SSE event model.
* ``api.client``: a small synchronous client (httpx, imported lazily); the Streamlit app talks to the service through it.

Importing this package does nothing.
"""
