"""Cross-encoder reranker, usable as the callable the Retriever accepts: ``reranker(query, texts) -> scores``.

Importing this module loads nothing. The model is loaded on first call, from the local HuggingFace cache only
(``local_files_only=True``) at a pinned revision; a missing cache raises a clear error instead of downloading.
"""

from __future__ import annotations

from collections.abc import Sequence


class CrossEncoderReranker:
    def __init__(self, model_name: str, revision: str, *, max_length: int = 512, batch_size: int = 16,
                 local_files_only: bool = True):
        self.name = model_name
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self._local_only = local_files_only
        self._model = None

    @classmethod
    def from_config(cls, cfg, **kwargs) -> CrossEncoderReranker:
        return cls(cfg.pipeline.reranker_model, cfg.pipeline.reranker_revision, **kwargs)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            try:
                self._model = CrossEncoder(self.name, revision=self.revision, max_length=self.max_length,
                                           local_files_only=self._local_only, device="cpu")
            except OSError as exc:
                raise RuntimeError(
                    f"{self.name}@{self.revision} is not in the local HuggingFace cache. "
                    "Download it once, explicitly, with huggingface_hub.snapshot_download."
                ) from exc
        return self._model

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def describe(self) -> dict:
        return {"class": type(self).__name__, "name": self.name, "revision": self.revision, "max_length": self.max_length,
                "batch_size": self.batch_size, "device": "cpu", "local_files_only": self._local_only}
