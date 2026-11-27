# ============================================
# IMPORTS
# ============================================

from typing import Any

from pydantic import BaseModel, Field

# ============================================
# INGESTION MODELS
# ============================================

class Section(BaseModel):

    content: str

    section_title: str = ""

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class Document(BaseModel):

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    sections: list[Section] = Field(
        default_factory=list
    )


class Chunk(BaseModel):

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================
# QUERY REQUEST
# ============================================

class QueryRequest(BaseModel):

    query: str


# ============================================
# SOURCE MODEL
# ============================================

class SourceDocument(BaseModel):

    source: str | None = None

    section_title: str | None = None

    content: str

    rerank_score: float


# ============================================
# RAGAS RESPONSE
# ============================================

class RagasMetrics(BaseModel):

    faithfulness: float | None = None

    answer_relevancy: float | None = None


# ============================================
# QUERY RESPONSE
# ============================================

class QueryResponse(BaseModel):

    query: str

    answer: str

    sources: list[SourceDocument]

    ragas: RagasMetrics | None = None