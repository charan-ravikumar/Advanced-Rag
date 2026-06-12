# ============================================
# IMPORTS
# ============================================

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================
# INGESTION MODELS
# ============================================

class Section(BaseModel):

    content: str

    section_title: str = ""

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


class Document(BaseModel):

    content: str

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    sections: List[Section] = Field(
        default_factory=list
    )


class Chunk(BaseModel):

    content: str

    metadata: Dict[str, Any] = Field(
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

    source: Optional[str] = None

    section_title: Optional[str] = None

    content: str

    rerank_score: float


# ============================================
# RAGAS RESPONSE
# ============================================

class RagasMetrics(BaseModel):

    faithfulness: Optional[float] = None

    answer_relevancy: Optional[float] = None


# ============================================
# QUERY RESPONSE
# ============================================

class QueryResponse(BaseModel):

    query: str

    answer: str

    sources: List[SourceDocument]

    ragas: Optional[RagasMetrics] = None