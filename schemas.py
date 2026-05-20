# ============================================
# IMPORTS
# ============================================

from typing import List, Optional

from pydantic import BaseModel


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