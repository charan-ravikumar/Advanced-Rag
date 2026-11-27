import bisect
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

from config import cfg
from observability.logger_config import get_logger
from rag.splitting import recursive_split_spans
from rag.tokens import max_chars, max_tokens_for, split_to_fit_spans, token_counter_for
from schemas import Chunk, Section

logger = get_logger(__name__)

# ============================================
# RECURSIVE CHUNKER
# ============================================

# Reference implementation only. recursive_chunk no longer calls it: it uses the offset-native
# rag.splitting.recursive_split_spans, which tests prove returns the same chunks with exact positions.
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.chunking.recursive_chunk_size,
    chunk_overlap=cfg.chunking.recursive_chunk_overlap
)


def _sections(document):
    """Sections to chunk within; a document without sections is one section."""
    if document.sections:
        return document.sections
    return [Section(content=document.content, section_title="", metadata={})]


def recursive_chunk(document, embedder=None, min_chunk_chars=None):
    """Split each section into overlapping fixed-size chunks (recursive character splitting).

    Chunks never cross a section boundary. A chunk shorter than ``min_chunk_chars`` is
    merged into its predecessor in the same section when the result fits the caps.
    Chunks over the token cap (character size does not bound tokens) are re-split.
    Every chunk records ``section_char_start`` / ``section_char_end``, computed while splitting.
    """
    min_chunk_chars = cfg.chunking.min_chunk_chars if min_chunk_chars is None else min_chunk_chars
    chunks = []
    count, max_tok = token_counter_for(embedder), max_tokens_for(embedder)

    def fits(s):
        return len(s) <= max_chars() and count(s) <= max_tok

    for sec_pos, section in enumerate(_sections(document)):
        text = section.content
        spans = recursive_split_spans(text, cfg.chunking.recursive_chunk_size, cfg.chunking.recursive_chunk_overlap)
        merged = []
        for s, e in spans:
            if merged and (e - s) < min_chunk_chars and fits(text[merged[-1][0]:e].strip()):
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        for s, e in merged:
            for a, b in split_to_fit_spans(text[s:e], max_chars(), max_tok, count):
                chunks.append(
                    Chunk(
                        content=text[s + a:s + b],
                        metadata={
                            **document.metadata,
                            "section_title": section.section_title,
                            "section_pos": sec_pos,
                            "section_char_start": s + a,
                            "section_char_end": s + b,
                            "chunk_id": len(chunks),
                            "chunking_strategy": "recursive"
                        }
                    )
                )
    return chunks


# ============================================
# SECTION / LAYOUT CHUNKER
# ============================================

def layout_chunk(document, embedder=None):
    """Split a document into chunks aligned with its pre-parsed section boundaries.

    A section larger than the embedding window is split into several chunks;
    ``section_pos`` records which section each chunk came from.
    """
    chunks = []
    count, max_tok = token_counter_for(embedder), max_tokens_for(embedder)
    for sec_pos, section in enumerate(document.sections):
        for a, b in split_to_fit_spans(section.content, max_chars(), max_tok, count):
            chunks.append(
                Chunk(
                    content=section.content[a:b],
                    metadata={
                        **document.metadata,
                        **section.metadata,
                        "section_title": section.section_title,
                        "section_pos": sec_pos,
                        "section_char_start": a,
                        "section_char_end": b,
                        "chunk_id": len(chunks),
                        "chunking_strategy": "layout"
                    }
                )
            )
    return chunks


# ============================================
# EMBEDDING MODEL
# ============================================

_default_embedder = None


def _get_embedding_model():
    """Default embedder for the semantic chunker, created lazily (never at import)."""
    global _default_embedder
    if _default_embedder is None:
        from rag.embedders import SentenceTransformerEmbedder

        _default_embedder = SentenceTransformerEmbedder.from_config(cfg)
    return _default_embedder


# ============================================
# SEMANTIC CHUNKER
# ============================================

def semantic_chunk(
    document,
    similarity_threshold=None,
    max_chunk_chars=None,
    min_chunk_chars=None,
    window_size=None,
    embedder=None
):
    """Split a document into semantically coherent chunks using embedding similarity."""
    c = cfg.chunking
    similarity_threshold = c.semantic_similarity_threshold if similarity_threshold is None else similarity_threshold
    max_chunk_chars = c.max_chunk_chars if max_chunk_chars is None else max_chunk_chars
    min_chunk_chars = c.min_chunk_chars if min_chunk_chars is None else min_chunk_chars
    window_size = c.semantic_window_size if window_size is None else window_size
    if embedder is None:
        embedder = _get_embedding_model()
    count, max_tok = token_counter_for(embedder), max_tokens_for(embedder)

    logger.info(
        f"Starting semantic chunking for: {document.metadata.get('source')}"
    )

    chunks = []
    for sec_pos, section in enumerate(_sections(document)):
        par_spans = _paragraph_spans(section.content)
        paragraphs = [section.content[a:b] for a, b in par_spans]
        if not paragraphs:
            continue
        groups = _group_paragraphs(
            paragraphs, embedder, similarity_threshold, max_chunk_chars, min_chunk_chars, window_size
        )
        if [p for g in groups for p in g] != paragraphs:
            raise AssertionError("semantic grouping must keep every paragraph, once, in order")
        first = 0
        for group in groups:
            spans = par_spans[first:first + len(group)]
            first += len(group)
            joined = "\n\n".join(group)
            # offset of each paragraph inside the joined text
            starts, pos = [], 0
            for p in group:
                starts.append(pos)
                pos += len(p) + 2
            # a single paragraph can exceed the caps on its own: re-split it
            for a, b in split_to_fit_spans(joined, max_chunk_chars, max_tok, count):
                i = max(0, bisect.bisect_right(starts, a) - 1)
                j = max(0, bisect.bisect_right(starts, b - 1) - 1)
                sec_start = spans[i][0] + max(0, a - starts[i])
                sec_end = spans[j][0] + (b - starts[j])
                piece = joined[a:b]
                chunks.append(
                    Chunk(
                        content=piece,
                        metadata={
                            **document.metadata,
                            "section_title": section.section_title,
                            "section_pos": sec_pos,
                            "section_char_start": sec_start,
                            "section_char_end": sec_end,
                            "chunk_id": len(chunks),
                            "chunking_strategy": "semantic",
                            "chunk_size_chars": len(piece),
                            "num_paragraphs": len(group),
                            "similarity_threshold": similarity_threshold
                        }
                    )
                )

    logger.info(f"Semantic chunking complete. Final chunks: {len(chunks)}")
    return chunks


def _paragraph_spans(content):
    """``(start, end)`` of each non-empty paragraph (blank-line separated, stripped), computed from the section text.

    Equivalent to ``re.sub(r"\\n{3,}", "\\n\\n", content).strip()`` followed by ``re.split(r"\\n\\s*\\n", ...)``
    and stripping each piece (proved on random text and the real corpus in tests/unit/test_splitting.py).
    """
    raw, pos = [], 0
    for m in re.finditer(r"\n\s*\n", content):
        raw.append((pos, m.start()))
        pos = m.end()
    raw.append((pos, len(content)))
    out = []
    for a, b in raw:
        seg = content[a:b]
        stripped = seg.strip()
        if stripped:
            left = len(seg) - len(seg.lstrip())
            out.append((a + left, a + left + len(stripped)))
    return out


def _group_paragraphs(paragraphs, model, similarity_threshold, max_chunk_chars, min_chunk_chars, window_size):
    """Group the paragraphs of ONE section by embedding similarity; never crosses sections."""
    sep = len("\n\n")
    # ----------------------------------------
    # STEP 2 - SLIDING WINDOW
    # ----------------------------------------

    windows = []
    for i in range(len(paragraphs)):
        start = max(0, i - window_size)
        end = min(len(paragraphs), i + window_size + 1)
        windows.append(" ".join(paragraphs[start:end]))

    # ----------------------------------------
    # STEP 3 - EMBEDDINGS
    # ----------------------------------------

    embeddings = model.encode(windows, show_progress_bar=False)

    # ----------------------------------------
    # STEP 4 - SEMANTIC GROUPING
    # ----------------------------------------

    semantic_groups = []
    current_group = [paragraphs[0]]
    current_chunk_size = len(paragraphs[0])

    for i in range(1, len(paragraphs)):
        similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        next_para = paragraphs[i]
        projected_size = current_chunk_size + sep + len(next_para)

        should_merge = (
            similarity >= similarity_threshold
            and projected_size <= max_chunk_chars
        )

        if should_merge:
            current_group.append(next_para)
            current_chunk_size = projected_size
        else:
            semantic_groups.append(current_group)
            current_group = [next_para]
            current_chunk_size = len(next_para)

    semantic_groups.append(current_group)
    logger.debug(f"Initial semantic groups: {len(semantic_groups)}")

    # ----------------------------------------
    # STEP 5 - SMALL CHUNK MERGING
    # ----------------------------------------

    balanced_groups = []
    temp_group = []

    def joined_size(*parts):
        return len("\n\n".join("\n\n".join(p) for p in parts))

    for group in semantic_groups:
        if joined_size(group) < min_chunk_chars:
            # flush rather than grow past the cap
            if temp_group and joined_size(temp_group, group) > max_chunk_chars:
                balanced_groups.append(temp_group)
                temp_group = []
            temp_group.extend(group)
        else:
            if temp_group:
                if joined_size(temp_group, group) <= max_chunk_chars:
                    group = temp_group + group
                else:
                    # previously the small group was silently dropped here
                    balanced_groups.append(temp_group)
                temp_group = []
            balanced_groups.append(group)

    if temp_group:
        if balanced_groups:
            combined_size = joined_size(balanced_groups[-1], temp_group)
            if combined_size <= max_chunk_chars:
                balanced_groups[-1].extend(temp_group)
            else:
                balanced_groups.append(temp_group)
        else:
            balanced_groups.append(temp_group)

    logger.debug(f"Balanced groups: {len(balanced_groups)}")
    return balanced_groups


# ============================================
# CHUNKER MAP
# ============================================

CHUNKER_MAP = {
    "recursive": recursive_chunk,
    "layout": layout_chunk,
    "semantic": semantic_chunk
}