# app/retriever.py
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import logging
import math
import re
import threading
import unicodedata

try:
    from .config_env import (
        DEFAULT_EMBED_MODEL,
        embedding_model_kwargs,
        format_embedding_document,
        hebrew_token_variants,
        load_project_environment,
        normalize_search_text,
        validate_tenant_id,
    )
    from .source_access import (
        source_metadata_complete,
        validate_source_id,
    )
    from .request_performance import measure, record_cache, timed_stage
    from .eval_trace import trace_retrieval
except ImportError:
    from config_env import (
        DEFAULT_EMBED_MODEL,
        embedding_model_kwargs,
        format_embedding_document,
        hebrew_token_variants,
        load_project_environment,
        normalize_search_text,
        validate_tenant_id,
    )
    from source_access import (
        source_metadata_complete,
        validate_source_id,
    )
    from request_performance import measure, record_cache, timed_stage
    from eval_trace import trace_retrieval

load_project_environment(Path(__file__).resolve().parent)

import numpy as np
from chromadb import PersistentClient
try:
    from chromadb.errors import NotFoundError as ChromaNotFoundError
except ImportError:  # pragma: no cover - compatibility with old clients
    class ChromaNotFoundError(Exception):
        """Fallback that deliberately does not swallow unrelated client errors."""

from sentence_transformers import SentenceTransformer


_EMBED_INFERENCE_LOCK = threading.RLock()
_RERANK_INFERENCE_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


class RetrievalCollectionNotFoundError(RuntimeError):
    """Raised when the serving path cannot find its pre-ingested collection.

    Collection creation belongs exclusively to the ingestion pipeline.  Keeping
    this error explicit lets the API distinguish an unknown collection-backed
    tenant from a missing shared metadata collection without mutating storage
    while serving a request.
    """

    def __init__(
        self,
        *,
        collection_name: str,
        tenant_id: str,
        tenant_strategy: str,
    ) -> None:
        self.collection_name = collection_name
        self.tenant_id = tenant_id
        self.tenant_strategy = tenant_strategy
        super().__init__(
            f"Retrieval collection {collection_name!r} is not initialized "
            f"for tenant {tenant_id!r}. Run ingestion before serving queries."
        )


def _bounded_env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Read an integer setting without making module import fragile."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Read a finite float setting and clamp it to its valid range."""
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


@lru_cache(maxsize=4)
def _shared_embedder(model_name: str, device: str = "cpu"):
    """Load one heavyweight embedding model per process/configuration."""
    model = SentenceTransformer(
        model_name, device=device, model_kwargs=embedding_model_kwargs(device),
    )
    _assert_selected_device(model, device, "embedder")
    return model


@lru_cache(maxsize=4)
def _shared_reranker(model_name: str, max_length: int, device: str = "cpu"):
    """Load one cross-encoder per process instead of one per tenant."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, max_length=max_length, device=device)
    _assert_selected_device(model, device, "reranker")
    return model


def _assert_selected_device(model, requested: str, component: str) -> str:
    selected = getattr(model, "device", None)
    if selected is None:
        selected = getattr(getattr(model, "model", None), "device", None)
    actual = str(selected if selected is not None else "unknown").casefold()
    if requested.casefold() == "cpu" and actual != "unknown" and not actual.startswith("cpu"):
        raise RuntimeError(f"{component} requested CPU but initialized on {actual}")
    logger.info(
        "Retrieval model device component=%s requested=%s actual=%s",
        component,
        requested,
        actual,
    )
    return actual

# ---------------- Env / defaults ----------------
CHROMA_PATH = os.getenv("CHROMA_PATH", "storage/chroma")
BASE_COLLECTION = os.getenv("RAG_COLLECTION", "company-knowledge")
TENANT_STRATEGY = os.getenv("TENANT_STRATEGY", "collection")  # "collection" | "metadata"

EMBED_MODEL = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu").strip() or "cpu"
EMBED_BATCH = _bounded_env_int("EMBED_BATCH", 64, minimum=1, maximum=1024)

# Retrieval knobs. Vector recall and reranker breadth are separate budgets:
# SEARCH_K=96 must not be silently truncated just because RERANK_POOL_K=48.
_MAX_POOL = _bounded_env_int(
    "VECTOR_SEARCH_POOL_K", 256, minimum=1, maximum=4096
)  # upper bound on how many vector candidates we ever pull
RERANK_POOL_K = _bounded_env_int(
    "RERANK_POOL_K", 64, minimum=1, maximum=_MAX_POOL
)
MAX_RETRIEVAL_QUERY_CHARS = _bounded_env_int(
    "MAX_RETRIEVAL_QUERY_CHARS", 4096, minimum=128, maximum=32768
)

# Reranker (cross-encoder) toggles
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"
RERANK_REQUIRED = os.getenv("RERANK_REQUIRED", "0") == "1"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = _bounded_env_int(
    "RERANK_CANDIDATES", 64, minimum=1, maximum=RERANK_POOL_K
)
RERANK_BATCH = _bounded_env_int("RERANK_BATCH", 16, minimum=1, maximum=256)
RERANK_MAX_LENGTH = _bounded_env_int(
    "RERANK_MAXLEN", 1024, minimum=128, maximum=4096
)
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu").strip() or "cpu"

# MMR toggles
USE_MMR = os.getenv("USE_MMR", "0") == "1"
MMR_LAMBDA = _bounded_env_float(
    "MMR_LAMBDA", 0.5, minimum=0.0, maximum=1.0
)

# Lexical fallback is deliberately paged and capped. It used to materialize an
# entire tenant collection on every hybrid query, making latency and memory
# proportional to corpus size rather than a configured budget.
LEXICAL_SCAN_BATCH = _bounded_env_int(
    "LEXICAL_SCAN_BATCH", 512, minimum=32, maximum=4096
)
LEXICAL_SCAN_MAX_DOCS = _bounded_env_int(
    "LEXICAL_SCAN_MAX_DOCS", 20000, minimum=LEXICAL_SCAN_BATCH, maximum=250000
)
LEXICAL_MAX_QUERY_TOKENS = _bounded_env_int(
    "LEXICAL_MAX_QUERY_TOKENS", 64, minimum=4, maximum=256
)
LEXICAL_MAX_VARIANTS = _bounded_env_int(
    "LEXICAL_MAX_VARIANTS", 192, minimum=8, maximum=1024
)
LEXICAL_MAX_RESULTS = _bounded_env_int(
    "LEXICAL_MAX_RESULTS", 2048, minimum=32, maximum=10000
)
EXACT_RETRIEVAL_MAX_TERM_CHARS = _bounded_env_int(
    "EXACT_RETRIEVAL_MAX_TERM_CHARS", 160, minimum=8, maximum=512
)
CONJUNCTIVE_RETRIEVAL_MAX_TERMS = _bounded_env_int(
    "CONJUNCTIVE_RETRIEVAL_MAX_TERMS", 4, minimum=2, maximum=8
)

# Optional: prefer cosine scoring; assume vectors normalized at ingest
HNSW_SPACE = os.getenv("HNSW_SPACE", "cosine")


_HEB_RE = re.compile(r"[\u0590-\u05FF]")

_LEXICAL_STOPWORDS = frozenset({
    # English question/document glue. These remain available inside an exact
    # phrase, but never qualify a document as an independent lexical hit.
    "a", "an", "and", "are", "as", "at", "based", "by", "did", "do",
    "does", "document", "file", "for", "from", "how", "in", "is", "it",
    "many", "much", "of", "on", "or", "please", "report", "say", "says",
    "show", "shows", "the", "this", "to", "what", "which", "who", "with",
    # Equivalent Hebrew glue.
    "או", "אודות", "איזה", "אילו", "איך", "איפה", "את", "בבקשה", "בדוח",
    "בקובץ", "במסמך", "דוח", "האם", "הדוח", "הוא", "היא", "הם",
    "הן", "המסמך", "הקובץ", "זה", "זו", "זאת", "יש", "כמה", "לגבי",
    "לפי", "מה", "מי", "מסמך", "נאמר", "על", "קובץ", "של", "כתוב",
})

# Deduplicate only inside the already-bounded retrieval pool.  Trigram overlap
# is deliberately conservative so ordinary chunk overlap does not collapse
# passages that still contain meaningfully different evidence.
_DEDUP_MIN_TOKENS = 8
_DEDUP_MIN_LENGTH_RATIO = 0.80
_DEDUP_JACCARD = 0.82
_DEDUP_CONTAINMENT = 0.92
_NORM_EPS = 1e-12


@(globals().get("timed_stage") or (lambda _name: lambda function: function))(
    "query_normalization_ms"
)
def _prepare_retrieval_query(value) -> str:
    """Normalize untrusted query text and enforce one retrieval-wide bound.

    This intentionally preserves punctuation and casing useful to embedding
    and reranker models. Lexical matching applies its stronger search-only
    normalization separately.
    """
    raw = unicodedata.normalize("NFKC", str(value or ""))
    cleaned: List[str] = []
    for character in raw:
        if character.isspace():
            cleaned.append(" ")
            continue
        category = unicodedata.category(character)
        if category.startswith("C"):
            # Drop bidi/zero-width controls, NULs, and other non-text controls.
            continue
        cleaned.append(character)
    query = re.sub(r"\s+", " ", "".join(cleaned)).strip()
    return query[:MAX_RETRIEVAL_QUERY_CHARS].rstrip()


def _l2_normalize_vector(vector) -> np.ndarray:
    """Return a finite unit vector, preserving a zero vector as zero.

    Ingestion stores normalized vectors, so query vectors must use the same
    geometry even when a custom/mock embedder ignores SentenceTransformers'
    ``normalize_embeddings`` option.
    """
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional embedding")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embedding contains non-finite values")

    # Accumulate in float64 so a finite float32 embedding cannot overflow
    # merely while its norm is being calculated.
    norm = float(np.linalg.norm(arr.astype(np.float64, copy=False)))
    if not np.isfinite(norm) or norm <= _NORM_EPS:
        return np.zeros_like(arr)
    return arr / norm


def _l2_normalize_rows(vectors) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize embedding rows and return ``(vectors, valid_row_mask)``.

    Chroma collections created before normalized ingestion was enabled may
    still contain vectors with arbitrary magnitudes.  MMR needs cosine
    similarity, so it normalizes every returned row itself instead of relying
    on the collection's history. Invalid/zero rows remain zero and are marked
    invalid so they cannot outrank usable evidence.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("expected a two-dimensional embedding matrix")
    if arr.shape[0] == 0:
        return arr, np.zeros(0, dtype=bool)

    finite_rows = np.all(np.isfinite(arr), axis=1)
    clean = np.where(np.isfinite(arr), arr, 0.0)
    norms = np.linalg.norm(clean.astype(np.float64, copy=False), axis=1)
    valid_rows = finite_rows & np.isfinite(norms) & (norms > _NORM_EPS)

    normalized = np.zeros_like(clean)
    normalized[valid_rows] = clean[valid_rows] / norms[valid_rows, None]
    return normalized, valid_rows


def _document_fingerprint(text: str) -> Tuple[str, int, frozenset]:
    """Return a language-agnostic, punctuation-insensitive chunk fingerprint."""
    normalized = normalize_search_text(text)
    # Keep numeric punctuation and currency symbols as atomic fact tokens.
    # Splitting ``1,234.56`` into three unrelated integers made deduplication
    # unable to distinguish locale-formatted values and their units reliably.
    tokens = re.findall(
        r"[-+]?\d+(?:[.,\u202f ]\d+)*%?|[%$\u20aa\u20ac]|"
        r"[\w\u0590-\u05FF]+",
        normalized,
        flags=re.UNICODE,
    )
    canonical = " ".join(tokens)
    shingles = frozenset(
        tuple(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))
    )
    return canonical, len(tokens), shingles


def _fingerprints_match(
    left: Tuple[str, int, frozenset], right: Tuple[str, int, frozenset]
) -> bool:
    """Conservatively identify exact and near-duplicate chunk text."""
    left_text, left_count, left_shingles = left
    right_text, right_count, right_shingles = right

    if left_text == right_text:
        return True

    def fact_signature(value: str):
        numbers = tuple(re.findall(
            r"(?<!\w)[-+]?\d+(?:[.,\u202f ]\d+)*%?(?!\w)", value
        ))
        units = tuple(sorted(set(re.findall(
            r"(?i)(?:[%$\u20aa\u20ac]|(?<!\w)(?:mw|kw|gw|mwh|kwh|gwh|percent|"
            r"usd|eur|ils|nis|dollars?|euros?|shekels?|million|billion|"
            r"years?|months?|days?|hours?|מגה.?ואט|קילו.?ואט|גיגה.?ואט|"
            r"אחוז(?:ים)?|שח|שקל(?:ים)?|דולר(?:ים)?|אירו|יורו|"
            r"מיליון|מיליארד|שנים?|חודשים?|ימים?|שעות?)(?!\w))",
            value,
        ))))
        negated = bool(re.search(
            r"(?i)(?:\b(?:not|no|never|without|didnt|doesnt|isnt|cannot|"
            r"did\s+not|does\s+not|is\s+not)\b|"
            r"(?:^|\s)(?:לא|אין|אינו|אינה|איננו|איננה|ללא|בלי)(?=\s|$))",
            value,
        ))
        exception = bool(re.search(
            r"(?i)(?:\b(?:except|unless|only|provided\s+that|subject\s+to)\b|"
            r"(?:^|\s)(?:למעט|אלא|רק|בתנאי|כפוף)(?=\s|$))",
            value,
        ))
        qualifiers = tuple(bool(re.search(pattern, value, re.IGNORECASE)) for pattern in (
            r"(?:at\s+least|minimum|no\s+less\s+than|לפחות|מינימום|לא\s+פחות\s+מ)",
            r"(?:at\s+most|maximum|up\s+to|no\s+more\s+than|לכל\s+היות|מקסימום|עד)",
            r"(?:approximately|about|roughly|estimated|בערך|בקירוב|משוער)",
            r"(?:between|from\s+.+\s+to|בין|מ.+\s+עד)",
            r"(?:if|provided\s+that|subject\s+to|אם|בתנאי|בכפוף)",
        ))
        return numbers, units, negated, exception, qualifiers

    # Similar boilerplate with different facts is not a duplicate.
    if fact_signature(left_text) != fact_signature(right_text):
        return False
    if min(left_count, right_count) < _DEDUP_MIN_TOKENS:
        return False

    length_ratio = min(left_count, right_count) / max(left_count, right_count)
    if length_ratio < _DEDUP_MIN_LENGTH_RATIO:
        return False

    overlap = len(left_shingles.intersection(right_shingles))
    if overlap == 0:
        return False

    union = len(left_shingles.union(right_shingles))
    shorter = min(len(left_shingles), len(right_shingles))
    jaccard = overlap / union if union else 0.0
    containment = overlap / shorter if shorter else 0.0
    return jaccard >= _DEDUP_JACCARD or containment >= _DEDUP_CONTAINMENT


def _dedup_scope(meta: Optional[Dict]) -> Tuple[str, str]:
    """Return the document/section identity within which dedup is safe."""
    value = meta if isinstance(meta, dict) else {}
    source = normalize_search_text(
        value.get("source") or value.get("path") or value.get("source_path") or ""
    )
    section = normalize_search_text(
        value.get("section")
        or value.get("title")
        or value.get("breadcrumb")
        or value.get("heading")
        or ""
    )
    return source, section


def _stable_unique_indices(
    order: List[int],
    docs: List[str],
    k: int,
    metas: Optional[List[Dict]] = None,
) -> List[int]:
    """Keep ranking order while filling up to ``k`` distinct, non-empty chunks.

    ``order`` may cover only a ranked prefix (for example, when a reranker has
    a configured candidate cap), so unseen candidate indices are appended in
    their original vector-search order before selection.
    """
    target = max(0, min(k, len(docs)))
    if target == 0:
        return []

    ranked: List[int] = []
    ranked_set = set()
    for value in order:
        try:
            index = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= index < len(docs) and index not in ranked_set:
            ranked.append(index)
            ranked_set.add(index)

    ranked.extend(i for i in range(len(docs)) if i not in ranked_set)

    selected: List[int] = []
    fingerprints: List[
        Tuple[Tuple[str, int, frozenset], Tuple[str, str]]
    ] = []
    for index in ranked:
        fingerprint = _document_fingerprint(docs[index])
        if not fingerprint[0]:
            continue
        meta = metas[index] if metas and index < len(metas) else {}
        scope = _dedup_scope(meta)
        if any(
            any(scope)
            and scope == seen_scope
            and _fingerprints_match(fingerprint, seen_fingerprint)
            for seen_fingerprint, seen_scope in fingerprints
        ):
            continue
        selected.append(index)
        fingerprints.append((fingerprint, scope))
        if len(selected) == target:
            break

    return selected

def _lexical_tokens(value: str) -> List[str]:
    normalized = normalize_search_text(value)
    return re.findall(r"[A-Za-z0-9_]+|[\u05d0-\u05ea]+", normalized)


@lru_cache(maxsize=8192)
def _cached_lexical_token_variants(token: str) -> frozenset[str]:
    """Memoize only the immutable, source-independent token transformation."""
    normalized = normalize_search_text(token)
    if not normalized:
        return frozenset()
    if re.fullmatch(r"[\u05d0-\u05ea]+", normalized):
        return frozenset(hebrew_token_variants(normalized))
    return frozenset((normalized,))


def _lexical_token_variants(token: str) -> frozenset[str]:
    # Bound retained key size as well as entry count. Unusually long OCR or
    # identifier tokens still use the exact transformation without retention.
    if len(token) > 128:
        return _cached_lexical_token_variants.__wrapped__(token)
    return _cached_lexical_token_variants(token)


def _kw_variants(value: str) -> list[str]:
    """Build deterministic, bounded lexical variants for a hybrid query.

    Individual stopwords are excluded so a query such as "what does the
    document say about Acme" cannot pull every chunk containing "the". They
    remain inside phrase variants, preserving exact natural-language matches.
    Hebrew reversal variants are limited to Hebrew runs; reversing Latin model
    names (ISO -> OSI) created unrelated exact hits in mixed-language queries.
    """
    query = _prepare_retrieval_query(value)
    tokens = _lexical_tokens(query)[:LEXICAL_MAX_QUERY_TOKENS]
    if not tokens:
        return []

    content_positions = {
        index
        for index, token in enumerate(tokens)
        if len(token) >= 2 and token.casefold() not in _LEXICAL_STOPWORDS
    }
    if not content_positions:
        return []

    output: List[str] = []
    seen: set[str] = set()

    def add(variant: str) -> None:
        normalized = " ".join(_lexical_tokens(variant))
        if (
            normalized
            and normalized not in seen
            and len(output) < LEXICAL_MAX_VARIANTS
        ):
            seen.add(normalized)
            output.append(normalized)

    canonical = " ".join(tokens)
    add(canonical)

    # Longer local phrases are more discriminative, so add trigrams before
    # bigrams. Every n-gram must contain at least one substantive query token.
    for width in (3, 2):
        for start in range(max(0, len(tokens) - width + 1)):
            positions = set(range(start, start + width))
            if positions & content_positions:
                add(" ".join(tokens[start:start + width]))

    for index in sorted(content_positions):
        add(tokens[index])

    # PyPDF2 can reverse Hebrew glyph runs. Reverse only all-Hebrew variants;
    # mixed Hebrew/Latin phrases keep identifiers and numbers in their real form.
    snapshot = list(output)
    for variant in snapshot:
        variant_tokens = _lexical_tokens(variant)
        if variant_tokens and all(
            re.fullmatch(r"[\u05d0-\u05ea]+", token) for token in variant_tokens
        ):
            add(variant[::-1])

    return output


def _lexical_variant_groups(variant: str) -> List[frozenset[str]]:
    return [
        variants
        for variants in (
            _lexical_token_variants(token) for token in _lexical_tokens(variant)
        )
        if variants
    ]


def _lexical_document_positions(
    document_groups: List[frozenset[str]],
) -> Dict[str, set[int]]:
    """Index the existing token forms within one freshly read document."""
    positions: Dict[str, set[int]] = {}
    for position, variants in enumerate(document_groups):
        for variant in variants:
            positions.setdefault(variant, set()).add(position)
    return positions


def _lexical_groups_match(
    query_groups: List[frozenset[str]],
    document_groups: List[frozenset[str]],
    positions: Optional[Dict[str, set[int]]] = None,
) -> bool:
    """Match identical ordered groups, checking only possible phrase starts."""
    if not query_groups or len(query_groups) > len(document_groups):
        return False
    width = len(query_groups)
    if positions is None:
        positions = _lexical_document_positions(document_groups)
    anchor_positions: Optional[set[int]] = None
    anchor_offset = 0
    for offset, query_group in enumerate(query_groups):
        matching_positions: set[int] = set()
        for variant in query_group:
            matching_positions.update(positions.get(variant, ()))
        if not matching_positions:
            return False
        if width == 1:
            return True
        if anchor_positions is None or len(matching_positions) < len(anchor_positions):
            anchor_positions = matching_positions
            anchor_offset = offset
            if len(matching_positions) == 1:
                break
    # Any match must contain the anchor at its declared offset. Retain the
    # original full ordered-group predicate at every such valid start.
    maximum_start = len(document_groups) - width
    return any(
        all(
            bool(query_groups[offset] & document_groups[start + offset])
            for offset in range(width)
        )
        for position in (anchor_positions or ())
        if 0 <= (start := position - anchor_offset) <= maximum_start
    )


def _lexical_variant_matches(variant: str, document_tokens: List[str]) -> bool:
    """Match complete Unicode tokens/phrases, including Hebrew clitics.

    Raw substring matching made short terms such as ``HR`` match unrelated
    words such as ``throughput``. Token-sequence matching keeps the Hebrew PDF
    variants while removing that source of false evidence.
    """
    query_groups = _lexical_variant_groups(variant)
    document_groups = [
        _lexical_token_variants(token) for token in document_tokens
    ]
    return _lexical_groups_match(query_groups, document_groups)


def _lexical_variant_weight(variant: str, canonical_query: str) -> float:
    """Give exact phrases more influence than isolated token matches."""
    token_count = len(_lexical_tokens(variant))
    if not token_count:
        return 0.0
    if variant == canonical_query:
        return float(min(10, 3 + token_count))
    if token_count >= 3:
        return 3.0
    if token_count == 2:
        return 2.0
    return 1.0


def _combine_where_filters(
    tenant_filter: Optional[Dict], requested_filter: Optional[Dict]
) -> Optional[Dict]:
    """Keep tenant isolation immutable when a caller adds a source filter."""
    for label, value in (
        ("tenant_filter", tenant_filter),
        ("requested_filter", requested_filter),
    ):
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"{label} must be a dictionary or None")

    def clone(value):
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return [clone(item) for item in value]
        return value

    base = clone(tenant_filter) if tenant_filter else None
    requested = clone(requested_filter) if requested_filter else None
    if base and requested:
        return {"$and": [base, requested]}
    return base or requested

def _safe_tolist(x):
    """Convert numpy arrays or torch tensors to plain Python lists; leave lists/dicts/None as-is."""
    if x is None:
        return None
    try:
        if hasattr(x, "tolist"):
            return x.tolist()
        return x
    except Exception:
        return x


def _first_query_row(response: Dict, key: str) -> list:
    """Read one Chroma query row while tolerating absent/malformed fields."""
    if not isinstance(response, dict):
        return []
    value = _safe_tolist(response.get(key))
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return []
    value = list(value)
    if not value:
        return []
    first = _safe_tolist(value[0])
    if isinstance(first, (list, tuple)):
        return list(first)
    # Chroma query responses are nested, but accepting a flat row makes the
    # fallback resilient to lightweight adapters and older client wrappers.
    return value


def _flat_get_field(response: Dict, key: str) -> list:
    """Read a Chroma ``get`` field, tolerating one accidental wrapper row."""
    if not isinstance(response, dict):
        return []
    value = _safe_tolist(response.get(key))
    if not isinstance(value, (list, tuple)):
        return []
    values = list(value)
    if len(values) == 1:
        first = _safe_tolist(values[0])
        if isinstance(first, (list, tuple)):
            return list(first)
    return values


def _finite_float_or_none(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _coerce_reranker_scores(value, expected: int) -> Optional[List[float]]:
    """Validate the cross-encoder contract before scores affect ordering."""
    raw = _safe_tolist(value)
    if not isinstance(raw, (list, tuple)) or len(raw) != expected:
        return None
    scores: List[float] = []
    for item in raw:
        # Some wrappers return shape [N, 1]. Accept that exact shape, but not
        # multi-class logits whose ordering semantics are undefined here.
        item = _safe_tolist(item)
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                return None
            item = item[0]
        score = _finite_float_or_none(item)
        if score is None:
            return None
        scores.append(score)
    return scores

E5_TASK = os.getenv(
    "E5_TASK",
    "Given a question, retrieve relevant passages that answer the question"
)

def _format_e5_query(q: str) -> str:
    q = q or ""
    m = EMBED_MODEL.lower()
    if "e5" not in m:
        return q
    if "instruct" in m:
        return f"Instruct: {E5_TASK}\nQuery: {q}"
    return "query: " + q

class Retriever:
    def __init__(self, tenant_id: Optional[str] = None) -> None:
        if TENANT_STRATEGY not in {"collection", "metadata"}:
            raise ValueError("TENANT_STRATEGY must be 'collection' or 'metadata'")
        tenant_id = validate_tenant_id(tenant_id or "")
        self.tenant_id = tenant_id
        self.client = PersistentClient(path=CHROMA_PATH)

        # Collection selection strategy
        if TENANT_STRATEGY == "collection":
            name = f"{BASE_COLLECTION}_{tenant_id}"
            try:
                self.collection = self.client.get_collection(name=name)
            except ChromaNotFoundError as exc:
                raise RetrievalCollectionNotFoundError(
                    collection_name=name,
                    tenant_id=tenant_id,
                    tenant_strategy=TENANT_STRATEGY,
                ) from exc
            self.where = None
        else:
            name = BASE_COLLECTION
            try:
                self.collection = self.client.get_collection(name=name)
            except ChromaNotFoundError as exc:
                raise RetrievalCollectionNotFoundError(
                    collection_name=name,
                    tenant_id=tenant_id,
                    tenant_strategy=TENANT_STRATEGY,
                ) from exc
            self.where = {"tenant_id": tenant_id}

        # Resolve storage before loading heavyweight models. A missing tenant
        # therefore fails quickly and cannot trigger model downloads or cache
        # pressure through repeated readiness/chat requests.
        self.embedder = _shared_embedder(EMBED_MODEL, EMBED_DEVICE)

        # Try to load reranker lazily
        self._reranker = None
        self._exact_document_filter_supported: Optional[bool] = None
        if USE_RERANKER:
            try:
                self._reranker = _shared_reranker(
                    RERANKER_MODEL, RERANK_MAX_LENGTH, RERANK_DEVICE
                )
            except Exception as exc:
                # Retrieval remains available, but the configured reranker
                # must never disappear silently—especially when CPU policy or
                # model initialization was ignored by a dependency.
                logger.warning("Reranker unavailable; continuing without it: %s", exc)
                self._reranker = None
                if RERANK_REQUIRED:
                    raise RuntimeError(
                        "The configured reranker is required but unavailable"
                    ) from exc
        self._source_links_ready: Optional[bool] = None

    def has_tenant_documents(self) -> bool:
        """Return whether this retriever can serve at least one tenant record.

        Collection-backed tenants can use Chroma's lightweight count endpoint.
        A shared metadata collection must instead perform a tenant-filtered,
        one-ID read: the collection may be populated while the requested
        tenant has no records at all.  Neither path creates or mutates storage,
        and document bodies are never fetched.
        """
        if self.where is None:
            return int(self.collection.count()) > 0

        result = self._tenant_safe_get(
            where=self.where,
            limit=1,
            include=["metadatas"],
        )
        return bool(_flat_get_field(result, "ids"))

    def source_links_ready(self) -> bool:
        """Verify the tenant index uses complete source-link metadata.

        The result is cached for the process lifetime. Ingestion is an offline
        maintenance operation, so a rebuilt index is activated by restarting
        the API service rather than changing this state underneath requests.
        """
        if self._source_links_ready is not None:
            return self._source_links_ready

        scan_limit = _bounded_env_int(
            "SOURCE_LINK_COMPAT_SCAN_LIMIT",
            100_000,
            minimum=1,
            maximum=1_000_000,
        )
        batch_size = min(2048, scan_limit)
        offset = 0
        saw_tenant_row = False
        seen_pages: set[Tuple[str, ...]] = set()
        while offset < scan_limit:
            request_size = min(batch_size, scan_limit - offset)
            kwargs = {
                "include": ["metadatas"],
                "limit": request_size,
                "offset": offset,
            }
            if self.where is not None:
                kwargs["where"] = self.where
            try:
                response = self.collection.get(**kwargs)
            except Exception as exc:
                logger.warning("Source-link compatibility scan failed: %s", exc)
                self._source_links_ready = False
                return False
            if not isinstance(response, dict):
                self._source_links_ready = False
                return False

            ids = _flat_get_field(response, "ids")
            metadatas = _flat_get_field(response, "metadatas")
            row_count = max(len(ids), len(metadatas))
            if row_count == 0:
                self._source_links_ready = saw_tenant_row
                return self._source_links_ready
            if len(metadatas) != row_count:
                self._source_links_ready = False
                return False
            signature = tuple(str(value) for value in ids[:4]) or tuple(
                str(meta.get("source_id") or "")
                for meta in metadatas[:4]
                if isinstance(meta, dict)
            )
            if signature in seen_pages:
                self._source_links_ready = False
                return False
            seen_pages.add(signature)

            for metadata in metadatas:
                if self.where is not None and not self._metadata_belongs_to_tenant(metadata):
                    continue
                saw_tenant_row = True
                if not source_metadata_complete(metadata, self.tenant_id):
                    self._source_links_ready = False
                    return False

            offset += row_count
            if row_count < request_size:
                self._source_links_ready = saw_tenant_row
                return self._source_links_ready

        logger.warning(
            "Source-link compatibility scan reached SOURCE_LINK_COMPAT_SCAN_LIMIT=%d",
            scan_limit,
        )
        self._source_links_ready = False
        return False

    def get_source_metadata(self, source_id: str) -> Optional[Dict]:
        """Resolve one opaque source ID inside this retriever's tenant scope."""
        try:
            source_id = validate_source_id(source_id)
        except ValueError:
            return None
        where = _combine_where_filters(
            self.where,
            {"source_id": {"$eq": source_id}},
        )
        kwargs = {"include": ["metadatas"], "limit": 16}
        if where is not None:
            kwargs["where"] = where
        try:
            response = self.collection.get(**kwargs)
        except Exception as exc:
            logger.warning("Source metadata lookup failed")
            return None
        if not isinstance(response, dict):
            return None

        matches: List[Dict] = []
        for raw_metadata in _flat_get_field(response, "metadatas"):
            metadata = self._normalize_meta(raw_metadata)
            if (
                metadata.get("source_id") != source_id
                or metadata.get("tenant_id") != self.tenant_id
                or not source_metadata_complete(metadata, self.tenant_id)
            ):
                continue
            matches.append(metadata)
        if not matches:
            return None

        identities = {
            (
                str(meta.get("source_relative_path") or ""),
                str(meta.get("source_sha256") or ""),
                str(meta.get("display_name") or ""),
                str(meta.get("mime_type") or ""),
            )
            for meta in matches
        }
        if len(identities) != 1:
            logger.warning("Rejected ambiguous source metadata for an opaque ID")
            return None
        return matches[0]

    def _metadata_belongs_to_tenant(self, metadata) -> bool:
        """Defense in depth if a shared-collection adapter ignores ``where``."""
        if getattr(self, "where", None) is None:
            return True
        return (
            isinstance(metadata, dict)
            and metadata.get("tenant_id") == getattr(self, "tenant_id", None)
        )

    def _tenant_safe_get(self, **kwargs) -> Dict:
        """Execute a Chroma get and post-filter every shared-collection row."""
        if self.where is not None:
            requested_include = kwargs.get("include")
            if requested_include is not None:
                requested_include = list(requested_include)
                if "metadatas" not in requested_include:
                    requested_include.append("metadatas")
                kwargs["include"] = requested_include

        response = self.collection.get(**kwargs)
        if self.where is None:
            return response
        if not isinstance(response, dict):
            logger.warning("Dropping malformed shared-tenant storage response")
            return {}

        ids = _flat_get_field(response, "ids")
        documents = _flat_get_field(response, "documents")
        metadatas = _flat_get_field(response, "metadatas")
        row_count = max(len(ids), len(documents), len(metadatas))
        if row_count == 0:
            return response
        if len(metadatas) < row_count:
            logger.warning(
                "Dropping shared-tenant response with missing tenant metadata rows=%d metadata=%d",
                row_count,
                len(metadatas),
            )
            return {key: [] for key in response}

        safe_indices = [
            index for index in range(row_count)
            if self._metadata_belongs_to_tenant(metadatas[index])
        ]
        dropped = row_count - len(safe_indices)
        if dropped:
            logger.warning(
                "Dropped %d storage row(s) outside the active tenant",
                dropped,
            )

        filtered = dict(response)
        for key in ("ids", "documents", "metadatas", "embeddings", "uris", "data"):
            if key not in response:
                continue
            values = _safe_tolist(response.get(key))
            if not isinstance(values, (list, tuple)) or len(values) < row_count:
                # Unknown or misaligned record data must never cross the tenant
                # boundary merely because its shape cannot be verified.
                filtered[key] = []
                continue
            values = list(values)
            filtered[key] = [values[index] for index in safe_indices]
        return filtered

    # ---------------- Embedding helpers ----------------
    def _embed(self, text: str):
        formatted = _format_e5_query(text)
        cache = getattr(self, "_query_embedding_cache", None)
        if cache is None:
            cache = {}
            self._query_embedding_cache = cache
        encoded = cache.get(formatted)
        cache_hit = encoded is not None
        if encoded is None:
            measure_stage = globals().get("measure")
            if measure_stage is None:
                with _EMBED_INFERENCE_LOCK:
                    raw = self.embedder.encode(
                        [formatted],
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )[0]
            else:
                with measure_stage("embedding_ms"):
                    with _EMBED_INFERENCE_LOCK:
                        raw = self.embedder.encode(
                            [formatted],
                            show_progress_bar=False,
                            normalize_embeddings=True,
                        )[0]
            encoded = tuple(float(value) for value in _l2_normalize_vector(raw))
            if len(cache) >= 256:
                cache.pop(next(iter(cache)))
            cache[formatted] = encoded
        getattr(logger, "info", lambda *_args, **_kwargs: None)(
            "RETRIEVAL_CACHE component=query_embedding state=%s",
            "hit" if cache_hit else "miss",
        )
        cache_recorder = globals().get("record_cache")
        if cache_recorder is not None:
            cache_recorder(
                "query_embedding",
                "hit" if cache_hit else "miss",
            )
        return np.asarray(encoded, dtype=np.float32)

    def _embed_documents(self, docs: list[str]):
        dd = [format_embedding_document(value, EMBED_MODEL) for value in (docs or [])]
        with _EMBED_INFERENCE_LOCK:
            encoded = self.embedder.encode(
                dd,
                batch_size=EMBED_BATCH,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        normalized, _ = _l2_normalize_rows(encoded)
        return normalized

    # ---------------- Initial candidate pull -------------------
    def _initial_candidates(
        self,
        query: str,
        n: int,
        where: dict | None = None,
    ) -> Dict:
        q = self._embed(query)
        try:
            if not any(abs(float(component)) > _NORM_EPS for component in q):
                logger.warning("Query embedder returned a zero vector; skipping vector search")
                return {}
        except (TypeError, ValueError, OverflowError):
            logger.warning("Query embedder returned an invalid vector; skipping vector search")
            return {}
        n = max(1, min(n, _MAX_POOL))

        # Valid include keys for Chroma; do NOT request "ids"
        include = ["documents", "metadatas", "distances"]
        # Fetch embeddings up-front when MMR is the effective fallback. The
        # reranker can be configured yet unavailable (for example, offline).
        if USE_MMR and self._reranker is None:
            include.append("embeddings")

        # Defensive: keep only supported keys
        _ALLOWED = {"documents", "metadatas", "distances", "embeddings", "uris", "data"}
        include = [x for x in include if x in _ALLOWED]

        query_kwargs = dict(
            query_embeddings=[q],
            n_results=n,
            include=include,
        )
        use_where = _combine_where_filters(self.where, where)
        if use_where is not None:
            query_kwargs["where"] = use_where
        measure_stage = globals().get("measure")
        if measure_stage is None:
            return self.collection.query(**query_kwargs)
        with measure_stage("vector_retrieval_ms"):
            return self.collection.query(**query_kwargs)


    # ---------------- Reranking (cross-encoder) ----------------
    def _apply_reranker(
        self, query: str, docs: List[str], metas: List[Dict], dists: List[float], ids: List[str], k: int
    ) -> Tuple[List[int], List[Optional[float]], str]:
        try:
            requested = max(0, min(int(k), len(docs)))
        except (TypeError, ValueError, OverflowError):
            requested = 0
        fallback_order = list(range(requested))
        fallback_scores = [
            _finite_float_or_none(dists[index]) if index < len(dists) else None
            for index in fallback_order
        ]
        if not self._reranker:
            return fallback_order, fallback_scores, "distance"

        # Build pairs for cross-encoder
        cand = min(RERANK_CANDIDATES, len(docs), requested)
        if cand <= 0:
            return fallback_order, fallback_scores, "distance"
        pairs = [[query, d] for d in docs[:cand]]

        try:
            measure_stage = globals().get("measure")
            if measure_stage is None:
                with _RERANK_INFERENCE_LOCK:
                    raw_scores = self._reranker.predict(
                        pairs,
                        batch_size=min(RERANK_BATCH, cand),
                        show_progress_bar=False,
                    )
            else:
                with measure_stage("rerank_ms"):
                    with _RERANK_INFERENCE_LOCK:
                        raw_scores = self._reranker.predict(
                            pairs,
                            batch_size=min(RERANK_BATCH, cand),
                            show_progress_bar=False,
                        )
        except Exception:
            # If reranker fails, fall back to vector distances (already ranked)
            return fallback_order, fallback_scores, "distance"

        scores = _coerce_reranker_scores(raw_scores, cand)
        if scores is None:
            logger.warning(
                "Ignoring malformed reranker output (expected %d finite scores)",
                cand,
            )
            return fallback_order, fallback_scores, "distance"

        # Higher CE score = more relevant
        ceiling = min(requested, cand)
        # Preserve the vector-search order for equal CE scores.  Besides being
        # deterministic, this prevents tied duplicate chunks from arbitrarily
        # displacing the earlier candidate before unique selection runs.
        order = sorted(
            range(cand), key=lambda i: (-scores[i], i)
        )[:ceiling]
        return order, [scores[i] for i in order], "reranker"

    # ---------------- MMR (Maximal Marginal Relevance) ----------------
    def _mmr(self, q_vec: List[float], doc_vecs: List[List[float]], top_k: int) -> List[int]:
        # Fall back to Chroma's stable vector-search order for malformed input.
        # Retrieval should still work even if a legacy collection contains an
        # unexpected embedding payload.
        try:
            candidate_count = len(doc_vecs)
            limit = max(0, min(int(top_k), candidate_count))
        except (TypeError, ValueError, OverflowError):
            return []
        fallback = list(range(limit))
        if candidate_count == 0 or limit == 0:
            return fallback

        try:
            q = _l2_normalize_vector(q_vec)
            D, valid_rows = _l2_normalize_rows(doc_vecs)
        except (TypeError, ValueError, OverflowError):
            return fallback

        if D.shape[0] != candidate_count or D.shape[1] != q.shape[0]:
            return fallback
        if not np.any(q) or not np.any(valid_rows):
            return fallback

        # Cosine relevance and diversity, independent of vector magnitude.
        sims = D @ q  # shape: [N]
        # Invalid/zero stored rows are retained as a last-resort backfill but
        # never outrank a valid row merely because their dot product is zero.
        sims[~valid_rows] = -1.0
        selected: List[int] = []
        candidates = set(range(D.shape[0]))
        mmr_lambda = min(1.0, max(0.0, float(MMR_LAMBDA)))

        while len(selected) < limit and candidates:
            # Consume all usable vectors before invalid legacy rows. Keeping
            # those rows in the order still lets the caller fill ``top_k``.
            cand_list = sorted(i for i in candidates if valid_rows[i])
            if not cand_list:
                cand_list = sorted(candidates)

            if not selected:
                i = int(cand_list[int(np.argmax(sims[cand_list]))])
                selected.append(i)
                candidates.remove(i)
                continue

            # Diversity term: max similarity to selected
            # Sorting makes tie-breaking stable across Python implementations.
            div = []
            for j in cand_list:
                max_sim = float(np.max(D[j] @ D[selected].T))
                div.append(max_sim)
            div = np.array(div, dtype=np.float32)

            relevance = sims[cand_list]
            mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * div
            i = int(cand_list[int(np.argmax(mmr_score))])

            selected.append(i)
            candidates.remove(i)

        return selected

    # ---------------- Public API ----------------
    @(globals().get("trace_retrieval") or (lambda _channel: lambda function: function))(
        "vector"
    )
    def search(
        self,
        query: str,
        k: int = 8,
        where: dict | None = None,
    ) -> List[Dict]:
        """
        Returns a list of {document, metadata, score} dicts.
        ``score_kind`` makes the score direction explicit: distance is lower-is-
        better, while reranker and lexical scores are higher-is-better. An
        optional source/metadata filter is always ANDed with the immutable
        tenant filter before it reaches Chroma.
        """
        try:
            target = max(0, min(int(k), _MAX_POOL))
        except (TypeError, ValueError, OverflowError):
            return []
        prepared_query = _prepare_retrieval_query(query)
        if target == 0 or not prepared_query:
            return []

        if self._reranker is not None:
            pool = max(target, min(_MAX_POOL, RERANK_POOL_K))
        elif USE_MMR:
            pool = max(target, min(_MAX_POOL, target * 4))
        else:
            # Leave bounded room to backfill past empty/duplicate chunks without
            # pulling the entire reranker pool for a plain vector search.
            pool = max(target, min(_MAX_POOL, target * 2))
        res = self._initial_candidates(prepared_query, n=pool, where=where)

        # Unpack and coerce types safely
        docs = _first_query_row(res, "documents")
        if not docs:
            return []
        docs = [value if isinstance(value, str) else str(value or "") for value in docs]
        metas = _first_query_row(res, "metadatas")
        dists = _first_query_row(res, "distances")
        ids = _first_query_row(res, "ids")
        embs = _first_query_row(res, "embeddings")
        if len(metas) < len(docs):
            metas.extend({} for _ in range(len(docs) - len(metas)))

        if getattr(self, "where", None) is not None:
            safe_indices = [
                index for index, metadata in enumerate(metas[:len(docs)])
                if self._metadata_belongs_to_tenant(metadata)
            ]
            dropped = len(docs) - len(safe_indices)
            if dropped:
                logger.warning(
                    "Dropped %d vector result(s) outside the active tenant",
                    dropped,
                )
            docs = [docs[index] for index in safe_indices]
            metas = [metas[index] for index in safe_indices]
            dists = [dists[index] if index < len(dists) else None for index in safe_indices]
            ids = [ids[index] if index < len(ids) else None for index in safe_indices]
            if embs and all(index < len(embs) for index in safe_indices):
                embs = [embs[index] for index in safe_indices]
            else:
                embs = []
            if not docs:
                return []

        # Optional cross-encoder reranking
        if self._reranker is not None and docs:
            # Rank the whole bounded reranker pool so duplicates in its first k
            # positions can be replaced by the next-best distinct passages.
            order, scores, score_kind = self._apply_reranker(
                prepared_query,
                docs,
                metas,
                dists,
                ids,
                k=min(len(docs), RERANK_CANDIDATES),
            )
            score_by_index = {i: score for i, score in zip(order, scores)}
            order = _stable_unique_indices(order, docs, target, metas)
            out: List[Dict] = []
            for i in order:
                meta = self._normalize_meta(metas[i])
                if "chunk_id" not in meta and i < len(ids) and ids[i] is not None:
                    meta["chunk_id"] = ids[i]
                reranked = i in score_by_index
                if reranked:
                    score = _finite_float_or_none(score_by_index[i])
                else:
                    score = _finite_float_or_none(dists[i]) if i < len(dists) else None
                out.append(
                    {
                        "document": docs[i],
                        "metadata": meta,
                        # Candidates beyond the configured reranker cap keep
                        # their vector order but have no comparable CE score.
                        "score": score,
                        "score_kind": score_kind if reranked else "distance",
                    }
                )
            return out

        # Otherwise MMR (if enabled) or original order
        if USE_MMR and docs:
            if not embs or (
                isinstance(embs, list)
                and (
                    len(embs) == 0
                    or (isinstance(embs[0], list) and len(embs[0]) == 0)
                )
            ):
                # We didn't fetch embeddings earlier → compute now
                embs = self._embed_documents(docs)

            qvec = self._embed(prepared_query)
            # Produce the complete bounded MMR order, then backfill past any
            # duplicate selections without changing MMR's relative ranking.
            order = self._mmr(qvec, embs, top_k=len(docs))
        else:
            order = list(range(len(docs)))

        order = _stable_unique_indices(order, docs, target, metas)

        results: List[Dict] = []
        for i in order:
            meta = self._normalize_meta(metas[i])
            if "chunk_id" not in meta and i < len(ids) and ids[i] is not None:
                meta["chunk_id"] = ids[i]
            results.append(
                {
                    "document": docs[i],
                    "metadata": meta,
                    "score": _finite_float_or_none(dists[i]) if i < len(dists) else None,
                    "score_kind": "distance",
                }
            )
        return results

    @(globals().get("trace_retrieval") or (lambda _channel: lambda function: function))(
        "vector"
    )
    def bounded_search(
        self,
        query: str,
        k: int = 4,
        where: dict | None = None,
        *,
        candidate_limit: int = 8,
        rerank_limit: int = 4,
    ) -> List[Dict]:
        """Run one small vector pull and one small rerank for latency gates."""
        try:
            target = max(0, min(int(k), 16))
            pool = max(
                target,
                min(int(candidate_limit), 16, _MAX_POOL),
            )
            rerank_cap = max(0, min(int(rerank_limit), 8, pool))
        except (TypeError, ValueError, OverflowError):
            return []
        prepared_query = _prepare_retrieval_query(query)
        if target == 0 or not prepared_query:
            return []
        response = self._initial_candidates(prepared_query, n=pool, where=where)
        docs = _first_query_row(response, "documents")
        if not docs:
            return []
        docs = [
            value if isinstance(value, str) else str(value or "")
            for value in docs[:pool]
        ]
        metas = _first_query_row(response, "metadatas")[:len(docs)]
        dists = _first_query_row(response, "distances")[:len(docs)]
        ids = _first_query_row(response, "ids")[:len(docs)]
        if len(metas) < len(docs):
            metas.extend({} for _ in range(len(docs) - len(metas)))

        if getattr(self, "where", None) is not None:
            safe = [
                index
                for index, metadata in enumerate(metas)
                if self._metadata_belongs_to_tenant(metadata)
            ]
            docs = [docs[index] for index in safe]
            metas = [metas[index] for index in safe]
            dists = [
                dists[index] if index < len(dists) else None
                for index in safe
            ]
            ids = [ids[index] if index < len(ids) else None for index in safe]
        if not docs:
            return []

        if self._reranker is not None and rerank_cap:
            order, scores, kind = self._apply_reranker(
                prepared_query,
                docs,
                metas,
                dists,
                ids,
                k=min(len(docs), rerank_cap),
            )
        else:
            order = list(range(min(len(docs), target)))
            scores = [
                dists[index] if index < len(dists) else None
                for index in order
            ]
            kind = "distance"
        order = _stable_unique_indices(order, docs, target, metas)
        score_by_index = dict(zip(order, scores))
        output: List[Dict] = []
        for index in order:
            metadata = self._normalize_meta(metas[index])
            if (
                "chunk_id" not in metadata
                and index < len(ids)
                and ids[index] is not None
            ):
                metadata["chunk_id"] = ids[index]
            output.append({
                "document": docs[index],
                "metadata": metadata,
                "score": _finite_float_or_none(score_by_index.get(index)),
                "score_kind": kind,
            })
        return output

    def get(self, *, where: dict | None = None, include=None):
        """Tenant-safe wrapper for raw Chroma reads used by context expansion."""
        use_where = _combine_where_filters(self.where, where)
        kwargs = {}
        if use_where is not None:
            kwargs["where"] = use_where
        if include is not None:
            kwargs["include"] = include
        return self._tenant_safe_get(**kwargs)

    def scan_records(
        self,
        *,
        limit: int,
        scan_limit: int,
        where: dict | None = None,
        batch_size: int = 512,
    ) -> Tuple[List[Dict], bool]:
        """Return a bounded, tenant-safe record scan plus completeness.

        Arithmetic and derived counts are correct only when every in-scope
        chunk was inspected. This paginates storage, post-validates tenant and
        source scope even if an adapter ignores ``where``, detects repeated
        pages, and returns ``complete=False`` whenever either hard bound is
        reached before storage exhaustion.
        """
        try:
            target = max(1, min(int(limit), 100_000))
            scan_cap = max(1, min(int(scan_limit), 1_000_000))
            page_size = max(1, min(int(batch_size), 2048))
        except (TypeError, ValueError, OverflowError):
            return [], False

        source_scope = None
        if where:
            if set(where) != {"source"} or not isinstance(where.get("source"), str):
                return [], False
            source_scope = where["source"].replace("\\", "/").casefold()
        use_where = _combine_where_filters(self.where, where)
        hits: List[Dict] = []
        scanned = 0
        offset = 0
        seen_pages: set[Tuple] = set()
        source_generations: Dict[str, Dict] = {}

        def scope_is_coherent() -> bool:
            """Validate new-schema per-source chunk generations when present."""
            for state in source_generations.values():
                expected = state.get("chunk_count")
                if expected is None:
                    continue
                if (
                    state.get("invalid")
                    or state["seen_rows"] != expected
                    or state["indices"] != set(range(expected))
                ):
                    return False
            return True

        while scanned < scan_cap:
            request_size = min(page_size, scan_cap - scanned)
            kwargs = {
                "include": ["documents", "metadatas"],
                "limit": request_size,
                "offset": offset,
            }
            if use_where is not None:
                kwargs["where"] = use_where
            try:
                response = self.collection.get(**kwargs)
            except Exception as exc:
                logger.warning("Bounded analytics scan failed: %s", exc)
                return hits, False
            if not isinstance(response, dict):
                return hits, False

            ids = _flat_get_field(response, "ids")
            documents = _flat_get_field(response, "documents")
            metadatas = _flat_get_field(response, "metadatas")
            raw_count = max(len(ids), len(documents), len(metadatas))
            if raw_count == 0:
                return hits, scope_is_coherent()
            # Complete documents and metadata are prerequisites for record
            # extraction, source deduplication, and tenant verification.
            if len(documents) != raw_count or len(metadatas) != raw_count:
                return hits, False
            page_signature = tuple(str(value) for value in ids[:4]) or tuple(
                str(value)[:160] for value in documents[:2]
            )
            if page_signature in seen_pages:
                logger.warning("Bounded analytics scan detected a repeated page")
                return hits, False
            seen_pages.add(page_signature)

            # Missing metadata makes shared-tenant or source scope impossible
            # to verify. Fail closed instead of accepting an unscoped row.
            if (self.where is not None or source_scope is not None) and len(metadatas) < raw_count:
                return hits, False

            for index in range(raw_count):
                raw_meta = metadatas[index] if index < len(metadatas) else {}
                if self.where is not None and not self._metadata_belongs_to_tenant(raw_meta):
                    continue
                meta = self._normalize_meta(raw_meta)
                if source_scope is not None:
                    actual_source = str(
                        meta.get("source") or meta.get("source_path") or meta.get("path") or ""
                    ).replace("\\", "/").casefold()
                    if actual_source != source_scope:
                        continue
                document = documents[index] if index < len(documents) else ""
                if not isinstance(document, str) or not document:
                    continue
                if len(hits) >= target:
                    return hits, False
                if "chunk_id" not in meta and index < len(ids) and ids[index] is not None:
                    meta["chunk_id"] = ids[index]

                source_key = str(
                    meta.get("source") or meta.get("source_path") or meta.get("path") or ""
                ).replace("\\", "/").casefold()
                raw_chunk_count = meta.get("chunk_count")
                raw_chunk_index = meta.get("chunk_index")
                if raw_chunk_count is not None or raw_chunk_index is not None:
                    state = source_generations.setdefault(
                        source_key,
                        {"chunk_count": None, "indices": set(), "seen_rows": 0, "invalid": False},
                    )
                    if (
                        not isinstance(raw_chunk_count, int)
                        or isinstance(raw_chunk_count, bool)
                        or raw_chunk_count < 1
                        or not isinstance(raw_chunk_index, int)
                        or isinstance(raw_chunk_index, bool)
                        or raw_chunk_index < 0
                    ):
                        state["invalid"] = True
                    elif state["chunk_count"] not in {None, raw_chunk_count}:
                        state["invalid"] = True
                    else:
                        state["chunk_count"] = raw_chunk_count
                        state["indices"].add(raw_chunk_index)
                        state["seen_rows"] += 1
                hits.append(
                    {
                        "id": ids[index] if index < len(ids) else None,
                        "document": document,
                        "metadata": meta,
                        "score": None,
                        "score_kind": "analytics_scan",
                    }
                )

            scanned += raw_count
            offset += raw_count
            if raw_count < request_size:
                return hits, scope_is_coherent()

        return hits, False

    @(globals().get("trace_retrieval") or (lambda _channel: lambda function: function))(
        "exact"
    )
    def exact_search(
        self,
        term: str,
        k: int = 64,
        where: dict | None = None,
    ) -> List[Dict]:
        """Find a literal anchor across the full tenant collection.

        This uses Chroma's server-side document filter instead of scanning the
        first N tenant chunks in Python. It is intentionally limited to short,
        high-value anchors (entity names, standards, and identifiers) and
        fails closed when an older storage adapter does not support
        ``where_document``. It never falls back to an unbounded corpus scan.
        """
        try:
            target = max(0, min(int(k), LEXICAL_MAX_RESULTS))
        except (TypeError, ValueError, OverflowError):
            return []
        prepared_term = _prepare_retrieval_query(term)
        prepared_term = prepared_term[:EXACT_RETRIEVAL_MAX_TERM_CHARS].strip()
        if (
            target == 0
            or len(prepared_term) < 2
            or getattr(self, "_exact_document_filter_supported", None) is False
        ):
            return []

        return self._document_filter_search(
            where_document={"$contains": prepared_term},
            target=target,
            where=where,
            lexical_weight=max(1, len(_lexical_tokens(prepared_term))),
        )

    @(globals().get("trace_retrieval") or (lambda _channel: lambda function: function))(
        "conjunctive"
    )
    def conjunctive_search(
        self,
        term_groups,
        k: int = 64,
        where: dict | None = None,
    ) -> List[Dict]:
        """Find chunks containing every meaningful query-term group.

        Each outer group is required; alternatives within a group are ORed.
        This supports conservative case and Hebrew morphology variants while
        keeping the full-collection filter selective. At least two groups are
        required, preventing a common word from becoming a broad corpus pull.
        """
        try:
            target = max(0, min(int(k), LEXICAL_MAX_RESULTS))
        except (TypeError, ValueError, OverflowError):
            return []
        if (
            target == 0
            or getattr(self, "_exact_document_filter_supported", None) is False
            or not isinstance(term_groups, (list, tuple))
        ):
            return []

        filters: List[Dict] = []
        seen_groups: set[Tuple[str, ...]] = set()
        for raw_group in term_groups:
            raw_values = (
                list(raw_group)
                if isinstance(raw_group, (list, tuple, set))
                else [raw_group]
            )
            values: List[str] = []
            seen_values: set[str] = set()
            for raw_value in raw_values:
                value = _prepare_retrieval_query(raw_value)
                value = value[:EXACT_RETRIEVAL_MAX_TERM_CHARS].strip()
                if len(value) < 2 or value in seen_values:
                    continue
                seen_values.add(value)
                values.append(value)
                if len(values) >= 6:
                    break
            group_key = tuple(sorted({value.casefold() for value in values}))
            if not values or group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            if len(values) == 1:
                filters.append({"$contains": values[0]})
            else:
                filters.append({"$or": [{"$contains": value} for value in values]})
            if len(filters) >= CONJUNCTIVE_RETRIEVAL_MAX_TERMS:
                break

        if len(filters) < 2:
            return []
        return self._document_filter_search(
            where_document={"$and": filters},
            target=target,
            where=where,
            lexical_weight=len(filters),
        )

    def _document_filter_search(
        self,
        *,
        where_document: Dict,
        target: int,
        where: dict | None,
        lexical_weight: int,
    ) -> List[Dict]:
        """Execute one bounded tenant-safe Chroma document-filter lookup."""

        use_where = _combine_where_filters(self.where, where)
        kwargs = {
            "include": ["documents", "metadatas"],
            "limit": target,
            "where_document": where_document,
        }
        if use_where is not None:
            kwargs["where"] = use_where
        try:
            response = self._tenant_safe_get(**kwargs)
            self._exact_document_filter_supported = True
        except TypeError as exc:
            # Lightweight and legacy adapters may not accept where_document.
            # Remember that capability result to avoid repeating failed calls.
            self._exact_document_filter_supported = False
            logger.warning("Exact document filtering is unsupported: %s", exc)
            return []
        except Exception as exc:
            # Exact lookup supplements semantic and bounded lexical retrieval;
            # a storage-side filter failure must not make chat unavailable.
            logger.warning("Exact document retrieval failed: %s", exc)
            return []

        docs = _flat_get_field(response, "documents")[:target]
        metas = _flat_get_field(response, "metadatas")
        ids = _flat_get_field(response, "ids")
        hits: List[Dict] = []
        for index, raw_doc in enumerate(docs):
            document = raw_doc if isinstance(raw_doc, str) else str(raw_doc or "")
            if not document:
                continue
            raw_meta = metas[index] if index < len(metas) else {}
            meta = self._normalize_meta(raw_meta)
            if (
                "chunk_id" not in meta
                and index < len(ids)
                and ids[index] is not None
            ):
                meta["chunk_id"] = ids[index]
            hits.append({
                "document": document,
                "metadata": meta,
                # Rank fusion uses list position, not this raw value. The
                # explicit score kind prevents accidental distance handling.
                "score": float(max(1, lexical_weight)),
                "score_kind": "lexical",
            })

        unique_order = _stable_unique_indices(
            list(range(len(hits))),
            [hit.get("document") or "" for hit in hits],
            target,
            [hit.get("metadata") or {} for hit in hits],
        )
        return [hits[index] for index in unique_order]

    @(globals().get("timed_stage") or (lambda _name: lambda function: function))(
        "lexical_retrieval_ms"
    )
    @(globals().get("trace_retrieval") or (lambda _channel: lambda function: function))(
        "lexical"
    )
    def keyword_search(
        self,
        query: str,
        k: int = 32,
        min_score: int = 1,
        where: dict | None = None,
    ):
        """
        Bounded lexical fallback for hybrid retrieval.

        Documents are scanned in pages under a hard tenant-local cap. Exact
        phrases receive more weight than isolated terms, and every requested
        metadata filter is ANDed with the retriever's immutable tenant filter.
        """
        try:
            target = max(0, min(int(k), LEXICAL_MAX_RESULTS))
        except (TypeError, ValueError, OverflowError):
            return []
        threshold = _finite_float_or_none(min_score)
        if target == 0 or threshold is None:
            return []
        threshold = max(0.0, threshold)

        prepared_query = _prepare_retrieval_query(query)
        variants = _kw_variants(prepared_query)
        if not variants:
            return []

        canonical_query = " ".join(
            _lexical_tokens(prepared_query)[:LEXICAL_MAX_QUERY_TOKENS]
        )
        variant_specs = [
            (
                _lexical_variant_groups(variant),
                _lexical_variant_weight(variant, canonical_query),
            )
            for variant in variants
        ]
        variant_specs = [
            (groups, weight)
            for groups, weight in variant_specs
            if groups and weight > 0
        ]
        if not variant_specs:
            return []

        use_where = _combine_where_filters(self.where, where)
        hits: List[Dict] = []
        scanned = 0
        offset = 0
        scan_order = 0
        legacy_unpaged = False
        seen_pages: set[Tuple] = set()
        candidate_limit = min(
            LEXICAL_SCAN_MAX_DOCS,
            max(256, target * 16),
        )

        while scanned < LEXICAL_SCAN_MAX_DOCS:
            page_limit = min(
                LEXICAL_SCAN_BATCH,
                LEXICAL_SCAN_MAX_DOCS - scanned,
            )
            kwargs = {
                "include": ["documents", "metadatas"],
                "limit": page_limit,
                "offset": offset,
            }
            if use_where is not None:
                kwargs["where"] = use_where
            try:
                res = self._tenant_safe_get(**kwargs)
            except TypeError:
                # Compatibility for a lightweight/older adapter without
                # pagination. Its returned row is still sliced to the hard cap.
                if offset:
                    logger.warning("Lexical pagination stopped at offset %d", offset)
                    break
                legacy_unpaged = True
                kwargs.pop("limit", None)
                kwargs.pop("offset", None)
                try:
                    res = self._tenant_safe_get(**kwargs)
                except Exception as exc:
                    logger.warning("Lexical retrieval failed: %s", exc)
                    break
            except Exception as exc:
                # Lexical retrieval is a fallback; keep any completed pages and
                # let semantic retrieval continue rather than failing the chat.
                logger.warning("Lexical retrieval failed: %s", exc)
                break

            docs = _flat_get_field(res, "documents")
            if not docs:
                break
            metas = _flat_get_field(res, "metadatas")
            ids = _flat_get_field(res, "ids")
            returned_count = len(docs)
            remaining = LEXICAL_SCAN_MAX_DOCS - scanned
            docs = docs[:remaining]

            # Defend against adapters that accept but ignore ``offset``.
            page_identity = tuple(ids[:len(docs)]) if ids else (
                len(docs),
                str(docs[0])[:80],
                str(docs[-1])[-80:],
            )
            if page_identity in seen_pages:
                logger.warning("Lexical pagination repeated a page at offset %d", offset)
                break
            seen_pages.add(page_identity)

            for page_index, raw_doc in enumerate(docs):
                document = raw_doc if isinstance(raw_doc, str) else str(raw_doc or "")
                document_tokens = _lexical_tokens(document)
                if not document_tokens:
                    scan_order += 1
                    continue
                document_groups = [
                    _lexical_token_variants(token) for token in document_tokens
                ]
                document_positions = _lexical_document_positions(document_groups)
                score = sum(
                    weight
                    for query_groups, weight in variant_specs
                    if _lexical_groups_match(
                        query_groups, document_groups, document_positions
                    )
                )
                if score >= threshold:
                    meta_value = metas[page_index] if page_index < len(metas) else {}
                    meta = self._normalize_meta(meta_value)
                    if (
                        "chunk_id" not in meta
                        and page_index < len(ids)
                        and ids[page_index] is not None
                    ):
                        meta["chunk_id"] = ids[page_index]
                    hits.append({
                        "document": document,
                        "metadata": meta,
                        "score": float(score),
                        "score_kind": "lexical",
                        "_scan_order": scan_order,
                    })
                scan_order += 1

            scanned += len(docs)
            offset += returned_count

            if len(hits) > candidate_limit * 2:
                hits.sort(key=lambda hit: (-hit["score"], hit["_scan_order"]))
                del hits[candidate_limit:]

            if legacy_unpaged or returned_count < page_limit or len(docs) < returned_count:
                break

        hits.sort(key=lambda hit: (-hit["score"], hit["_scan_order"]))
        if len(hits) > candidate_limit:
            hits = hits[:candidate_limit]
        unique_order = _stable_unique_indices(
            list(range(len(hits))),
            [hit.get("document") or "" for hit in hits],
            target,
            [hit.get("metadata") or {} for hit in hits],
        )
        output = [hits[index] for index in unique_order]
        for hit in output:
            hit.pop("_scan_order", None)
        return output

    # ---------------- Utilities ----------------
    @staticmethod
    def _normalize_meta(m: Optional[Dict]) -> Dict:
        if not isinstance(m, dict) or not m:
            return {}
        meta: Dict[str, object] = {}
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                try:
                    import json
                    meta[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    meta[k] = str(v)
        return meta
