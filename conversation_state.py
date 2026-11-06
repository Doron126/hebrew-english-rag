"""Bounded, tenant-scoped conversation state for retrieval operations.

History helps resolve operations, references, scope, and novelty. It is never
returned as a factual evidence unit and never changes tenant authorization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable, Mapping, Optional

from request_understanding import (
    AnswerLanguage,
    AnswerShape,
    ConversationOperation,
    EVIDENCE_CHUNK_ID_RE,
    MAX_SUBQUESTIONS,
    QuestionType,
    RequestIntent,
    RequestPlan,
    RequestScope,
    SOURCE_ID_RE,
    Subquestion,
    deterministic_analyze,
    normalize_request_text,
)
from semantic_operations import material_fact_signature


CONVERSATION_STATE_SCHEMA_VERSION = 3
MAX_CONVERSATION_TURNS = 12
MAX_STATE_SOURCES = 8
MAX_STATE_EVIDENCE = 48
MAX_STATE_PAGES = 48
MAX_STATE_ASSISTANT_TEXTS = 6
MAX_STATE_TEXT_CHARS = 1200
MAX_STORED_CONVERSATIONS = 256
CONVERSATION_TTL_SECONDS = 6 * 60 * 60
CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u05d0-\u05ea]{2,}")
_CITATION_RE = re.compile(r"\[\d{1,3}\]")
_IDK_RE = re.compile(
    r"(?:"
    r"אני\s+לא\s+יודע\s+על\s+סמך\s+המסמכים|"
    r"\bI\s+(?:do\s+not|don't)\s+know\s+based\s+on\s+the\s+available"
    r")",
    re.IGNORECASE,
)
_PARTIAL_RE = re.compile(
    r"(?:"
    r"המסמכים\s+תומכים.+אבל|לא\s+מצאתי\s+מספיק\s+ראיות|"
    r"לא\s+הצלחתי\s+לאמת\s+בקטעים\s+שאותרו|"
    r"\b(?:the\s+documents\s+support|I\s+found\s+support).+\bbut\b|"
    r"\binsufficient\s+evidence\s+for\b|"
    r"\bI\s+could\s+not\s+verify\s+from\s+the\s+passages\s+found\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_SEMANTIC_STOPWORDS = frozenset({
    "the", "and", "for", "from", "that", "this", "with", "based", "source",
    "document", "documents", "report", "answer", "result", "fact", "finding",
    "של", "את", "על", "עם", "לפי", "המסמך", "המסמכים", "הדוח", "מקור",
    "תשובה", "תוצאה", "עובדה", "ממצא", "זה", "זאת",
})


def _value(turn: Any, field: str, default: Any = None) -> Any:
    if isinstance(turn, Mapping):
        return turn.get(field, default)
    return getattr(turn, field, default)


def _safe_list(value: Any, pattern: re.Pattern[str], maximum: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    output: list[str] = []
    for item in value[:maximum]:
        text = str(item or "")
        if pattern.fullmatch(text) and text not in output:
            output.append(text)
    return tuple(output)


def semantic_terms(value: str) -> frozenset[str]:
    """Return bounded language-agnostic terms for novelty comparison."""
    normalized = normalize_request_text(_CITATION_RE.sub(" ", value)).casefold()
    terms = [
        token
        for token in _TOKEN_RE.findall(normalized)
        if token not in _SEMANTIC_STOPWORDS
    ]
    return frozenset(terms[:160])


def semantic_fingerprint(value: str) -> str:
    terms = sorted(semantic_terms(value))
    payload = "\x1f".join(terms).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def topic_fingerprint(value: str) -> Optional[str]:
    terms = sorted(semantic_terms(value))
    if len(terms) < 2:
        return None
    return hashlib.sha256("\x1f".join(terms[:32]).encode("utf-8")).hexdigest()


def semantically_equivalent(
    left: str,
    right: str,
    *,
    threshold: float = 0.72,
) -> bool:
    left_terms = semantic_terms(left)
    right_terms = semantic_terms(right)
    if not left_terms or not right_terms:
        return False
    overlap = len(left_terms & right_terms)
    containment = overlap / max(1, min(len(left_terms), len(right_terms)))
    union = len(left_terms | right_terms)
    jaccard = overlap / max(1, union)
    left_numbers = tuple(re.findall(r"(?<!\w)\d+(?:[.,]\d+)*%?", left))
    right_numbers = tuple(re.findall(r"(?<!\w)\d+(?:[.,]\d+)*%?", right))
    return (
        left_numbers == right_numbers
        and (containment >= threshold or jaccard >= threshold)
    )


@dataclass(frozen=True)
class QuizRecord:
    document_id: str
    section: str
    page: Optional[int]
    concept: str
    evidence_id: str
    verified_answer: str


@dataclass(frozen=True)
class SourceReference:
    """A source actually returned by the server, never model-generated text."""
    source_id: str
    display_name: str
    source_relative_path: str
    page: Optional[int] = None
    status: str = 'returned'
    identity_text: str = ''
    source_sha256: str = ''


@dataclass(frozen=True)
class ConversationState:
    schema_version: int
    tenant_id: str
    previous_intent: Optional[RequestIntent]
    previous_answer_shape: Optional[AnswerShape]
    previous_requested_items: int
    previous_subquestions: tuple[Subquestion, ...]
    answer_language: Optional[AnswerLanguage]
    active_source_id: Optional[str]
    source_ids: frozenset[str]
    evidence_chunk_ids: frozenset[str]
    source_page_pairs: frozenset[tuple[str, int]]
    evidence_fingerprints: frozenset[str]
    semantic_fingerprints: frozenset[str]
    recent_assistant_texts: tuple[str, ...]
    previous_complete: bool
    previous_partial: bool
    safe_topic_fingerprint: Optional[str]
    supported_assistant_turns: int
    previous_user_question: Optional[str] = None
    previous_request_plan: Optional[RequestPlan] = None
    previous_requirements: tuple[Any, ...] = ()
    previous_accepted_facts: tuple[str, ...] = ()
    previous_answer: Optional[str] = None
    previous_citations: tuple[tuple[str, Optional[int]], ...] = ()
    previous_source_scope: tuple[str, ...] = ()
    previous_semantic_contract: Optional[Any] = None
    previous_operator_execution: Optional[Any] = None
    previous_operator_candidate_order: tuple[Any, ...] = ()
    previous_operator_hits: tuple[Mapping[str, Any], ...] = ()
    quiz_records: tuple[QuizRecord, ...] = ()
    reference_blocked_by_failed_turn: bool = False
    listing_query: str = ''
    listing_source_id: str = ''
    listing_next_offset: int = 0
    listing_catalog_sha256: str = ''
    listing_proposal: str = ''
    listing_semantic_review: str = ''
    document_references: tuple[SourceReference, ...] = ()


@dataclass(frozen=True)
class ConversationResolution:
    plan: RequestPlan
    state: ConversationState
    resolved: bool
    needs_clarification: bool
    reason_code: str
    referent_queries: tuple[str, ...]
    novelty_evidence_ids: frozenset[str]
    novelty_source_page_pairs: frozenset[tuple[str, int]]
    novelty_semantic_fingerprints: frozenset[str]

    @property
    def effective_source_scope(self) -> tuple[str, ...]:
        """Return request scope, never a previous answer's citation selection.

        Prior evidence is useful for resolving a genuine reference. It is not
        an implicit filter for a new question, and an explicit selected source
        takes precedence over every contextual source.
        """
        if self.plan.scope in {
            RequestScope.SELECTED_SOURCE,
            RequestScope.ACTIVE_CONVERSATION_SOURCE,
        }:
            return (self.plan.source_id,) if self.plan.source_id else ()
        if (
            self.plan.inherit_previous_scope
            and self.plan.conversation_operation is not ConversationOperation.NONE
        ):
            return self.state.previous_source_scope
        return ()


class ConversationTenantMismatch(ValueError):
    """A conversation handle exists but belongs to another tenant."""


class TenantConversationStore:
    """Bounded process-local state keyed by both tenant and conversation.

    Clients receive an opaque handle and send it on subsequent turns.  History
    remains a recovery path, but accepted plans and facts no longer have to be
    reverse-engineered from assistant prose during a normal conversation.
    """

    def __init__(
        self,
        *,
        maximum: int = MAX_STORED_CONVERSATIONS,
        ttl_seconds: float = CONVERSATION_TTL_SECONDS,
        persistence_path: Optional[str] = None,
    ) -> None:
        self.maximum = max(1, int(maximum))
        self.ttl_seconds = max(60.0, float(ttl_seconds))
        self._lock = threading.RLock()
        self._states: "OrderedDict[tuple[str, str], tuple[float, ConversationState]]" = OrderedDict()
        self._owners: dict[str, str] = {}
        self._persistence_path = str(persistence_path or '')
        if self._persistence_path:
            with self._connect() as db:
                db.execute('CREATE TABLE IF NOT EXISTS conversation_references ('
                    'conversation_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, '
                    'updated REAL NOT NULL, payload TEXT NOT NULL)')

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self._persistence_path, timeout=10.0)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _persist(self, identifier: str, state: ConversationState) -> None:
        # Store only server-produced references and the context needed to
        # interpret them. Source access is rechecked when a reference is used.
        payload = json.dumps({
            'question': state.previous_user_question or '',
            'answer': state.previous_answer or '',
            'references': [asdict(ref) for ref in state.document_references],
            'scope': list(state.previous_source_scope),
            'blocked': state.reference_blocked_by_failed_turn,
            'listing': ({'query': state.listing_query, 'source_id': state.listing_source_id,
                         'offset': state.listing_next_offset, 'catalog_sha256': state.listing_catalog_sha256,
                         'proposal': state.listing_proposal, 'semantic_review': state.listing_semantic_review}
                        if state.listing_next_offset > 0 else None),
        }, ensure_ascii=False, separators=(',', ':'))
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM conversation_references WHERE updated < ?',
                       (time.time() - self.ttl_seconds,))
            owner = db.execute('SELECT tenant_id FROM conversation_references WHERE conversation_id=?',
                               (identifier,)).fetchone()
            if owner and owner[0] != state.tenant_id:
                raise ConversationTenantMismatch(identifier)
            db.execute('INSERT INTO conversation_references VALUES (?, ?, ?, ?) '
                'ON CONFLICT(conversation_id) DO UPDATE SET updated=excluded.updated, payload=excluded.payload',
                (identifier, state.tenant_id, time.time(), payload))
            db.execute('DELETE FROM conversation_references WHERE conversation_id IN '
                '(SELECT conversation_id FROM conversation_references ORDER BY updated DESC, '
                'conversation_id DESC LIMIT -1 OFFSET ?)', (self.maximum,))

    def _restore(self, tenant: str, identifier: str) -> Optional[ConversationState]:
        with self._connect() as db:
            row = db.execute('SELECT tenant_id, updated, payload FROM conversation_references '
                             'WHERE conversation_id=?', (identifier,)).fetchone()
        if not row:
            return None
        if time.time() - row[1] > self.ttl_seconds:
            # Delete only the expired version observed above; a concurrent
            # refresh must not be erased by this cleanup.
            with self._connect() as db:
                db.execute('DELETE FROM conversation_references WHERE conversation_id=? AND updated=?',
                           (identifier, row[1]))
            return None
        if row[0] != tenant:
            raise ConversationTenantMismatch(identifier)
        payload = json.loads(row[2])
        refs = tuple(SourceReference(**item) for item in payload['references'])
        scope = tuple(payload['scope'])
        history = []
        if payload['question']:
            history.append({'role': 'user', 'content': payload['question']})
        if payload['answer']:
            history.append({'role': 'assistant', 'content': payload['answer'], 'source_ids': list(scope)})
        state = build_conversation_state(history, tenant_id=tenant)
        listing = payload.get('listing')
        listing_values = {}
        if isinstance(listing, dict):
            query, source = listing.get('query'), listing.get('source_id')
            offset, catalog = listing.get('offset'), listing.get('catalog_sha256')
            proposal, review = listing.get('proposal', ''), listing.get('semantic_review', '')
            if (isinstance(query, str) and 0 < len(query) <= 4000 and isinstance(source, str)
                    and len(source) <= 128 and type(offset) is int and 0 < offset <= 100000
                    and isinstance(catalog, str) and re.fullmatch(r'[a-f0-9]{64}', catalog)
                    and isinstance(proposal, str) and len(proposal) <= 20000
                    and isinstance(review, str) and len(review) <= 20000):
                listing_values = dict(listing_query=query, listing_source_id=source,
                    listing_next_offset=offset, listing_catalog_sha256=catalog,
                    listing_proposal=proposal, listing_semantic_review=review)
        return replace(state, document_references=refs, previous_source_scope=scope,
                       reference_blocked_by_failed_turn=bool(payload['blocked']), **listing_values)

    def _touch(self, tenant: str, identifier: str) -> None:
        # Match the in-memory store's sliding access TTL. Do not recreate a
        # record that another transaction has expired or evicted.
        with self._connect() as db:
            db.execute('UPDATE conversation_references SET updated=? '
                       'WHERE tenant_id=? AND conversation_id=?',
                       (time.time(), tenant, identifier))

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def _prune(self, now: float) -> None:
        expired = [
            key for key, (updated, _state) in self._states.items()
            if now - updated > self.ttl_seconds
        ]
        for key in expired:
            self._states.pop(key, None)
            conversation_id = key[1]
            if not any(other[1] == conversation_id for other in self._states):
                self._owners.pop(conversation_id, None)
        while len(self._states) > self.maximum:
            (_tenant, conversation_id), _value = self._states.popitem(last=False)
            if not any(other[1] == conversation_id for other in self._states):
                self._owners.pop(conversation_id, None)

    def load(self, tenant_id: str, conversation_id: str) -> Optional[ConversationState]:
        if not CONVERSATION_ID_RE.fullmatch(str(conversation_id or "")):
            return None
        tenant = str(tenant_id)
        identifier = str(conversation_id)
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            owner = self._owners.get(identifier)
            if owner is not None and owner != tenant:
                raise ConversationTenantMismatch(identifier)
            key = (tenant, identifier)
            stored = self._states.get(key)
            if stored is None:
                if not self._persistence_path:
                    return None
                restored = self._restore(tenant, identifier)
                if restored is None:
                    return None
                self._touch(tenant, identifier)
                self._owners[identifier] = tenant
                self._states[key] = (now, restored)
                self._prune(now)
                return restored
            self._states.move_to_end(key)
            if self._persistence_path:
                self._touch(tenant, identifier)
            self._states[key] = (now, stored[1])
            return stored[1]

    def save(self, conversation_id: str, state: ConversationState) -> None:
        identifier = str(conversation_id)
        if not CONVERSATION_ID_RE.fullmatch(identifier):
            raise ValueError("invalid_conversation_id")
        tenant = str(state.tenant_id)
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            owner = self._owners.get(identifier)
            if owner is not None and owner != tenant:
                raise ConversationTenantMismatch(identifier)
            if self._persistence_path:
                self._persist(identifier, state)
            key = (tenant, identifier)
            self._owners[identifier] = tenant
            self._states[key] = (now, state)
            self._states.move_to_end(key)
            self._prune(now)


def build_conversation_state(
    history: Iterable[Any],
    *,
    tenant_id: str,
) -> ConversationState:
    """Reconstruct safe state from at most twelve recent client turns."""
    turns = list(history or [])[-MAX_CONVERSATION_TURNS:]
    source_ids: set[str] = set()
    evidence_ids: set[str] = set()
    source_pages: set[tuple[str, int]] = set()
    assistant_texts: list[str] = []
    semantic_fingerprints: set[str] = set()
    evidence_fingerprints: set[str] = set()
    active_source: Optional[str] = None
    latest_sources: tuple[str, ...] = ()
    supported_turns = 0
    previous_complete = False
    previous_partial = False
    user_queries: list[str] = []
    user_turn_indices: list[int] = []

    for turn_index, turn in enumerate(turns):
        role = str(_value(turn, "role", "")).casefold()
        content = normalize_request_text(_value(turn, "content", ""))
        if role == "user" and content:
            user_queries.append(content)
            user_turn_indices.append(turn_index)
            continue
        if role != "assistant" or not content:
            continue
        turn_sources = _safe_list(
            _value(turn, "source_ids", ()),
            SOURCE_ID_RE,
            MAX_STATE_SOURCES,
        )
        turn_evidence = _safe_list(
            _value(turn, "evidence_chunk_ids", ()),
            EVIDENCE_CHUNK_ID_RE,
            MAX_STATE_EVIDENCE,
        )
        raw_pages = _value(turn, "pages", ())
        pages = tuple(
            page for page in (
                int(item)
                for item in raw_pages[:MAX_STATE_PAGES]
                if isinstance(item, int) or str(item).isdigit()
            )
            if 1 <= page <= 1_000_000
        ) if isinstance(raw_pages, (list, tuple)) else ()
        # The active referent belongs to the latest answer. A multi-source or
        # unsupported answer cannot leave an older single source active.
        latest_sources = turn_sources
        active_source = turn_sources[0] if len(turn_sources) == 1 else None
        source_ids.update(turn_sources)
        evidence_ids.update(turn_evidence)
        for source in turn_sources:
            for page in pages:
                source_pages.add((source, page))
        for evidence_id in turn_evidence:
            evidence_fingerprints.add(
                hashlib.sha256(evidence_id.encode("ascii")).hexdigest()
            )
        bounded_text = content[:MAX_STATE_TEXT_CHARS]
        assistant_texts.append(bounded_text)
        semantic_fingerprints.add(semantic_fingerprint(bounded_text))
        if not _IDK_RE.search(content):
            supported_turns += 1
            previous_partial = bool(_PARTIAL_RE.search(content))
            previous_complete = not previous_partial
        else:
            previous_partial = False
            previous_complete = False

    previous_plan: Optional[RequestPlan] = None
    anchor_plan: Optional[RequestPlan] = None
    anchor_query: Optional[str] = None
    if user_queries:
        try:
            previous_plan = deterministic_analyze(
                user_queries[-1], turns[:user_turn_indices[-1]]).plan
        except Exception:
            previous_plan = None
        for user_query, user_turn_index in reversed(list(zip(user_queries, user_turn_indices))):
            try:
                # Reconstruct each turn with the context it actually had.
                # A deictic attribute question must not become an independent
                # anchor merely because history recovery analyzed it alone.
                candidate = deterministic_analyze(user_query, turns[:user_turn_index]).plan
            except Exception:
                continue
            if (
                candidate is not None
                and candidate.conversation_operation is ConversationOperation.NONE
                and candidate.intent
                not in {
                    RequestIntent.UNSUPPORTED_OR_CONVERSATIONAL,
                    RequestIntent.CLARIFICATION,
                }
            ):
                anchor_plan = candidate
                anchor_query = user_query
                break
    structural_plan = anchor_plan or previous_plan
    structural_subquestions = (
        structural_plan.subquestions[:MAX_SUBQUESTIONS] if structural_plan else ())
    if anchor_query:
        from document_reading import operation as source_reading_operation
        if source_reading_operation(anchor_query):
            # A project prefix separated by a comma is part of an entity's
            # complete user-side identity. Generic clause decomposition must
            # not turn the prefix into an independent retrieval referent.
            structural_subquestions = (Subquestion(
                id='conversation-anchor', question_type=QuestionType.DIRECT_FACT,
                query=anchor_query[:1200], subject=anchor_query[:1200]),)

    recent_texts = tuple(
        reversed(assistant_texts[-MAX_STATE_ASSISTANT_TEXTS:])
    )
    previous_answer = assistant_texts[-1] if assistant_texts else None
    accepted_signature = (
        material_fact_signature(previous_answer or "")[0]
        if previous_answer and previous_complete
        else ()
    )
    return ConversationState(
        schema_version=CONVERSATION_STATE_SCHEMA_VERSION,
        tenant_id=str(tenant_id),
        previous_intent=previous_plan.intent if previous_plan else None,
        previous_answer_shape=previous_plan.answer_shape if previous_plan else None,
        previous_requested_items=(
            previous_plan.requested_items if previous_plan else 1
        ),
        previous_subquestions=structural_subquestions,
        answer_language=(
            previous_plan.answer_language if previous_plan else None
        ),
        active_source_id=active_source,
        source_ids=frozenset(list(source_ids)[:MAX_STATE_SOURCES]),
        evidence_chunk_ids=frozenset(
            list(evidence_ids)[:MAX_STATE_EVIDENCE]
        ),
        source_page_pairs=frozenset(
            list(source_pages)[:MAX_STATE_PAGES]
        ),
        evidence_fingerprints=frozenset(evidence_fingerprints),
        semantic_fingerprints=frozenset(semantic_fingerprints),
        recent_assistant_texts=recent_texts,
        previous_complete=previous_complete,
        previous_partial=previous_partial,
        safe_topic_fingerprint=topic_fingerprint(anchor_query or ""),
        supported_assistant_turns=supported_turns,
        previous_user_question=anchor_query,
        previous_request_plan=structural_plan,
        previous_requirements=(
            structural_plan.evidence_requirements if structural_plan else ()
        ),
        previous_accepted_facts=tuple(accepted_signature),
        previous_answer=previous_answer,
        previous_citations=tuple(
            (source, page) for source, page in list(source_pages)[:MAX_STATE_PAGES]
        ),
        previous_source_scope=latest_sources,
    )


def resolve_conversation_plan(
    plan: RequestPlan,
    state: ConversationState,
    *,
    tenant_id: str,
) -> ConversationResolution:
    """Resolve a contextual operation without treating history as evidence."""
    if state.tenant_id != str(tenant_id):
        # A caller must build state for the current authorized tenant. If it
        # does not, discard every inherited field.
        empty = build_conversation_state((), tenant_id=str(tenant_id))
        selected = plan.scope is RequestScope.SELECTED_SOURCE
        return ConversationResolution(
            plan=replace(
                plan,
                scope=plan.scope if selected else RequestScope.TENANT_CORPUS,
                source_id=plan.source_id if selected else None,
                inherit_previous_scope=False,
                requires_context_resolution=False,
            ),
            state=empty,
            resolved=False,
            needs_clarification=(
                plan.conversation_operation is not ConversationOperation.NONE
            ),
            reason_code="tenant_context_mismatch",
            referent_queries=(),
            novelty_evidence_ids=frozenset(),
            novelty_source_page_pairs=frozenset(),
            novelty_semantic_fingerprints=frozenset(),
        )

    operation = plan.conversation_operation
    contextual = operation is not ConversationOperation.NONE
    if (
        contextual and state.reference_blocked_by_failed_turn
        and operation is not ConversationOperation.VERIFY_PREVIOUS
    ):
        return ConversationResolution(
            plan=replace(plan, scope=RequestScope.TENANT_CORPUS, source_id=None,
                         inherit_previous_scope=False, requires_context_resolution=True),
            state=state, resolved=False, needs_clarification=True,
            reason_code='blocked_by_prior_turn_failure', referent_queries=(),
            novelty_evidence_ids=frozenset(), novelty_source_page_pairs=frozenset(),
            novelty_semantic_fingerprints=frozenset())
    if contextual and state.supported_assistant_turns == 0:
        return ConversationResolution(
            plan=plan,
            state=state,
            resolved=False,
            needs_clarification=True,
            reason_code="blocked_by_prior_turn_failure",
            referent_queries=(),
            novelty_evidence_ids=frozenset(),
            novelty_source_page_pairs=frozenset(),
            novelty_semantic_fingerprints=frozenset(),
        )

    if contextual and not state.previous_subquestions and state.previous_user_question:
        # Source-reading contracts can answer a whole section without scalar
        # subquestions. Their accepted user query still supplies the referent
        # for a later attribute question. This is a retrieval anchor only;
        # assistant prose and previous factual values remain outside evidence.
        anchor = normalize_request_text(state.previous_user_question)[:1200]
        if anchor:
            state = replace(state, previous_subquestions=(Subquestion(
                id='conversation-anchor', question_type=QuestionType.DIRECT_FACT,
                query=anchor, subject=anchor),))

    resolved_plan = plan
    if (
        plan.inherit_previous_scope
        and plan.scope is RequestScope.TENANT_CORPUS
        and state.active_source_id
    ):
        resolved_plan = replace(
            plan,
            scope=RequestScope.ACTIVE_CONVERSATION_SOURCE,
            source_id=state.active_source_id,
            requires_context_resolution=True,
        )

    if (operation is ConversationOperation.CONSTRAIN_TO_SOURCE
            and not (resolved_plan.scope in {RequestScope.SELECTED_SOURCE,
                                             RequestScope.ACTIVE_CONVERSATION_SOURCE}
                     and resolved_plan.source_id)):
        # An explicit "this document only" with multiple or absent recent
        # sources cannot silently expand to the tenant corpus.
        return ConversationResolution(
            plan=replace(resolved_plan, requires_context_resolution=True),
            state=state, resolved=False, needs_clarification=True,
            reason_code="source_reference_ambiguous", referent_queries=(),
            novelty_evidence_ids=frozenset(), novelty_source_page_pairs=frozenset(),
            novelty_semantic_fingerprints=frozenset())

    referent_queries = tuple(
        item.query for item in state.previous_subquestions[:MAX_SUBQUESTIONS]
    )
    if contextual and not referent_queries and state.safe_topic_fingerprint:
        # The fingerprint proves a bounded topic existed but cannot itself be
        # searched. A low-information history therefore needs clarification.
        return ConversationResolution(
            plan=resolved_plan,
            state=state,
            resolved=False,
            needs_clarification=True,
            reason_code="reference_query_unavailable",
            referent_queries=(),
            novelty_evidence_ids=frozenset(),
            novelty_source_page_pairs=frozenset(),
            novelty_semantic_fingerprints=frozenset(),
        )

    another = operation is ConversationOperation.ANOTHER_RESULT
    referent_operations = {
        ConversationOperation.EXPLAIN_REFERENT,
        ConversationOperation.ELABORATE,
        ConversationOperation.COMPARE_WITH_PREVIOUS,
        ConversationOperation.SUMMARIZE_FINDINGS,
        ConversationOperation.CHANGE_ITEM_COUNT,
        ConversationOperation.TIME_FOCUS,
        ConversationOperation.ATTRIBUTE_FOCUS,
        ConversationOperation.CONSTRAIN_TO_SOURCE,
        ConversationOperation.VERIFY_PREVIOUS,
        ConversationOperation.CORRECT_PREVIOUS,
        ConversationOperation.EXPAND_PREVIOUS,
    }
    if operation in referent_operations and state.previous_subquestions:
        # Search the current operation together with the preceding user-side
        # referent. Assistant prose remains novelty/context only and never
        # becomes factual evidence.
        resolved_subquestions: list[Subquestion] = []
        for index, current in enumerate(resolved_plan.subquestions):
            current_terms = semantic_terms(
                f"{current.subject} {current.query}"
            )
            scored_previous = []
            for previous_index, candidate in enumerate(
                state.previous_subquestions
            ):
                current_subject_terms = semantic_terms(current.subject)
                candidate_subject_terms = semantic_terms(candidate.subject)
                subject_overlap = len(
                    current_subject_terms & candidate_subject_terms
                )
                subject_containment = (
                    subject_overlap / max(1, min(
                        len(current_subject_terms),
                        len(candidate_subject_terms),
                    ))
                )
                candidate_terms = semantic_terms(
                    f"{candidate.subject} {candidate.query}"
                )
                overlap = len(current_terms & candidate_terms)
                containment = (
                    overlap / max(1, min(
                        len(current_terms),
                        len(candidate_terms),
                    ))
                )
                scored_previous.append((
                    subject_containment,
                    subject_overlap,
                    containment,
                    overlap,
                    -abs(previous_index - index),
                    -previous_index,
                    candidate,
                ))
            best = max(scored_previous)
            previous = (
                best[-1]
                if best[0] > 0.0 or best[2] > 0.0
                else state.previous_subquestions[
                    min(index, len(state.previous_subquestions) - 1)
                ]
            )
            combined_query = normalize_request_text(
                f"{previous.query} {current.query}"
            )[:1200]
            resolved_subquestions.append(replace(
                current,
                query=combined_query,
                subject=previous.subject or current.subject,
                question_type=(
                    previous.question_type
                    if operation in {
                        ConversationOperation.ELABORATE,
                        ConversationOperation.TIME_FOCUS,
                        ConversationOperation.ATTRIBUTE_FOCUS,
                        ConversationOperation.VERIFY_PREVIOUS,
                        ConversationOperation.CORRECT_PREVIOUS,
                        ConversationOperation.EXPAND_PREVIOUS,
                    }
                    else current.question_type
                ),
                required_dimensions=tuple(dict.fromkeys(
                    previous.required_dimensions
                    + current.required_dimensions
                )),
                required_attributes=tuple(dict.fromkeys(
                    previous.required_attributes
                    + current.required_attributes
                )),
                relationship_group=(
                    current.relationship_group
                    or previous.relationship_group
                ),
            ))
        resolved_plan = replace(
            resolved_plan,
            subquestions=tuple(resolved_subquestions),
            needs_multi_query=(
                resolved_plan.needs_multi_query
                or len(resolved_subquestions) > 1
            ),
        )
    return ConversationResolution(
        plan=resolved_plan,
        state=state,
        resolved=True,
        needs_clarification=False,
        reason_code=(
            "another_result_resolved"
            if another
            else "context_reference_resolved"
            if contextual
            else "independent_request"
        ),
        referent_queries=referent_queries if contextual else (),
        novelty_evidence_ids=(
            state.evidence_chunk_ids if another else frozenset()
        ),
        novelty_source_page_pairs=(
            state.source_page_pairs if another else frozenset()
        ),
        novelty_semantic_fingerprints=(
            state.semantic_fingerprints if another else frozenset()
        ),
    )


def is_novel_result(candidate_text: str, state: ConversationState) -> bool:
    """Reject claim-level semantic duplicates across the bounded history."""
    candidate_fingerprint = semantic_fingerprint(candidate_text)
    if candidate_fingerprint in state.semantic_fingerprints:
        return False
    return not any(
        semantically_equivalent(candidate_text, previous)
        for previous in state.recent_assistant_texts
    )
