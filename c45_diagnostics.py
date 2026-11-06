"""C45 observational diagnostics: closed payload and bounded local sink.

Application adapters enter containment before importing this module. No
application object is read, coerced, serialized or delivered here. Request
correlation copies the existing immutable request_id into a diagnostic-only
ContextVar; an absent or oversized identity makes the attempt unobservable.

The fixed builtin sink never calls a supplied callback or external backend.
Ordinary Python/I/O failures are contained. BaseException control flow remains
Python control flow; process exit, native/fatal failures and OOM survival are
not guaranteed. Record loss is allowed; no diagnostic return is app authority.
"""
from collections import deque
from contextvars import ContextVar, Token
from dataclasses import dataclass
import json

C45_CONCRETE_PAYLOAD_SCHEMA_VERSION = 1
PROJECT_X_C45_DIAGNOSTIC_SINK_V1 = "PROJECT_X_C45_DIAGNOSTIC_SINK_V1"
PAYLOAD_FIELDS = (
    "lifecycle_id", "witness_family_id", "anchor_id", "primitive_id",
    "stage", "outcome", "failure_class", "source_anchor",
)
_MAX_SCALAR_CHARS = 256
_SINK_CAPACITY = 64
_LIFECYCLE = ContextVar("c45_diagnostic_lifecycle", default=None)
_RECORDS = deque(maxlen=_SINK_CAPACITY)

# Source-derived constants from the authenticated C44E2E2 family/anchor census.
# These identify prospective obligations, not executed historical observations.
_BINDINGS = {
    'C44E2E2_FAMILY_01_c9eefcabc9ba': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_DOWNSTREAM_UNRESOLVED_CITATION_BINDING__structural_94e99072167e',
        'C44_PRIMITIVE_UNIT_TERMINAL_DOWNSTREAM_UNRESOLVED_CITATION_BINDING__structural_94e99072167e',
        'chat_api.py@edd516b7413184be9dc50c2ec559825346037b4a:_attach_verified_evidence_quotes',
    ),
    'C44E2E2_FAMILY_02_7668230fb810': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_DOWNSTREAM_UNRESOLVED_CITATION_BINDING__structural_9c7c514c870f',
        'C44_PRIMITIVE_UNIT_TERMINAL_DOWNSTREAM_UNRESOLVED_CITATION_BINDING__structural_9c7c514c870f',
        'chat_api.py@edd516b7413184be9dc50c2ec559825346037b4a:_attach_verified_evidence_quotes',
    ),
    'C44E2E2_FAMILY_03_0f14aac5cc1b': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_DOWNSTREAM_UNRESOLVED_SYNTHESIS_RELEVANCE__structural_f94b4e601adb',
        'C44_PRIMITIVE_UNIT_TERMINAL_DOWNSTREAM_UNRESOLVED_SYNTHESIS_RELEVANCE__structural_f94b4e601adb',
        'semantic_contracts.py@3df2c3593c1516ef95427c5be738ccec13bb69af:bind_operator_output_facts',
    ),
    'C44E2E2_FAMILY_04_20ba9dc99d4b': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_DOWNSTREAM_UNRESOLVED_SYNTHESIS_RELEVANCE__structural_f95d15ce6072',
        'C44_PRIMITIVE_UNIT_TERMINAL_DOWNSTREAM_UNRESOLVED_SYNTHESIS_RELEVANCE__structural_f95d15ce6072',
        'semantic_contracts.py@3df2c3593c1516ef95427c5be738ccec13bb69af:build_claim_graph',
    ),
    'C44E2E2_FAMILY_05_462e07fc25b7': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_CANDIDATE_RETRIEVAL',
        'C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_CANDIDATE_RETRIEVAL',
        'enterprise_retrieval.py@9f4de59a8d748dbdae5be93d2b864f1a502a483b:AdaptiveRetriever.retrieve',
    ),
    'C44E2E2_FAMILY_06_4206ccaa33c2': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_EVIDENCE_COVERAGE',
        'C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_EVIDENCE_COVERAGE',
        'enterprise_retrieval.py@9f4de59a8d748dbdae5be93d2b864f1a502a483b:build_evidence_coverage',
    ),
    'C44E2E2_FAMILY_07_4e0f40837437': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_FACT_ADMISSIBILITY',
        'C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_FACT_ADMISSIBILITY',
        'enterprise_retrieval.py@9f4de59a8d748dbdae5be93d2b864f1a502a483b:canonical_fact_admissibility',
    ),
    'C44E2E2_FAMILY_08_ffcc7a246f0b': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_MODEL_ASSISTED_EVIDENCE_SEMANTICS',
        'C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_MODEL_ASSISTED_EVIDENCE_SEMANTICS',
        'enterprise_retrieval.py@9f4de59a8d748dbdae5be93d2b864f1a502a483b:_maybe_resolve_evidence_semantics',
    ),
    'C44E2E2_FAMILY_09_118762662a9e': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_STRUCTURAL_EVIDENCE_CONSTRUCTION',
        'C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_STRUCTURAL_EVIDENCE_CONSTRUCTION',
        'enterprise_retrieval.py@9f4de59a8d748dbdae5be93d2b864f1a502a483b:classify_structural_unit',
    ),
    'C44E2E2_FAMILY_10_999f7411d1fa': (
        'C44_ANCHOR_C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_TYPED_FACT_EXTRACTION',
        'C44_PRIMITIVE_UNIT_TERMINAL_UPSTREAM_UNRESOLVED_TYPED_FACT_EXTRACTION',
        'enterprise_retrieval.py@9f4de59a8d748dbdae5be93d2b864f1a502a483b:_source_text_fact_for_proposition',
    ),
    'C44E2E2_FAMILY_11_a429090e7827': (
        'C44_ANCHOR_OPERATOR_FINALIZER_MEMBERSHIP',
        'C44_PRIMITIVE_OPERATOR_FINALIZER_OPERAND_MEMBERSHIP',
        'chat_api.py@edd516b7413184be9dc50c2ec559825346037b4a:_finalize_fact_bound_response',
    ),
    'C44E2E2_FAMILY_12_4a61fb9f4ebf': (
        'C44_ANCHOR_REQUEST_LIFECYCLE_IDENTITY',
        'C44_CORRELATION_REQUEST_LIFECYCLE_IDENTITY',
        'chat_api.py@edd516b7413184be9dc50c2ec559825346037b4a:_request_performance_scope',
    ),
}

_FAILURE_CLASSES = (
    "ordinary Python exception", "control-flow/cancellation exception",
    "I/O failure", "resource exhaustion", "allocation failure",
    "nontermination/blocking/backpressure", "process exit", "fatal/native failure",
)


@dataclass(frozen=True, slots=True)
class Payload:
    lifecycle_id: str
    witness_family_id: str
    anchor_id: str
    primitive_id: str
    stage: str
    outcome: str
    failure_class: str | None
    source_anchor: str

    def __post_init__(self):
        values = (
            self.lifecycle_id, self.witness_family_id, self.anchor_id,
            self.primitive_id, self.stage, self.outcome, self.source_anchor,
        )
        # Exact builtin types preclude descriptors/conversion/iteration hooks.
        for value in values:
            if type(value) is not str:
                raise TypeError("C45 requires already-materialized builtin scalars")
            if not value or len(value) > _MAX_SCALAR_CHARS:
                raise ValueError("C45 scalar outside bounded envelope")
        if self.failure_class is not None:
            if type(self.failure_class) is not str:
                raise TypeError("C45 failure class must be a canonical scalar")
            if self.failure_class not in _FAILURE_CLASSES:
                raise ValueError("C45 failure class outside governing model")
        if self.stage not in ("PREPARATION", "SINK", "CONTINUATION"):
            raise ValueError("C45 stage outside governing domain")
        if self.outcome not in ("RETURNED", "RAISED", "DROPPED", "UNOBSERVABLE"):
            raise ValueError("C45 outcome outside governing domain")
        # Owner-bound null means no bounded diagnostic failure was observed.
        if self.outcome == "RAISED" and self.failure_class is None:
            raise ValueError("C45 raised outcome requires a failure classification")
        if (self.stage == "PREPARATION" and self.outcome == "RETURNED"
                and self.failure_class is not None):
            raise ValueError("C45 successful preparation cannot classify a failure")
        expected = _BINDINGS[self.witness_family_id]
        if (self.anchor_id, self.primitive_id, self.source_anchor) != expected:
            raise ValueError("C45 source/family identity mismatch")


def _snapshot(witness_family_id):
    # PAY-00: read only diagnostic-owned immutable identity/state.
    if type(witness_family_id) is not str:
        raise TypeError("C45 family must be a builtin scalar")
    return _LIFECYCLE.get(), witness_family_id, _BINDINGS[witness_family_id]


def _prepare(snapshot):
    # PAY-01: copy bounded canonical scalars, never an application value.
    lifecycle_id, witness_family_id, binding = snapshot
    anchor_id, primitive_id, source_anchor = binding
    return Payload(
        lifecycle_id=lifecycle_id,
        witness_family_id=witness_family_id,
        anchor_id=anchor_id,
        primitive_id=primitive_id,
        stage="PREPARATION",
        outcome="RETURNED",
        failure_class=None,
        source_anchor=source_anchor,
    )


def serialize(payload):
    # PAY-02: exact closed schema in the owner-bound presentation order.
    if type(payload) is not Payload:
        raise TypeError("C45 serializer requires its closed payload")
    return json.dumps({
        "lifecycle_id": payload.lifecycle_id,
        "witness_family_id": payload.witness_family_id,
        "anchor_id": payload.anchor_id,
        "primitive_id": payload.primitive_id,
        "stage": payload.stage,
        "outcome": payload.outcome,
        "failure_class": payload.failure_class,
        "source_anchor": payload.source_anchor,
    }, ensure_ascii=True, separators=(",", ":"))


def _sink(serialized_record):
    # SINK-01: only this fixed builtin buffer is admitted; no callbacks/waits.
    if type(_RECORDS) is not deque or _RECORDS.maxlen != _SINK_CAPACITY:
        return "DROPPED"
    if len(_RECORDS) >= _SINK_CAPACITY:
        return "DROPPED"
    _RECORDS.append(serialized_record)
    return "DELIVERED"


def _complete(outcome):
    # Local bookkeeping is diagnostic-only and inside observe's containment.
    return outcome


def observe(witness_family_id):
    """Attempt PAY-00 -> PAY-01 -> PAY-02 -> SINK-01; caller ignores result."""
    try:
        snapshot = _snapshot(witness_family_id)
        if snapshot[0] is None:
            return "UNOBSERVABLE"
        payload = _prepare(snapshot)
        serialized_record = serialize(payload)
        outcome = _sink(serialized_record)
        return _complete(outcome)
    except Exception:
        return "CONTAINED_FAILURE"


def begin_lifecycle(lifecycle_id):
    """Transport the original request scalar; never allocate an app identity."""
    try:
        # An invalid nested identity must shadow, not reuse, the outer one.
        snapshot = None
        if type(lifecycle_id) is str:
            if lifecycle_id and len(lifecycle_id) <= _MAX_SCALAR_CHARS:
                snapshot = lifecycle_id
        return _LIFECYCLE.set(snapshot)
    except Exception:
        return None


def end_lifecycle(token):
    """Reset only diagnostic correlation, including during an app exception."""
    try:
        if type(token) is Token:
            _LIFECYCLE.reset(token)
    except Exception:
        pass
