"""Prospective, synthetic-only repository observer substrate for R7R13.

It deliberately has no real-epoch activation path.  A future separately
authorized bootstrap must install a verified guard before any governed target
can run; absent or unhealthy observer state fails closed.
"""
from __future__ import annotations
import hashlib
import json
import re
import copy
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, TypeVar

from r7r13_contracts import CONTROL_BOUNDARY, DIMENSIONS

T = TypeVar("T")
LEDGER_SCHEMA_VERSION = "r7r14-1.0.0"
SYNTHETIC_EPOCH_KIND = "SYNTHETIC_TEST_EPOCH"

# Executable source authority; installed primitives must resolve from here.
R7R13_GUARD_REGISTRY: dict[str, dict[str, Any]] = {
    "candidate-launch": {"dimension_ids": ("CANDIDATE_STARTED",), "primitive": "tools.launch_isolated_candidate.execute_candidate_launch", "classifier": "always_candidate_launch"},
    "benchmark-runner": {"dimension_ids": ("BENCHMARK_TRAFFIC_SENT",), "primitive": "tools.run_real_corpus_baseline._run_case", "classifier": "post_chat_case_dispatch"},
    "scorer": {"dimension_ids": ("SCORER_RERUN",), "primitive": "tools.score_real_corpus_results._emit_score_reports", "classifier": "score_report_emission"},
    "eval-runner-request": {"dimension_ids": ("REAL_SUBJECT_EXECUTION_COUNT",), "primitive": "eval_runner.ask_api", "classifier": "tenant_subject_post_chat"},
    "chat-api-request": {"dimension_ids": ("REAL_SUBJECT_EXECUTION_COUNT",), "primitive": "chat_api._guarded_chat_operation", "classifier": "candidate_mode_chat_endpoint"},
    "webui-streamlit-request": {"dimension_ids": ("REAL_SUBJECT_EXECUTION_COUNT",), "primitive": "webui_streamlit.module_chat_post", "classifier": "streamlit_user_post_chat"},
}

def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))

def _sha(v: Any) -> str:
    return hashlib.sha256(_canon(v).encode("utf-8")).hexdigest()

def guard_registry_sha256() -> str:
    return _sha(R7R13_GUARD_REGISTRY)

class GuardFailure(ValueError):
    """A governed target must not run after this error."""


REAL_AUTHORIZATION_SCHEMA_VERSION = "r7r16-real-activity-authorization-1.0.0"
R19_CAMPAIGN_COMPOSITION_SCHEMA = "r7r19-campaign-composition-1.0.0"
R20_CAMPAIGN_COMPOSITION_SCHEMA = "r7r20-campaign-composition-1.0.0"
REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION = "r7r16-real-activity-receipt-1.0.0"
REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17 = "r7r17-real-activity-receipt-1.0.0"
REAL_EFFECT_ATTESTATION_SCHEMA_VERSION = "r7r16-real-effect-attestation-1.0.0"
REAL_ACTIVITY_COVERAGE_SCHEMA_VERSION = "r7r16-real-activity-coverage-1.0.0"
REAL_INTERVAL_BOUNDARY_SCHEMA_VERSION = "r7r16-real-activity-boundary-1.0.0"
AUTHORIZATION_USE_CLAIM_SCHEMA_VERSION = "r7r16-authorization-use-claim-1.1.0"
REAL_EPOCH_KIND = "REAL_AUTHORIZED_EPOCH"
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REPOSITORY_CONTROLLED_DIMENSIONS = frozenset(
    row["dimension_id"]
    for row in CONTROL_BOUNDARY
    if row["classification"] == "FULLY_REPOSITORY_CONTROLLED"
)


@dataclass(frozen=True)
class SignatureVerification:
    """Result from an explicitly selected authorization/observer trust root.

    ``SYNTHETIC_TEST_ONLY`` permits protocol tests but never live activation.
    This repository deliberately provides no production trust-root adapter.
    """

    verified: bool
    authority_id: str
    authority_class: str


class RecordSignatureVerifier(Protocol):
    def verify(self, signer_id: str, canonical_payload: bytes, signature: str) -> SignatureVerification: ...


@dataclass(frozen=True)
class RealOperationAuthorization:
    execution_id: str
    dimension_id: str
    surface_id: str
    operation_identity: str
    source_commit: str
    source_tree: str
    process_identity: str
    request_identity: str
    initiating_record_sha256: str
    frozen_input_sha256: str


@dataclass(frozen=True)
class RealAuthorizationScope:
    campaign_id: str
    interval_id: str
    B0: str
    B1: str
    source_commit: str
    source_tree: str
    operations: tuple[RealOperationAuthorization, ...]
    campaign_protocol: str | None = None
    interval_role: str | None = None


@dataclass(frozen=True)
class ParsedRealAuthorization:
    """Authenticated, scope-checked authorization that has not been consumed."""

    decision_id: str
    epoch_id: str
    campaign_id: str
    interval_id: str
    B0: str
    B1: str
    issued_at: str
    expires_at: str
    source_commit: str
    source_tree: str
    nonce: str
    operations: tuple[RealOperationAuthorization, ...]
    authority_id: str
    authority_class: str
    authorization_sha256: str
    consumption_state: str = "NOT_CONSUMED"
    campaign_protocol: str | None = None
    interval_role: str | None = None

    @property
    def live_authority_qualified(self) -> bool:
        return self.authority_class == "REPOSITORY_ACTION_AUTHORITY"


@dataclass(frozen=True)
class ConsumedRealAuthorization:
    authorization: ParsedRealAuthorization
    claim_id: str
    reservation_id: str
    consumption_receipt_sha256: str
    store_identity: str
    consumption_claim: Mapping[str, Any]


@dataclass(frozen=True)
class AuthorizationReservationRequest:
    """Complete interval scope the durable store must atomically reserve."""

    authorization_sha256: str
    decision_id: str
    campaign_id: str
    interval_id: str
    B0: str
    B1: str
    nonce: str
    execution_ids: tuple[str, ...]


class AuthorizationUseStore(Protocol):
    """Future atomic, durable, cross-process replay and interval authority.

    ``claim_once`` MUST atomically reject any half-open interval overlap
    ``[B0, B1)`` against every existing reservation, irrespective of campaign
    or interval ID, while also enforcing decision/nonce one-time use. Adjacent
    intervals where one B1 equals the next B0 do not overlap. It returns a
    role-signed claim binding the entire request and an explicit reservation
    status; a caller-side precheck or unsigned boolean is not authority.
    """

    def claim_once(
        self,
        request: AuthorizationReservationRequest,
    ) -> Mapping[str, Any]: ...


def _utc_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise GuardFailure(f"REAL_AUTH_{field_name}_MISSING")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GuardFailure(f"REAL_AUTH_{field_name}_INVALID") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GuardFailure(f"REAL_AUTH_{field_name}_MUST_BE_TIMEZONE_AWARE")
    return parsed.astimezone(timezone.utc)


def _canonical_record_payload(record: Mapping[str, Any]) -> bytes:
    return json.dumps(
        {key: value for key, value in record.items() if key != "signature"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _verify_signed_record(
    record: Mapping[str, Any],
    verifier: RecordSignatureVerifier,
    *,
    permitted_authority_classes: frozenset[str],
) -> SignatureVerification:
    signer_id = record.get("signer_id")
    signature = record.get("signature")
    if not isinstance(signer_id, str) or not signer_id or not isinstance(signature, str) or not signature:
        raise GuardFailure("REAL_RECORD_SIGNATURE_MISSING")
    result = verifier.verify(signer_id, _canonical_record_payload(record), signature)
    if (not isinstance(result, SignatureVerification) or not result.verified or not result.authority_id
            or result.authority_class not in permitted_authority_classes):
        raise GuardFailure("REAL_RECORD_SIGNATURE_UNVERIFIED")
    return result


def _authority_class_for_epoch(authorization: ParsedRealAuthorization, real_class: str) -> frozenset[str]:
    """Test signatures can validate only test epochs; they never cross into real authority."""
    if authorization.authority_class == "SYNTHETIC_TEST_ONLY":
        return frozenset({"SYNTHETIC_TEST_ONLY"})
    if authorization.authority_class == "REPOSITORY_ACTION_AUTHORITY":
        return frozenset({real_class})
    raise GuardFailure("REAL_AUTH_AUTHORITY_CLASS_UNSUPPORTED")


def _operation_from_record(raw: Any) -> RealOperationAuthorization:
    required = {
        "execution_id", "dimension_id", "surface_id", "operation_identity",
        "source_commit", "source_tree", "process_identity", "request_identity",
        "initiating_record_sha256", "frozen_input_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise GuardFailure("REAL_AUTH_OPERATION_SCHEMA_INVALID")
    values = {key: raw[key] for key in required}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise GuardFailure("REAL_AUTH_OPERATION_FIELD_MISSING")
    if values["execution_id"].lower() == "synthetic":
        raise GuardFailure("REAL_AUTH_SYNTHETIC_EXECUTION_ID_FORBIDDEN")
    if values["dimension_id"] not in _REPOSITORY_CONTROLLED_DIMENSIONS:
        raise GuardFailure("REAL_AUTH_EXTERNAL_DIMENSION_NOT_REPOSITORY_GOVERNED")
    surface = R7R13_GUARD_REGISTRY.get(values["surface_id"])
    if surface is None or values["dimension_id"] not in surface["dimension_ids"]:
        raise GuardFailure("REAL_AUTH_SURFACE_DIMENSION_MISMATCH")
    if not _GIT_OBJECT_ID.fullmatch(values["source_commit"]) or not _GIT_OBJECT_ID.fullmatch(values["source_tree"]):
        raise GuardFailure("REAL_AUTH_SOURCE_IDENTITY_INVALID")
    if any(not re.fullmatch(r"[0-9a-f]{64}", values[field]) for field in ("initiating_record_sha256", "frozen_input_sha256")):
        raise GuardFailure("REAL_AUTH_INITIATING_INPUT_IDENTITY_INVALID")
    return RealOperationAuthorization(**values)


def parse_real_epoch_authorization(
    document: Any,
    expected: RealAuthorizationScope,
    verifier: RecordSignatureVerifier,
    *,
    now: datetime,
) -> ParsedRealAuthorization:
    """Authenticate and bind an explicit authorization without consuming it.

    This parser has no production verifier, installer, or runtime caller in R16.
    A test verifier is deliberately marked non-live and cannot qualify activity.
    """
    required = {
        "schema_version", "decision_id", "decision", "campaign_id", "epoch_id",
        "interval_id", "B0", "B1", "source_commit", "source_tree", "issued_at",
        "expires_at", "nonce", "operations", "signer_id", "signature",
    }
    r19_fields = {"campaign_protocol", "interval_role"}
    if not isinstance(document, dict) or set(document) not in (required, required | r19_fields):
        raise GuardFailure("REAL_AUTH_SCHEMA_INVALID")
    if document["schema_version"] != REAL_AUTHORIZATION_SCHEMA_VERSION:
        raise GuardFailure("REAL_AUTH_SCHEMA_VERSION_MISMATCH")
    if document["decision"] != "ALLOW":
        raise GuardFailure("REAL_AUTH_DECISION_NOT_ALLOW")
    for key in ("decision_id", "campaign_id", "epoch_id", "interval_id", "nonce"):
        if not isinstance(document[key], str) or not document[key]:
            raise GuardFailure(f"REAL_AUTH_{key.upper()}_MISSING")
    if document["campaign_id"] != expected.campaign_id:
        raise GuardFailure("REAL_AUTH_CAMPAIGN_MISMATCH")
    if document["interval_id"] != expected.interval_id:
        raise GuardFailure("REAL_AUTH_INTERVAL_ID_MISMATCH")
    if document["B0"] != expected.B0 or document["B1"] != expected.B1:
        raise GuardFailure("REAL_AUTH_INTERVAL_BOUNDARY_MISMATCH")
    if document["source_commit"] != expected.source_commit or document["source_tree"] != expected.source_tree:
        raise GuardFailure("REAL_AUTH_SOURCE_IDENTITY_MISMATCH")
    protocol = document.get("campaign_protocol")
    interval_role = document.get("interval_role")
    if (protocol is None) != (interval_role is None):
        raise GuardFailure("REAL_AUTH_PROTOCOL_BINDING_INCOMPLETE")
    if protocol is not None:
        if (protocol == R19_CAMPAIGN_COMPOSITION_SCHEMA
                and interval_role in {"B", "C"}):
            pass
        elif protocol == R20_CAMPAIGN_COMPOSITION_SCHEMA and interval_role == "C":
            pass
        else:
            raise GuardFailure("REAL_AUTH_PROTOCOL_BINDING_UNSUPPORTED")
    if (protocol, interval_role) != (expected.campaign_protocol, expected.interval_role):
        raise GuardFailure("REAL_AUTH_PROTOCOL_SCOPE_MISMATCH")
    if not _GIT_OBJECT_ID.fullmatch(document["source_commit"]) or not _GIT_OBJECT_ID.fullmatch(document["source_tree"]):
        raise GuardFailure("REAL_AUTH_SOURCE_IDENTITY_INVALID")
    b0 = _utc_timestamp(document["B0"], "B0")
    b1 = _utc_timestamp(document["B1"], "B1")
    issued = _utc_timestamp(document["issued_at"], "ISSUED_AT")
    expires = _utc_timestamp(document["expires_at"], "EXPIRES_AT")
    if b0 >= b1 or issued >= expires:
        raise GuardFailure("REAL_AUTH_INTERVAL_OR_EXPIRY_INVALID")
    if b0 < issued or b1 > expires:
        raise GuardFailure("REAL_AUTH_INTERVAL_OUTSIDE_AUTHORITY_LIFETIME")
    if now.tzinfo is None or now.utcoffset() is None:
        raise GuardFailure("REAL_AUTH_CLOCK_MUST_BE_TIMEZONE_AWARE")
    now_utc = now.astimezone(timezone.utc)
    if not issued <= now_utc < expires:
        raise GuardFailure("REAL_AUTH_EXPIRED_OR_NOT_YET_VALID")
    raw_operations = document["operations"]
    if not isinstance(raw_operations, list) or not raw_operations:
        raise GuardFailure("REAL_AUTH_OPERATIONS_MISSING")
    operations = tuple(_operation_from_record(row) for row in raw_operations)
    if len({row.execution_id for row in operations}) != len(operations):
        raise GuardFailure("REAL_AUTH_DUPLICATE_EXECUTION_ID")
    if len({(row.dimension_id, row.surface_id, row.operation_identity) for row in operations}) != len(operations):
        raise GuardFailure("REAL_AUTH_DUPLICATE_OPERATION_BINDING")
    expected_operations = tuple(sorted(expected.operations, key=lambda row: row.execution_id))
    if tuple(sorted(operations, key=lambda row: row.execution_id)) != expected_operations:
        raise GuardFailure("REAL_AUTH_OPERATION_SCOPE_MISMATCH")
    if any(row.source_commit != document["source_commit"] or row.source_tree != document["source_tree"] for row in operations):
        raise GuardFailure("REAL_AUTH_OPERATION_SOURCE_MISMATCH")
    verification = _verify_signed_record(
        document, verifier,
        permitted_authority_classes=frozenset({"REPOSITORY_ACTION_AUTHORITY", "SYNTHETIC_TEST_ONLY"}),
    )
    digest = hashlib.sha256(_canonical_record_payload(document)).hexdigest()
    return ParsedRealAuthorization(
        decision_id=document["decision_id"], epoch_id=document["epoch_id"],
        campaign_id=document["campaign_id"], interval_id=document["interval_id"],
        B0=document["B0"], B1=document["B1"], source_commit=document["source_commit"],
        issued_at=document["issued_at"], expires_at=document["expires_at"],
        source_tree=document["source_tree"], nonce=document["nonce"], operations=operations,
        authority_id=verification.authority_id, authority_class=verification.authority_class,
        authorization_sha256=digest,
        campaign_protocol=protocol, interval_role=interval_role,
    )


def validate_authorization_use_claim(
    authorization: ParsedRealAuthorization,
    claim: Mapping[str, Any],
    verifier: RecordSignatureVerifier,
    *,
    now: datetime,
) -> str:
    """Validate one atomic durable-store claim; replayed claims fail closed."""
    required = {
        "schema_version", "claim_id", "reservation_id", "reservation_status",
        "decision_id", "campaign_id", "interval_id", "B0", "B1", "nonce",
        "execution_ids", "authorization_sha256", "claimed_at", "store_identity", "signer_id", "signature",
    }
    if not isinstance(claim, Mapping) or set(claim) != required:
        raise GuardFailure("REAL_AUTH_CLAIM_SCHEMA_INVALID")
    if claim["schema_version"] != AUTHORIZATION_USE_CLAIM_SCHEMA_VERSION:
        raise GuardFailure("REAL_AUTH_CLAIM_SCHEMA_VERSION_MISMATCH")
    expected_ids = sorted(row.execution_id for row in authorization.operations)
    claim_ids = claim["execution_ids"]
    if (not isinstance(claim_ids, list) or any(not isinstance(item, str) or not item for item in claim_ids)
            or claim_ids != sorted(set(claim_ids))):
        raise GuardFailure("REAL_AUTH_CLAIM_EXECUTION_IDS_INVALID")
    if (claim["decision_id"] != authorization.decision_id
            or claim["campaign_id"] != authorization.campaign_id
            or claim["interval_id"] != authorization.interval_id
            or claim["B0"] != authorization.B0 or claim["B1"] != authorization.B1
            or claim["nonce"] != authorization.nonce
            or claim_ids != expected_ids
            or claim["authorization_sha256"] != authorization.authorization_sha256):
        raise GuardFailure("REAL_AUTH_CLAIM_BINDING_MISMATCH")
    if (not isinstance(claim["claim_id"], str) or not claim["claim_id"]
            or not isinstance(claim["reservation_id"], str) or not claim["reservation_id"]
            or not isinstance(claim["store_identity"], str) or not claim["store_identity"]):
        raise GuardFailure("REAL_AUTH_CLAIM_IDENTITY_MISSING")
    claimed_at = _utc_timestamp(claim["claimed_at"], "CLAIMED_AT")
    if now.tzinfo is None or now.utcoffset() is None or claimed_at > now.astimezone(timezone.utc):
        raise GuardFailure("REAL_AUTH_CLAIM_FROM_FUTURE")
    issued = _utc_timestamp(authorization.issued_at, "ISSUED_AT")
    expires = _utc_timestamp(authorization.expires_at, "EXPIRES_AT")
    if not issued <= claimed_at < expires:
        raise GuardFailure("REAL_AUTH_CLAIM_OUTSIDE_AUTHORITY_LIFETIME")
    if claimed_at >= _utc_timestamp(authorization.B0, "B0"):
        raise GuardFailure("REAL_AUTH_CLAIM_NOT_BEFORE_B0")
    verification = _verify_signed_record(
        claim, verifier,
        permitted_authority_classes=frozenset({"REPLAY_PROTECTION_AUTHORITY"}),
    )
    status = claim["reservation_status"]
    if status == "INTERVAL_CONFLICT":
        raise GuardFailure("REAL_AUTH_INTERVAL_CONFLICT")
    if status == "ALREADY_CONSUMED":
        raise GuardFailure("REAL_AUTHORIZATION_REPLAYED_OR_ALREADY_CONSUMED")
    if status != "NEW_NONCONFLICTING":
        raise GuardFailure("REAL_AUTH_RESERVATION_STATUS_INVALID")
    return hashlib.sha256(_canonical_record_payload(claim)).hexdigest()


def _validate_consumed_authorization(
    consumed: ConsumedRealAuthorization,
    verifier: RecordSignatureVerifier,
) -> None:
    """Revalidate a persisted receipt's historical bindings, not a fresh clock observation."""
    claim = consumed.consumption_claim
    claimed_at = _utc_timestamp(claim.get("claimed_at"), "CLAIMED_AT") if isinstance(claim, Mapping) else None
    if claimed_at is None:
        raise GuardFailure("REAL_AUTH_CLAIM_SCHEMA_INVALID")
    digest = validate_authorization_use_claim(
        consumed.authorization, claim, verifier, now=claimed_at,
    )
    if (digest != consumed.consumption_receipt_sha256
            or claim.get("claim_id") != consumed.claim_id
            or claim.get("reservation_id") != consumed.reservation_id
            or claim.get("store_identity") != consumed.store_identity):
        raise GuardFailure("REAL_AUTH_CONSUMPTION_RECEIPT_MISMATCH")


def validate_real_epoch_boundaries(
    authorization: ParsedRealAuthorization,
    boundary_receipts: list[dict[str, Any]],
    verifier: RecordSignatureVerifier,
) -> dict[str, Any]:
    """Require separately authenticated, ordered B0/B1 receipts for this epoch."""
    if len(boundary_receipts) != 2:
        raise GuardFailure("REAL_BOUNDARY_RECEIPT_COUNT_INVALID")
    expected = (("B0", authorization.B0), ("B1", authorization.B1))
    seen_ids: set[str] = set()
    for row, (boundary_name, timestamp) in zip(boundary_receipts, expected, strict=True):
        if not isinstance(row, Mapping):
            raise GuardFailure("REAL_BOUNDARY_RECEIPT_SCHEMA_INVALID")
        required = {
            "schema_version", "receipt_id", "epoch_id", "campaign_id", "interval_id",
            "boundary", "timestamp", "source_commit", "source_tree", "signer_id", "signature",
        }
        if set(row) != required or row["schema_version"] != REAL_INTERVAL_BOUNDARY_SCHEMA_VERSION:
            raise GuardFailure("REAL_BOUNDARY_RECEIPT_SCHEMA_INVALID")
        if row["boundary"] != boundary_name or row["timestamp"] != timestamp:
            raise GuardFailure("REAL_BOUNDARY_RECEIPT_ORDER_OR_TIME_MISMATCH")
        if (row["epoch_id"], row["campaign_id"], row["interval_id"], row["source_commit"], row["source_tree"]) != (
            authorization.epoch_id, authorization.campaign_id, authorization.interval_id,
            authorization.source_commit, authorization.source_tree,
        ):
            raise GuardFailure("REAL_BOUNDARY_RECEIPT_AUTHORITY_MISMATCH")
        if not isinstance(row["receipt_id"], str) or not row["receipt_id"] or row["receipt_id"] in seen_ids:
            raise GuardFailure("REAL_BOUNDARY_RECEIPT_DUPLICATE_ID")
        seen_ids.add(row["receipt_id"])
        _verify_signed_record(
            row, verifier,
            permitted_authority_classes=_authority_class_for_epoch(authorization, "INTERVAL_BOUNDARY_AUTHORITY"),
        )
    if _utc_timestamp(authorization.B0, "B0") >= _utc_timestamp(authorization.B1, "B1"):
        raise GuardFailure("REAL_BOUNDARY_INTERVAL_OVERLAP_OR_ORDER_INVALID")
    return {"B0": authorization.B0, "B1": authorization.B1, "boundary_receipt_ids": sorted(seen_ids), "complete": True}


def reject_overlapping_real_epoch_scopes(authorizations: list[ParsedRealAuthorization]) -> None:
    """Optional diagnostic for half-open interval overlap; not authorization."""
    ordered = sorted(authorizations, key=lambda row: _utc_timestamp(row.B0, "B0"))
    for left, right in zip(ordered, ordered[1:]):
        if _utc_timestamp(left.B1, "B1") > _utc_timestamp(right.B0, "B0"):
            raise GuardFailure("REAL_AUTH_INTERVAL_OVERLAP")


def consume_real_epoch_authorization(
    authorization: ParsedRealAuthorization,
    store: AuthorizationUseStore,
    verifier: RecordSignatureVerifier,
    *,
    now: datetime,
    validation_clock: Callable[[], datetime] | None = None,
) -> ConsumedRealAuthorization:
    """Admit at PRE, then validate the fresh signed claim against independent POST.

    The post-claim UTC sample occurs only after the atomic store returns. It is
    never derived from the claim timestamp; rollback fails closed without retry.
    ``validation_clock`` permits deterministic clock observations in tests.
    """
    if not authorization.live_authority_qualified:
        raise GuardFailure("REAL_AUTH_TEST_OR_UNTRUSTED_AUTHORITY_CANNOT_BE_CONSUMED")
    if now.tzinfo is None or now.utcoffset() is None or not (
        _utc_timestamp(authorization.issued_at, "ISSUED_AT") <= now.astimezone(timezone.utc)
        < _utc_timestamp(authorization.expires_at, "EXPIRES_AT")
    ):
        raise GuardFailure("REAL_AUTH_EXPIRED_OR_NOT_YET_VALID")
    if now.astimezone(timezone.utc) >= _utc_timestamp(authorization.B0, "B0"):
        raise GuardFailure("REAL_AUTH_CONSUMED_AFTER_B0")
    request = AuthorizationReservationRequest(
        authorization_sha256=authorization.authorization_sha256,
        decision_id=authorization.decision_id,
        campaign_id=authorization.campaign_id,
        interval_id=authorization.interval_id,
        B0=authorization.B0,
        B1=authorization.B1,
        nonce=authorization.nonce,
        execution_ids=tuple(sorted(row.execution_id for row in authorization.operations)),
    )
    claim = store.claim_once(request)
    post_claim_now = (datetime.now(timezone.utc) if validation_clock is None
                      else validation_clock())
    if (not isinstance(post_claim_now, datetime) or post_claim_now.tzinfo is None
            or post_claim_now.utcoffset() is None):
        raise GuardFailure("REAL_AUTH_POST_CLAIM_CLOCK_INVALID")
    if post_claim_now.astimezone(timezone.utc) < now.astimezone(timezone.utc):
        raise GuardFailure("REAL_AUTH_POST_CLAIM_CLOCK_ROLLBACK")
    digest = validate_authorization_use_claim(authorization, claim, verifier, now=post_claim_now)
    return ConsumedRealAuthorization(
        authorization=authorization,
        claim_id=str(claim["claim_id"]),
        reservation_id=str(claim["reservation_id"]),
        consumption_receipt_sha256=digest,
        store_identity=str(claim["store_identity"]),
        consumption_claim=dict(claim),
    )


def validate_real_activity_receipts(
    authorization: ParsedRealAuthorization | ConsumedRealAuthorization,
    boundary_receipts: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    verifier: RecordSignatureVerifier,
    *,
    prior_receipt_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate an already-observed future ledger; never invokes a target."""
    consumed = isinstance(authorization, ConsumedRealAuthorization)
    parsed = authorization.authorization if consumed else authorization
    if parsed.live_authority_qualified and not consumed:
        raise GuardFailure("REAL_AUTHORIZATION_NOT_CONSUMED")
    if consumed:
        _validate_consumed_authorization(authorization, verifier)
    authorization = parsed
    validate_real_epoch_boundaries(authorization, boundary_receipts, verifier)
    prior_receipt_ids = prior_receipt_ids or set()
    allowed = {row.execution_id: row for row in authorization.operations}
    event_ids: set[str] = set()
    phases: dict[str, list[str]] = {}
    attempted: set[str] = set()
    dimension_owner_execution_ids: set[str] = set()
    causal_bindings: dict[str, dict[str, Any]] = {}
    last_by_execution: dict[str, str] = {}
    execution_links: dict[str, dict[str, str]] = {}
    receipt_schema_versions = {record.get("schema_version") for record in receipts if isinstance(record, Mapping)}
    if len(receipt_schema_versions) > 1:
        raise GuardFailure("REAL_RECEIPT_MIXED_SCHEMA_VERSIONS")
    # R16's first receipt used an empty predecessor marker. R17's durable
    # ledger is a hash chain rooted at SHA-256(empty), matching its checkpoint
    # and append path. Keep the version-specific root explicit here so the
    # independent receipt validator agrees with the R17 writer.
    previous_hash: str | None = (
        hashlib.sha256(b"").hexdigest()
        if receipt_schema_versions == {REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17}
        else None
    )
    previous_time: datetime | None = None
    for expected_sequence, record in enumerate(receipts, start=1):
        if not isinstance(record, Mapping):
            raise GuardFailure("REAL_RECEIPT_SCHEMA_INVALID")
        required = {
            "schema_version", "epoch_kind", "event_id", "sequence", "epoch_id",
            "authorization_decision_id", "campaign_id", "interval_id", "execution_id",
            "timestamp", "dimension_id", "surface_id", "operation_identity",
            "source_commit", "source_tree", "process_identity", "request_identity",
            "initiating_record_sha256", "frozen_input_sha256", "behavior_observation_sha256",
            "outcome_attestation_id",
            "phase", "reason_code", "previous_receipt_sha256", "signer_id", "signature",
        }
        r17_receipt = record.get("schema_version") == REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17
        r17_fields = {"parent_execution_id", "correlation_id", "dimension_count_owner"}
        if r17_receipt:
            required |= r17_fields
        if set(record) != required:
            raise GuardFailure("REAL_RECEIPT_SCHEMA_INVALID")
        if (record["schema_version"] not in {
                REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION,
                REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17,
        } or record["epoch_kind"] != REAL_EPOCH_KIND):
            raise GuardFailure("REAL_RECEIPT_EPOCH_KIND_OR_SCHEMA_MISMATCH")
        if (not isinstance(record["sequence"], int) or isinstance(record["sequence"], bool)
                or not isinstance(record["execution_id"], str) or not record["execution_id"]):
            raise GuardFailure("REAL_RECEIPT_SEQUENCE_OR_EXECUTION_ID_INVALID")
        op = allowed.get(record["execution_id"])
        if op is None:
            raise GuardFailure("REAL_RECEIPT_UNKNOWN_EXECUTION_ID")
        binding = (
            record["dimension_id"], record["surface_id"], record["operation_identity"],
            record["source_commit"], record["source_tree"], record["process_identity"], record["request_identity"],
            record["initiating_record_sha256"], record["frozen_input_sha256"],
        )
        if binding != (op.dimension_id, op.surface_id, op.operation_identity, op.source_commit, op.source_tree, op.process_identity, op.request_identity,
                       op.initiating_record_sha256, op.frozen_input_sha256):
            raise GuardFailure("REAL_RECEIPT_OPERATION_BINDING_MISMATCH")
        if r17_receipt:
            parent_id = record["parent_execution_id"]
            correlation_id = record["correlation_id"]
            owns_dimension = record["dimension_count_owner"]
            if ((parent_id is not None and
                    (not isinstance(parent_id, str) or not parent_id or parent_id == record["execution_id"]))
                    or not isinstance(correlation_id, str) or not correlation_id
                    or correlation_id != op.request_identity
                    or not isinstance(owns_dimension, bool)):
                raise GuardFailure("R17_RECEIPT_CAUSAL_BINDING_INVALID")
            causal = {
                "parent_execution_id": parent_id,
                "correlation_id": correlation_id,
                "dimension_count_owner": owns_dimension,
            }
            previous_causal = causal_bindings.get(record["execution_id"])
            if previous_causal is not None and previous_causal != causal:
                raise GuardFailure("R17_RECEIPT_CAUSAL_BINDING_CHANGED_WITHIN_LIFECYCLE")
            causal_bindings[record["execution_id"]] = causal
            if owns_dimension:
                dimension_owner_execution_ids.add(record["execution_id"])
        if (not re.fullmatch(r"[0-9a-f]{64}", str(record["behavior_observation_sha256"]))
                or not isinstance(record["outcome_attestation_id"], str) or not record["outcome_attestation_id"]):
            raise GuardFailure("REAL_RECEIPT_BEHAVIOR_OR_OUTCOME_LINK_INVALID")
        if (record["epoch_id"], record["authorization_decision_id"], record["campaign_id"], record["interval_id"]) != (authorization.epoch_id, authorization.decision_id, authorization.campaign_id, authorization.interval_id):
            raise GuardFailure("REAL_RECEIPT_AUTHORITY_BINDING_MISMATCH")
        if record["sequence"] != expected_sequence:
            raise GuardFailure("REAL_RECEIPT_SEQUENCE_GAP_OR_REORDER")
        event_id = record["event_id"]
        if not isinstance(event_id, str) or not event_id or event_id in event_ids or event_id in prior_receipt_ids:
            raise GuardFailure("REAL_RECEIPT_DUPLICATE_OR_REPLAYED_ID")
        event_ids.add(event_id)
        if record["previous_receipt_sha256"] != previous_hash:
            raise GuardFailure("REAL_RECEIPT_CHAIN_MISMATCH")
        timestamp = _utc_timestamp(record["timestamp"], "RECEIPT_TIMESTAMP")
        before_b0 = timestamp < _utc_timestamp(authorization.B0, "B0") if r17_receipt else timestamp <= _utc_timestamp(authorization.B0, "B0")
        if before_b0 or timestamp >= _utc_timestamp(authorization.B1, "B1"):
            raise GuardFailure("REAL_RECEIPT_OUTSIDE_B0_B1")
        if previous_time is not None and timestamp < previous_time:
            raise GuardFailure("REAL_RECEIPT_TIME_ORDER_MISMATCH")
        previous_time = timestamp
        phase = record["phase"]
        if phase not in {"ATTEMPTED", "AUTHORIZED", "BLOCKED", "STARTED", "COMPLETED", "FAILED", "CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLOSED"}:
            raise GuardFailure("REAL_RECEIPT_PHASE_UNKNOWN")
        reason_code = record["reason_code"]
        if phase in {"BLOCKED", "FAILED"}:
            if not isinstance(reason_code, str) or not reason_code:
                raise GuardFailure("REAL_RECEIPT_TERMINAL_REASON_MISSING")
        elif reason_code is not None:
            raise GuardFailure("REAL_RECEIPT_REASON_ON_NONTERMINAL_PHASE")
        if phase == "ATTEMPTED":
            if record["execution_id"] in attempted:
                raise GuardFailure("REAL_RECEIPT_REPLAYED_EXECUTION_ID")
            attempted.add(record["execution_id"])
        phases.setdefault(record["execution_id"], []).append(phase)
        last_by_execution[record["execution_id"]] = phase
        link = {
            "execution_id": record["execution_id"],
            "initiating_record_sha256": record["initiating_record_sha256"],
            "frozen_input_sha256": record["frozen_input_sha256"],
            "behavior_observation_sha256": record["behavior_observation_sha256"],
            "outcome_attestation_id": record["outcome_attestation_id"],
        }
        if r17_receipt:
            link.update(causal_bindings[record["execution_id"]])
        previous_link = execution_links.get(record["execution_id"])
        if previous_link is not None and previous_link != link:
            raise GuardFailure("REAL_RECEIPT_EXECUTION_LINK_CHANGED_WITHIN_LIFECYCLE")
        execution_links[record["execution_id"]] = link
        _verify_signed_record(
            record, verifier,
            permitted_authority_classes=_authority_class_for_epoch(authorization, "REPOSITORY_OBSERVER_AUTHORITY"),
        )
        previous_hash = hashlib.sha256(_canonical_record_payload(record)).hexdigest()
    for execution_id, observed_phases in phases.items():
        if observed_phases not in (
            ["ATTEMPTED", "AUTHORIZED", "STARTED", "COMPLETED", "CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLOSED"],
            ["ATTEMPTED", "AUTHORIZED", "STARTED", "FAILED", "CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLOSED"],
            ["ATTEMPTED", "BLOCKED", "CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLOSED"],
        ):
            raise GuardFailure("REAL_RECEIPT_LIFECYCLE_INCOMPLETE_OR_INVALID")
        if last_by_execution[execution_id] != "CLOSED":
            raise GuardFailure("REAL_RECEIPT_LIFECYCLE_NOT_CLOSED")
    if any(record.get("schema_version") == REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17 for record in receipts):
        from r7r17_activity import R17_GOVERNED_CALL_TOPOLOGY

        for execution_id, causal in causal_bindings.items():
            parent_id = causal["parent_execution_id"]
            child = allowed.get(execution_id)
            if child is None:
                raise GuardFailure("R17_RECEIPT_EXECUTION_NOT_AUTHORIZED")
            child_surface = R17_GOVERNED_CALL_TOPOLOGY["surfaces"].get(child.surface_id)
            if (child_surface is None or child.dimension_id != child_surface["dimension_id"]
                    or child.operation_identity != child_surface["primitive"]):
                raise GuardFailure("R17_RECEIPT_SURFACE_CONTRACT_MISMATCH")
            if parent_id is None:
                if causal["dimension_count_owner"] is not True:
                    raise GuardFailure("R17_TOP_LEVEL_DIMENSION_OWNER_MISMATCH")
                continue
            parent = allowed.get(parent_id)
            if (parent is None or child is None or parent.request_identity != child.request_identity
                    or parent_id not in phases or not phases[parent_id]):
                raise GuardFailure("R17_RECEIPT_PARENT_NOT_AUTHORIZED_OR_CORRELATED")
            parent_attempt = next((row["sequence"] for row in receipts
                                   if row.get("execution_id") == parent_id and row.get("phase") == "ATTEMPTED"), None)
            child_attempt = next((row["sequence"] for row in receipts
                                  if row.get("execution_id") == execution_id and row.get("phase") == "ATTEMPTED"), None)
            if parent_attempt is None or child_attempt is None or parent_attempt >= child_attempt:
                raise GuardFailure("R17_RECEIPT_PARENT_ORDER_INVALID")
            edge = R17_GOVERNED_CALL_TOPOLOGY["edges"].get((parent.surface_id, child.surface_id))
            expected_owner = edge is not None and edge["dimension_count_owner"] != "PARENT"
            expected_same_dimension = parent.dimension_id == child.dimension_id
            if (edge is None
                    or edge["dimension_relation"] != ("SAME" if expected_same_dimension else "DIFFERENT")
                    or causal["dimension_count_owner"] is not expected_owner):
                raise GuardFailure("R17_RECEIPT_DIMENSION_OWNER_MISMATCH")
    ledger_head = previous_hash or hashlib.sha256(b"").hexdigest()
    return {
        "schema_version": (
            REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17
            if receipt_schema_versions == {REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17}
            else REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION
        ),
        "receipt_count": len(receipts),
        "unique_execution_count": len(phases),
        "attempted_execution_ids": sorted(attempted),
        "dimension_owner_execution_ids": sorted(
            dimension_owner_execution_ids if any(
                record.get("schema_version") == REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17
                for record in receipts
            ) else attempted
        ),
        "execution_links": {key: execution_links[key] for key in sorted(execution_links)},
        "ledger_head_sha256": ledger_head,
        "receipt_records": copy.deepcopy(receipts),
        "boundary_records": copy.deepcopy(boundary_receipts),
        "prior_receipt_ids": sorted(prior_receipt_ids),
        "ledger_chain_valid": True,
        "lifecycle_complete": all(last_by_execution[key] == "CLOSED" for key in phases),
        "authorization_consumed": consumed,
        "real_authority_qualified": authorization.live_authority_qualified,
    }


def validate_real_effect_attestations(
    authorization: ParsedRealAuthorization | ConsumedRealAuthorization,
    receipt_validation: Mapping[str, Any],
    attestations: list[dict[str, Any]],
    verifier: RecordSignatureVerifier,
) -> dict[str, Any]:
    """Bind one independent effect/absence observation to each executed operation."""
    consumed = isinstance(authorization, ConsumedRealAuthorization)
    parsed = authorization.authorization if consumed else authorization
    if parsed.live_authority_qualified and not consumed:
        raise GuardFailure("REAL_AUTHORIZATION_NOT_CONSUMED")
    if consumed:
        _validate_consumed_authorization(authorization, verifier)
    links = receipt_validation.get("execution_links")
    ledger_head = receipt_validation.get("ledger_head_sha256")
    if not isinstance(links, Mapping) or not isinstance(ledger_head, str):
        raise GuardFailure("REAL_EFFECT_RECEIPT_LEDGER_AUTHORITY_MISSING")
    allowed_operations = {row.execution_id: row for row in parsed.operations}
    seen_ids: set[str] = set()
    by_execution: dict[str, dict[str, Any]] = {}
    for row in attestations:
        if not isinstance(row, Mapping):
            raise GuardFailure("REAL_EFFECT_ATTESTATION_SCHEMA_INVALID")
        required = {
            "schema_version", "attestation_id", "epoch_id", "authorization_decision_id", "campaign_id", "interval_id",
            "execution_id", "initiating_record_sha256", "frozen_input_sha256", "behavior_observation_sha256",
            "ledger_head_sha256", "effect_kind", "before_state_sha256", "after_state_sha256", "absence_proof_sha256",
            "source_commit", "source_tree", "timestamp", "signer_id", "signature",
        }
        if set(row) != required or row["schema_version"] != REAL_EFFECT_ATTESTATION_SCHEMA_VERSION:
            raise GuardFailure("REAL_EFFECT_ATTESTATION_SCHEMA_INVALID")
        ident = row["attestation_id"]
        execution_id = row["execution_id"]
        if not isinstance(ident, str) or not ident or ident in seen_ids or execution_id in by_execution:
            raise GuardFailure("REAL_EFFECT_ATTESTATION_DUPLICATE_OR_REPLAYED")
        seen_ids.add(ident)
        op = allowed_operations.get(execution_id)
        link = links.get(execution_id)
        if op is None or not isinstance(link, Mapping):
            raise GuardFailure("REAL_EFFECT_ATTESTATION_UNKNOWN_EXECUTION")
        authority_binding = (
            row["epoch_id"], row["authorization_decision_id"], row["campaign_id"], row["interval_id"],
            row["source_commit"], row["source_tree"], row["initiating_record_sha256"], row["frozen_input_sha256"],
            row["behavior_observation_sha256"], row["ledger_head_sha256"],
        )
        expected_binding = (
            parsed.epoch_id, parsed.decision_id, parsed.campaign_id, parsed.interval_id,
            parsed.source_commit, parsed.source_tree, op.initiating_record_sha256, op.frozen_input_sha256,
            link["behavior_observation_sha256"], ledger_head,
        )
        if authority_binding != expected_binding or link["outcome_attestation_id"] != ident:
            raise GuardFailure("REAL_EFFECT_ATTESTATION_CAUSAL_BINDING_MISMATCH")
        timestamp = _utc_timestamp(row["timestamp"], "EFFECT_TIMESTAMP")
        if timestamp <= _utc_timestamp(parsed.B0, "B0") or timestamp >= _utc_timestamp(parsed.B1, "B1"):
            raise GuardFailure("REAL_EFFECT_ATTESTATION_OUTSIDE_B0_B1")
        if row["effect_kind"] == "STATE_TRANSITION":
            before, after = row["before_state_sha256"], row["after_state_sha256"]
            if (not isinstance(before, str) or not re.fullmatch(r"[0-9a-f]{64}", before)
                    or not isinstance(after, str) or not re.fullmatch(r"[0-9a-f]{64}", after)
                    or before == after or row["absence_proof_sha256"] is not None):
                raise GuardFailure("REAL_EFFECT_STATE_TRANSITION_EVIDENCE_INVALID")
        elif row["effect_kind"] == "NO_EFFECT_VERIFIED":
            absence = row["absence_proof_sha256"]
            if (row["before_state_sha256"] is not None or row["after_state_sha256"] is not None
                    or not isinstance(absence, str) or not re.fullmatch(r"[0-9a-f]{64}", absence)):
                raise GuardFailure("REAL_EFFECT_ABSENCE_EVIDENCE_INVALID")
        else:
            raise GuardFailure("REAL_EFFECT_KIND_UNKNOWN")
        _verify_signed_record(row, verifier, permitted_authority_classes=_authority_class_for_epoch(parsed, "STATE_TRANSITION_OBSERVER_AUTHORITY"))
        by_execution[execution_id] = {
            "attestation_id": ident,
            "execution_id": execution_id,
            "effect_kind": row["effect_kind"],
            "attestation_sha256": hashlib.sha256(_canonical_record_payload(row)).hexdigest(),
        }
    expected_executions = set(links)
    if set(by_execution) != expected_executions:
        raise GuardFailure("REAL_EFFECT_ATTESTATION_COVERAGE_INCOMPLETE")
    rows = [by_execution[key] for key in sorted(by_execution)]
    return {
        "effect_records": copy.deepcopy(attestations),
        "effects_by_execution": {key: by_execution[key] for key in sorted(by_execution)},
        "effect_evidence_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "complete": True,
    }


def _pair_evidence_digests(
    surface: str,
    dimension: str,
    receipt_validation: Mapping[str, Any],
    effect_validation: Mapping[str, Any],
) -> tuple[list[str], list[str], str, str, str]:
    links = receipt_validation["execution_links"]
    effects = effect_validation["effects_by_execution"]
    operations = {row.execution_id: row for row in receipt_validation["authorized_operations"]}
    executions = sorted(
        ident for ident, operation in operations.items()
        if (operation.surface_id, operation.dimension_id) == (surface, dimension) and ident in links
    )
    linked_rows = [links[ident] for ident in executions]
    effect_ids = sorted(effects[ident]["attestation_id"] for ident in executions if ident in effects)
    effect_rows = [effects[ident] for ident in executions if ident in effects]
    execution_digest = hashlib.sha256(json.dumps(linked_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    effect_digest = hashlib.sha256(json.dumps(effect_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    census_digest = hashlib.sha256(json.dumps({
        "ledger_head_sha256": receipt_validation["ledger_head_sha256"],
        "execution_evidence_sha256": execution_digest,
        "effect_evidence_sha256": effect_digest,
        "enumerated_execution_ids": executions,
        "effect_attestation_ids": effect_ids,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return executions, effect_ids, execution_digest, effect_digest, census_digest


def derive_real_repository_activity_state(
    authorization: ParsedRealAuthorization | ConsumedRealAuthorization,
    receipt_validation: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    verifier: RecordSignatureVerifier,
    *,
    effect_attestations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Revalidate all five linked evidence classes; missing links remain UNKNOWN."""
    consumed = isinstance(authorization, ConsumedRealAuthorization)
    parsed = authorization.authorization if consumed else authorization
    if not isinstance(receipt_validation, Mapping) or not isinstance(receipt_validation.get("receipt_records"), list) or not isinstance(receipt_validation.get("boundary_records"), list):
        raise GuardFailure("REAL_RECEIPT_LEDGER_RAW_EVIDENCE_MISSING")
    revalidated = validate_real_activity_receipts(
        authorization, receipt_validation["boundary_records"], receipt_validation["receipt_records"], verifier,
        prior_receipt_ids=set(receipt_validation.get("prior_receipt_ids", [])),
    )
    if revalidated["ledger_head_sha256"] != receipt_validation.get("ledger_head_sha256"):
        raise GuardFailure("REAL_RECEIPT_LEDGER_AGGREGATE_MISMATCH")
    revalidated["authorized_operations"] = parsed.operations
    effect_validation = validate_real_effect_attestations(authorization, revalidated, effect_attestations or [], verifier)
    authorization = parsed
    expected_pairs = {
        (surface, dimension)
        for surface, record in R7R13_GUARD_REGISTRY.items()
        for dimension in record["dimension_ids"]
        if dimension in _REPOSITORY_CONTROLLED_DIMENSIONS
    }
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    coverage_ids: set[str] = set()
    for row in coverage_rows:
        if not isinstance(row, Mapping):
            raise GuardFailure("REAL_COVERAGE_SCHEMA_MISMATCH")
        required_coverage_fields = {
            "schema_version", "coverage_id", "epoch_id", "authorization_decision_id", "campaign_id", "interval_id",
            "B0", "B1", "surface_id", "dimension_id", "source_commit", "source_tree",
            "complete_enumeration", "unknown_execution_count", "enumerated_execution_ids", "effect_attestation_ids",
            "ledger_head_sha256", "execution_evidence_sha256", "effect_evidence_sha256", "census_evidence_sha256",
            "census_scope_identity", "signer_id", "signature",
        }
        if set(row) != required_coverage_fields or row.get("schema_version") != REAL_ACTIVITY_COVERAGE_SCHEMA_VERSION:
            raise GuardFailure("REAL_COVERAGE_SCHEMA_MISMATCH")
        if (not isinstance(row["coverage_id"], str) or not row["coverage_id"] or row["coverage_id"] in coverage_ids
                or not isinstance(row["census_scope_identity"], str) or not row["census_scope_identity"]):
            raise GuardFailure("REAL_COVERAGE_CENSUS_IDENTITY_INVALID")
        coverage_ids.add(row["coverage_id"])
        if (not isinstance(row["unknown_execution_count"], int) or isinstance(row["unknown_execution_count"], bool)
                or row["unknown_execution_count"] < 0 or not isinstance(row["complete_enumeration"], bool)
                or not isinstance(row["enumerated_execution_ids"], list) or not isinstance(row["effect_attestation_ids"], list)):
            raise GuardFailure("REAL_COVERAGE_ENUMERATION_SCHEMA_INVALID")
        if (any(not isinstance(item, str) or not item for item in row["enumerated_execution_ids"] + row["effect_attestation_ids"])
                or row["enumerated_execution_ids"] != sorted(set(row["enumerated_execution_ids"]))
                or row["effect_attestation_ids"] != sorted(set(row["effect_attestation_ids"]))):
            raise GuardFailure("REAL_COVERAGE_ENUMERATION_NONCANONICAL")
        pair = (row.get("surface_id"), row.get("dimension_id"))
        if pair not in expected_pairs or pair in by_pair:
            raise GuardFailure("REAL_COVERAGE_UNKNOWN_OR_DUPLICATE_SURFACE")
        if (row.get("epoch_id"), row.get("authorization_decision_id"), row.get("campaign_id"), row.get("interval_id")) != (
            authorization.epoch_id, authorization.decision_id, authorization.campaign_id, authorization.interval_id
        ):
            raise GuardFailure("REAL_COVERAGE_INTERVAL_BINDING_MISMATCH")
        if row.get("B0") != authorization.B0 or row.get("B1") != authorization.B1:
            raise GuardFailure("REAL_COVERAGE_BOUNDARY_MISMATCH")
        if (row.get("source_commit"), row.get("source_tree")) != (authorization.source_commit, authorization.source_tree):
            raise GuardFailure("REAL_COVERAGE_SOURCE_OR_INTEGRITY_PROOF_MISSING")
        executions, effect_ids, execution_digest, effect_digest, census_digest = _pair_evidence_digests(
            pair[0], pair[1], revalidated, effect_validation,
        )
        if (row.get("ledger_head_sha256") != revalidated["ledger_head_sha256"]
                or row.get("execution_evidence_sha256") != execution_digest
                or row.get("effect_evidence_sha256") != effect_digest
                or row.get("census_evidence_sha256") != census_digest
                or row.get("enumerated_execution_ids") != executions
                or row.get("effect_attestation_ids") != effect_ids):
            raise GuardFailure("REAL_COVERAGE_LEDGER_RECONCILIATION_MISMATCH")
        _verify_signed_record(
            row, verifier,
            permitted_authority_classes=_authority_class_for_epoch(authorization, "INDEPENDENT_CENSUS_AUTHORITY"),
        )
        by_pair[pair] = row
    observed_by_dimension: dict[str, list[str]] = {dimension: [] for dimension in _REPOSITORY_CONTROLLED_DIMENSIONS}
    all_observed_by_dimension: dict[str, list[str]] = {dimension: [] for dimension in _REPOSITORY_CONTROLLED_DIMENSIONS}
    for operation in authorization.operations:
        if operation.execution_id in revalidated["attempted_execution_ids"]:
            all_observed_by_dimension[operation.dimension_id].append(operation.execution_id)
            if operation.execution_id in revalidated.get(
                "dimension_owner_execution_ids", revalidated["attempted_execution_ids"]
            ):
                observed_by_dimension[operation.dimension_id].append(operation.execution_id)
    rows: list[dict[str, Any]] = []
    for dimension in sorted(_REPOSITORY_CONTROLLED_DIMENSIONS):
        pairs = sorted(pair for pair in expected_pairs if pair[1] == dimension)
        complete = len(pairs) > 0 and all(
            (pair in by_pair and by_pair[pair].get("complete_enumeration") is True
             and by_pair[pair].get("unknown_execution_count") == 0
             and isinstance(by_pair[pair].get("enumerated_execution_ids"), list))
            for pair in pairs
        )
        enumerated = set().union(*(set(by_pair[pair].get("enumerated_execution_ids", [])) for pair in pairs if pair in by_pair))
        all_observed = set(all_observed_by_dimension[dimension])
        if complete and enumerated != all_observed:
            raise GuardFailure("REAL_COVERAGE_LEDGER_RECONCILIATION_MISMATCH")
        observed = set(observed_by_dimension[dimension])
        status = "POSITIVE_OBSERVED" if observed else "ZERO_WITH_COMPLETE_COVERAGE" if complete else "UNKNOWN"
        rows.append({"dimension_id": dimension, "status": status, "observed_execution_ids": sorted(observed), "coverage_complete": complete})
    all_repo_covered = all(row["coverage_complete"] for row in rows)
    repository_scope_zero_authority_valid = (
        bool(rows)
        and all_repo_covered
        and all(row["status"] == "ZERO_WITH_COMPLETE_COVERAGE" for row in rows)
        and consumed
        and authorization.live_authority_qualified
        and revalidated["ledger_chain_valid"]
        and revalidated["lifecycle_complete"]
        and effect_validation["complete"]
    )
    external_dimensions = {
        dimension: "EXTERNAL_PROVIDER_REQUIRED_UNAVAILABLE"
        for dimension in DIMENSIONS
        if dimension not in _REPOSITORY_CONTROLLED_DIMENSIONS
    }
    full_nine_dimension_authority = repository_scope_zero_authority_valid and not external_dimensions
    return {
        "repository_dimension_rows": rows,
        "repository_coverage_complete": all_repo_covered,
        "repository_scope_zero_authority_valid": repository_scope_zero_authority_valid,
        "external_dimensions": external_dimensions,
        "FULL_9_DIMENSION_OBSERVABILITY": full_nine_dimension_authority,
        "ZERO_ACTIVITY_AUTHORITY_VALID": full_nine_dimension_authority,
        "R7_TO_R11_ZERO_ACTIVITY_RETROACTIVE_AUTHORITY": "UNPROVABLE",
    }


@dataclass
class LedgerCheckpoint:
    """Authoritative head retained independently from mutable ledger entries."""

    epoch_id: str
    ledger_id: str
    checkpoint_authority_id: str
    checkpoint_sequence: int = 0
    entry_count: int = 0
    first_entry_hash: str | None = None
    last_entry_id: str | None = None
    last_entry_hash: str | None = None
    chain_root: str = ""
    observer_identity: str = "r7r14-observer"
    guard_registry_identity: str = ""

    def __post_init__(self) -> None:
        if not self.guard_registry_identity:
            self.guard_registry_identity = guard_registry_sha256()
        if not self.chain_root:
            self.chain_root = _sha([self.epoch_id, self.ledger_id, "root"])

    def commitment(self) -> str:
        return _sha({
            "epoch_id": self.epoch_id,
            "ledger_id": self.ledger_id,
            "checkpoint_authority_id": self.checkpoint_authority_id,
            "checkpoint_sequence": self.checkpoint_sequence,
            "entry_count": self.entry_count,
            "first_entry_hash": self.first_entry_hash,
            "last_entry_id": self.last_entry_id,
            "last_entry_hash": self.last_entry_hash,
            "chain_root": self.chain_root,
            "observer_identity": self.observer_identity,
            "guard_registry_identity": self.guard_registry_identity,
        })


@dataclass(frozen=True)
class CheckpointAuthority:
    """Higher-level bound checkpoint identity; never recomputed from a ledger."""

    checkpoint_authority_id: str
    expected_commitment: str


@dataclass(frozen=True)
class ExternalLedgerCheckpointAuthority:
    """Checkpoint commitment kept outside the mutable ledger object."""
    authority_id: str
    expected_commitment: str
    epoch_id: str
    ledger_id: str
    checkpoint_sequence: int
    entry_count: int
    checkpoint_payload_hash: str

    def payload_hash(self) -> str:
        return _sha(self.__dict__)


@dataclass(frozen=True)
class TrustedCheckpointBinding:
    """Immutable fixture/artifact binding; it is not ledger-owned state."""
    authority_id: str
    authority_payload_hash: str


def bind_synthetic_external_checkpoint(ledger: "Ledger") -> tuple[ExternalLedgerCheckpointAuthority, TrustedCheckpointBinding]:
    if not ledger.verify():
        raise GuardFailure("CHECKPOINT_PRECONDITION_FAILURE")
    checkpoint = ledger.checkpoint
    assert checkpoint is not None
    authority = ExternalLedgerCheckpointAuthority(
        authority_id=_sha(["r7r15-synthetic-external", checkpoint.epoch_id, checkpoint.ledger_id, checkpoint.commitment()]),
        expected_commitment=checkpoint.commitment(), epoch_id=checkpoint.epoch_id,
        ledger_id=checkpoint.ledger_id, checkpoint_sequence=checkpoint.checkpoint_sequence,
        entry_count=checkpoint.entry_count, checkpoint_payload_hash=checkpoint.commitment(),
    )
    return authority, TrustedCheckpointBinding(authority.authority_id, authority.payload_hash())

@dataclass
class Ledger:
    epoch_id: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    ledger_id: str = "repository-observer-ledger"
    checkpoint: LedgerCheckpoint | None = None
    checkpoint_authority: CheckpointAuthority | None = None

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            self.checkpoint = LedgerCheckpoint(
                epoch_id=self.epoch_id,
                ledger_id=self.ledger_id,
                checkpoint_authority_id=f"synthetic:{self.epoch_id}",
            )
        if self.checkpoint.epoch_id != self.epoch_id or self.checkpoint.ledger_id != self.ledger_id:
            raise GuardFailure("CHECKPOINT_BINDING_FAILURE")

    def append(self, receipt: dict[str, Any]) -> dict[str, Any]:
        if receipt.get("activation_epoch_id") != self.epoch_id:
            raise GuardFailure("EPOCH_BINDING_FAILURE")
        expected = self.entries[-1]["receipt_payload_hash"] if self.entries else None
        if receipt.get("previous_ledger_entry_hash") != expected:
            raise GuardFailure("CHAIN_POSITION_FAILURE")
        payload = {k: v for k, v in receipt.items() if k != "receipt_payload_hash"}
        receipt = {**payload, "receipt_payload_hash": _sha(payload)}
        if any(x["receipt_payload_hash"] == receipt["receipt_payload_hash"] for x in self.entries):
            raise GuardFailure("DUPLICATE_RECEIPT")
        checkpoint = self.checkpoint
        assert checkpoint is not None
        if self.checkpoint_authority is not None:
            raise GuardFailure("CHECKPOINT_SEALED")
        old = (
            checkpoint.checkpoint_sequence, checkpoint.entry_count,
            checkpoint.first_entry_hash, checkpoint.last_entry_id,
            checkpoint.last_entry_hash, checkpoint.chain_root,
        )
        self.entries.append(receipt)
        try:
            checkpoint.checkpoint_sequence += 1
            checkpoint.entry_count += 1
            checkpoint.first_entry_hash = checkpoint.first_entry_hash or receipt["receipt_payload_hash"]
            checkpoint.last_entry_id = receipt["event_id"]
            checkpoint.last_entry_hash = receipt["receipt_payload_hash"]
            checkpoint.chain_root = _sha([checkpoint.chain_root, receipt["receipt_payload_hash"]])
        except Exception:
            self.entries.pop()
            (
                checkpoint.checkpoint_sequence, checkpoint.entry_count,
                checkpoint.first_entry_hash, checkpoint.last_entry_id,
                checkpoint.last_entry_hash, checkpoint.chain_root,
            ) = old
            raise GuardFailure("CHECKPOINT_ADVANCE_FAILURE")
        return receipt

    def verification_failure_reason(self) -> str | None:
        """Return a source-derived rejection reason without repairing state."""
        previous = None
        for entry in self.entries:
            payload = {k: v for k, v in entry.items() if k != "receipt_payload_hash"}
            if entry.get("activation_epoch_id") != self.epoch_id:
                return "EPOCH_SUBSTITUTION"
            if entry.get("previous_ledger_entry_hash") != previous:
                return "CHAIN_POSITION_MISMATCH"
            if entry.get("receipt_payload_hash") != _sha(payload):
                return "RECEIPT_HASH_MISMATCH"
            previous = entry["receipt_payload_hash"]
        checkpoint = self.checkpoint
        authority = self.checkpoint_authority
        if checkpoint is None:
            return "CHECKPOINT_MISSING"
        if checkpoint.epoch_id != self.epoch_id or checkpoint.ledger_id != self.ledger_id:
            return "CHECKPOINT_BINDING_MISMATCH"
        if checkpoint.entry_count != len(self.entries):
            return "ENTRY_COUNT_MISMATCH"
        if checkpoint.last_entry_hash != previous:
            return "LAST_HASH_MISMATCH"
        expected_root = _sha([self.epoch_id, self.ledger_id, "root"])
        for entry in self.entries:
            expected_root = _sha([expected_root, entry["receipt_payload_hash"]])
        if checkpoint.chain_root != expected_root:
            return "CHAIN_ROOT_MISMATCH"
        if checkpoint.first_entry_hash != (self.entries[0]["receipt_payload_hash"] if self.entries else None):
            return "FIRST_HASH_MISMATCH"
        if checkpoint.last_entry_id != (self.entries[-1]["event_id"] if self.entries else None):
            return "LAST_ENTRY_ID_MISMATCH"
        if authority is not None and (
            authority.checkpoint_authority_id != checkpoint.checkpoint_authority_id
            or authority.expected_commitment != checkpoint.commitment()
        ):
            return "CHECKPOINT_AUTHORITY_MISMATCH"
        return None

    def verify(self) -> bool:
        return self.verification_failure_reason() is None

    def seal_checkpoint(self) -> CheckpointAuthority:
        """Bind the current head; subsequent appends require a new interval."""
        if not self.verify():
            raise GuardFailure("CHECKPOINT_PRECONDITION_FAILURE")
        assert self.checkpoint is not None
        self.checkpoint_authority = CheckpointAuthority(
            self.checkpoint.checkpoint_authority_id,
            self.checkpoint.commitment(),
        )
        return self.checkpoint_authority

    def verify_checkpoint(
        self,
        expected_authority: ExternalLedgerCheckpointAuthority,
        trusted_binding: TrustedCheckpointBinding,
    ) -> bool:
        """Verify mutable state only against independently supplied authority."""
        return self.verify_checkpoint_reason(expected_authority, trusted_binding) is None

    def verify_checkpoint_reason(
        self,
        expected_authority: ExternalLedgerCheckpointAuthority,
        trusted_binding: TrustedCheckpointBinding,
    ) -> str | None:
        """Return a typed external-authority rejection without mutating state."""
        if (expected_authority.authority_id != trusted_binding.authority_id or
                expected_authority.payload_hash() != trusted_binding.authority_payload_hash):
            return "EXTERNAL_AUTHORITY_BINDING_MISMATCH"
        checkpoint = self.checkpoint
        if checkpoint is None:
            return "CHECKPOINT_MISSING"
        internal_failure = self.verification_failure_reason()
        if internal_failure:
            return f"LEDGER_{internal_failure}"
        if checkpoint.commitment() != expected_authority.expected_commitment:
            return "EXTERNAL_CHECKPOINT_COMMITMENT_MISMATCH"
        if checkpoint.epoch_id != expected_authority.epoch_id:
            return "EXTERNAL_EPOCH_MISMATCH"
        if checkpoint.ledger_id != expected_authority.ledger_id:
            return "EXTERNAL_LEDGER_MISMATCH"
        if checkpoint.checkpoint_sequence != expected_authority.checkpoint_sequence:
            return "EXTERNAL_CHECKPOINT_SEQUENCE_MISMATCH"
        if checkpoint.entry_count != expected_authority.entry_count:
            return "EXTERNAL_ENTRY_COUNT_MISMATCH"
        if checkpoint.commitment() != expected_authority.checkpoint_payload_hash:
            return "EXTERNAL_CHECKPOINT_PAYLOAD_MISMATCH"
        return None

class Guard:
    """Receipt-before-target guard; R13 permits synthetic epochs only."""

    def __init__(
        self,
        epoch: dict[str, Any],
        dimensions: set[str],
        surfaces: set[str],
        ledger: Ledger | None,
    ) -> None:
        self.epoch, self.dimensions, self.surfaces, self.ledger = (
            epoch,
            dimensions,
            surfaces,
            ledger,
        )

    def _validate(self, dimension: str, surface: str) -> Ledger:
        if self.epoch.get("kind") != SYNTHETIC_EPOCH_KIND:
            raise GuardFailure("INVALID_EPOCH")
        if dimension not in self.dimensions:
            raise GuardFailure("FOREIGN_DIMENSION")
        if surface not in self.surfaces or surface not in R7R13_GUARD_REGISTRY:
            raise GuardFailure("FOREIGN_SURFACE")
        if dimension not in R7R13_GUARD_REGISTRY[surface]["dimension_ids"]:
            raise GuardFailure("FOREIGN_SURFACE_DIMENSION_PAIR")
        if self.ledger is None or not self.ledger.verify():
            raise GuardFailure("OBSERVER_UNAVAILABLE")
        return self.ledger

    def _append(
        self,
        ledger: Ledger,
        dimension: str,
        surface: str,
        operation: str,
        *,
        phase: str,
        allowed: bool,
        blocked: bool,
        failed: bool = False,
        completed: bool = False,
    ) -> dict[str, Any]:
        return ledger.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "event_id": _sha(
                    [dimension, surface, operation, phase, len(ledger.entries)]
                ),
                "dimension_id": dimension,
                "surface_id": surface,
                "activation_epoch_id": self.epoch["epoch_id"],
                "execution_id": "synthetic",
                "operation_identity": operation,
                "phase": phase,
                "attempted": phase in {"ATTEMPTED", "BLOCKED"},
                "blocked": blocked,
                "allowed": allowed,
                "failed": failed,
                "completed": completed,
                "previous_ledger_entry_hash": (
                    ledger.entries[-1]["receipt_payload_hash"] if ledger.entries else None
                ),
            }
        )

    def attempt(self, dimension: str, surface: str, operation: str) -> dict[str, Any]:
        ledger = self._validate(dimension, surface)
        return self._append(
            ledger,
            dimension,
            surface,
            operation,
            phase="BLOCKED",
            allowed=False,
            blocked=True,
        )

    def execute(
        self,
        dimension: str,
        surface: str,
        operation: str,
        target: Callable[[], T],
        *,
        policy_allows: bool,
    ) -> T:
        ledger = self._validate(dimension, surface)
        if not policy_allows:
            self._append(
                ledger, dimension, surface, operation,
                phase="BLOCKED", allowed=False, blocked=True,
            )
            raise GuardFailure("POLICY_PROHIBITS_OPERATION")
        # The attempt record exists before the target is evaluated.
        self._append(
            ledger, dimension, surface, operation,
            phase="ATTEMPTED", allowed=True, blocked=False,
        )
        self._append(
            ledger, dimension, surface, operation,
            phase="AUTHORIZED", allowed=True, blocked=False,
        )
        self._append(
            ledger, dimension, surface, operation,
            phase="STARTED", allowed=True, blocked=False,
        )
        try:
            result = target()
        except Exception:
            self._append(
                ledger, dimension, surface, operation,
                phase="FAILED", allowed=True, blocked=False, failed=True,
            )
            raise
        self._append(
            ledger, dimension, surface, operation,
            phase="COMPLETED", allowed=True, blocked=False, completed=True,
        )
        return result

_REAL_ACTIVITY_DISPATCHER: Any = None
_REAL_ACTIVITY_DISPATCHER_LOCK = threading.RLock()


def install_real_activity_dispatcher(dispatcher: Any) -> None:
    """Install an already validated R17 campaign runtime in this process."""
    from r7r17_activity import R17CampaignRuntime

    if type(dispatcher) is not R17CampaignRuntime:
        raise GuardFailure("R17_DISPATCHER_TYPE_INVALID")
    if dispatcher.trust.test_only:
        raise GuardFailure("R17_TEST_ONLY_RUNTIME_CANNOT_BE_INSTALLED")
    if not dispatcher.trust.readiness()["real_activation_eligible"]:
        raise GuardFailure("R17_NONTEST_TRUST_REQUIRED_TO_INSTALL_REAL_RUNTIME")
    dispatcher.assert_activated()
    global _REAL_ACTIVITY_DISPATCHER
    with _REAL_ACTIVITY_DISPATCHER_LOCK:
        if _REAL_ACTIVITY_DISPATCHER is not None:
            raise GuardFailure("R17_DISPATCHER_ALREADY_INSTALLED")
        _REAL_ACTIVITY_DISPATCHER = dispatcher


def uninstall_real_activity_dispatcher(dispatcher: Any) -> None:
    global _REAL_ACTIVITY_DISPATCHER
    with _REAL_ACTIVITY_DISPATCHER_LOCK:
        if _REAL_ACTIVITY_DISPATCHER is not dispatcher:
            raise GuardFailure("R17_DISPATCHER_IDENTITY_MISMATCH")
        dispatcher.assert_can_deactivate()
        _REAL_ACTIVITY_DISPATCHER = None


def active_real_activity_dispatcher() -> Any:
    return _REAL_ACTIVITY_DISPATCHER


_ACTIVE_GUARD: Guard | None = None


def install_synthetic_guard(guard: Guard) -> None:
    if guard.epoch.get("kind") != SYNTHETIC_EPOCH_KIND:
        raise GuardFailure("INVALID_EPOCH")
    global _ACTIVE_GUARD
    _ACTIVE_GUARD = guard


def clear_installed_guard() -> None:
    global _ACTIVE_GUARD
    _ACTIVE_GUARD = None


def require_guard(guard: Guard | None = None) -> Guard:
    active = guard or _ACTIVE_GUARD
    if active is None:
        raise GuardFailure("OBSERVER_UNAVAILABLE")
    return active


def execute_governed(
    *,
    dimension: str,
    surface: str,
    operation: str,
    target: Callable[[], T],
    guard: Guard | None = None,
    policy_allows: bool = True,
    r17_input: Mapping[str, Any] | None = None,
) -> T:
    """The sole guard-to-target path used by installed primitives."""
    dispatcher = active_real_activity_dispatcher()
    if dispatcher is not None:
        return dispatcher.execute(
            dimension=dimension,
            surface=surface,
            operation=operation,
            target=target,
            policy_allows=policy_allows,
            r17_input=r17_input,
        )
    return require_guard(guard).execute(
        dimension, surface, operation, target, policy_allows=policy_allows
    )


async def execute_governed_async(
    *,
    dimension: str,
    surface: str,
    operation: str,
    target: Callable[[], Awaitable[T]],
    guard: Guard | None = None,
    policy_allows: bool = True,
    r17_input: Mapping[str, Any] | None = None,
) -> T:
    """Async equivalent used where the guarded target is the HTTP operation."""
    dispatcher = active_real_activity_dispatcher()
    if dispatcher is not None:
        return await dispatcher.execute_async(
            dimension=dimension,
            surface=surface,
            operation=operation,
            target=target,
            policy_allows=policy_allows,
            r17_input=r17_input,
        )
    active = require_guard(guard)
    ledger = active._validate(dimension, surface)
    if not policy_allows:
        active._append(ledger, dimension, surface, operation, phase="BLOCKED", allowed=False, blocked=True)
        raise GuardFailure("POLICY_PROHIBITS_OPERATION")
    active._append(ledger, dimension, surface, operation, phase="ATTEMPTED", allowed=True, blocked=False)
    active._append(ledger, dimension, surface, operation, phase="AUTHORIZED", allowed=True, blocked=False)
    active._append(ledger, dimension, surface, operation, phase="STARTED", allowed=True, blocked=False)
    try:
        result = await target()
    except Exception:
        active._append(ledger, dimension, surface, operation, phase="FAILED", allowed=True, blocked=False, failed=True)
        raise
    active._append(ledger, dimension, surface, operation, phase="COMPLETED", allowed=True, blocked=False, completed=True)
    return result


def derive_repository_activity_state(
    epoch: dict[str, Any], ledger: Ledger, covered: bool
) -> dict[str, Any]:
    integrity = ledger.verify()
    attempts = [item for item in ledger.entries if item.get("phase") in {"ATTEMPTED", "BLOCKED"}]
    terminals = [item for item in ledger.entries if item.get("phase") in {"COMPLETED", "FAILED"}]
    return {
        "coverage_complete": bool(covered),
        "observer_continuity_valid": integrity,
        "ledger_integrity_valid": integrity,
        "interval_valid": epoch.get("kind") == SYNTHETIC_EPOCH_KIND,
        "attempt_count": len(attempts),
        "blocked_attempt_count": sum(bool(item["blocked"]) for item in attempts),
        "allowed_count": sum(bool(item["allowed"]) for item in attempts),
        "completed_count": sum(bool(item["completed"]) for item in terminals),
        "failed_count": sum(bool(item["failed"]) for item in terminals),
        # R13 only establishes the repository-controlled synthetic subset.
        "ZERO_ACTIVITY_AUTHORITY_VALID": False,
    }
