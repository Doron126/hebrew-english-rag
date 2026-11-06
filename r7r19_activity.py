"""Staged, source-bound A→B→C campaign composition for C44E3R7R19.

R17 remains the authority for operation receipts, claims, ledgers, and the
legacy A→B protocol.  R19 adds a signed B/C discriminator and requires the
closed B result to be realized before the scorer-only C authority can attach.
Nothing in this module starts a process or invokes a governed target.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from r7r13_observer import (
    GuardFailure,
    ParsedRealAuthorization,
    R7R13_GUARD_REGISTRY,
    R19_CAMPAIGN_COMPOSITION_SCHEMA,
    R20_CAMPAIGN_COMPOSITION_SCHEMA,
    _canonical_record_payload,
    _utc_timestamp,
    _verify_signed_record,
    parse_real_epoch_authorization,
    validate_real_activity_receipts,
    validate_real_effect_attestations,
)
from r7r17_activity import (
    PreparedCampaign,
    R17CampaignRuntime,
    R17TrustConfiguration,
    RuntimeIdentityProvider,
    SQLiteActivityLedger,
    SQLiteAuthorizationUseStore,
    _R19_ACTIVATION_TOKEN,
    ConsumedRealAuthorization,
    _consumed_prepared_interval,
    canonical_json_bytes,
    canonical_sha256,
    validate_r17_dimension_activity_attestations,
)


R19_REALIZATION_SCHEMA = "r7r19-scorer-input-realization-1.0.0"
R19_B_TO_C_AUTHORITY_SCHEMA = "r7r19-b-to-c-composition-authority-1.0.0"
R19_REPOSITORY_REPORT_WRITER = "repository-report-writer"
_R19_PROTOCOL = R19_CAMPAIGN_COMPOSITION_SCHEMA
_ALL_PAIRS = {
    (surface, dimension)
    for surface, details in R7R13_GUARD_REGISTRY.items()
    for dimension in details["dimension_ids"]
}
_A_PAIRS = {("candidate-launch", "CANDIDATE_STARTED")}
_C_PAIRS = {("scorer", "SCORER_RERUN")}
_B_PAIRS = _ALL_PAIRS - _A_PAIRS - _C_PAIRS


def r19_realization_implementation_identity() -> dict[str, Any]:
    """Hash the exact source files that compute and dispatch the realization."""
    root = Path(__file__).resolve().parent
    paths = {
        "r7r19_activity.py": Path(__file__),
        "tools/score_real_corpus_results.py": root / "tools" / "score_real_corpus_results.py",
    }
    files: dict[str, str] = {}
    for relative, path in paths.items():
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or path.resolve(strict=True) != path:
                raise GuardFailure("R19_IMPLEMENTATION_SOURCE_NOT_CANONICAL_REGULAR_FILE")
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise GuardFailure("R19_IMPLEMENTATION_SOURCE_UNAVAILABLE") from exc
    return {"files": files, "sha256": canonical_sha256(files)}


def persist_r19_realization(record: Mapping[str, Any], path_value: str | Path) -> tuple[Path, str]:
    """Persist a canonical, exclusive, read-only realization artifact."""
    path = Path(path_value)
    if not path.is_absolute() or path.resolve(strict=False) != path or not path.parent.is_dir():
        raise GuardFailure("R19_REALIZATION_OUTPUT_PATH_NOT_CANONICAL")
    encoded = serialize_r19_realization(record)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o400)
    except FileExistsError as exc:
        raise GuardFailure("R19_REALIZATION_OUTPUT_ALREADY_EXISTS") from exc
    except OSError as exc:
        raise GuardFailure("R19_REALIZATION_OUTPUT_CREATE_FAILED") from exc
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o400)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        stored = path.read_bytes()
        if stored != encoded:
            raise GuardFailure("R19_REALIZATION_OUTPUT_READBACK_MISMATCH")
        _immutable_artifact(path)
    except Exception:
        # A partial file is evidence of a failed exclusive write; do not erase it.
        raise
    return path, hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class R19IntervalBCompositionContext:
    interval_a: PreparedCampaign
    candidate_activity_attestations: Sequence[Mapping[str, Any]]
    interval_a_ledger: SQLiteActivityLedger
    interval_a_effect_attestations: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class R19IntervalCCompositionContext:
    interval_a: PreparedCampaign
    candidate_activity_attestations: Sequence[Mapping[str, Any]]
    interval_a_ledger: SQLiteActivityLedger
    interval_a_effect_attestations: Sequence[Mapping[str, Any]]
    interval_b: PreparedCampaign
    interval_b_ledger: SQLiteActivityLedger
    interval_b_effect_attestations: Sequence[Mapping[str, Any]]
    interval_b_activity_attestations: Sequence[Mapping[str, Any]]
    scorer_input_realization: Mapping[str, Any]
    b_to_c_authority: Mapping[str, Any]
    expected_result_artifact_path: Path


def r19_scorer_input(
    report: Mapping[str, Any], *, report_json: Path | str,
    report_markdown: Path | str, writer: str = R19_REPOSITORY_REPORT_WRITER,
) -> dict[str, str]:
    """The single canonical input object used by realization and scorer dispatch."""
    if not isinstance(report, Mapping):
        raise GuardFailure("R19_REALIZED_REPORT_OBJECT_REQUIRED")
    if writer not in {R19_REPOSITORY_REPORT_WRITER, "provided"}:
        raise GuardFailure("R19_NONCANONICAL_REPORT_WRITER_REJECTED")
    return {
        "report_sha256": canonical_sha256(dict(report)),
        "report_json": str(report_json),
        "report_markdown": str(report_markdown),
        "writer": writer,
    }


def r19_scorer_input_sha256(
    report: Mapping[str, Any], *, report_json: Path | str,
    report_markdown: Path | str,
) -> str:
    return canonical_sha256(r19_scorer_input(
        report, report_json=report_json, report_markdown=report_markdown,
    ))


def _pairs(authorization: ParsedRealAuthorization) -> set[tuple[str, str]]:
    return {(row.surface_id, row.dimension_id) for row in authorization.operations}


def _require_operation_primitives(authorization: ParsedRealAuthorization) -> None:
    from r7r17_activity import R17_GOVERNED_CALL_TOPOLOGY

    for operation in authorization.operations:
        contract = R17_GOVERNED_CALL_TOPOLOGY["surfaces"].get(operation.surface_id)
        if (contract is None or operation.dimension_id != contract["dimension_id"]
                or operation.operation_identity != contract["primitive"]
                or (operation.surface_id, operation.dimension_id) not in _ALL_PAIRS):
            raise GuardFailure("R19_OPERATION_PRIMITIVE_OR_SURFACE_INVALID")


def _require_role(
    authorization: ParsedRealAuthorization, role: str,
    expected_pairs: set[tuple[str, str]], *, exact_cardinality: bool,
    protocol: str = _R19_PROTOCOL,
) -> None:
    if (authorization.campaign_protocol != protocol
            or authorization.interval_role != role or _pairs(authorization) != expected_pairs
            or (exact_cardinality and len(authorization.operations) != len(expected_pairs))):
        raise GuardFailure(f"R19_INTERVAL_{role}_AUTHORIZATION_SCOPE_INVALID")
    _require_operation_primitives(authorization)


def _closed_state(
    authorization: ParsedRealAuthorization, ledger: SQLiteActivityLedger, *, label: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(ledger, SQLiteActivityLedger) or ledger.authorization != authorization:
        raise GuardFailure(f"R19_{label}_LEDGER_AUTHORIZATION_MISMATCH")
    state = ledger.verify()
    closure = state.get("closure")
    if (state.get("integrity") != "PASS" or not state.get("closed")
            or not isinstance(closure, Mapping)
            or _utc_timestamp(closure.get("closed_at"), f"{label}.CLOSED_AT")
            < _utc_timestamp(authorization.B1, f"{label}.B1")
            or closure.get("final_sequence") != state.get("record_count")
            or closure.get("final_head_sha256") != state.get("head_sha256")):
        raise GuardFailure(f"R19_{label}_LEDGER_NOT_CLOSED_AFTER_B1")
    return state, closure


def _validated_interval_evidence(
    authorization: ParsedRealAuthorization,
    consumed: ConsumedRealAuthorization,
    ledger: SQLiteActivityLedger, effect_attestations: Sequence[Mapping[str, Any]],
    activity_attestations: Sequence[Mapping[str, Any]], trust: R17TrustConfiguration,
    *, label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _state, _closure = _closed_state(authorization, ledger, label=label)
    receipts = validate_real_activity_receipts(
        consumed, ledger.boundary_records(), ledger.receipt_records(), trust,
    )
    effects = validate_real_effect_attestations(
        consumed, receipts, [dict(row) for row in effect_attestations], trust,
    )
    activity = validate_r17_dimension_activity_attestations(
        authorization, receipts, effects,
        [dict(row) for row in activity_attestations], trust,
    )
    if not receipts.get("lifecycle_complete") or not effects.get("complete"):
        raise GuardFailure(f"R19_{label}_EVIDENCE_LIFECYCLE_INCOMPLETE")
    return receipts, effects, activity


def validate_r19_a_to_b(
    interval_a: PreparedCampaign, interval_b: PreparedCampaign, *,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: Sequence[Mapping[str, Any]],
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
) -> dict[str, Any]:
    """Bind launch-only R17 A to the exact scorer-excluding R19 B partition."""
    auth_a, consumed_a = _consumed_prepared_interval(interval_a, trust, reservation_store)
    auth_b, consumed_b = _consumed_prepared_interval(interval_b, trust, reservation_store)
    _require_role(auth_b, "B", _B_PAIRS, exact_cardinality=False)
    if (_pairs(auth_a) != _A_PAIRS or len(auth_a.operations) != 1
            or auth_a.campaign_protocol is not None
            or (auth_a.campaign_id, auth_a.source_commit, auth_a.source_tree)
            != (auth_b.campaign_id, auth_b.source_commit, auth_b.source_tree)
            or auth_a.interval_id == auth_b.interval_id
            or auth_a.decision_id == auth_b.decision_id
            or auth_a.nonce == auth_b.nonce
            or {op.execution_id for op in auth_a.operations}
            & {op.execution_id for op in auth_b.operations}
            or _utc_timestamp(auth_a.B1, "A.B1") > _utc_timestamp(auth_b.B0, "B.B0")):
        raise GuardFailure("R19_A_TO_B_INTERVAL_COMPOSITION_INVALID")
    _require_operation_primitives(auth_a)
    _receipts, effects, activity = _validated_interval_evidence(
        auth_a, consumed_a, interval_a_ledger,
        interval_a_effect_attestations, candidate_activity_attestations,
        trust, label="A",
    )
    launch = auth_a.operations[0]
    binding = activity["positive_by_execution"].get(launch.execution_id)
    if (binding is None or binding.get("event_kind") != "CANDIDATE_PROCESS_STARTED"
            or not binding.get("candidate_process_identity")):
        raise GuardFailure("R19_CANDIDATE_IDENTITY_BINDING_MISSING")
    candidate_identity = binding["candidate_process_identity"]
    activity_row = next(
        (row for row in candidate_activity_attestations
         if row.get("execution_id") == launch.execution_id), None,
    )
    if not isinstance(activity_row, Mapping):
        raise GuardFailure("R19_CANDIDATE_IDENTITY_ATTESTATION_MISSING")
    launch_effect = effects["effects_by_execution"].get(launch.execution_id)
    if not isinstance(launch_effect, Mapping):
        raise GuardFailure("R19_CANDIDATE_LAUNCH_EFFECT_BINDING_MISSING")
    a_state, a_closure = _closed_state(auth_a, interval_a_ledger, label="A")
    a_closed_at = _utc_timestamp(
        interval_a_ledger.verify()["closure"]["closed_at"], "A.CLOSED_AT"
    )
    binding_at = _utc_timestamp(activity_row.get("timestamp"), "A.CANDIDATE_BINDING_AT")
    b_claimed_at = _utc_timestamp(interval_b.claim.get("claimed_at"), "B.CLAIMED_AT")
    b_issued_at = _utc_timestamp(interval_b.authorization_document.get("issued_at"), "B.ISSUED_AT")
    b_b0 = _utc_timestamp(auth_b.B0, "B.B0")
    if (b_claimed_at <= a_closed_at or b_issued_at <= a_closed_at
            or b_claimed_at <= binding_at or b_issued_at <= binding_at
            or b_claimed_at >= b_b0 or b_issued_at >= b_b0):
        raise GuardFailure("R19_B_AUTHORITY_NOT_AFTER_A_CLOSURE_AND_CANDIDATE_BINDING")
    chat = [op for op in auth_b.operations if op.surface_id == "chat-api-request"]
    if not chat or any(op.process_identity != candidate_identity for op in chat):
        raise GuardFailure("R19_B_CHAT_PROCESS_NOT_BOUND_TO_CANDIDATE")
    return {
        "schema_version": R19_CAMPAIGN_COMPOSITION_SCHEMA,
        "status": "READY_FOR_INTERVAL_B",
        "campaign_id": auth_a.campaign_id,
        "source_commit": auth_a.source_commit,
        "source_tree": auth_a.source_tree,
        "interval_a_decision_id": auth_a.decision_id,
        "interval_a_interval_id": auth_a.interval_id,
        "interval_a_authorization_sha256": auth_a.authorization_sha256,
        "interval_a_claim_id": consumed_a.claim_id,
        "interval_a_reservation_id": consumed_a.reservation_id,
        "interval_a_store_identity": consumed_a.store_identity,
        "interval_a_claim_sha256": hashlib.sha256(
            _canonical_record_payload(interval_a.claim)
        ).hexdigest(),
        "interval_a_final_sequence": a_state["record_count"],
        "interval_a_ledger_head_sha256": a_state["head_sha256"],
        "interval_a_closed_at": a_closure["closed_at"],
        "interval_a_launch_execution_id": launch.execution_id,
        "candidate_process_identity": candidate_identity,
        "candidate_binding_activity_id": activity_row["activity_id"],
        "candidate_binding_activity_sha256": binding["activity_sha256"],
        "candidate_binding_effect_attestation_id": launch_effect["attestation_id"],
        "candidate_binding_effect_attestation_sha256": launch_effect["attestation_sha256"],
        "interval_b_decision_id": auth_b.decision_id,
        "interval_b_interval_id": auth_b.interval_id,
        "interval_b_authorization_sha256": auth_b.authorization_sha256,
        "interval_b_claim_id": consumed_b.claim_id,
        "interval_b_reservation_id": consumed_b.reservation_id,
        "interval_b_store_identity": consumed_b.store_identity,
        "interval_b_claim_sha256": hashlib.sha256(
            _canonical_record_payload(interval_b.claim)
        ).hexdigest(),
        "interval_b_execution_ids": sorted(op.execution_id for op in auth_b.operations),
        "required_b_surface_pairs": sorted([list(pair) for pair in _B_PAIRS]),
        "scorer_excluded_from_b": True,
        "live_activity_performed_by_composition_check": False,
    }


def _immutable_artifact(path_value: str | Path) -> tuple[Path, str]:
    path = Path(path_value)
    try:
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise GuardFailure("R19_RESULT_ARTIFACT_PATH_NOT_CANONICAL")
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) & 0o222):
            raise GuardFailure("R19_RESULT_ARTIFACT_NOT_IMMUTABLE_REGULAR_FILE")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GuardFailure("R19_RESULT_ARTIFACT_UNAVAILABLE") from exc
    return path, digest


def build_r19_scorer_input_realization(
    *, interval_a: PreparedCampaign,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b: PreparedCampaign, interval_b_ledger: SQLiteActivityLedger,
    reservation_store: SQLiteAuthorizationUseStore,
    interval_b_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_activity_attestations: Sequence[Mapping[str, Any]],
    result_artifact_path: Path, benchmark_execution_ids: Sequence[str],
    report: Mapping[str, Any], report_json: Path, report_markdown: Path,
    trust: R17TrustConfiguration, realized_at: str,
) -> dict[str, Any]:
    """Sign, but do not emit, the exact report object and its future C input."""
    auth_b, consumed_b = _consumed_prepared_interval(
        interval_b, trust, reservation_store,
    )
    _require_role(auth_b, "B", _B_PAIRS, exact_cardinality=False)
    a_to_b_composition = validate_r19_a_to_b(
        interval_a, interval_b,
        candidate_activity_attestations=candidate_activity_attestations,
        interval_a_ledger=interval_a_ledger,
        interval_a_effect_attestations=interval_a_effect_attestations,
        trust=trust, reservation_store=reservation_store,
    )
    state, closure = _closed_state(
        auth_b, interval_b_ledger, label="B",
    )
    if _utc_timestamp(realized_at, "REALIZATION_TIMESTAMP") <= _utc_timestamp(
        closure["closed_at"], "B.CLOSED_AT"
    ):
        raise GuardFailure("R19_REALIZATION_PRECEDES_B_CLOSURE")
    benchmark_ids = sorted(
        op.execution_id for op in auth_b.operations
        if op.surface_id == "benchmark-runner"
    )
    if (not benchmark_ids or sorted(set(benchmark_execution_ids)) != benchmark_ids
            or len(benchmark_execution_ids) != len(benchmark_ids)):
        raise GuardFailure("R19_BENCHMARK_EXECUTION_SET_MISMATCH")
    output_paths = (Path(report_json), Path(report_markdown))
    if any(not path.is_absolute() or path.resolve(strict=False) != path for path in output_paths):
        raise GuardFailure("R19_SCORE_REPORT_OUTPUT_PATHS_MUST_BE_ABSOLUTE")
    _path, result_digest = _immutable_artifact(result_artifact_path)
    receipts, _effects, activity = _validated_interval_evidence(
        auth_b, consumed_b, interval_b_ledger,
        interval_b_effect_attestations, interval_b_activity_attestations,
        trust, label="B",
    )
    if any(ident not in activity["positive_by_execution"] for ident in benchmark_ids):
        raise GuardFailure("R19_BENCHMARK_ACTIVITY_NOT_OBSERVED")
    phases: dict[str, list[str]] = defaultdict(list)
    for receipt in receipts["receipt_records"]:
        phases[str(receipt.get("execution_id"))].append(str(receipt.get("phase")))
    required_benchmark_lifecycle = [
        "ATTEMPTED", "AUTHORIZED", "STARTED", "COMPLETED",
        "CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLOSED",
    ]
    if any(phases.get(ident) != required_benchmark_lifecycle for ident in benchmark_ids):
        raise GuardFailure("R19_BENCHMARK_LIFECYCLE_NOT_COMPLETED")
    try:
        result_rows = [
            json.loads(line) for line in _path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardFailure("R19_RESULT_ARTIFACT_JSONL_INVALID") from exc
    result_case_ids = [row.get("case_id") for row in result_rows if isinstance(row, Mapping)]
    if (len(result_rows) != len(benchmark_ids)
            or len(result_case_ids) != len(result_rows)
            or any(not isinstance(case_id, str) or not case_id for case_id in result_case_ids)
            or len(set(result_case_ids)) != len(result_case_ids)):
        raise GuardFailure("R19_RESULT_ROWS_NOT_ONE_TO_ONE_WITH_BENCHMARK_EXECUTIONS")
    if (not isinstance(report, Mapping)
            or not isinstance(report.get("run"), Mapping)
            or report["run"].get("result_sha256") != result_digest
            or report["run"].get("result_count") != len(result_rows)
            or report["run"].get("result_count") != len(benchmark_ids)):
        raise GuardFailure("R19_REPORT_NOT_BOUND_TO_CLOSED_B_RESULT_ARTIFACT")
    input_row = r19_scorer_input(
        report, report_json=report_json, report_markdown=report_markdown,
    )
    record: dict[str, Any] = {
        "schema_version": R19_REALIZATION_SCHEMA,
        "campaign_protocol": _R19_PROTOCOL,
        "campaign_id": auth_b.campaign_id,
        "source_commit": auth_b.source_commit,
        "source_tree": auth_b.source_tree,
        "a_to_b_composition_sha256": canonical_sha256(a_to_b_composition),
        "interval_b_id": auth_b.interval_id,
        "interval_b_decision_id": auth_b.decision_id,
        "interval_b_authorization_sha256": auth_b.authorization_sha256,
        "interval_b_claim_id": consumed_b.claim_id,
        "interval_b_reservation_id": consumed_b.reservation_id,
        "interval_b_store_identity": consumed_b.store_identity,
        "interval_b_claim_sha256": hashlib.sha256(
            _canonical_record_payload(interval_b.claim)
        ).hexdigest(),
        "interval_b_final_sequence": state["record_count"],
        "interval_b_final_head_sha256": state["head_sha256"],
        "interval_b_closed_at": closure["closed_at"],
        "benchmark_execution_ids": benchmark_ids,
        "benchmark_process_identities": sorted({
            op.process_identity for op in auth_b.operations
            if op.execution_id in benchmark_ids
        }),
        "result_artifact_path": str(_path),
        "result_artifact_sha256": result_digest,
        "report_object": dict(report),
        "report_sha256": input_row["report_sha256"],
        "report_json": input_row["report_json"],
        "report_markdown": input_row["report_markdown"],
        "writer": input_row["writer"],
        "scorer_input_sha256": canonical_sha256(input_row),
        "realization_implementation": r19_realization_implementation_identity(),
        "realized_at": realized_at,
    }
    signer = trust.signer_for("STATE_TRANSITION_OBSERVER_AUTHORITY")
    record["signer_id"] = signer.signer_id
    record["signature"] = signer.sign(_canonical_record_payload(record))
    return record


def serialize_r19_realization(record: Mapping[str, Any]) -> bytes:
    if not isinstance(record, Mapping) or record.get("schema_version") != R19_REALIZATION_SCHEMA:
        raise GuardFailure("R19_REALIZATION_SCHEMA_INVALID")
    return canonical_json_bytes(dict(record)) + b"\n"


def deserialize_r19_realization(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardFailure("R19_REALIZATION_SERIALIZATION_INVALID") from exc
    if not isinstance(value, dict) or value.get("schema_version") != R19_REALIZATION_SCHEMA:
        raise GuardFailure("R19_REALIZATION_SCHEMA_INVALID")
    if serialize_r19_realization(value) != data:
        raise GuardFailure("R19_REALIZATION_SERIALIZATION_NOT_CANONICAL")
    return value


def _validate_realization(
    record: Mapping[str, Any], *,
    auth_b: ParsedRealAuthorization, interval_b: PreparedCampaign,
    consumed_b: ConsumedRealAuthorization, interval_b_ledger: SQLiteActivityLedger,
    a_to_b_composition_sha256: str,
    benchmark_execution_ids: Sequence[str], expected_result_artifact_path: Path,
    trust: R17TrustConfiguration, interval_c: PreparedCampaign,
    authorized_c_input_sha256: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version", "campaign_protocol", "campaign_id", "source_commit", "source_tree",
        "a_to_b_composition_sha256", "interval_b_id", "interval_b_decision_id",
        "interval_b_authorization_sha256", "interval_b_claim_id",
        "interval_b_reservation_id", "interval_b_store_identity",
        "interval_b_claim_sha256", "interval_b_final_sequence", "interval_b_final_head_sha256",
        "interval_b_closed_at", "benchmark_execution_ids", "benchmark_process_identities",
        "result_artifact_path", "result_artifact_sha256", "report_object", "report_sha256",
        "report_json", "report_markdown", "writer", "scorer_input_sha256",
        "realization_implementation", "realized_at",
        "signer_id", "signature",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise GuardFailure("R19_REALIZATION_SCHEMA_INVALID")
    if record.get("schema_version") != R19_REALIZATION_SCHEMA or record.get("campaign_protocol") != _R19_PROTOCOL:
        raise GuardFailure("R19_REALIZATION_PROTOCOL_MISMATCH")
    if record.get("writer") != R19_REPOSITORY_REPORT_WRITER:
        raise GuardFailure("R19_REALIZATION_NONCANONICAL_REPORT_WRITER")
    _verify_signed_record(
        record, trust,
        permitted_authority_classes=frozenset({"STATE_TRANSITION_OBSERVER_AUTHORITY"}),
    )
    state, closure = _closed_state(auth_b, interval_b_ledger, label="B")
    expected_ids = sorted(
        op.execution_id for op in auth_b.operations
        if op.surface_id == "benchmark-runner"
    )
    expected_processes = sorted({
        op.process_identity for op in auth_b.operations
        if op.surface_id == "benchmark-runner"
    })
    expected_path, expected_file_hash = _immutable_artifact(expected_result_artifact_path)
    bindings = {
        "campaign_id": auth_b.campaign_id,
        "source_commit": auth_b.source_commit,
        "source_tree": auth_b.source_tree,
        "a_to_b_composition_sha256": a_to_b_composition_sha256,
        "interval_b_id": auth_b.interval_id,
        "interval_b_decision_id": auth_b.decision_id,
        "interval_b_authorization_sha256": auth_b.authorization_sha256,
        "interval_b_claim_id": consumed_b.claim_id,
        "interval_b_reservation_id": consumed_b.reservation_id,
        "interval_b_store_identity": consumed_b.store_identity,
        "interval_b_claim_sha256": hashlib.sha256(
            _canonical_record_payload(interval_b.claim)
        ).hexdigest(),
        "interval_b_final_sequence": state["record_count"],
        "interval_b_final_head_sha256": state["head_sha256"],
        "interval_b_closed_at": closure["closed_at"],
        "benchmark_execution_ids": expected_ids,
        "benchmark_process_identities": expected_processes,
        "result_artifact_path": str(expected_path),
        "result_artifact_sha256": expected_file_hash,
    }
    if any(record.get(key) != value for key, value in bindings.items()):
        raise GuardFailure("R19_REALIZATION_B_OR_RESULT_BINDING_MISMATCH")
    expected_implementation = r19_realization_implementation_identity()
    if record.get("realization_implementation") != expected_implementation:
        raise GuardFailure("R19_REALIZATION_IMPLEMENTATION_IDENTITY_MISMATCH")
    if sorted(set(benchmark_execution_ids)) != expected_ids:
        raise GuardFailure("R19_BENCHMARK_EXECUTION_SET_MISMATCH")
    realized_at = _utc_timestamp(record.get("realized_at"), "REALIZATION_TIMESTAMP")
    if (realized_at <= _utc_timestamp(closure["closed_at"], "B.CLOSED_AT")
            or realized_at >= _utc_timestamp(interval_c.authorization_document.get("issued_at"), "C.ISSUED_AT")
            or realized_at >= _utc_timestamp(interval_c.claim.get("claimed_at"), "C.CLAIMED_AT")):
        raise GuardFailure("R19_REALIZATION_NOT_BETWEEN_B_CLOSURE_AND_C_AUTHORITY")
    report = record.get("report_object")
    if not isinstance(report, Mapping):
        raise GuardFailure("R19_REALIZED_REPORT_OBJECT_INVALID")
    report_run = report.get("run")
    if not isinstance(report_run, Mapping):
        raise GuardFailure("R19_REALIZED_REPORT_RUN_INVALID")
    output_paths = (
        Path(str(record.get("report_json", ""))),
        Path(str(record.get("report_markdown", ""))),
    )
    if any(not path.is_absolute() or path.resolve(strict=False) != path for path in output_paths):
        raise GuardFailure("R19_SCORE_REPORT_OUTPUT_PATHS_MUST_BE_ABSOLUTE")
    expected_input = r19_scorer_input(
        report, report_json=record.get("report_json"),
        report_markdown=record.get("report_markdown"), writer=record.get("writer"),
    )
    if (record.get("report_sha256") != expected_input["report_sha256"]
            or record.get("scorer_input_sha256") != canonical_sha256(expected_input)
            or report_run.get("result_sha256") != expected_file_hash
            or report_run.get("result_count") != len(expected_ids)):
        raise GuardFailure("R19_REALIZATION_REPORT_OR_SCORER_INPUT_MISMATCH")
    c_claimed_at = _utc_timestamp(interval_c.claim.get("claimed_at"), "C.CLAIMED_AT")
    c_issued_at = _utc_timestamp(interval_c.authorization_document.get("issued_at"), "C.ISSUED_AT")
    b_closed_at = _utc_timestamp(closure["closed_at"], "B.CLOSED_AT")
    if c_claimed_at <= max(b_closed_at, realized_at) or c_issued_at <= max(b_closed_at, realized_at):
        raise GuardFailure("R19_C_AUTHORITY_PRECEDES_B_REALIZATION")
    c_b0 = _utc_timestamp(interval_c.expected_scope.B0, "C.B0")
    if c_claimed_at >= c_b0 or c_issued_at >= c_b0:
        raise GuardFailure("R19_C_AUTHORITY_NOT_PRE_B0")
    c_operation = interval_c.expected_scope.operations[0]
    expected_c_input_sha256 = authorized_c_input_sha256 or record["scorer_input_sha256"]
    if c_operation.frozen_input_sha256 != expected_c_input_sha256:
        raise GuardFailure("R19_C_AUTHORIZATION_NOT_BOUND_TO_REALIZED_SCORER_INPUT")
    return dict(report)


def _derive_r19_b_to_c_composition(
    interval_a: PreparedCampaign,
    interval_b: PreparedCampaign, interval_c: PreparedCampaign, *,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_ledger: SQLiteActivityLedger,
    interval_b_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_activity_attestations: Sequence[Mapping[str, Any]],
    scorer_input_realization: Mapping[str, Any],
    expected_result_artifact_path: Path,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
    c_protocol: str = _R19_PROTOCOL,
    authorized_c_input_sha256: str | None = None,
) -> dict[str, Any]:
    """Require closed B receipts and a signed, exact result-derived C input."""
    if c_protocol not in {_R19_PROTOCOL, R20_CAMPAIGN_COMPOSITION_SCHEMA}:
        raise GuardFailure("R19_B_TO_C_PROTOCOL_UNSUPPORTED")
    auth_b, consumed_b = _consumed_prepared_interval(interval_b, trust, reservation_store)
    auth_c, consumed_c = _consumed_prepared_interval(interval_c, trust, reservation_store)
    a_to_b_composition = validate_r19_a_to_b(
        interval_a, interval_b,
        candidate_activity_attestations=candidate_activity_attestations,
        interval_a_ledger=interval_a_ledger,
        interval_a_effect_attestations=interval_a_effect_attestations,
        trust=trust, reservation_store=reservation_store,
    )
    _require_role(auth_b, "B", _B_PAIRS, exact_cardinality=False)
    _require_role(
        auth_c, "C", _C_PAIRS, exact_cardinality=True, protocol=c_protocol,
    )
    if ((auth_b.campaign_id, auth_b.source_commit, auth_b.source_tree)
            != (auth_c.campaign_id, auth_c.source_commit, auth_c.source_tree)
            or auth_b.interval_id == auth_c.interval_id
            or auth_b.decision_id == auth_c.decision_id or auth_b.nonce == auth_c.nonce
            or {op.execution_id for op in auth_b.operations}
            & {op.execution_id for op in auth_c.operations}
            or _utc_timestamp(auth_b.B1, "B.B1") > _utc_timestamp(auth_c.B0, "C.B0")):
        raise GuardFailure("R19_B_TO_C_INTERVAL_COMPOSITION_INVALID")
    receipts, _effects, activity = _validated_interval_evidence(
        auth_b, consumed_b, interval_b_ledger,
        interval_b_effect_attestations, interval_b_activity_attestations,
        trust, label="B",
    )
    benchmark_ops = [op for op in auth_b.operations if op.surface_id == "benchmark-runner"]
    if not benchmark_ops or any(
        op.execution_id not in activity["positive_by_execution"]
        for op in benchmark_ops
    ):
        raise GuardFailure("R19_BENCHMARK_ACTIVITY_NOT_OBSERVED")
    expected_ids = sorted(op.execution_id for op in benchmark_ops)
    phases: dict[str, list[str]] = defaultdict(list)
    for row in receipts["receipt_records"]:
        phases[str(row.get("execution_id"))].append(str(row.get("phase")))
    if any(
        phases.get(ident) != [
            "ATTEMPTED", "AUTHORIZED", "STARTED", "COMPLETED",
            "CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLOSED",
        ]
        for ident in expected_ids
    ):
        raise GuardFailure("R19_BENCHMARK_LIFECYCLE_NOT_COMPLETED")
    report = _validate_realization(
        scorer_input_realization, auth_b=auth_b, interval_b=interval_b,
        consumed_b=consumed_b,
        interval_b_ledger=interval_b_ledger, benchmark_execution_ids=expected_ids,
        a_to_b_composition_sha256=canonical_sha256(a_to_b_composition),
        expected_result_artifact_path=expected_result_artifact_path,
        trust=trust, interval_c=interval_c,
        authorized_c_input_sha256=authorized_c_input_sha256,
    )
    scorer_op = auth_c.operations[0]
    expected_c_input_sha256 = (
        authorized_c_input_sha256 or scorer_input_realization.get("scorer_input_sha256")
    )
    if scorer_op.frozen_input_sha256 != expected_c_input_sha256:
        raise GuardFailure("R19_C_SCORER_INPUT_HASH_MISMATCH")
    result = {
        "schema_version": R19_CAMPAIGN_COMPOSITION_SCHEMA,
        "status": "READY_FOR_INTERVAL_C",
        "campaign_id": auth_b.campaign_id,
        "source_commit": auth_b.source_commit,
        "source_tree": auth_b.source_tree,
        "a_to_b_composition_sha256": canonical_sha256(a_to_b_composition),
        "interval_a_decision_id": interval_a.authorization_document["decision_id"],
        "interval_a_interval_id": interval_a.expected_scope.interval_id,
        "interval_a_authorization_sha256": a_to_b_composition["interval_a_authorization_sha256"],
        "interval_a_claim_id": a_to_b_composition["interval_a_claim_id"],
        "interval_a_reservation_id": a_to_b_composition["interval_a_reservation_id"],
        "interval_a_store_identity": a_to_b_composition["interval_a_store_identity"],
        "interval_b_decision_id": auth_b.decision_id,
        "interval_b_interval_id": auth_b.interval_id,
        "interval_b_authorization_sha256": auth_b.authorization_sha256,
        "interval_b_claim_id": consumed_b.claim_id,
        "interval_b_reservation_id": consumed_b.reservation_id,
        "interval_b_store_identity": consumed_b.store_identity,
        "interval_b_claim_sha256": hashlib.sha256(
            _canonical_record_payload(interval_b.claim)
        ).hexdigest(),
        "interval_b_final_sequence": interval_b_ledger.verify()["record_count"],
        "interval_b_final_head_sha256": interval_b_ledger.verify()["head_sha256"],
        "interval_b_closed_at": interval_b_ledger.verify()["closure"]["closed_at"],
        "benchmark_execution_ids": expected_ids,
        "benchmark_process_identities": sorted({op.process_identity for op in benchmark_ops}),
        "result_artifact_sha256": scorer_input_realization["result_artifact_sha256"],
        "realization_sha256": canonical_sha256(dict(scorer_input_realization)),
        "report_sha256": scorer_input_realization["report_sha256"],
        "report_json": scorer_input_realization["report_json"],
        "report_markdown": scorer_input_realization["report_markdown"],
        "writer": scorer_input_realization["writer"],
        "scorer_input_sha256": scorer_input_realization["scorer_input_sha256"],
        "interval_c_decision_id": auth_c.decision_id,
        "interval_c_interval_id": auth_c.interval_id,
        "interval_c_authorization_sha256": auth_c.authorization_sha256,
        "interval_c_claim_id": consumed_c.claim_id,
        "interval_c_reservation_id": consumed_c.reservation_id,
        "interval_c_store_identity": consumed_c.store_identity,
        "interval_c_claim_sha256": hashlib.sha256(
            _canonical_record_payload(interval_c.claim)
        ).hexdigest(),
        "scorer_operation_execution_id": scorer_op.execution_id,
        "scorer_process_identity": scorer_op.process_identity,
        "scorer_surface_id": scorer_op.surface_id,
        "scorer_dimension_id": scorer_op.dimension_id,
        "scorer_primitive": scorer_op.operation_identity,
        "report_object": report,
        "live_activity_performed_by_composition_check": False,
    }
    result["composition_sha256"] = canonical_sha256(result)
    return result


def seal_r19_b_to_c_authority(
    interval_a: PreparedCampaign,
    interval_b: PreparedCampaign, interval_c: PreparedCampaign, *,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_ledger: SQLiteActivityLedger,
    interval_b_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_activity_attestations: Sequence[Mapping[str, Any]],
    scorer_input_realization: Mapping[str, Any],
    expected_result_artifact_path: Path,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
    issued_at: str,
) -> dict[str, Any]:
    """Seal the post-C-authorization binding to one exact B→C composition."""
    composition = _derive_r19_b_to_c_composition(
        interval_a, interval_b, interval_c,
        candidate_activity_attestations=candidate_activity_attestations,
        interval_a_ledger=interval_a_ledger,
        interval_a_effect_attestations=interval_a_effect_attestations,
        interval_b_ledger=interval_b_ledger,
        interval_b_effect_attestations=interval_b_effect_attestations,
        interval_b_activity_attestations=interval_b_activity_attestations,
        scorer_input_realization=scorer_input_realization,
        expected_result_artifact_path=expected_result_artifact_path,
        trust=trust, reservation_store=reservation_store,
    )
    auth_c, _consumed_c = _consumed_prepared_interval(interval_c, trust, reservation_store)
    issued = _utc_timestamp(issued_at, "B_TO_C_AUTHORITY_ISSUED_AT")
    floor = max(
        _utc_timestamp(composition["interval_b_closed_at"], "B.CLOSED_AT"),
        _utc_timestamp(scorer_input_realization["realized_at"], "REALIZATION_TIMESTAMP"),
        _utc_timestamp(interval_c.authorization_document.get("issued_at"), "C.ISSUED_AT"),
        _utc_timestamp(interval_c.claim.get("claimed_at"), "C.CLAIMED_AT"),
    )
    if issued <= floor or issued >= _utc_timestamp(auth_c.B0, "C.B0"):
        raise GuardFailure("R19_B_TO_C_AUTHORITY_NOT_AFTER_C_AUTHORITY_AND_BEFORE_B0")
    record: dict[str, Any] = {
        "schema_version": R19_B_TO_C_AUTHORITY_SCHEMA,
        "campaign_protocol": _R19_PROTOCOL,
        "composition": composition,
        "issued_at": issued_at,
    }
    signer = trust.signer_for("STATE_TRANSITION_OBSERVER_AUTHORITY")
    record["signer_id"] = signer.signer_id
    record["signature"] = signer.sign(_canonical_record_payload(record))
    return record


def serialize_r19_b_to_c_authority(record: Mapping[str, Any]) -> bytes:
    if (not isinstance(record, Mapping)
            or record.get("schema_version") != R19_B_TO_C_AUTHORITY_SCHEMA):
        raise GuardFailure("R19_B_TO_C_AUTHORITY_SCHEMA_INVALID")
    return canonical_json_bytes(dict(record)) + b"\n"


def validate_r19_b_to_c(
    interval_a: PreparedCampaign,
    interval_b: PreparedCampaign, interval_c: PreparedCampaign, *,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_ledger: SQLiteActivityLedger,
    interval_b_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_activity_attestations: Sequence[Mapping[str, Any]],
    scorer_input_realization: Mapping[str, Any],
    b_to_c_authority: Mapping[str, Any],
    expected_result_artifact_path: Path,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
) -> dict[str, Any]:
    """Re-derive and verify a sealed B→C authority for the exact C interval."""
    composition = _derive_r19_b_to_c_composition(
        interval_a, interval_b, interval_c,
        candidate_activity_attestations=candidate_activity_attestations,
        interval_a_ledger=interval_a_ledger,
        interval_a_effect_attestations=interval_a_effect_attestations,
        interval_b_ledger=interval_b_ledger,
        interval_b_effect_attestations=interval_b_effect_attestations,
        interval_b_activity_attestations=interval_b_activity_attestations,
        scorer_input_realization=scorer_input_realization,
        expected_result_artifact_path=expected_result_artifact_path,
        trust=trust, reservation_store=reservation_store,
    )
    required = {"schema_version", "campaign_protocol", "composition", "issued_at", "signer_id", "signature"}
    if not isinstance(b_to_c_authority, Mapping) or set(b_to_c_authority) != required:
        raise GuardFailure("R19_B_TO_C_AUTHORITY_MISSING_OR_INVALID")
    if (b_to_c_authority.get("schema_version") != R19_B_TO_C_AUTHORITY_SCHEMA
            or b_to_c_authority.get("campaign_protocol") != _R19_PROTOCOL
            or b_to_c_authority.get("composition") != composition):
        raise GuardFailure("R19_B_TO_C_AUTHORITY_COMPOSITION_MISMATCH")
    _verify_signed_record(
        b_to_c_authority, trust,
        permitted_authority_classes=frozenset({"STATE_TRANSITION_OBSERVER_AUTHORITY"}),
    )
    auth_c, _consumed_c = _consumed_prepared_interval(interval_c, trust, reservation_store)
    issued = _utc_timestamp(b_to_c_authority.get("issued_at"), "B_TO_C_AUTHORITY_ISSUED_AT")
    floor = max(
        _utc_timestamp(composition["interval_b_closed_at"], "B.CLOSED_AT"),
        _utc_timestamp(scorer_input_realization["realized_at"], "REALIZATION_TIMESTAMP"),
        _utc_timestamp(interval_c.authorization_document.get("issued_at"), "C.ISSUED_AT"),
        _utc_timestamp(interval_c.claim.get("claimed_at"), "C.CLAIMED_AT"),
    )
    if issued <= floor or issued >= _utc_timestamp(auth_c.B0, "C.B0"):
        raise GuardFailure("R19_B_TO_C_AUTHORITY_TIME_INVALID")
    result = dict(composition)
    result["b_to_c_authority_sha256"] = hashlib.sha256(
        _canonical_record_payload(b_to_c_authority)
    ).hexdigest()
    return result


def attach_r19_interval_after_b0(
    prepared: PreparedCampaign, *,
    composition_context: R19IntervalBCompositionContext | R19IntervalCCompositionContext,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
    ledger: SQLiteActivityLedger,
    identity_provider: RuntimeIdentityProvider,
    b0_record: Mapping[str, Any],
    now: datetime,
) -> R17CampaignRuntime:
    authorization = parse_real_epoch_authorization(
        dict(prepared.authorization_document), prepared.expected_scope, trust, now=now,
    )
    if authorization.campaign_protocol != _R19_PROTOCOL:
        raise GuardFailure("R19_PROTOCOL_AUTHORIZATION_REQUIRED")
    return R17CampaignRuntime._activate_r19_after_protocol_validation(
        prepared, authorization, protocol_validation_token=_R19_ACTIVATION_TOKEN,
        composition_context=composition_context,
        trust=trust, reservation_store=reservation_store, ledger=ledger,
        identity_provider=identity_provider, b0_record=b0_record, now=now,
    )


def realized_report_for_active_c(
    realization: Mapping[str, Any], *, report_json: Path | str,
    report_markdown: Path | str,
) -> dict[str, Any]:
    """Return the already-realized object only under its matching active C."""
    from r7r17_activity import current_campaign_runtime

    runtime = current_campaign_runtime()
    if (runtime is None or runtime.authorization.campaign_protocol != _R19_PROTOCOL
            or runtime.authorization.interval_role != "C"):
        raise GuardFailure("R19_SCORER_DISPATCH_REQUIRES_ACTIVE_INTERVAL_C")
    if canonical_sha256(dict(realization)) != getattr(runtime, "_r19_realization_sha256", None):
        raise GuardFailure("R19_SCORER_REALIZATION_DIFFERS_FROM_CLOSURE")
    report = realization.get("report_object")
    if not isinstance(report, Mapping):
        raise GuardFailure("R19_REALIZED_REPORT_OBJECT_INVALID")
    expected_input = r19_scorer_input(
        report, report_json=report_json, report_markdown=report_markdown,
    )
    scorer_ops = [op for op in runtime.authorization.operations if op.surface_id == "scorer"]
    if (len(scorer_ops) != 1
            or canonical_sha256(expected_input) != scorer_ops[0].frozen_input_sha256
            or expected_input["report_sha256"] != realization.get("report_sha256")):
        raise GuardFailure("R19_SCORER_DISPATCH_INPUT_NOT_EXACTLY_AUTHORIZED")
    return dict(report)
