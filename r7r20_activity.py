"""R20 exact-byte scorer snapshots and mandatory R20 C-boundary enforcement."""
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
    R19_CAMPAIGN_COMPOSITION_SCHEMA,
    R20_CAMPAIGN_COMPOSITION_SCHEMA,
    _canonical_record_payload,
    _utc_timestamp,
    _verify_signed_record,
    parse_real_epoch_authorization,
)
from r7r17_activity import (
    PreparedCampaign,
    R17CampaignRuntime,
    R17TrustConfiguration,
    RuntimeIdentityProvider,
    SQLiteActivityLedger,
    SQLiteAuthorizationUseStore,
    _consumed_prepared_interval,
    canonical_json_bytes,
    canonical_sha256,
    current_campaign_runtime,
)
from r7r19_activity import (
    R19_REALIZATION_SCHEMA,
    R19_REPOSITORY_REPORT_WRITER,
    _B_PAIRS,
    _closed_state,
    _derive_r19_b_to_c_composition,
    _require_role,
    _validated_interval_evidence,
    deserialize_r19_realization,
    r19_realization_implementation_identity,
    r19_scorer_input,
    validate_r19_a_to_b,
)


R20_SNAPSHOT_MANIFEST_SCHEMA = "r7r20-scorer-input-snapshot-manifest-1.0.0"
R20_REALIZATION_SCHEMA = "r7r20-exact-scorer-input-realization-1.0.0"
R20_B_TO_C_AUTHORITY_SCHEMA = "r7r20-b-to-c-snapshot-composition-authority-1.0.0"
R20_SCORER_IDENTITY_SCHEMA = "r7r20-scorer-identity-1.0.0"
R20_SCORER_INPUT_SCHEMA = "r7r20-scorer-input-1.0.0"
_REPORT_SNAPSHOT_KEY = "r20_input_snapshot"
_REPORT_SCORER_IDENTITY_KEY = "r20_scorer_identity"
_REPORT_INPUT_BINDING_KEY = "r20_scorer_input_binding"
_ARTIFACT_NAMES = ("dataset", "schema", "results")
_OPTIONAL_ARTIFACT_NAMES = ("archived_dataset",)


@dataclass(frozen=True)
class ArtifactSnapshot:
    artifact_id: str
    path: Path
    content: bytes
    sha256: str

    @property
    def byte_length(self) -> int:
        return len(self.content)

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class R20IntervalCCompositionContext:
    """Trusted R20 C boundary input; the signed outer envelope is mandatory."""

    interval_a: PreparedCampaign
    candidate_activity_attestations: Sequence[Mapping[str, Any]]
    interval_a_ledger: SQLiteActivityLedger
    interval_a_effect_attestations: Sequence[Mapping[str, Any]]
    interval_b: PreparedCampaign
    interval_b_ledger: SQLiteActivityLedger
    interval_b_effect_attestations: Sequence[Mapping[str, Any]]
    interval_b_activity_attestations: Sequence[Mapping[str, Any]]
    scorer_input_envelope: Mapping[str, Any]
    b_to_c_authority: Mapping[str, Any]
    expected_result_artifact_path: Path


def capture_artifact_snapshot(
    path_value: str | Path, *, artifact_id: str, require_read_only: bool = False,
) -> ArtifactSnapshot:
    """Read one canonical regular file through a stable, no-follow descriptor."""
    path = Path(path_value)
    if artifact_id not in {*_ARTIFACT_NAMES, *_OPTIONAL_ARTIFACT_NAMES}:
        raise GuardFailure("R20_SNAPSHOT_ARTIFACT_ID_INVALID")
    try:
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise GuardFailure("R20_SNAPSHOT_PATH_NOT_CANONICAL")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
    except GuardFailure:
        raise
    except OSError as exc:
        raise GuardFailure("R20_SNAPSHOT_ARTIFACT_UNAVAILABLE") from exc
    try:
        before = os.fstat(fd)
        named_before = path.lstat()
        if (not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino) != (named_before.st_dev, named_before.st_ino)):
            raise GuardFailure("R20_SNAPSHOT_NOT_STABLE_REGULAR_FILE")
        if require_read_only and stat.S_IMODE(before.st_mode) & 0o222:
            raise GuardFailure("R20_RESULTS_SNAPSHOT_NOT_READ_ONLY")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(fd)
        named_after = path.lstat()
        identity_before = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns, before.st_mode,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns, after.st_mode,
        )
        if (identity_before != identity_after
                or (named_after.st_dev, named_after.st_ino) != (before.st_dev, before.st_ino)
                or len(content) != before.st_size):
            raise GuardFailure("R20_SNAPSHOT_ARTIFACT_CHANGED_DURING_READ")
    except GuardFailure:
        raise
    except OSError as exc:
        raise GuardFailure("R20_SNAPSHOT_ARTIFACT_READ_FAILED") from exc
    finally:
        os.close(fd)
    return ArtifactSnapshot(
        artifact_id=artifact_id,
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def snapshot_manifest(
    dataset: ArtifactSnapshot, schema: ArtifactSnapshot, results: ArtifactSnapshot,
    archived_dataset: ArtifactSnapshot | None = None,
) -> dict[str, Any]:
    supplied = (dataset, schema, results) + ((archived_dataset,) if archived_dataset else ())
    snapshots = {item.artifact_id: item for item in supplied}
    expected = set(_ARTIFACT_NAMES) | ({"archived_dataset"} if archived_dataset else set())
    if set(snapshots) != expected:
        raise GuardFailure("R20_SNAPSHOT_MANIFEST_ARTIFACT_SET_INVALID")
    for name, item in snapshots.items():
        if (not item.path.is_absolute()
                or item.path.resolve(strict=False) != item.path
                or hashlib.sha256(item.content).hexdigest() != item.sha256):
            raise GuardFailure("R20_SNAPSHOT_MANIFEST_SOURCE_INVALID")
    return {
        "schema_version": R20_SNAPSHOT_MANIFEST_SCHEMA,
        "artifacts": {
            name: snapshots[name].manifest_entry()
            for name in (*_ARTIFACT_NAMES, *(('archived_dataset',) if archived_dataset else ()))
        },
    }


def parse_snapshot_inputs(
    dataset: ArtifactSnapshot, schema: ArtifactSnapshot, results: ArtifactSnapshot,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int]:
    """Parse and validate dataset/results only from the captured byte objects."""
    try:
        import jsonschema
        schema_value = json.loads(schema.content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardFailure("R20_SNAPSHOT_JSON_INPUT_INVALID") from exc
    if not isinstance(schema_value, dict):
        raise GuardFailure("R20_SCHEMA_SNAPSHOT_NOT_OBJECT")
    cases = _parse_snapshot_jsonl(dataset.content, str(dataset.path))
    try:
        jsonschema.Draft202012Validator.check_schema(schema_value)
        validator = jsonschema.Draft202012Validator(schema_value)
    except jsonschema.SchemaError as exc:
        raise GuardFailure("R20_SCHEMA_SNAPSHOT_SCHEMA_INVALID") from exc
    problems: list[str] = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id") or "")
        if case_id in seen:
            problems.append(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        for issue in validator.iter_errors(case):
            location = ".".join(str(part) for part in issue.absolute_path)
            problems.append(f"{case_id}:{location or '<root>'}: {issue.message}")
    if problems:
        raise GuardFailure("R20_SNAPSHOT_SCHEMA_VALIDATION_FAILED:" + "\n".join(problems[:40]))

    result_rows = _parse_snapshot_jsonl(results.content, str(results.path))
    indexed: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in result_rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise GuardFailure("R20_RESULT_ROW_CASE_ID_MISSING")
        duplicates += int(case_id in indexed)
        indexed[case_id] = row
    return cases, indexed, duplicates


def _parse_snapshot_jsonl(content: bytes, source: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GuardFailure(f"R20_SNAPSHOT_JSONL_UTF8_INVALID:{source}") from exc
    output: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GuardFailure(
                f"R20_SNAPSHOT_JSONL_INVALID:{source}:{line_number}"
            ) from exc
        if not isinstance(item, dict):
            raise GuardFailure(f"R20_SNAPSHOT_JSONL_ROW_NOT_OBJECT:{source}:{line_number}")
        output.append(item)
    if not output:
        raise GuardFailure(f"R20_SNAPSHOT_JSONL_EMPTY:{source}")
    return output


def validate_result_set(
    cases: list[dict[str, Any]], results: Mapping[str, Mapping[str, Any]],
    duplicate_rows: int,
) -> None:
    case_ids = {case["case_id"] for case in cases}
    extra = sorted(set(results) - case_ids)
    missing = sorted(case_ids - set(results))
    if extra or missing or duplicate_rows:
        raise GuardFailure(
            "R20_CLOSED_B_RESULT_SET_INVALID:"
            f"extra={len(extra)}:missing={len(missing)}:duplicate={duplicate_rows}"
        )


def validate_snapshot_manifest(manifest: Any) -> dict[str, Any]:
    if (not isinstance(manifest, Mapping)
            or set(manifest) != {"schema_version", "artifacts"}
            or manifest.get("schema_version") != R20_SNAPSHOT_MANIFEST_SCHEMA
            or not isinstance(manifest.get("artifacts"), Mapping)
            or not set(_ARTIFACT_NAMES).issubset(manifest["artifacts"])
            or not set(manifest["artifacts"]).issubset(set(_ARTIFACT_NAMES) | set(_OPTIONAL_ARTIFACT_NAMES))):
        raise GuardFailure("R20_SNAPSHOT_MANIFEST_INVALID")
    normalized: dict[str, Any] = {"schema_version": R20_SNAPSHOT_MANIFEST_SCHEMA, "artifacts": {}}
    for name in (*_ARTIFACT_NAMES, *(('archived_dataset',) if "archived_dataset" in manifest["artifacts"] else ())):
        item = manifest["artifacts"][name]
        if (not isinstance(item, Mapping) or set(item) != {"path", "byte_length", "sha256"}
                or not isinstance(item.get("path"), str)
                or not Path(item["path"]).is_absolute()
                or Path(item["path"]).resolve(strict=False) != Path(item["path"])
                or not isinstance(item.get("byte_length"), int)
                or isinstance(item.get("byte_length"), bool)
                or item["byte_length"] < 0
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in item["sha256"])):
            raise GuardFailure("R20_SNAPSHOT_MANIFEST_ARTIFACT_INVALID")
        normalized["artifacts"][name] = dict(item)
    return normalized


def scorer_identity() -> dict[str, Any]:
    """Identify every implementation module used to snapshot and score R20 input."""
    root = Path(__file__).resolve().parent
    sources = {
        "r7r20_activity.py": Path(__file__),
        "r7r19_activity.py": root / "r7r19_activity.py",
        "evaluation_lifecycle.py": root / "evaluation_lifecycle.py",
        "tools/score_real_corpus_results.py": root / "tools" / "score_real_corpus_results.py",
    }
    files: dict[str, str] = {}
    try:
        for relative, path in sources.items():
            metadata = path.lstat()
            if (not stat.S_ISREG(metadata.st_mode)
                    or path.resolve(strict=True) != path):
                raise GuardFailure("R20_SCORER_IDENTITY_SOURCE_NOT_CANONICAL")
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GuardFailure("R20_SCORER_IDENTITY_SOURCE_UNAVAILABLE") from exc
    return {
        "schema_version": R20_SCORER_IDENTITY_SCHEMA,
        "entrypoint": "tools.score_real_corpus_results.realize_r20_score_report_from_closed_b",
        "files": files,
        "sha256": canonical_sha256(files),
    }


def scorer_input_binding(
    manifest: Mapping[str, Any], identity: Mapping[str, Any], *,
    report_json: str | Path, report_markdown: str | Path,
    writer: str = R19_REPOSITORY_REPORT_WRITER,
) -> dict[str, Any]:
    normalized_manifest = validate_snapshot_manifest(manifest)
    output_paths = (Path(report_json), Path(report_markdown))
    if any(not path.is_absolute() or path.resolve(strict=False) != path for path in output_paths):
        raise GuardFailure("R20_REPORT_OUTPUT_PATH_NOT_CANONICAL")
    if not isinstance(identity, Mapping):
        raise GuardFailure("R20_SCORER_IDENTITY_INVALID")
    return {
        "schema_version": R20_SCORER_INPUT_SCHEMA,
        "snapshot_manifest_sha256": canonical_sha256(normalized_manifest),
        "scorer_identity_sha256": canonical_sha256(dict(identity)),
        "report_json": str(report_json),
        "report_markdown": str(report_markdown),
        "writer": writer,
    }


def r20_scorer_input(
    report: Mapping[str, Any], *, report_json: str | Path,
    report_markdown: str | Path,
    writer: str = R19_REPOSITORY_REPORT_WRITER,
) -> dict[str, str]:
    """Use the frozen R19 C-input shape over an R20-augmented report object.

    The report digest contains the full R20 snapshot manifest and scorer identity;
    preserving this exact object shape lets the unchanged R19 C validator enforce it.
    """
    if (not isinstance(report, Mapping)
            or not isinstance(report.get(_REPORT_SNAPSHOT_KEY), Mapping)
            or not isinstance(report.get(_REPORT_SCORER_IDENTITY_KEY), Mapping)
            or not isinstance(report.get(_REPORT_INPUT_BINDING_KEY), Mapping)):
        raise GuardFailure("R20_SCORER_INPUT_SNAPSHOT_BINDING_REQUIRED")
    manifest = validate_snapshot_manifest(report[_REPORT_SNAPSHOT_KEY])
    identity = report[_REPORT_SCORER_IDENTITY_KEY]
    expected_binding = scorer_input_binding(
        manifest, identity, report_json=report_json,
        report_markdown=report_markdown, writer=writer,
    )
    if dict(report[_REPORT_INPUT_BINDING_KEY]) != expected_binding:
        raise GuardFailure("R20_SCORER_INPUT_BINDING_MISMATCH")
    return r19_scorer_input(
        report,
        report_json=report_json,
        report_markdown=report_markdown,
        writer=writer,
    )


def r20_scorer_input_sha256(
    report: Mapping[str, Any], *, report_json: str | Path,
    report_markdown: str | Path,
    writer: str = R19_REPOSITORY_REPORT_WRITER,
) -> str:
    return canonical_sha256(r20_scorer_input(
        report, report_json=report_json, report_markdown=report_markdown, writer=writer,
    ))


def r20_c_scorer_input(envelope: Mapping[str, Any]) -> dict[str, str]:
    """Canonical C authorization input, distinct from the legacy R19 scorer input."""
    if (not isinstance(envelope, Mapping)
            or envelope.get("schema_version") != R20_REALIZATION_SCHEMA
            or envelope.get("campaign_protocol") != R20_CAMPAIGN_COMPOSITION_SCHEMA):
        raise GuardFailure("R20_C_REQUIRES_SNAPSHOT_ENVELOPE")
    required = (
        "snapshot_manifest_sha256", "scorer_identity_sha256", "report_sha256",
        "r19_realization_sha256", "scorer_input_sha256",
    )
    if any(not isinstance(envelope.get(key), str) or len(envelope[key]) != 64 for key in required):
        raise GuardFailure("R20_C_SCORER_INPUT_BINDING_INVALID")
    return {
        "schema_version": R20_SCORER_INPUT_SCHEMA,
        "campaign_protocol": R20_CAMPAIGN_COMPOSITION_SCHEMA,
        "r20_envelope_sha256": canonical_sha256(dict(envelope)),
        "snapshot_manifest_sha256": envelope["snapshot_manifest_sha256"],
        "scorer_identity_sha256": envelope["scorer_identity_sha256"],
        "report_sha256": envelope["report_sha256"],
        "r19_realization_sha256": envelope["r19_realization_sha256"],
        "r19_scorer_input_sha256": envelope["scorer_input_sha256"],
    }


def r20_c_scorer_input_sha256(envelope: Mapping[str, Any]) -> str:
    return canonical_sha256(r20_c_scorer_input(envelope))


def _derive_r20_b_to_c_composition(
    interval_a: PreparedCampaign,
    interval_b: PreparedCampaign, interval_c: PreparedCampaign, *,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_ledger: SQLiteActivityLedger,
    interval_b_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_activity_attestations: Sequence[Mapping[str, Any]],
    scorer_input_envelope: Mapping[str, Any],
    expected_result_artifact_path: Path,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
) -> dict[str, Any]:
    if (not isinstance(scorer_input_envelope, Mapping)
            or scorer_input_envelope.get("schema_version") != R20_REALIZATION_SCHEMA
            or scorer_input_envelope.get("campaign_protocol") != R20_CAMPAIGN_COMPOSITION_SCHEMA):
        raise GuardFailure("R20_C_REQUIRES_SNAPSHOT_ENVELOPE")
    inner = validate_r20_realization_for_c(
        scorer_input_envelope,
        result_artifact_path=expected_result_artifact_path,
        trust=trust,
    )
    c_input_sha256 = r20_c_scorer_input_sha256(scorer_input_envelope)
    composition = _derive_r19_b_to_c_composition(
        interval_a, interval_b, interval_c,
        candidate_activity_attestations=candidate_activity_attestations,
        interval_a_ledger=interval_a_ledger,
        interval_a_effect_attestations=interval_a_effect_attestations,
        interval_b_ledger=interval_b_ledger,
        interval_b_effect_attestations=interval_b_effect_attestations,
        interval_b_activity_attestations=interval_b_activity_attestations,
        scorer_input_realization=inner,
        expected_result_artifact_path=expected_result_artifact_path,
        trust=trust,
        reservation_store=reservation_store,
        c_protocol=R20_CAMPAIGN_COMPOSITION_SCHEMA,
        authorized_c_input_sha256=c_input_sha256,
    )
    result = dict(composition)
    result.update({
        "schema_version": R20_B_TO_C_AUTHORITY_SCHEMA,
        "campaign_protocol": R20_CAMPAIGN_COMPOSITION_SCHEMA,
        "r19_realization_sha256": canonical_sha256(dict(inner)),
        "realization_sha256": canonical_sha256(dict(scorer_input_envelope)),
        "r19_scorer_input_sha256": inner["scorer_input_sha256"],
        "scorer_input_sha256": c_input_sha256,
        "r20_envelope_sha256": canonical_sha256(dict(scorer_input_envelope)),
        "r20_snapshot_manifest_sha256": scorer_input_envelope["snapshot_manifest_sha256"],
        "result_artifact_path": str(expected_result_artifact_path),
    })
    result.pop("composition_sha256", None)
    result["composition_sha256"] = canonical_sha256(result)
    return result


def seal_r20_b_to_c_authority(
    interval_a: PreparedCampaign,
    interval_b: PreparedCampaign, interval_c: PreparedCampaign, *,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_ledger: SQLiteActivityLedger,
    interval_b_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_activity_attestations: Sequence[Mapping[str, Any]],
    scorer_input_envelope: Mapping[str, Any],
    expected_result_artifact_path: Path,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
    issued_at: str,
) -> dict[str, Any]:
    composition = _derive_r20_b_to_c_composition(
        interval_a, interval_b, interval_c,
        candidate_activity_attestations=candidate_activity_attestations,
        interval_a_ledger=interval_a_ledger,
        interval_a_effect_attestations=interval_a_effect_attestations,
        interval_b_ledger=interval_b_ledger,
        interval_b_effect_attestations=interval_b_effect_attestations,
        interval_b_activity_attestations=interval_b_activity_attestations,
        scorer_input_envelope=scorer_input_envelope,
        expected_result_artifact_path=expected_result_artifact_path,
        trust=trust, reservation_store=reservation_store,
    )
    auth_c, _consumed_c = _consumed_prepared_interval(interval_c, trust, reservation_store)
    issued = _utc_timestamp(issued_at, "R20_B_TO_C_AUTHORITY_ISSUED_AT")
    floor = max(
        _utc_timestamp(composition["interval_b_closed_at"], "B.CLOSED_AT"),
        _utc_timestamp(scorer_input_envelope["realized_at"], "R20.REALIZED_AT"),
        _utc_timestamp(interval_c.authorization_document.get("issued_at"), "C.ISSUED_AT"),
        _utc_timestamp(interval_c.claim.get("claimed_at"), "C.CLAIMED_AT"),
    )
    if issued <= floor or issued >= _utc_timestamp(auth_c.B0, "C.B0"):
        raise GuardFailure("R20_B_TO_C_AUTHORITY_TIME_INVALID")
    record: dict[str, Any] = {
        "schema_version": R20_B_TO_C_AUTHORITY_SCHEMA,
        "campaign_protocol": R20_CAMPAIGN_COMPOSITION_SCHEMA,
        "composition": composition,
        "issued_at": issued_at,
    }
    signer = trust.signer_for("STATE_TRANSITION_OBSERVER_AUTHORITY")
    record["signer_id"] = signer.signer_id
    record["signature"] = signer.sign(_canonical_record_payload(record))
    return record


def validate_r20_b_to_c(
    interval_a: PreparedCampaign,
    interval_b: PreparedCampaign, interval_c: PreparedCampaign, *,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_ledger: SQLiteActivityLedger,
    interval_b_effect_attestations: Sequence[Mapping[str, Any]],
    interval_b_activity_attestations: Sequence[Mapping[str, Any]],
    scorer_input_envelope: Mapping[str, Any],
    b_to_c_authority: Mapping[str, Any],
    expected_result_artifact_path: Path,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
) -> dict[str, Any]:
    composition = _derive_r20_b_to_c_composition(
        interval_a, interval_b, interval_c,
        candidate_activity_attestations=candidate_activity_attestations,
        interval_a_ledger=interval_a_ledger,
        interval_a_effect_attestations=interval_a_effect_attestations,
        interval_b_ledger=interval_b_ledger,
        interval_b_effect_attestations=interval_b_effect_attestations,
        interval_b_activity_attestations=interval_b_activity_attestations,
        scorer_input_envelope=scorer_input_envelope,
        expected_result_artifact_path=expected_result_artifact_path,
        trust=trust, reservation_store=reservation_store,
    )
    required = {"schema_version", "campaign_protocol", "composition", "issued_at", "signer_id", "signature"}
    if not isinstance(b_to_c_authority, Mapping) or set(b_to_c_authority) != required:
        raise GuardFailure("R20_B_TO_C_AUTHORITY_MISSING_OR_INVALID")
    if (b_to_c_authority.get("schema_version") != R20_B_TO_C_AUTHORITY_SCHEMA
            or b_to_c_authority.get("campaign_protocol") != R20_CAMPAIGN_COMPOSITION_SCHEMA
            or b_to_c_authority.get("composition") != composition):
        raise GuardFailure("R20_B_TO_C_AUTHORITY_COMPOSITION_MISMATCH")
    _verify_signed_record(
        b_to_c_authority, trust,
        permitted_authority_classes=frozenset({"STATE_TRANSITION_OBSERVER_AUTHORITY"}),
    )
    auth_c, _consumed_c = _consumed_prepared_interval(interval_c, trust, reservation_store)
    issued = _utc_timestamp(b_to_c_authority.get("issued_at"), "R20_B_TO_C_AUTHORITY_ISSUED_AT")
    floor = max(
        _utc_timestamp(composition["interval_b_closed_at"], "B.CLOSED_AT"),
        _utc_timestamp(scorer_input_envelope["realized_at"], "R20.REALIZED_AT"),
        _utc_timestamp(interval_c.authorization_document.get("issued_at"), "C.ISSUED_AT"),
        _utc_timestamp(interval_c.claim.get("claimed_at"), "C.CLAIMED_AT"),
    )
    if issued <= floor or issued >= _utc_timestamp(auth_c.B0, "C.B0"):
        raise GuardFailure("R20_B_TO_C_AUTHORITY_TIME_INVALID")
    result = dict(composition)
    result["b_to_c_authority_sha256"] = hashlib.sha256(
        _canonical_record_payload(b_to_c_authority)
    ).hexdigest()
    return result


def attach_r20_interval_after_b0(
    prepared: PreparedCampaign, *,
    composition_context: R20IntervalCCompositionContext,
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
    if (authorization.campaign_protocol != R20_CAMPAIGN_COMPOSITION_SCHEMA
            or authorization.interval_role != "C"):
        raise GuardFailure("R20_PROTOCOL_AUTHORIZATION_REQUIRED")
    return R17CampaignRuntime._activate_r20_after_protocol_validation(
        prepared, authorization, composition_context=composition_context,
        trust=trust, reservation_store=reservation_store, ledger=ledger,
        identity_provider=identity_provider, b0_record=b0_record, now=now,
    )


def realized_report_for_active_c(
    envelope: Mapping[str, Any], *, report_json: str | Path,
    report_markdown: str | Path, result_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Dispatch the report only under a matching validated R20 C runtime."""
    runtime = current_campaign_runtime()
    if (runtime is None
            or runtime.authorization.campaign_protocol != R20_CAMPAIGN_COMPOSITION_SCHEMA
            or runtime.authorization.interval_role != "C"):
        raise GuardFailure("R20_SCORER_DISPATCH_REQUIRES_ACTIVE_INTERVAL_C")
    envelope_digest = canonical_sha256(dict(envelope)) if isinstance(envelope, Mapping) else None
    if envelope_digest != getattr(runtime, "_r20_realization_sha256", None):
        raise GuardFailure("R20_SCORER_REALIZATION_DIFFERS_FROM_CLOSURE")
    authorized_result_path = getattr(runtime, "_r20_result_artifact_path", None)
    if (not isinstance(authorized_result_path, str)
            or (result_artifact_path is not None
                and str(result_artifact_path) != authorized_result_path)):
        raise GuardFailure("R20_SCORER_RESULT_PATH_DIFFERS_FROM_CLOSURE")
    inner = validate_r20_realization_for_c(
        envelope, result_artifact_path=authorized_result_path, trust=runtime.trust,
    )
    if (str(report_json) != inner.get("report_json")
            or str(report_markdown) != inner.get("report_markdown")):
        raise GuardFailure("R20_SCORER_OUTPUT_PATHS_NOT_EXACTLY_AUTHORIZED")
    expected_input_hash = r20_c_scorer_input_sha256(envelope)
    scorer_ops = [op for op in runtime.authorization.operations if op.surface_id == "scorer"]
    if len(scorer_ops) != 1 or scorer_ops[0].frozen_input_sha256 != expected_input_hash:
        raise GuardFailure("R20_SCORER_DISPATCH_INPUT_NOT_EXACTLY_AUTHORIZED")
    report = inner.get("report_object")
    if not isinstance(report, Mapping):
        raise GuardFailure("R20_REALIZED_REPORT_OBJECT_INVALID")
    return dict(report)


def build_r19_c_compatible_realization_from_snapshot(
    *, interval_a: PreparedCampaign,
    candidate_activity_attestations: list[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: list[Mapping[str, Any]],
    interval_b: PreparedCampaign,
    interval_b_ledger: SQLiteActivityLedger,
    reservation_store: SQLiteAuthorizationUseStore,
    interval_b_effect_attestations: list[Mapping[str, Any]],
    interval_b_activity_attestations: list[Mapping[str, Any]],
    result_artifact_path: Path,
    result_bytes: bytes,
    result_rows: Mapping[str, Mapping[str, Any]],
    benchmark_execution_ids: list[str],
    report: Mapping[str, Any],
    report_json: Path,
    report_markdown: Path,
    trust: R17TrustConfiguration,
    realized_at: str,
) -> dict[str, Any]:
    """Create the exact R19 C-gate shape without reopening an R20 input path.

    This is deliberately in the versioned R20 module: R19's builder, schema,
    serializer, and validator remain unchanged. The resulting inner record uses
    the unchanged R19 contract so its existing B→C proof remains authoritative.
    """
    auth_b, consumed_b = _consumed_prepared_interval(
        interval_b, trust, reservation_store,
    )
    _require_role(auth_b, "B", _B_PAIRS, exact_cardinality=False)
    a_to_b = validate_r19_a_to_b(
        interval_a, interval_b,
        candidate_activity_attestations=candidate_activity_attestations,
        interval_a_ledger=interval_a_ledger,
        interval_a_effect_attestations=interval_a_effect_attestations,
        trust=trust, reservation_store=reservation_store,
    )
    state, closure = _closed_state(auth_b, interval_b_ledger, label="B")
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
    result_path = Path(result_artifact_path)
    if not result_path.is_absolute() or result_path.resolve(strict=True) != result_path:
        raise GuardFailure("R20_RESULT_SNAPSHOT_PATH_NOT_CANONICAL")
    if not isinstance(result_bytes, bytes):
        raise GuardFailure("R20_RESULT_SNAPSHOT_BYTES_REQUIRED")
    result_digest = hashlib.sha256(result_bytes).hexdigest()
    parsed_result_rows = _parse_snapshot_jsonl(result_bytes, str(result_path))
    parsed_result_map: dict[str, dict[str, Any]] = {}
    parsed_duplicates = 0
    for row in parsed_result_rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise GuardFailure("R20_RESULT_ROW_CASE_ID_MISSING")
        parsed_duplicates += int(case_id in parsed_result_map)
        parsed_result_map[case_id] = row
    if parsed_duplicates or parsed_result_map != dict(result_rows):
        raise GuardFailure("R20_RESULT_ROWS_DIFFER_FROM_SNAPSHOT_BYTES")
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
    required_lifecycle = [
        "ATTEMPTED", "AUTHORIZED", "STARTED", "COMPLETED",
        "CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLOSED",
    ]
    if any(phases.get(ident) != required_lifecycle for ident in benchmark_ids):
        raise GuardFailure("R19_BENCHMARK_LIFECYCLE_NOT_COMPLETED")
    result_case_ids = [row.get("case_id") for row in result_rows.values()]
    if (len(result_rows) != len(benchmark_ids)
            or len(result_case_ids) != len(result_rows)
            or any(not isinstance(case_id, str) or not case_id for case_id in result_case_ids)
            or len(set(result_case_ids)) != len(result_case_ids)):
        raise GuardFailure("R19_RESULT_ROWS_NOT_ONE_TO_ONE_WITH_BENCHMARK_EXECUTIONS")
    report_run = report.get("run") if isinstance(report, Mapping) else None
    if (not isinstance(report_run, Mapping)
            or report_run.get("result_sha256") != result_digest
            or report_run.get("result_count") != len(result_rows)
            or report_run.get("result_count") != len(benchmark_ids)):
        raise GuardFailure("R19_REPORT_NOT_BOUND_TO_CLOSED_B_RESULT_ARTIFACT")
    scorer_input = r19_scorer_input(
        report, report_json=report_json, report_markdown=report_markdown,
    )
    record: dict[str, Any] = {
        "schema_version": R19_REALIZATION_SCHEMA,
        "campaign_protocol": auth_b.campaign_protocol,
        "campaign_id": auth_b.campaign_id,
        "source_commit": auth_b.source_commit,
        "source_tree": auth_b.source_tree,
        "a_to_b_composition_sha256": canonical_sha256(a_to_b),
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
        "result_artifact_path": str(result_path),
        "result_artifact_sha256": result_digest,
        "report_object": dict(report),
        "report_sha256": scorer_input["report_sha256"],
        "report_json": scorer_input["report_json"],
        "report_markdown": scorer_input["report_markdown"],
        "writer": scorer_input["writer"],
        "scorer_input_sha256": canonical_sha256(scorer_input),
        "realization_implementation": r19_realization_implementation_identity(),
        "realized_at": realized_at,
    }
    signer = trust.signer_for("STATE_TRANSITION_OBSERVER_AUTHORITY")
    record["signer_id"] = signer.signer_id
    record["signature"] = signer.sign(_canonical_record_payload(record))
    return record


def seal_r20_snapshot_realization(
    r19_realization: Mapping[str, Any], *,
    manifest: Mapping[str, Any], scorer: Mapping[str, Any],
    trust: R17TrustConfiguration,
) -> dict[str, Any]:
    normalized_manifest = validate_snapshot_manifest(manifest)
    report = r19_realization.get("report_object")
    if not isinstance(report, Mapping):
        raise GuardFailure("R20_REPORT_OBJECT_REQUIRED")
    if (report.get(_REPORT_SNAPSHOT_KEY) != normalized_manifest
            or report.get(_REPORT_SCORER_IDENTITY_KEY) != scorer
            or scorer != scorer_identity()):
        raise GuardFailure("R20_REPORT_SNAPSHOT_OR_SCORER_BINDING_MISMATCH")
    input_hash = r20_scorer_input_sha256(
        report,
        report_json=r19_realization.get("report_json"),
        report_markdown=r19_realization.get("report_markdown"),
        writer=r19_realization.get("writer"),
    )
    if (r19_realization.get("schema_version") != R19_REALIZATION_SCHEMA
            or r19_realization.get("scorer_input_sha256") != input_hash
            or r19_realization.get("report_sha256") != canonical_sha256(dict(report))):
        raise GuardFailure("R20_R19_REALIZATION_BINDING_MISMATCH")
    _verify_signed_record(
        r19_realization, trust,
        permitted_authority_classes=frozenset({"STATE_TRANSITION_OBSERVER_AUTHORITY"}),
    )
    record: dict[str, Any] = {
        "schema_version": R20_REALIZATION_SCHEMA,
        "campaign_protocol": R20_CAMPAIGN_COMPOSITION_SCHEMA,
        "snapshot_manifest": normalized_manifest,
        "snapshot_manifest_sha256": canonical_sha256(normalized_manifest),
        "scorer_identity": dict(scorer),
        "scorer_identity_sha256": canonical_sha256(dict(scorer)),
        "report_object": dict(report),
        "report_sha256": canonical_sha256(dict(report)),
        "scorer_input_sha256": input_hash,
        "r19_realization": dict(r19_realization),
        "r19_realization_sha256": canonical_sha256(dict(r19_realization)),
        "realized_at": r19_realization.get("realized_at"),
    }
    signer = trust.signer_for("STATE_TRANSITION_OBSERVER_AUTHORITY")
    record["signer_id"] = signer.signer_id
    record["signature"] = signer.sign(_canonical_record_payload(record))
    return record


def serialize_r20_realization(record: Mapping[str, Any]) -> bytes:
    if (not isinstance(record, Mapping)
            or record.get("schema_version") != R20_REALIZATION_SCHEMA
            or record.get("campaign_protocol") != R20_CAMPAIGN_COMPOSITION_SCHEMA):
        raise GuardFailure("R20_REALIZATION_SCHEMA_INVALID")
    return canonical_json_bytes(dict(record)) + b"\n"


def deserialize_r20_realization(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardFailure("R20_REALIZATION_SERIALIZATION_INVALID") from exc
    if (not isinstance(value, dict)
            or value.get("schema_version") != R20_REALIZATION_SCHEMA
            or value.get("campaign_protocol") != R20_CAMPAIGN_COMPOSITION_SCHEMA):
        raise GuardFailure("R20_REALIZATION_SCHEMA_INVALID")
    if serialize_r20_realization(value) != data:
        raise GuardFailure("R20_REALIZATION_SERIALIZATION_NOT_CANONICAL")
    return value


def validate_r20_realization_for_c(
    record: Mapping[str, Any], *, result_artifact_path: str | Path,
    trust: R17TrustConfiguration, require_current_scorer: bool = True,
) -> dict[str, Any]:
    required = {
        "schema_version", "campaign_protocol", "snapshot_manifest",
        "snapshot_manifest_sha256", "scorer_identity", "scorer_identity_sha256",
        "report_object", "report_sha256", "scorer_input_sha256", "r19_realization",
        "r19_realization_sha256", "realized_at", "signer_id", "signature",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise GuardFailure("R20_REALIZATION_SCHEMA_INVALID")
    if (record.get("schema_version") != R20_REALIZATION_SCHEMA
            or record.get("campaign_protocol") != R20_CAMPAIGN_COMPOSITION_SCHEMA):
        raise GuardFailure("R20_REALIZATION_SCHEMA_INVALID")
    _verify_signed_record(
        record, trust,
        permitted_authority_classes=frozenset({"STATE_TRANSITION_OBSERVER_AUTHORITY"}),
    )
    manifest = validate_snapshot_manifest(record.get("snapshot_manifest"))
    if canonical_sha256(manifest) != record.get("snapshot_manifest_sha256"):
        raise GuardFailure("R20_SNAPSHOT_MANIFEST_HASH_MISMATCH")
    scorer = record.get("scorer_identity")
    if (not isinstance(scorer, Mapping)
            or scorer.get("schema_version") != R20_SCORER_IDENTITY_SCHEMA
            or not isinstance(scorer.get("files"), Mapping)
            or canonical_sha256(dict(scorer["files"])) != scorer.get("sha256")
            or canonical_sha256(dict(scorer)) != record.get("scorer_identity_sha256")):
        raise GuardFailure("R20_SCORER_IDENTITY_INVALID")
    if require_current_scorer and dict(scorer) != scorer_identity():
        raise GuardFailure("R20_SCORER_IDENTITY_NOT_CURRENT")
    inner = record.get("r19_realization")
    if not isinstance(inner, Mapping) or inner.get("schema_version") != R19_REALIZATION_SCHEMA:
        raise GuardFailure("R20_R19_REALIZATION_REQUIRED")
    if inner.get("campaign_protocol") != R19_CAMPAIGN_COMPOSITION_SCHEMA:
        raise GuardFailure("R20_INNER_R19_PROTOCOL_MISMATCH")
    # The inner object is also the unchanged R19 C-gate input; require canonical form and signature.
    if deserialize_r19_realization(canonical_json_bytes(dict(inner)) + b"\n") != dict(inner):
        raise GuardFailure("R20_R19_REALIZATION_NOT_CANONICAL")
    _verify_signed_record(
        inner, trust,
        permitted_authority_classes=frozenset({"STATE_TRANSITION_OBSERVER_AUTHORITY"}),
    )
    report = record.get("report_object")
    if not isinstance(report, Mapping) or dict(report) != dict(inner.get("report_object", {})):
        raise GuardFailure("R20_INNER_REPORT_MISMATCH")
    if (report.get(_REPORT_SNAPSHOT_KEY) != manifest
            or report.get(_REPORT_SCORER_IDENTITY_KEY) != dict(scorer)):
        raise GuardFailure("R20_REPORT_PROVENANCE_MISMATCH")
    dataset = report.get("dataset")
    run = report.get("run")
    artifacts = manifest["artifacts"]
    schema = report.get("schema")
    if (not isinstance(dataset, Mapping) or not isinstance(schema, Mapping)
            or not isinstance(run, Mapping)
            or dataset.get("path") != artifacts["dataset"]["path"]
            or dataset.get("sha256") != artifacts["dataset"]["sha256"]
            or dataset.get("byte_length") != artifacts["dataset"]["byte_length"]
            or schema.get("path") != artifacts["schema"]["path"]
            or schema.get("sha256") != artifacts["schema"]["sha256"]
            or schema.get("byte_length") != artifacts["schema"]["byte_length"]
            or run.get("result_path") != artifacts["results"]["path"]
            or run.get("result_sha256") != artifacts["results"]["sha256"]
            or run.get("result_byte_length") != artifacts["results"]["byte_length"]
            or not Path(artifacts["schema"]["path"]).is_absolute()):
        raise GuardFailure("R20_REPORT_SNAPSHOT_CONTENT_MISMATCH")
    provenance = report.get("provenance")
    archived_report = provenance.get("archived_executed_dataset") if isinstance(provenance, Mapping) else None
    if "archived_dataset" in artifacts:
        if (not isinstance(archived_report, Mapping)
                or archived_report.get("path") != artifacts["archived_dataset"]["path"]
                or archived_report.get("sha256") != artifacts["archived_dataset"]["sha256"]):
            raise GuardFailure("R20_ARCHIVED_DATASET_PROVENANCE_MISMATCH")
    elif archived_report is not None:
        raise GuardFailure("R20_UNSNAPSHOTTED_ARCHIVED_DATASET_PROVENANCE")
    expected_path = Path(result_artifact_path)
    if (not expected_path.is_absolute()
            or expected_path.resolve(strict=False) != expected_path
            or inner.get("result_artifact_path") != str(expected_path)
            or inner.get("result_artifact_sha256") != artifacts["results"]["sha256"]):
        raise GuardFailure("R20_RESULT_ARTIFACT_BINDING_MISMATCH")
    expected_input = r20_scorer_input(
        report,
        report_json=inner.get("report_json"),
        report_markdown=inner.get("report_markdown"),
        writer=inner.get("writer"),
    )
    if (canonical_sha256(dict(report)) != record.get("report_sha256")
            or record.get("report_sha256") != inner.get("report_sha256")
            or canonical_sha256(expected_input) != record.get("scorer_input_sha256")
            or record.get("scorer_input_sha256") != inner.get("scorer_input_sha256")
            or canonical_sha256(dict(inner)) != record.get("r19_realization_sha256")
            or record.get("campaign_protocol") != R20_CAMPAIGN_COMPOSITION_SCHEMA
            or record.get("realized_at") != inner.get("realized_at")):
        raise GuardFailure("R20_SCORER_INPUT_BINDING_MISMATCH")
    return dict(inner)


def extract_r19_realization_for_c(
    record: Mapping[str, Any], *, result_artifact_path: str | Path,
    trust: R17TrustConfiguration,
) -> dict[str, Any]:
    return validate_r20_realization_for_c(
        record, result_artifact_path=result_artifact_path, trust=trust,
    )
