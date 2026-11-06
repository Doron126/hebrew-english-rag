"""Default-disabled, non-production real-activity integration for C44E3R7R17.

No verifier, signer, campaign file, database path, or activation is installed
by import.  A real dispatcher is installed only after explicit external trust
adapters validate the authorization, one-time reservation, source identity,
B0 evidence, and durable ledger.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import fcntl
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import tempfile
import threading
import uuid
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from r7r13_observer import (
    AUTHORIZATION_USE_CLAIM_SCHEMA_VERSION,
    REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17,
    REAL_EPOCH_KIND,
    R19_CAMPAIGN_COMPOSITION_SCHEMA,
    R7R13_GUARD_REGISTRY,
    AuthorizationReservationRequest,
    ConsumedRealAuthorization,
    GuardFailure,
    ParsedRealAuthorization,
    RealAuthorizationScope,
    RecordSignatureVerifier,
    SignatureVerification,
    _canonical_record_payload,
    _verify_signed_record,
    _utc_timestamp,
    consume_real_epoch_authorization,
    derive_real_repository_activity_state,
    parse_real_epoch_authorization,
    validate_authorization_use_claim,
    validate_real_effect_attestations,
    validate_real_activity_receipts,
    validate_real_epoch_boundaries,
)


R17_TRUST_CONFIGURATION_SCHEMA = "r7r17-trust-configuration-1.0.0"
R17_RESERVATION_STORE_SCHEMA = "r7r17-reservation-store-1.0.0"
R17_LEDGER_SCHEMA = "r7r17-durable-ledger-1.0.0"
R17_GOVERNED_CALL_TOPOLOGY_SCHEMA = "r7r17-governed-call-topology-1.0.0"
R17_INVOCATION_BINDING_SCHEMA = "r7r17-invocation-binding-1.0.0"
R17_EXECUTION_RECOVERY_SCHEMA = "r7r17-execution-recovery-attestation-1.0.0"
R17_DIMENSION_ACTIVITY_SCHEMA = "r7r17-dimension-activity-attestation-1.0.0"
R17_CAMPAIGN_COMPOSITION_SCHEMA = "r7r17-campaign-composition-1.0.0"

# Positive activity is a source-level proposition, not a lifecycle phase.
# Each event is deliberately surface-specific and independently attested.
R17_DIMENSION_ACTIVITY_EVENTS = {
    "REAL_SUBJECT_EXECUTION_COUNT": {
        "event_kind": "SUBJECT_REQUEST_ACCEPTED",
        "evidence_surface_id": "chat-api-request",
    },
    "CANDIDATE_STARTED": {
        "event_kind": "CANDIDATE_PROCESS_STARTED",
        "evidence_surface_id": "candidate-launch",
    },
    "BENCHMARK_TRAFFIC_SENT": {
        "event_kind": "BENCHMARK_HTTP_REQUEST_DISPATCHED",
        "evidence_surface_id": "benchmark-runner",
    },
    "SCORER_RERUN": {
        "event_kind": "SCORER_REPORTS_COMMITTED",
        "evidence_surface_id": "scorer",
    },
}

AUTHORITY_CLASSES = (
    "REPOSITORY_ACTION_AUTHORITY",
    "REPLAY_PROTECTION_AUTHORITY",
    "INTERVAL_BOUNDARY_AUTHORITY",
    "REPOSITORY_OBSERVER_AUTHORITY",
    "STATE_TRANSITION_OBSERVER_AUTHORITY",
    "INDEPENDENT_CENSUS_AUTHORITY",
)
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_LINUX_PROCESS_IDENTITY = re.compile(r"^linux:[0-9a-f-]{36}:[1-9][0-9]*:[0-9]+:[0-9a-f]{64}$")
_R17_ACTIVATION_TOKEN = object()
_R19_ACTIVATION_TOKEN = object()
_CURRENT_BINDING: ContextVar["InvocationBinding | None"] = ContextVar(
    "r7r17_current_invocation_binding", default=None
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _receipt_hash_for_execution(records: Sequence[Mapping[str, Any]], execution_id: str) -> str:
    """Return the digest of an execution's latest receipt in a verified ledger."""
    matching = [record for record in records if record.get("execution_id") == execution_id]
    if not matching:
        raise GuardFailure("R17_RECOVERY_EXECUTION_RECEIPT_MISSING")
    return hashlib.sha256(_canonical_record_payload(matching[-1])).hexdigest()


def _timestamp_microseconds(value: str) -> int:
    parsed = _utc_timestamp(value, "RESERVATION_BOUNDARY")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    return ((delta.days * 86400 + delta.seconds) * 1_000_000) + delta.microseconds


def _secure_private_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise GuardFailure("R17_PRIVATE_DATABASE_METADATA_UNAVAILABLE") from exc
        if not stat.S_ISREG(info.st_mode):
            raise GuardFailure("R17_PRIVATE_DATABASE_FILE_TYPE_INVALID")
        try:
            os.chmod(candidate, 0o600, follow_symlinks=False)
        except OSError as exc:
            raise GuardFailure("R17_PRIVATE_DATABASE_PERMISSION_FAILURE") from exc


class SignatureBackend(Protocol):
    """Externally supplied verification/signing implementation; no key ships here."""

    def verify(self, signer_id: str, payload: bytes, signature: str) -> SignatureVerification: ...


class RecordSigner(Protocol):
    signer_id: str
    authority_id: str
    authority_class: str

    def sign(self, payload: bytes) -> str: ...


@dataclass(frozen=True)
class TrustRoleConfiguration:
    authority_class: str
    authority_id: str
    signer_ids: tuple[str, ...]
    backend: SignatureBackend
    signer: RecordSigner | None = None


class _RoleVerifier:
    def __init__(self, role: TrustRoleConfiguration) -> None:
        self.role = role

    def verify(self, signer_id: str, canonical_payload: bytes, signature: str) -> SignatureVerification:
        if signer_id not in self.role.signer_ids:
            return SignatureVerification(False, self.role.authority_id, self.role.authority_class)
        result = self.role.backend.verify(signer_id, canonical_payload, signature)
        if (not isinstance(result, SignatureVerification) or not result.verified
                or result.authority_id != self.role.authority_id
                or result.authority_class != self.role.authority_class):
            return SignatureVerification(False, self.role.authority_id, self.role.authority_class)
        return result


class R17TrustConfiguration(RecordSignatureVerifier):
    """Six separately identified authority roles, all explicitly configured."""

    def __init__(self, roles: Mapping[str, TrustRoleConfiguration], *, test_only: bool = False) -> None:
        if set(roles) != set(AUTHORITY_CLASSES):
            missing = sorted(set(AUTHORITY_CLASSES) - set(roles))
            extra = sorted(set(roles) - set(AUTHORITY_CLASSES))
            raise GuardFailure(f"R17_TRUST_CONFIGURATION_INCOMPLETE:missing={missing}:extra={extra}")
        authorities: set[str] = set()
        signers: set[str] = set()
        for role_name, role in roles.items():
            if role.authority_class != role_name or not role.authority_id:
                raise GuardFailure("R17_TRUST_AUTHORITY_IDENTITY_INVALID")
            if not role.signer_ids or len(set(role.signer_ids)) != len(role.signer_ids):
                raise GuardFailure("R17_TRUST_SIGNER_SET_INVALID")
            if role.authority_id in authorities or signers.intersection(role.signer_ids):
                raise GuardFailure("R17_TRUST_ROLE_IDENTITY_COLLISION")
            if role.signer is not None and (
                role.signer.authority_class != role_name
                or role.signer.authority_id != role.authority_id
                or role.signer.signer_id not in role.signer_ids
            ):
                raise GuardFailure("R17_TRUST_SIGNER_ROLE_MISMATCH")
            authorities.add(role.authority_id)
            signers.update(role.signer_ids)
        self._roles = dict(roles)
        self._verifiers = {name: _RoleVerifier(role) for name, role in roles.items()}
        self._signer_to_role = {
            signer_id: role_name
            for role_name, role in roles.items()
            for signer_id in role.signer_ids
        }
        self._test_only_requested = bool(test_only)

    @property
    def test_only(self) -> bool:
        # The legacy setter below keeps synthetic fixtures usable, but cannot
        # promote their HMAC backend into real authority.
        if self._test_only_requested:
            return True
        if (set(self._verifiers) != set(self._roles)
                or any(type(self._verifiers[name]) is not _RoleVerifier
                       or self._verifiers[name].role is not role
                       for name, role in self._roles.items())
                or self._signer_to_role != {
                    signer_id: name
                    for name, role in self._roles.items()
                    for signer_id in role.signer_ids
                }):
            return True
        from r7r22_trust import real_trust_configuration_verified

        return not real_trust_configuration_verified(self._roles)

    @test_only.setter
    def test_only(self, value: bool) -> None:
        self._test_only_requested = bool(value)

    def verify(self, signer_id: str, canonical_payload: bytes, signature: str) -> SignatureVerification:
        role_name = self._signer_to_role.get(signer_id)
        if role_name is None:
            return SignatureVerification(False, "", "")
        return self._verifiers[role_name].verify(signer_id, canonical_payload, signature)

    def signer_for(self, authority_class: str) -> RecordSigner:
        if authority_class not in self._roles:
            raise GuardFailure("R17_TRUST_ROLE_MISSING")
        signer = self._roles[authority_class].signer
        if signer is None:
            raise GuardFailure(f"R17_TRUST_ROLE_SIGNER_UNAVAILABLE:{authority_class}")
        return signer

    def readiness(self) -> dict[str, Any]:
        # A mutable test_only flag is not real authority.  The R22 provider
        # rechecks its exact six-role implementation and on-disk key bindings.
        effective_test_only = self.test_only
        return {
            "schema_version": R17_TRUST_CONFIGURATION_SCHEMA,
            "configured_authority_classes": sorted(self._roles),
            "authority_ids_distinct": len({role.authority_id for role in self._roles.values()}) == len(AUTHORITY_CLASSES),
            "signer_ids_distinct": len(self._signer_to_role) == sum(len(role.signer_ids) for role in self._roles.values()),
            "test_only": effective_test_only,
            "real_activation_eligible": not effective_test_only,
        }


def load_trust_configuration(
    path: Path,
    backend_factory: Callable[[Mapping[str, Any]], tuple[SignatureBackend, RecordSigner | None]],
) -> R17TrustConfiguration:
    """Load role metadata; a host-owned provider supplies key/HSM operations.

    The config format contains references and identities only.  Inline secret
    material is not part of the schema and is rejected by exact field checking.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardFailure("R17_TRUST_CONFIGURATION_UNREADABLE") from exc
    if not isinstance(document, dict) or set(document) != {"schema_version", "test_only", "roles"}:
        raise GuardFailure("R17_TRUST_CONFIGURATION_SCHEMA_INVALID")
    if document["schema_version"] != R17_TRUST_CONFIGURATION_SCHEMA or not isinstance(document["test_only"], bool):
        raise GuardFailure("R17_TRUST_CONFIGURATION_SCHEMA_INVALID")
    if not isinstance(document["roles"], list):
        raise GuardFailure("R17_TRUST_CONFIGURATION_SCHEMA_INVALID")
    roles: dict[str, TrustRoleConfiguration] = {}
    for row in document["roles"]:
        required = {"authority_class", "authority_id", "signer_ids", "provider_ref"}
        if not isinstance(row, dict) or set(row) != required:
            raise GuardFailure("R17_TRUST_CONFIGURATION_ROLE_SCHEMA_INVALID")
        role_name = row["authority_class"]
        signer_ids = row["signer_ids"]
        if (role_name not in AUTHORITY_CLASSES or role_name in roles
                or not isinstance(row["authority_id"], str) or not row["authority_id"]
                or not isinstance(signer_ids, list) or not signer_ids
                or any(not isinstance(item, str) or not item for item in signer_ids)
                or not isinstance(row["provider_ref"], str) or not row["provider_ref"]):
            raise GuardFailure("R17_TRUST_CONFIGURATION_ROLE_SCHEMA_INVALID")
        backend, signer = backend_factory(dict(row))
        roles[role_name] = TrustRoleConfiguration(
            role_name, row["authority_id"], tuple(signer_ids), backend, signer,
        )
    return R17TrustConfiguration(roles, test_only=document["test_only"])


class SQLiteAuthorizationUseStore:
    """Concrete cross-process one-time reservation store for R16 claims."""

    def __init__(
        self,
        database_path: Path,
        *,
        store_identity: str,
        claim_signer: RecordSigner,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.path = Path(database_path).expanduser().absolute()
        self.store_identity = store_identity
        self.claim_signer = claim_signer
        self.timeout_seconds = float(timeout_seconds)
        if (not store_identity or not _HEX_256.fullmatch(store_identity)
                or claim_signer.authority_class != "REPLAY_PROTECTION_AUTHORITY"
                or not claim_signer.signer_id or not claim_signer.authority_id):
            raise GuardFailure("R17_RESERVATION_STORE_CONFIGURATION_INVALID")
        if self.path.exists() and self.path.is_symlink():
            raise GuardFailure("R17_RESERVATION_DATABASE_MUST_NOT_BE_SYMLINK")
        if not self.path.parent.is_dir():
            raise GuardFailure("R17_RESERVATION_DATABASE_PARENT_MISSING")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self.timeout_seconds, isolation_level=None)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS r17_store_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version TEXT NOT NULL,
                    store_identity TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS r17_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL UNIQUE,
                    decision_id TEXT NOT NULL UNIQUE,
                    nonce TEXT NOT NULL UNIQUE,
                    campaign_id TEXT NOT NULL,
                    interval_id TEXT NOT NULL,
                    authorization_sha256 TEXT NOT NULL,
                    B0 TEXT NOT NULL,
                    B1 TEXT NOT NULL,
                    B0_us INTEGER NOT NULL,
                    B1_us INTEGER NOT NULL CHECK (B0_us < B1_us),
                    execution_ids_json TEXT NOT NULL,
                    claim_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS r17_reservation_overlap_idx
                    ON r17_reservations(B0_us, B1_us);
                CREATE TRIGGER IF NOT EXISTS r17_reservations_no_update
                    BEFORE UPDATE ON r17_reservations BEGIN SELECT RAISE(ABORT, 'R17_RESERVATION_APPEND_ONLY'); END;
                CREATE TRIGGER IF NOT EXISTS r17_reservations_no_delete
                    BEFORE DELETE ON r17_reservations BEGIN SELECT RAISE(ABORT, 'R17_RESERVATION_APPEND_ONLY'); END;
            """)
            row = conn.execute("SELECT schema_version, store_identity FROM r17_store_meta WHERE singleton=1").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO r17_store_meta(singleton,schema_version,store_identity) VALUES(1,?,?)",
                    (R17_RESERVATION_STORE_SCHEMA, self.store_identity),
                )
            elif row != (R17_RESERVATION_STORE_SCHEMA, self.store_identity):
                raise GuardFailure("R17_RESERVATION_STORE_IDENTITY_MISMATCH")
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise GuardFailure("R17_RESERVATION_STORE_INTEGRITY_FAILURE")
        except sqlite3.DatabaseError as exc:
            raise GuardFailure("R17_RESERVATION_STORE_INITIALIZATION_FAILURE") from exc
        finally:
            conn.close()
        _secure_private_sqlite_files(self.path)

    def claim_once(self, request: AuthorizationReservationRequest) -> Mapping[str, Any]:
        execution_ids = list(request.execution_ids)
        if (not execution_ids or execution_ids != sorted(set(execution_ids))
                or not _HEX_256.fullmatch(request.authorization_sha256)):
            raise GuardFailure("R17_RESERVATION_REQUEST_INVALID")
        b0_us, b1_us = _timestamp_microseconds(request.B0), _timestamp_microseconds(request.B1)
        if b0_us >= b1_us:
            raise GuardFailure("R17_RESERVATION_INTERVAL_INVALID")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute(
                "SELECT reservation_id FROM r17_reservations WHERE decision_id=? OR nonce=?",
                (request.decision_id, request.nonce),
            ).fetchone()
            overlap = conn.execute(
                "SELECT reservation_id FROM r17_reservations WHERE B0_us < ? AND ? < B1_us LIMIT 1",
                (b1_us, b0_us),
            ).fetchone()
            status = "ALREADY_CONSUMED" if prior else "INTERVAL_CONFLICT" if overlap else "NEW_NONCONFLICTING"
            reservation_id = prior[0] if prior else str(uuid.uuid4())
            claim: dict[str, Any] = {
                "schema_version": AUTHORIZATION_USE_CLAIM_SCHEMA_VERSION,
                "claim_id": str(uuid.uuid4()),
                "reservation_id": reservation_id,
                "reservation_status": status,
                "decision_id": request.decision_id,
                "campaign_id": request.campaign_id,
                "interval_id": request.interval_id,
                "B0": request.B0,
                "B1": request.B1,
                "nonce": request.nonce,
                "execution_ids": execution_ids,
                "authorization_sha256": request.authorization_sha256,
                "claimed_at": now.isoformat().replace("+00:00", "Z"),
                "store_identity": self.store_identity,
                "signer_id": self.claim_signer.signer_id,
            }
            claim["signature"] = self.claim_signer.sign(_canonical_record_payload(claim))
            if not isinstance(claim["signature"], str) or not claim["signature"]:
                raise GuardFailure("R17_RESERVATION_CLAIM_SIGNATURE_MISSING")
            if status == "NEW_NONCONFLICTING":
                conn.execute(
                    """INSERT INTO r17_reservations(
                        reservation_id,claim_id,decision_id,nonce,campaign_id,interval_id,
                        authorization_sha256,B0,B1,B0_us,B1_us,execution_ids_json,claim_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        reservation_id, claim["claim_id"], request.decision_id, request.nonce,
                        request.campaign_id, request.interval_id, request.authorization_sha256,
                        request.B0, request.B1, b0_us, b1_us,
                        json.dumps(execution_ids, separators=(",", ":")),
                        json.dumps(claim, sort_keys=True, separators=(",", ":")),
                    ),
                )
            conn.execute("COMMIT")
            _secure_private_sqlite_files(self.path)
            return claim
        except (sqlite3.DatabaseError, OSError) as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if isinstance(exc, GuardFailure):
                raise
            raise GuardFailure("R17_RESERVATION_ATOMIC_CLAIM_FAILURE") from exc
        finally:
            conn.close()

    def reservation_for_claim(self, claim: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if not isinstance(claim, Mapping):
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT claim_json FROM r17_reservations WHERE reservation_id=? AND claim_id=?",
                (claim.get("reservation_id"), claim.get("claim_id")),
            ).fetchone()
            return json.loads(row[0]) if row else None
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            raise GuardFailure("R17_RESERVATION_READBACK_FAILURE") from exc
        finally:
            conn.close()

    def recovery_snapshot(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            check = conn.execute("PRAGMA integrity_check").fetchone()
            if check != ("ok",):
                raise GuardFailure("R17_RESERVATION_STORE_INTEGRITY_FAILURE")
            rows = conn.execute(
                "SELECT reservation_id,decision_id,nonce,campaign_id,interval_id,authorization_sha256,B0,B1,execution_ids_json,claim_json FROM r17_reservations ORDER BY B0_us,decision_id"
            ).fetchall()
            reservations = [
                {
                    "reservation_id": row[0], "decision_id": row[1], "nonce": row[2],
                    "campaign_id": row[3], "interval_id": row[4], "authorization_sha256": row[5],
                    "B0": row[6], "B1": row[7], "execution_ids": json.loads(row[8]),
                    "claim": json.loads(row[9]),
                }
                for row in rows
            ]
            return {"schema_version": R17_RESERVATION_STORE_SCHEMA, "store_identity": self.store_identity,
                    "reservation_count": len(reservations), "reservations": reservations}
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            raise GuardFailure("R17_RESERVATION_RECOVERY_FAILURE") from exc
        finally:
            conn.close()


class SQLiteAuthorizationClaimReader(SQLiteAuthorizationUseStore):
    """Verify an existing reservation without possessing claim authority.

    The URI read-only connection and overridden claim method deliberately make
    this a capability-restricted subtype for the existing R17/R19 readback API.
    It never initializes or changes the reservation database.
    """

    def __init__(self, database_path: Path, *, store_identity: str) -> None:
        self.path = Path(database_path).expanduser().absolute()
        self.store_identity = store_identity
        self.timeout_seconds = 30.0
        if not _HEX_256.fullmatch(store_identity):
            raise GuardFailure("R26_RESERVATION_READER_IDENTITY_INVALID")
        info = self.path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise GuardFailure("R26_RESERVATION_READER_FILE_INVALID")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT schema_version, store_identity FROM r17_store_meta WHERE singleton=1"
            ).fetchone()
            if row != (R17_RESERVATION_STORE_SCHEMA, store_identity):
                raise GuardFailure("R26_RESERVATION_READER_STORE_MISMATCH")
            if conn.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise GuardFailure("R26_RESERVATION_READER_INTEGRITY_FAILURE")
        except sqlite3.DatabaseError as exc:
            raise GuardFailure("R26_RESERVATION_READER_INVALID") from exc
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self.path.as_posix()}?mode=ro", uri=True,
            timeout=self.timeout_seconds, isolation_level=None,
        )
        conn.execute("PRAGMA query_only=ON")
        return conn

    def claim_once(self, request: AuthorizationReservationRequest) -> Mapping[str, Any]:
        raise GuardFailure("R26_CANDIDATE_CANNOT_CLAIM_AUTHORIZATION")


class RuntimeIdentityProvider(Protocol):
    def source_identity(self) -> tuple[str, str]: ...
    def process_identity(self) -> str: ...
    def process_identity_alive(self, identity: str) -> bool: ...


class GitProcessIdentityProvider:
    """Read-only identity for a clean Linux Git worktree and current process."""

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = Path(repository_root or Path(__file__).resolve().parent).resolve()

    def source_identity(self) -> tuple[str, str]:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.repository_root, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=self.repository_root, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.repository_root, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise GuardFailure("R17_RUNTIME_SOURCE_IDENTITY_UNAVAILABLE") from exc
        if status:
            raise GuardFailure("R17_RUNTIME_SOURCE_WORKTREE_NOT_CLEAN")
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit) or not re.fullmatch(r"[0-9a-f]{40,64}", tree):
            raise GuardFailure("R17_RUNTIME_SOURCE_IDENTITY_INVALID")
        return commit, tree

    def process_identity(self) -> str:
        return self._process_identity_for_pid(os.getpid())

    def _process_identity_for_pid(self, pid: int) -> str:
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
            stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            start_ticks = stat_line[stat_line.rfind(")") + 2 :].split()[19]
            executable = os.path.realpath(f"/proc/{pid}/exe")
        except (OSError, IndexError) as exc:
            raise GuardFailure("R17_PROCESS_IDENTITY_UNAVAILABLE") from exc
        if not boot_id or not start_ticks or not executable:
            raise GuardFailure("R17_PROCESS_IDENTITY_INVALID")
        return f"linux:{boot_id}:{pid}:{start_ticks}:{canonical_sha256(executable)}"

    def process_identity_alive(self, identity: str) -> bool:
        parts = identity.split(":")
        if len(parts) != 5 or parts[0] != "linux":
            return False
        try:
            pid = int(parts[2])
            return self._process_identity_for_pid(pid) == identity
        except (ValueError, GuardFailure):
            return False


_ALLOWED_NEXT_PHASES = {
    None: {"ATTEMPTED"},
    "ATTEMPTED": {"AUTHORIZED", "BLOCKED"},
    "AUTHORIZED": {"STARTED"},
    "STARTED": {"COMPLETED", "FAILED"},
    "COMPLETED": {"CLEANUP_STARTED"},
    "FAILED": {"CLEANUP_STARTED"},
    "BLOCKED": {"CLEANUP_STARTED"},
    "CLEANUP_STARTED": {"CLEANUP_COMPLETED"},
    "CLEANUP_COMPLETED": {"CLOSED"},
    "CLOSED": set(),
}
_R17_RECEIPT_FIELDS = {
    "schema_version", "epoch_kind", "event_id", "sequence", "epoch_id",
    "authorization_decision_id", "campaign_id", "interval_id", "execution_id",
    "timestamp", "dimension_id", "surface_id", "operation_identity",
    "source_commit", "source_tree", "process_identity", "request_identity",
    "initiating_record_sha256", "frozen_input_sha256", "behavior_observation_sha256",
    "outcome_attestation_id", "parent_execution_id", "correlation_id",
    "dimension_count_owner", "phase", "reason_code", "previous_receipt_sha256",
    "signer_id", "signature",
}


class SQLiteActivityLedger:
    """Append-only, restart-checked SQLite receipt chain plus external head file."""

    def __init__(
        self,
        database_path: Path,
        checkpoint_path: Path,
        *,
        ledger_id: str,
        authorization: ParsedRealAuthorization,
        observer_signer: RecordSigner,
        trust: R17TrustConfiguration,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.path = Path(database_path).expanduser().absolute()
        self.checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        self.ledger_id = ledger_id
        self.authorization = authorization
        self._authorized_operations = {operation.execution_id: operation for operation in authorization.operations}
        self.namespace = {
            "epoch_id": authorization.epoch_id,
            "authorization_decision_id": authorization.decision_id,
            "campaign_id": authorization.campaign_id,
            "interval_id": authorization.interval_id,
            "B0": authorization.B0,
            "B1": authorization.B1,
            "source_commit": authorization.source_commit,
            "source_tree": authorization.source_tree,
            "authorized_operations": [operation.__dict__ for operation in authorization.operations],
        }
        self.observer_signer = observer_signer
        self.trust = trust
        self.clock = clock
        self._poisoned = False
        if (not ledger_id or not authorization.epoch_id or not authorization.campaign_id or not authorization.interval_id
                or observer_signer.authority_class != "REPOSITORY_OBSERVER_AUTHORITY"
                or observer_signer.signer_id not in trust._roles["REPOSITORY_OBSERVER_AUTHORITY"].signer_ids
                or observer_signer.authority_id != trust._roles["REPOSITORY_OBSERVER_AUTHORITY"].authority_id
                or self.path == self.checkpoint_path
                or (self.path.exists() and self.path.is_symlink())
                or (self.checkpoint_path.exists() and self.checkpoint_path.is_symlink())
                or not self.path.parent.is_dir() or not self.checkpoint_path.parent.is_dir()):
            raise GuardFailure("R17_LEDGER_CONFIGURATION_INVALID")
        _timestamp_microseconds(authorization.B0)
        _timestamp_microseconds(authorization.B1)
        if _timestamp_microseconds(authorization.B0) >= _timestamp_microseconds(authorization.B1):
            raise GuardFailure("R17_LEDGER_INTERVAL_INVALID")
        created = not self.path.exists()
        self._initialize(created=created)
        self.verify()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self, *, created: bool) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS r17_ledger_meta (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    schema_version TEXT NOT NULL,
                    ledger_id TEXT NOT NULL,
                    namespace_json TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    head_sha256 TEXT NOT NULL,
                    closed INTEGER NOT NULL CHECK(closed IN (0,1))
                );
                CREATE TABLE IF NOT EXISTS r17_events (
                    sequence INTEGER PRIMARY KEY CHECK(sequence > 0),
                    event_id TEXT NOT NULL UNIQUE,
                    execution_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    receipt_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS r17_events_execution_idx ON r17_events(execution_id, sequence);
                CREATE TABLE IF NOT EXISTS r17_lifecycles (
                    execution_id TEXT PRIMARY KEY,
                    last_phase TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL UNIQUE
                );
                CREATE TRIGGER IF NOT EXISTS r17_events_lifecycle_guard
                    BEFORE INSERT ON r17_events
                    WHEN
                      (NEW.phase = 'ATTEMPTED' AND EXISTS(
                          SELECT 1 FROM r17_lifecycles WHERE execution_id=NEW.execution_id))
                      OR (NEW.phase != 'ATTEMPTED' AND NOT EXISTS(
                          SELECT 1 FROM r17_lifecycles WHERE execution_id=NEW.execution_id))
                      OR (NEW.phase != CASE
                          WHEN NEW.phase = 'ATTEMPTED' THEN 'ATTEMPTED'
                          ELSE (SELECT CASE last_phase
                              WHEN 'ATTEMPTED' THEN CASE WHEN NEW.phase IN ('AUTHORIZED','BLOCKED') THEN NEW.phase ELSE '' END
                              WHEN 'AUTHORIZED' THEN CASE WHEN NEW.phase='STARTED' THEN NEW.phase ELSE '' END
                              WHEN 'STARTED' THEN CASE WHEN NEW.phase IN ('COMPLETED','FAILED') THEN NEW.phase ELSE '' END
                              WHEN 'COMPLETED' THEN CASE WHEN NEW.phase='CLEANUP_STARTED' THEN NEW.phase ELSE '' END
                              WHEN 'FAILED' THEN CASE WHEN NEW.phase='CLEANUP_STARTED' THEN NEW.phase ELSE '' END
                              WHEN 'BLOCKED' THEN CASE WHEN NEW.phase='CLEANUP_STARTED' THEN NEW.phase ELSE '' END
                              WHEN 'CLEANUP_STARTED' THEN CASE WHEN NEW.phase='CLEANUP_COMPLETED' THEN NEW.phase ELSE '' END
                              WHEN 'CLEANUP_COMPLETED' THEN CASE WHEN NEW.phase='CLOSED' THEN NEW.phase ELSE '' END
                              ELSE '' END
                           FROM r17_lifecycles WHERE execution_id=NEW.execution_id)
                          END)
                    BEGIN SELECT RAISE(ABORT, 'R17_LIFECYCLE_PHASE_OR_ID_REPLAY'); END;
                CREATE TRIGGER IF NOT EXISTS r17_events_lifecycle_insert
                    AFTER INSERT ON r17_events
                    BEGIN
                      INSERT INTO r17_lifecycles(execution_id,last_phase,last_sequence)
                        SELECT NEW.execution_id,NEW.phase,NEW.sequence WHERE NEW.phase='ATTEMPTED';
                      UPDATE r17_lifecycles SET last_phase=NEW.phase,last_sequence=NEW.sequence
                        WHERE execution_id=NEW.execution_id AND NEW.phase!='ATTEMPTED';
                    END;
                CREATE TRIGGER IF NOT EXISTS r17_events_no_update
                    BEFORE UPDATE ON r17_events BEGIN SELECT RAISE(ABORT, 'R17_LEDGER_APPEND_ONLY'); END;
                CREATE TRIGGER IF NOT EXISTS r17_events_no_delete
                    BEFORE DELETE ON r17_events BEGIN SELECT RAISE(ABORT, 'R17_LEDGER_APPEND_ONLY'); END;
                CREATE TRIGGER IF NOT EXISTS r17_events_insert_order
                    BEFORE INSERT ON r17_events
                    WHEN NEW.sequence != (SELECT sequence + 1 FROM r17_ledger_meta WHERE singleton=1)
                    BEGIN SELECT RAISE(ABORT, 'R17_LEDGER_SEQUENCE_GAP'); END;
                CREATE TRIGGER IF NOT EXISTS r17_events_closed_guard
                    BEFORE INSERT ON r17_events
                    WHEN (SELECT closed FROM r17_ledger_meta WHERE singleton=1) != 0
                    BEGIN SELECT RAISE(ABORT, 'R17_LEDGER_CLOSED'); END;
                CREATE TABLE IF NOT EXISTS r17_boundaries (
                    boundary TEXT PRIMARY KEY CHECK(boundary IN ('B0','B1')),
                    record_json TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS r17_boundaries_no_update
                    BEFORE UPDATE ON r17_boundaries BEGIN SELECT RAISE(ABORT, 'R17_BOUNDARY_APPEND_ONLY'); END;
                CREATE TRIGGER IF NOT EXISTS r17_boundaries_no_delete
                    BEFORE DELETE ON r17_boundaries BEGIN SELECT RAISE(ABORT, 'R17_BOUNDARY_APPEND_ONLY'); END;
                CREATE TABLE IF NOT EXISTS r17_execution_recoveries (
                    recovery_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL UNIQUE,
                    record_json TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS r17_execution_recoveries_no_update
                    BEFORE UPDATE ON r17_execution_recoveries
                    BEGIN SELECT RAISE(ABORT, 'R17_RECOVERY_APPEND_ONLY'); END;
                CREATE TRIGGER IF NOT EXISTS r17_execution_recoveries_no_delete
                    BEFORE DELETE ON r17_execution_recoveries
                    BEGIN SELECT RAISE(ABORT, 'R17_RECOVERY_APPEND_ONLY'); END;
                CREATE TABLE IF NOT EXISTS r17_closure (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    closed_at TEXT NOT NULL,
                    boundary_json TEXT NOT NULL,
                    final_sequence INTEGER NOT NULL,
                    final_head_sha256 TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS r17_closure_no_update
                    BEFORE UPDATE ON r17_closure BEGIN SELECT RAISE(ABORT, 'R17_CLOSURE_IMMUTABLE'); END;
                CREATE TRIGGER IF NOT EXISTS r17_closure_no_delete
                    BEFORE DELETE ON r17_closure BEGIN SELECT RAISE(ABORT, 'R17_CLOSURE_IMMUTABLE'); END;
            """)
            raw_namespace = json.dumps(self.namespace, sort_keys=True, separators=(",", ":"))
            root = hashlib.sha256(b"").hexdigest()
            row = conn.execute(
                "SELECT schema_version,ledger_id,namespace_json,sequence,head_sha256,closed FROM r17_ledger_meta WHERE singleton=1"
            ).fetchone()
            if row is None:
                if not created:
                    raise GuardFailure("R17_LEDGER_META_MISSING")
                conn.execute(
                    "INSERT INTO r17_ledger_meta VALUES(1,?,?,?,?,?,0)",
                    (R17_LEDGER_SCHEMA, self.ledger_id, raw_namespace, 0, root),
                )
            elif row[:3] != (R17_LEDGER_SCHEMA, self.ledger_id, raw_namespace):
                raise GuardFailure("R17_LEDGER_NAMESPACE_MISMATCH")
            if created and not self.checkpoint_path.exists():
                self._write_checkpoint({
                    "schema_version": R17_LEDGER_SCHEMA, "ledger_id": self.ledger_id,
                    "namespace": self.namespace, "sequence": 0, "head_sha256": root,
                })
            elif not self.checkpoint_path.is_file():
                raise GuardFailure("R17_LEDGER_CHECKPOINT_MISSING")
        except (sqlite3.DatabaseError, OSError) as exc:
            if isinstance(exc, GuardFailure):
                raise
            raise GuardFailure("R17_LEDGER_INITIALIZATION_FAILURE") from exc
        finally:
            conn.close()
        os.chmod(self.path, 0o600)

    @contextmanager
    def _io_lock(self) -> Iterator[None]:
        """Serialize DB/head-file transactions across independent processes."""
        lock_path = Path(f"{self.path}.lock")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(lock_path, flags, 0o600)
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise GuardFailure("R17_LEDGER_CROSS_PROCESS_LOCK_UNAVAILABLE") from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _write_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        encoded = canonical_json_bytes(dict(checkpoint)) + b"\n"
        fd, temporary_name = tempfile.mkstemp(prefix=".r17-head-", dir=self.checkpoint_path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.checkpoint_path)
            directory_fd = os.open(self.checkpoint_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise GuardFailure("R17_LEDGER_CHECKPOINT_DURABILITY_FAILURE") from exc

    def _read_checkpoint(self) -> dict[str, Any]:
        try:
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GuardFailure("R17_LEDGER_CHECKPOINT_INVALID") from exc
        if (not isinstance(checkpoint, dict)
                or set(checkpoint) != {"schema_version", "ledger_id", "namespace", "sequence", "head_sha256"}
                or checkpoint.get("schema_version") != R17_LEDGER_SCHEMA
                or checkpoint.get("ledger_id") != self.ledger_id
                or checkpoint.get("namespace") != self.namespace
                or not isinstance(checkpoint.get("sequence"), int)
                or not _HEX_256.fullmatch(str(checkpoint.get("head_sha256")))):
            raise GuardFailure("R17_LEDGER_CHECKPOINT_IDENTITY_MISMATCH")
        return checkpoint

    def _events(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        try:
            rows = conn.execute(
                "SELECT sequence,event_id,execution_id,phase,receipt_sha256,record_json "
                "FROM r17_events ORDER BY sequence"
            ).fetchall()
            records = []
            for sequence, event_id, execution_id, phase, digest, encoded in rows:
                record = json.loads(encoded)
                if (not isinstance(record, dict)
                        or (record.get("sequence"), record.get("event_id"), record.get("execution_id"),
                            record.get("phase")) != (sequence, event_id, execution_id, phase)
                        or hashlib.sha256(_canonical_record_payload(record)).hexdigest() != digest):
                    raise GuardFailure("R17_LEDGER_DATABASE_ROW_BINDING_MISMATCH")
                records.append(record)
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            raise GuardFailure("R17_LEDGER_RECORD_READ_FAILURE") from exc
        if any(not isinstance(record, dict) for record in records):
            raise GuardFailure("R17_LEDGER_RECORD_SCHEMA_INVALID")
        return records

    def _verify_records(self, records: Sequence[Mapping[str, Any]]) -> tuple[int, str, dict[str, str]]:
        previous = hashlib.sha256(b"").hexdigest()
        last_by_execution: dict[str, str] = {}
        seen_events: set[str] = set()
        attempt_sequence: dict[str, int] = {}
        start_sequence: dict[str, int] = {}
        terminal_sequence: dict[str, int] = {}
        causal_by_execution: dict[str, tuple[str | None, str, bool]] = {}
        previous_time: datetime | None = None
        for sequence, record in enumerate(records, start=1):
            if set(record) != _R17_RECEIPT_FIELDS:
                raise GuardFailure("R17_LEDGER_RECORD_SCHEMA_INVALID")
            if record.get("sequence") != sequence or record.get("previous_receipt_sha256") != previous:
                raise GuardFailure("R17_LEDGER_CHAIN_OR_SEQUENCE_INVALID")
            if record.get("schema_version") != REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17 or record.get("epoch_kind") != REAL_EPOCH_KIND:
                raise GuardFailure("R17_LEDGER_SCHEMA_IDENTITY_INVALID")
            if any(record.get(key) != self.namespace[value] for key, value in (
                ("epoch_id", "epoch_id"), ("campaign_id", "campaign_id"),
                ("interval_id", "interval_id"), ("source_commit", "source_commit"),
                ("source_tree", "source_tree"),
            )):
                raise GuardFailure("R17_LEDGER_EVENT_NAMESPACE_MISMATCH")
            if record.get("authorization_decision_id") != self.namespace["authorization_decision_id"]:
                raise GuardFailure("R17_LEDGER_DECISION_ID_MISMATCH")
            event_id = record.get("event_id")
            execution_id = record.get("execution_id")
            phase = record.get("phase")
            if not isinstance(event_id, str) or not event_id or event_id in seen_events:
                raise GuardFailure("R17_LEDGER_DUPLICATE_EVENT")
            if not isinstance(execution_id, str) or not execution_id:
                raise GuardFailure("R17_LEDGER_EXECUTION_ID_INVALID")
            operation = self._authorized_operations.get(execution_id)
            if operation is None or any(record.get(field) != expected for field, expected in (
                ("dimension_id", operation.dimension_id), ("surface_id", operation.surface_id),
                ("operation_identity", operation.operation_identity),
                ("source_commit", operation.source_commit), ("source_tree", operation.source_tree),
                ("process_identity", operation.process_identity), ("request_identity", operation.request_identity),
                ("initiating_record_sha256", operation.initiating_record_sha256),
                ("frozen_input_sha256", operation.frozen_input_sha256),
            )):
                raise GuardFailure("R17_LEDGER_EVENT_OPERATION_NOT_AUTHORIZED")
            seen_events.add(event_id)
            timestamp = _utc_timestamp(record.get("timestamp"), "RECEIPT_TIMESTAMP")
            if (timestamp < _utc_timestamp(self.namespace["B0"], "B0")
                    or timestamp >= _utc_timestamp(self.namespace["B1"], "B1")
                    or (previous_time is not None and timestamp < previous_time)):
                raise GuardFailure("R17_LEDGER_EVENT_OUTSIDE_INTERVAL_OR_REORDERED")
            previous_time = timestamp
            if phase not in _ALLOWED_NEXT_PHASES.get(last_by_execution.get(execution_id), set()):
                raise GuardFailure("R17_LEDGER_LIFECYCLE_PHASE_REORDER")
            if phase in {"BLOCKED", "FAILED"}:
                if not isinstance(record.get("reason_code"), str) or not record["reason_code"]:
                    raise GuardFailure("R17_LEDGER_TERMINAL_REASON_MISSING")
            elif record.get("reason_code") is not None:
                raise GuardFailure("R17_LEDGER_REASON_ON_NONTERMINAL_PHASE")
            correlation = record.get("correlation_id")
            parent_id = record.get("parent_execution_id")
            owner = record.get("dimension_count_owner")
            if (not isinstance(correlation, str) or not correlation
                    or correlation != record.get("request_identity")
                    or (parent_id is not None and (not isinstance(parent_id, str) or not parent_id or parent_id == execution_id))
                    or not isinstance(owner, bool)):
                raise GuardFailure("R17_LEDGER_CAUSAL_BINDING_INVALID")
            _verify_signed_record(
                record, self.trust,
                permitted_authority_classes=frozenset({"REPOSITORY_OBSERVER_AUTHORITY"}),
            )
            last_by_execution[execution_id] = str(phase)
            if phase == "ATTEMPTED":
                attempt_sequence[execution_id] = sequence
            elif phase == "STARTED":
                start_sequence[execution_id] = sequence
            elif phase in {"COMPLETED", "FAILED", "BLOCKED"}:
                terminal_sequence[execution_id] = sequence
            causal = (
                parent_id,
                str(correlation),
                bool(owner),
            )
            if execution_id in causal_by_execution and causal_by_execution[execution_id] != causal:
                raise GuardFailure("R17_LEDGER_CAUSAL_BINDING_CHANGED")
            causal_by_execution[execution_id] = causal
            previous = hashlib.sha256(_canonical_record_payload(record)).hexdigest()
        for execution_id, (parent_id, correlation, owner) in causal_by_execution.items():
            child = self._authorized_operations[execution_id]
            if correlation != child.request_identity:
                raise GuardFailure("R17_LEDGER_CORRELATION_MISMATCH")
            if parent_id is None:
                if not owner:
                    raise GuardFailure("R17_LEDGER_TOP_LEVEL_OWNER_INVALID")
                continue
            parent = self._authorized_operations.get(parent_id)
            edge = R17_GOVERNED_CALL_TOPOLOGY["edges"].get((parent.surface_id, child.surface_id)) if parent else None
            child_attempt = attempt_sequence.get(execution_id)
            parent_attempt = attempt_sequence.get(parent_id)
            parent_started = start_sequence.get(parent_id)
            parent_terminal = terminal_sequence.get(parent_id)
            expected_owner = edge is not None and edge["dimension_count_owner"] != "PARENT"
            if (parent is None or edge is None or parent.request_identity != child.request_identity
                    or edge["dimension_relation"] != (
                        "SAME" if parent.dimension_id == child.dimension_id else "DIFFERENT"
                    )
                    or owner is not expected_owner
                    or child_attempt is None or parent_attempt is None or parent_started is None
                    or parent_attempt >= parent_started or parent_started >= child_attempt
                    or (parent_terminal is not None and child_attempt >= parent_terminal)):
                raise GuardFailure("R17_LEDGER_CAUSAL_BINDING_OR_ORDER_INVALID")
        return len(records), previous, last_by_execution

    def verify(self) -> dict[str, Any]:
        with self._io_lock():
            return self._verify_locked()

    def _verify_locked(self) -> dict[str, Any]:
        if self._poisoned:
            raise GuardFailure("R17_LEDGER_POISONED_AFTER_DURABILITY_FAILURE")
        conn = self._connect()
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise GuardFailure("R17_LEDGER_SQLITE_INTEGRITY_FAILURE")
            row = conn.execute(
                "SELECT sequence,head_sha256,closed FROM r17_ledger_meta WHERE singleton=1"
            ).fetchone()
            if row is None:
                raise GuardFailure("R17_LEDGER_META_MISSING")
            records = self._events(conn)
            sequence, head, last_by_execution = self._verify_records(records)
            if (row[0] != sequence or row[1] != head
                    or (sequence == 0 and head != hashlib.sha256(b"").hexdigest())):
                raise GuardFailure("R17_LEDGER_DATABASE_HEAD_MISMATCH")
            checkpoint = self._read_checkpoint()
            if checkpoint["sequence"] != sequence or checkpoint["head_sha256"] != head:
                raise GuardFailure("R17_LEDGER_TRUNCATED_OR_CHECKPOINT_DIVERGENCE")
            lifecycle_rows = conn.execute(
                "SELECT execution_id,last_phase,last_sequence FROM r17_lifecycles ORDER BY execution_id"
            ).fetchall()
            expected_lifecycles = {}
            for index, record in enumerate(records, start=1):
                expected_lifecycles[record["execution_id"]] = (record["phase"], index)
            if {row[0]: (row[1], row[2]) for row in lifecycle_rows} != expected_lifecycles:
                raise GuardFailure("R17_LEDGER_LIFECYCLE_INDEX_MISMATCH")
            closure = conn.execute("SELECT closed_at,boundary_json,final_sequence,final_head_sha256 FROM r17_closure WHERE singleton=1").fetchone()
            if bool(row[2]) != bool(closure):
                raise GuardFailure("R17_LEDGER_CLOSURE_STATE_MISMATCH")
            if closure and (closure[2] != sequence or closure[3] != head):
                raise GuardFailure("R17_LEDGER_CLOSURE_HEAD_MISMATCH")
            if closure and _utc_timestamp(closure[0], "LEDGER_CLOSED_AT") < _utc_timestamp(self.authorization.B1, "B1"):
                raise GuardFailure("R17_LEDGER_CLOSED_BEFORE_B1")
            boundaries = [json.loads(row[0]) for row in conn.execute(
                "SELECT record_json FROM r17_boundaries ORDER BY CASE boundary WHEN 'B0' THEN 0 ELSE 1 END"
            ).fetchall()]
            if closure:
                if (len(boundaries) != 2 or boundaries[0].get("boundary") != "B0"
                        or boundaries[1].get("boundary") != "B1"
                        or json.dumps(boundaries[1], sort_keys=True, separators=(",", ":")) != closure[1]):
                    raise GuardFailure("R17_LEDGER_CLOSURE_BOUNDARY_MISMATCH")
                validate_real_epoch_boundaries(self.authorization, boundaries, self.trust)
            elif len(boundaries) not in {0, 1} or (boundaries and boundaries[0].get("boundary") != "B0"):
                raise GuardFailure("R17_LEDGER_BOUNDARY_SEQUENCE_INVALID")
            if boundaries and boundaries[0].get("boundary") == "B0":
                _validate_single_boundary(self.authorization, boundaries[0], "B0", self.trust)
            current = self.clock()
            recovery_rows = conn.execute(
                "SELECT record_json FROM r17_execution_recoveries ORDER BY execution_id"
            ).fetchall()
            recovered_ids: set[str] = set()
            for (encoded_recovery,) in recovery_rows:
                recovery = json.loads(encoded_recovery)
                execution_id = _validate_execution_recovery_record(
                    self.authorization, recovery, self.trust, now=current,
                )
                if (execution_id in recovered_ids
                        or last_by_execution.get(execution_id) in {None, "CLOSED"}
                        or recovery["ledger_head_sha256"] != head
                        or recovery["last_receipt_sha256"] != _receipt_hash_for_execution(records, execution_id)):
                    raise GuardFailure("R17_RECOVERY_LEDGER_BINDING_MISMATCH")
                recovered_ids.add(execution_id)
            lifecycle_open_ids = sorted(
                execution_id for execution_id, phase in last_by_execution.items() if phase != "CLOSED"
            )
            open_ids = sorted(set(lifecycle_open_ids) - recovered_ids)
            if closure and open_ids:
                raise GuardFailure("R17_LEDGER_CLOSED_WITH_UNRESOLVED_EXECUTIONS")
            return {
                "schema_version": R17_LEDGER_SCHEMA,
                "ledger_id": self.ledger_id,
                "sequence": sequence,
                "head_sha256": head,
                "integrity": "PASS",
                "closed": bool(row[2]),
                "closure": ({
                    "closed_at": closure[0],
                    "boundary": json.loads(closure[1]),
                    "final_sequence": closure[2],
                    "final_head_sha256": closure[3],
                } if closure else None),
                "open_execution_ids": open_ids,
                "lifecycle_open_execution_ids": lifecycle_open_ids,
                "recovered_execution_ids": sorted(recovered_ids),
                "record_count": len(records),
            }
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            raise GuardFailure("R17_LEDGER_SQLITE_READ_FAILURE") from exc
        finally:
            conn.close()

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._io_lock():
            return self._append_locked(record)

    def _append_locked(self, record: dict[str, Any]) -> dict[str, Any]:
        if self._poisoned:
            raise GuardFailure("R17_LEDGER_POISONED_AFTER_DURABILITY_FAILURE")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            state = conn.execute("SELECT sequence,head_sha256,closed FROM r17_ledger_meta WHERE singleton=1").fetchone()
            if state is None or state[2]:
                raise GuardFailure("R17_LEDGER_CLOSED_OR_UNAVAILABLE")
            b0_row = conn.execute(
                "SELECT record_json FROM r17_boundaries WHERE boundary='B0'"
            ).fetchone()
            if b0_row is None:
                raise GuardFailure("R17_B0_BOUNDARY_NOT_DURABLY_OPENED")
            try:
                b0_record = json.loads(b0_row[0])
            except json.JSONDecodeError as exc:
                raise GuardFailure("R17_B0_BOUNDARY_RECORD_INVALID") from exc
            _validate_single_boundary(self.authorization, b0_record, "B0", self.trust)
            records = self._events(conn)
            sequence, head, last_by_execution = self._verify_records(records)
            if state[0] != sequence or state[1] != head:
                raise GuardFailure("R17_LEDGER_DATABASE_HEAD_MISMATCH")
            if not self.checkpoint_path.is_file():
                raise GuardFailure("R17_LEDGER_CHECKPOINT_MISSING")
            checkpoint = self._read_checkpoint()
            if checkpoint["sequence"] != sequence or checkpoint["head_sha256"] != head:
                raise GuardFailure("R17_LEDGER_TRUNCATED_OR_CHECKPOINT_DIVERGENCE")
            execution_id = record.get("execution_id")
            phase = record.get("phase")
            if phase not in _ALLOWED_NEXT_PHASES.get(last_by_execution.get(execution_id), set()):
                raise GuardFailure("R17_LEDGER_LIFECYCLE_PHASE_REORDER")
            timestamp = _utc_timestamp(record.get("timestamp"), "RECEIPT_TIMESTAMP")
            if timestamp < _utc_timestamp(self.namespace["B0"], "B0") or timestamp >= _utc_timestamp(self.namespace["B1"], "B1"):
                raise GuardFailure("R17_LEDGER_EVENT_OUTSIDE_B0_B1")
            if records:
                prior_time = _utc_timestamp(records[-1]["timestamp"], "RECEIPT_TIMESTAMP")
                if timestamp < prior_time:
                    raise GuardFailure("R17_LEDGER_EVENT_TIME_REORDER")
            record.update({"sequence": sequence + 1, "previous_receipt_sha256": head})
            if set(record) - {"signature"} != _R17_RECEIPT_FIELDS - {"signature"}:
                raise GuardFailure("R17_LEDGER_RECORD_SCHEMA_INVALID")
            record["signature"] = self.observer_signer.sign(_canonical_record_payload(record))
            if not isinstance(record["signature"], str) or not record["signature"]:
                raise GuardFailure("R17_LEDGER_SIGNING_FAILURE")
            self._verify_records([*records, record])
            receipt_hash = hashlib.sha256(_canonical_record_payload(record)).hexdigest()
            conn.execute(
                "INSERT INTO r17_events(sequence,event_id,execution_id,phase,receipt_sha256,record_json) VALUES(?,?,?,?,?,?)",
                (sequence + 1, record["event_id"], execution_id, phase, receipt_hash,
                 json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
            )
            conn.execute(
                "UPDATE r17_ledger_meta SET sequence=?,head_sha256=? WHERE singleton=1",
                (sequence + 1, receipt_hash),
            )
            conn.execute("COMMIT")
        except (sqlite3.DatabaseError, OSError) as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if isinstance(exc, GuardFailure):
                raise
            raise GuardFailure("R17_LEDGER_APPEND_FAILURE") from exc
        finally:
            conn.close()
        try:
            self._write_checkpoint({
                "schema_version": R17_LEDGER_SCHEMA, "ledger_id": self.ledger_id,
                "namespace": self.namespace, "sequence": sequence + 1,
                "head_sha256": receipt_hash,
            })
        except GuardFailure:
            self._poisoned = True
            raise
        _secure_private_sqlite_files(self.path)
        return record

    def append_lifecycle(
        self,
        *,
        authorization: ParsedRealAuthorization,
        operation: Any,
        execution_id: str,
        request_identity: str,
        initiating_record_sha256: str,
        frozen_input_sha256: str,
        behavior_observation_sha256: str,
        outcome_attestation_id: str,
        parent_execution_id: str | None,
        dimension_count_owner: bool,
        phase: str,
        reason_code: str | None = None,
    ) -> dict[str, Any]:
        if (authorization != self.authorization
                or self._authorized_operations.get(operation.execution_id) != operation):
            raise GuardFailure("R17_LEDGER_APPEND_OPERATION_NOT_AUTHORIZED")
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise GuardFailure("R17_LEDGER_CLOCK_MUST_BE_TIMEZONE_AWARE")
        record = {
            "schema_version": REAL_ACTIVITY_RECEIPT_SCHEMA_VERSION_R17,
            "epoch_kind": REAL_EPOCH_KIND,
            "event_id": str(uuid.uuid4()),
            "sequence": 0,
            "epoch_id": authorization.epoch_id,
            "authorization_decision_id": authorization.decision_id,
            "campaign_id": authorization.campaign_id,
            "interval_id": authorization.interval_id,
            "execution_id": execution_id,
            "timestamp": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dimension_id": operation.dimension_id,
            "surface_id": operation.surface_id,
            "operation_identity": operation.operation_identity,
            "source_commit": operation.source_commit,
            "source_tree": operation.source_tree,
            "process_identity": operation.process_identity,
            "request_identity": request_identity,
            "initiating_record_sha256": initiating_record_sha256,
            "frozen_input_sha256": frozen_input_sha256,
            "behavior_observation_sha256": behavior_observation_sha256,
            "outcome_attestation_id": outcome_attestation_id,
            "parent_execution_id": parent_execution_id,
            "correlation_id": request_identity,
            "dimension_count_owner": dimension_count_owner,
            "phase": phase,
            "reason_code": reason_code,
            "previous_receipt_sha256": "",
            "signer_id": self.observer_signer.signer_id,
            "signature": "",
        }
        return self._append(record)

    def bind_b0(self, record: Mapping[str, Any], authorization: ParsedRealAuthorization) -> None:
        _validate_single_boundary(authorization, record, "B0", self.trust)
        self._append_boundary("B0", record)

    def _append_boundary(self, boundary: str, record: Mapping[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute("SELECT record_json FROM r17_boundaries WHERE boundary=?", (boundary,)).fetchone()
            encoded = json.dumps(dict(record), sort_keys=True, separators=(",", ":"))
            if prior is not None:
                if prior[0] != encoded:
                    raise GuardFailure("R17_BOUNDARY_REPLAY_MISMATCH")
            else:
                if conn.execute("SELECT closed FROM r17_ledger_meta WHERE singleton=1").fetchone() != (0,):
                    raise GuardFailure("R17_LEDGER_CLOSED_OR_UNAVAILABLE")
                conn.execute("INSERT INTO r17_boundaries(boundary,record_json) VALUES(?,?)", (boundary, encoded))
            conn.execute("COMMIT")
        except (sqlite3.DatabaseError, OSError) as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if isinstance(exc, GuardFailure):
                raise
            raise GuardFailure("R17_BOUNDARY_PERSISTENCE_FAILURE") from exc
        finally:
            conn.close()

    def boundary_records(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT record_json FROM r17_boundaries ORDER BY CASE boundary WHEN 'B0' THEN 0 ELSE 1 END").fetchall()
            return [json.loads(row[0]) for row in rows]
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            raise GuardFailure("R17_BOUNDARY_READ_FAILURE") from exc
        finally:
            conn.close()

    def receipt_records(self) -> list[dict[str, Any]]:
        with self._io_lock():
            conn = self._connect()
            try:
                return self._events(conn)
            finally:
                conn.close()

    def execution_last_phase(self, execution_id: str) -> str | None:
        with self._io_lock():
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT phase FROM r17_events WHERE execution_id=? ORDER BY sequence DESC LIMIT 1",
                    (execution_id,),
                ).fetchone()
                return str(row[0]) if row else None
            except sqlite3.DatabaseError as exc:
                raise GuardFailure("R17_LEDGER_EXECUTION_STATE_READ_FAILURE") from exc
            finally:
                conn.close()

    def execution_is_started(self, execution_id: str) -> bool:
        state = self.verify()
        if state["closed"]:
            return False
        return self.execution_last_phase(execution_id) == "STARTED"

    def record_execution_recovery(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Persist an independent post-B1 termination attestation outside the receipt chain."""
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None or now.astimezone(timezone.utc) < _utc_timestamp(self.authorization.B1, "B1"):
            raise GuardFailure("R17_RECOVERY_REQUIRES_B1_OR_LATER")
        with self._io_lock():
            state = self._verify_locked()
            if state["closed"]:
                raise GuardFailure("R17_LEDGER_CLOSED_OR_UNAVAILABLE")
            execution_id = _validate_execution_recovery_record(
                self.authorization, record, self.trust, now=now,
            )
            if execution_id not in state["open_execution_ids"]:
                raise GuardFailure("R17_RECOVERY_EXECUTION_NOT_UNRESOLVED")
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                records = self._events(conn)
                sequence, head, last_by_execution = self._verify_records(records)
                if (last_by_execution.get(execution_id) in {None, "CLOSED"}
                        or record.get("ledger_head_sha256") != head
                        or record.get("last_receipt_sha256") != _receipt_hash_for_execution(records, execution_id)):
                    raise GuardFailure("R17_RECOVERY_LEDGER_BINDING_MISMATCH")
                conn.execute(
                    "INSERT INTO r17_execution_recoveries(recovery_id,execution_id,record_json) VALUES(?,?,?)",
                    (record["recovery_id"], execution_id,
                     json.dumps(dict(record), sort_keys=True, separators=(",", ":"))),
                )
                conn.execute("COMMIT")
            except (sqlite3.DatabaseError, OSError) as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                if isinstance(exc, GuardFailure):
                    raise
                raise GuardFailure("R17_RECOVERY_PERSISTENCE_FAILURE") from exc
            finally:
                conn.close()
            _secure_private_sqlite_files(self.path)
            return self._verify_locked()

    def close_campaign(
        self,
        b1_record: Mapping[str, Any],
        authorization: ParsedRealAuthorization,
        trust: R17TrustConfiguration,
        *,
        claim: Mapping[str, Any],
        reservation_store: SQLiteAuthorizationUseStore,
        identity_provider: RuntimeIdentityProvider,
        expected_process_identity: str,
    ) -> dict[str, Any]:
        if authorization != self.authorization:
            raise GuardFailure("R17_LEDGER_CLOSE_AUTHORIZATION_MISMATCH")
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise GuardFailure("R17_CAMPAIGN_CLOCK_MUST_BE_TIMEZONE_AWARE")
        now = now.astimezone(timezone.utc)
        if now < _utc_timestamp(authorization.B1, "B1"):
            raise GuardFailure("R17_B1_TEMPORAL_BOUNDARY_NOT_REACHED")
        if not isinstance(reservation_store, SQLiteAuthorizationUseStore):
            raise GuardFailure("R17_RESERVATION_STORE_REQUIRED_FOR_CLOSE")
        claimed_at = _utc_timestamp(claim.get("claimed_at"), "CLAIMED_AT")
        validate_authorization_use_claim(authorization, claim, trust, now=claimed_at)
        if reservation_store.reservation_for_claim(claim) != dict(claim):
            raise GuardFailure("R17_CLOSE_RESERVATION_READBACK_MISMATCH")
        try:
            if (identity_provider.source_identity() != (authorization.source_commit, authorization.source_tree)
                    or identity_provider.process_identity() != expected_process_identity
                    or not identity_provider.process_identity_alive(expected_process_identity)):
                raise GuardFailure("R17_CLOSE_RUNTIME_IDENTITY_MISMATCH")
        except Exception as exc:
            if isinstance(exc, GuardFailure):
                raise
            raise GuardFailure("R17_CLOSE_RUNTIME_IDENTITY_UNAVAILABLE") from exc
        boundaries = self.boundary_records()
        if len(boundaries) != 1 or boundaries[0].get("boundary") != "B0":
            raise GuardFailure("R17_B0_BOUNDARY_MISSING_OR_INVALID")
        validate_real_epoch_boundaries(authorization, [boundaries[0], dict(b1_record)], trust)
        state = self.verify()
        if state["open_execution_ids"]:
            raise GuardFailure("R17_B1_OPEN_EXECUTIONS")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            sequence, head, last_by_execution = self._verify_records(self._events(conn))
            recovered_ids = {
                row[0] for row in conn.execute(
                    "SELECT execution_id FROM r17_execution_recoveries"
                ).fetchall()
            }
            if any(phase != "CLOSED" and execution_id not in recovered_ids
                   for execution_id, phase in last_by_execution.items()):
                raise GuardFailure("R17_B1_OPEN_EXECUTIONS")
            closed_at = now.isoformat().replace("+00:00", "Z")
            conn.execute(
                "INSERT INTO r17_boundaries(boundary,record_json) VALUES('B1',?)",
                (json.dumps(dict(b1_record), sort_keys=True, separators=(",", ":")),),
            )
            conn.execute(
                "INSERT INTO r17_closure(singleton,closed_at,boundary_json,final_sequence,final_head_sha256) VALUES(1,?,?,?,?)",
                (closed_at, json.dumps(dict(b1_record), sort_keys=True, separators=(",", ":")), sequence, head),
            )
            conn.execute("UPDATE r17_ledger_meta SET closed=1 WHERE singleton=1")
            conn.execute("COMMIT")
            _secure_private_sqlite_files(self.path)
            return {"closed": True, "closed_at": closed_at, "final_sequence": sequence, "final_head_sha256": head}
        except (sqlite3.DatabaseError, OSError) as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if isinstance(exc, GuardFailure):
                raise
            raise GuardFailure("R17_LEDGER_CLOSE_FAILURE") from exc
        finally:
            conn.close()


def _validate_single_boundary(
    authorization: ParsedRealAuthorization,
    row: Mapping[str, Any],
    expected_boundary: str,
    verifier: RecordSignatureVerifier,
) -> None:
    required = {
        "schema_version", "receipt_id", "epoch_id", "campaign_id", "interval_id",
        "boundary", "timestamp", "source_commit", "source_tree", "signer_id", "signature",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise GuardFailure("REAL_BOUNDARY_RECEIPT_SCHEMA_INVALID")
    expected_timestamp = authorization.B0 if expected_boundary == "B0" else authorization.B1
    if row["schema_version"] != "r7r16-real-activity-boundary-1.0.0":
        raise GuardFailure("REAL_BOUNDARY_RECEIPT_SCHEMA_INVALID")
    if row["boundary"] != expected_boundary or row["timestamp"] != expected_timestamp:
        raise GuardFailure("REAL_BOUNDARY_RECEIPT_ORDER_OR_TIME_MISMATCH")
    if (row["epoch_id"], row["campaign_id"], row["interval_id"], row["source_commit"], row["source_tree"]) != (
        authorization.epoch_id, authorization.campaign_id, authorization.interval_id,
        authorization.source_commit, authorization.source_tree,
    ):
        raise GuardFailure("REAL_BOUNDARY_RECEIPT_AUTHORITY_MISMATCH")
    if not isinstance(row["receipt_id"], str) or not row["receipt_id"]:
        raise GuardFailure("REAL_BOUNDARY_RECEIPT_DUPLICATE_ID")
    _verify_signed_record(
        row, verifier,
        permitted_authority_classes=frozenset({"INTERVAL_BOUNDARY_AUTHORITY"}),
    )


def _validate_execution_recovery_record(
    authorization: ParsedRealAuthorization,
    record: Mapping[str, Any],
    verifier: RecordSignatureVerifier,
    *,
    now: datetime,
) -> str:
    required = {
        "schema_version", "recovery_id", "epoch_id", "authorization_decision_id",
        "campaign_id", "interval_id", "execution_id", "dimension_id", "surface_id",
        "operation_identity", "process_identity", "last_receipt_sha256",
        "ledger_head_sha256", "source_commit", "source_tree", "termination_kind",
        "observed_at", "signer_id", "signature",
    }
    if not isinstance(record, Mapping) or set(record) != required or record.get("schema_version") != R17_EXECUTION_RECOVERY_SCHEMA:
        raise GuardFailure("R17_RECOVERY_ATTESTATION_SCHEMA_INVALID")
    if record.get("termination_kind") != "PROCESS_TERMINATED":
        raise GuardFailure("R17_RECOVERY_TERMINATION_KIND_INVALID")
    op = next((row for row in authorization.operations if row.execution_id == record.get("execution_id")), None)
    if op is None or (
        record.get("epoch_id"), record.get("authorization_decision_id"), record.get("campaign_id"),
        record.get("interval_id"), record.get("dimension_id"), record.get("surface_id"),
        record.get("operation_identity"), record.get("process_identity"), record.get("source_commit"),
        record.get("source_tree"),
    ) != (
        authorization.epoch_id, authorization.decision_id, authorization.campaign_id,
        authorization.interval_id, op.dimension_id, op.surface_id, op.operation_identity,
        op.process_identity, authorization.source_commit, authorization.source_tree,
    ):
        raise GuardFailure("R17_RECOVERY_AUTHORITY_BINDING_MISMATCH")
    if (not isinstance(record.get("recovery_id"), str) or not record["recovery_id"]
            or any(not _HEX_256.fullmatch(str(record.get(field, ""))) for field in (
                "last_receipt_sha256", "ledger_head_sha256",
            ))):
        raise GuardFailure("R17_RECOVERY_EVIDENCE_IDENTITY_INVALID")
    observed = _utc_timestamp(record.get("observed_at"), "RECOVERY_OBSERVED_AT")
    if now.tzinfo is None or now.utcoffset() is None:
        raise GuardFailure("R17_CAMPAIGN_CLOCK_MUST_BE_TIMEZONE_AWARE")
    if observed < _utc_timestamp(authorization.B1, "B1") or observed > now.astimezone(timezone.utc):
        raise GuardFailure("R17_RECOVERY_OUTSIDE_POST_B1_OBSERVATION_WINDOW")
    _verify_signed_record(
        record, verifier,
        permitted_authority_classes=frozenset({"STATE_TRANSITION_OBSERVER_AUTHORITY"}),
    )
    return op.execution_id


R17_GOVERNED_CALL_TOPOLOGY = {
    "schema_version": R17_GOVERNED_CALL_TOPOLOGY_SCHEMA,
    "surfaces": {
        "candidate-launch": {"dimension_id": "CANDIDATE_STARTED", "primitive": "tools.launch_isolated_candidate.execute_candidate_launch", "operation_identity": "systemd-run-candidate"},
        "benchmark-runner": {"dimension_id": "BENCHMARK_TRAFFIC_SENT", "primitive": "tools.run_real_corpus_baseline._run_case", "operation_identity": "benchmark-case-post-chat"},
        "scorer": {"dimension_id": "SCORER_RERUN", "primitive": "tools.score_real_corpus_results._emit_score_reports", "operation_identity": "emit-deterministic-score-reports"},
        "eval-runner-request": {"dimension_id": "REAL_SUBJECT_EXECUTION_COUNT", "primitive": "eval_runner.ask_api", "operation_identity": "tenant-subject-post-chat"},
        "chat-api-request": {"dimension_id": "REAL_SUBJECT_EXECUTION_COUNT", "primitive": "chat_api._guarded_chat_operation", "operation_identity": "candidate-runtime-chat-endpoint"},
        "webui-streamlit-request": {"dimension_id": "REAL_SUBJECT_EXECUTION_COUNT", "primitive": "webui_streamlit.module_chat_post", "operation_identity": "streamlit-user-post-chat", "source_path": "webui_streamlit.py", "source_symbol": "<module-level-guarded-lambda>"},
    },
    "edges": {
        ("eval-runner-request", "chat-api-request"): {
            "kind": "SAME_LOGICAL_SUBJECT_REQUEST",
            "dimension_relation": "SAME",
            "dimension_count_owner": "PARENT",
        },
        ("webui-streamlit-request", "chat-api-request"): {
            "kind": "SAME_LOGICAL_SUBJECT_REQUEST",
            "dimension_relation": "SAME",
            "dimension_count_owner": "PARENT",
        },
        ("benchmark-runner", "chat-api-request"): {
            "kind": "BENCHMARK_DISPATCH_TO_SUBJECT_EXECUTION",
            "dimension_relation": "DIFFERENT",
            "dimension_count_owner": "BOTH",
        },
    },
    "semantics": {
        "same_dimension_nested_call": "record both surface lifecycles; only outer client owns the logical subject count",
        "cross_dimension_nested_call": "each surface owns its distinct dimension count and child receipt binds to parent execution",
        "candidate_launch": "launcher is a terminal process-start surface; later campaign operations are correlated by campaign authorization, not falsely made lexical children",
        "scorer": "terminal report-emission surface; no nested governed surface identified",
        "direct_chat_api": "when directly authorized with no parent, chat API owns the subject count",
    },
}


def validate_r17_dimension_activity_attestations(
    authorization: ParsedRealAuthorization,
    receipt_validation: Mapping[str, Any],
    effect_validation: Mapping[str, Any],
    attestations: Sequence[Mapping[str, Any]],
    verifier: RecordSignatureVerifier,
) -> dict[str, Any]:
    """Validate dimension-specific positive events; ATTEMPTED is never evidence."""
    records = receipt_validation.get("receipt_records")
    effects = effect_validation.get("effects_by_execution")
    effect_records = effect_validation.get("effect_records")
    ledger_head = receipt_validation.get("ledger_head_sha256")
    if (not isinstance(records, list) or not isinstance(effects, Mapping)
            or not isinstance(effect_records, list) or not isinstance(ledger_head, str)):
        raise GuardFailure("R17_ACTIVITY_RECEIPT_OR_EFFECT_AUTHORITY_MISSING")
    effect_times = {
        str(record.get("execution_id")): _utc_timestamp(record.get("timestamp"), "ACTIVITY_EFFECT_TIMESTAMP")
        for record in effect_records if isinstance(record, Mapping)
    }
    operations = {op.execution_id: op for op in authorization.operations}
    phases: dict[str, list[str]] = {}
    by_execution: dict[str, list[Mapping[str, Any]]] = {}
    dimension_owners: set[str] = set()
    for receipt in records:
        if not isinstance(receipt, Mapping):
            raise GuardFailure("R17_ACTIVITY_RECEIPT_SCHEMA_INVALID")
        execution_id = str(receipt.get("execution_id", ""))
        phases.setdefault(execution_id, []).append(str(receipt.get("phase", "")))
        by_execution.setdefault(execution_id, []).append(receipt)
        if receipt.get("dimension_count_owner") is True:
            dimension_owners.add(execution_id)
    seen_activity_ids: set[str] = set()
    positive_by_execution: dict[str, dict[str, Any]] = {}
    covered_started_ids: set[str] = set()
    for row in attestations:
        required = {
            "schema_version", "activity_id", "epoch_id", "authorization_decision_id",
            "campaign_id", "interval_id", "execution_id", "dimension_id", "surface_id",
            "operation_identity", "event_kind", "evidence_execution_ids", "evidence_sha256",
            "ledger_head_sha256", "source_commit", "source_tree", "process_identity",
            "candidate_process_identity", "request_identity", "timestamp", "signer_id", "signature",
        }
        if (not isinstance(row, Mapping) or set(row) != required
                or row.get("schema_version") != R17_DIMENSION_ACTIVITY_SCHEMA):
            raise GuardFailure("R17_ACTIVITY_ATTESTATION_SCHEMA_INVALID")
        activity_id = row.get("activity_id")
        execution_id = row.get("execution_id")
        op = operations.get(execution_id)
        if (not isinstance(activity_id, str) or not activity_id or activity_id in seen_activity_ids
                or not isinstance(execution_id, str) or execution_id in positive_by_execution or op is None):
            raise GuardFailure("R17_ACTIVITY_ATTESTATION_DUPLICATE_OR_UNKNOWN_EXECUTION")
        seen_activity_ids.add(activity_id)
        contract = R17_DIMENSION_ACTIVITY_EVENTS.get(op.dimension_id)
        authority_binding = (
            row.get("epoch_id"), row.get("authorization_decision_id"), row.get("campaign_id"),
            row.get("interval_id"), row.get("dimension_id"), row.get("surface_id"),
            row.get("operation_identity"), row.get("event_kind"), row.get("ledger_head_sha256"),
            row.get("source_commit"), row.get("source_tree"), row.get("process_identity"),
            row.get("request_identity"),
        )
        expected_binding = (
            authorization.epoch_id, authorization.decision_id, authorization.campaign_id,
            authorization.interval_id, op.dimension_id, op.surface_id, op.operation_identity,
            contract["event_kind"] if contract else None, ledger_head, authorization.source_commit,
            authorization.source_tree, op.process_identity, op.request_identity,
        )
        if contract is None or authority_binding != expected_binding:
            raise GuardFailure("R17_ACTIVITY_ATTESTATION_OPERATION_BINDING_MISMATCH")
        if execution_id not in dimension_owners or "STARTED" not in phases.get(execution_id, []):
            raise GuardFailure("R17_ACTIVITY_WITHOUT_STARTED_DIMENSION_OWNER")
        if "BLOCKED" in phases.get(execution_id, []):
            raise GuardFailure("R17_BLOCKED_EXECUTION_CANNOT_HAVE_POSITIVE_ACTIVITY")
        evidence = row.get("evidence_execution_ids")
        if (not isinstance(evidence, list) or not evidence
                or any(not isinstance(item, str) for item in evidence)
                or evidence != sorted(set(evidence))
                or any(item not in operations for item in evidence)):
            raise GuardFailure("R17_ACTIVITY_EVIDENCE_EXECUTIONS_INVALID")
        evidence_receipts: list[Mapping[str, Any]] = []
        for evidence_id in evidence:
            evidence_op = operations[evidence_id]
            evidence_phases = phases.get(evidence_id, [])
            if (evidence_op.surface_id != contract["evidence_surface_id"]
                    or "STARTED" not in evidence_phases or "BLOCKED" in evidence_phases):
                raise GuardFailure("R17_ACTIVITY_EVIDENCE_SURFACE_OR_LIFECYCLE_MISMATCH")
            evidence_receipts.extend(
                receipt for receipt in by_execution[evidence_id]
                if receipt.get("phase") == "STARTED"
            )
        if op.dimension_id == "REAL_SUBJECT_EXECUTION_COUNT":
            if op.surface_id == "chat-api-request":
                if evidence != [execution_id]:
                    raise GuardFailure("R17_SUBJECT_ACTIVITY_DIRECT_BINDING_INVALID")
            else:
                if len(evidence) != 1:
                    raise GuardFailure("R17_SUBJECT_ACTIVITY_CHILD_BINDING_INVALID")
                child_id = evidence[0]
                edge = R17_GOVERNED_CALL_TOPOLOGY["edges"].get((op.surface_id, "chat-api-request"))
                child_receipts = by_execution[child_id]
                if (edge is None or edge["dimension_relation"] != "SAME"
                        or any(item.get("parent_execution_id") != execution_id for item in child_receipts)
                        or any(item.get("dimension_count_owner") is not False for item in child_receipts)):
                    raise GuardFailure("R17_SUBJECT_ACTIVITY_CAUSAL_CHILD_INVALID")
        elif evidence != [execution_id]:
            raise GuardFailure("R17_ACTIVITY_SINGLE_SURFACE_BINDING_INVALID")
        if not _HEX_256.fullmatch(str(row.get("evidence_sha256", ""))):
            raise GuardFailure("R17_ACTIVITY_EVIDENCE_DIGEST_INVALID")
        actual_evidence_sha = hashlib.sha256(canonical_json_bytes(evidence_receipts)).hexdigest()
        if actual_evidence_sha != row.get("evidence_sha256"):
            raise GuardFailure("R17_ACTIVITY_EVIDENCE_DIGEST_MISMATCH")
        if op.dimension_id == "CANDIDATE_STARTED":
            candidate_identity = row.get("candidate_process_identity")
            if (not isinstance(candidate_identity, str)
                    or not _LINUX_PROCESS_IDENTITY.fullmatch(candidate_identity)):
                raise GuardFailure("R17_CANDIDATE_RUNTIME_IDENTITY_MISSING")
        elif row.get("candidate_process_identity") is not None:
            raise GuardFailure("R17_UNRELATED_CANDIDATE_IDENTITY_PRESENT")
        if (not isinstance(effects.get(evidence[0]), Mapping)
                or effects[evidence[0]].get("effect_kind") != "STATE_TRANSITION"):
            raise GuardFailure("R17_ACTIVITY_LACKS_INDEPENDENT_STATE_TRANSITION")
        timestamp = _utc_timestamp(row.get("timestamp"), "ACTIVITY_TIMESTAMP")
        if not (_utc_timestamp(authorization.B0, "B0") < timestamp < _utc_timestamp(authorization.B1, "B1")):
            raise GuardFailure("R17_ACTIVITY_ATTESTATION_OUTSIDE_B0_B1")
        if any(evidence_id not in effect_times or effect_times[evidence_id] > timestamp for evidence_id in evidence):
            raise GuardFailure("R17_ACTIVITY_PRECEDES_INDEPENDENT_EFFECT_OBSERVATION")
        if any(
            timestamp < _utc_timestamp(start_receipt["timestamp"], "ACTIVITY_START_RECEIPT_TIMESTAMP")
            for evidence_id in evidence for start_receipt in by_execution[evidence_id]
            if start_receipt.get("phase") == "STARTED"
        ):
            raise GuardFailure("R17_ACTIVITY_ATTESTATION_PRECEDES_OPERATION_START")
        _verify_signed_record(
            row, verifier,
            permitted_authority_classes=frozenset({"STATE_TRANSITION_OBSERVER_AUTHORITY"}),
        )
        positive_by_execution[execution_id] = {
            "activity_id": activity_id,
            "execution_id": execution_id,
            "dimension_id": op.dimension_id,
            "event_kind": contract["event_kind"],
            "activity_sha256": hashlib.sha256(_canonical_record_payload(row)).hexdigest(),
            "candidate_process_identity": row.get("candidate_process_identity"),
        }
        covered_started_ids.add(execution_id)
        covered_started_ids.update(evidence)
    return {
        "schema_version": R17_DIMENSION_ACTIVITY_SCHEMA,
        "positive_by_execution": {key: positive_by_execution[key] for key in sorted(positive_by_execution)},
        "covered_started_execution_ids": sorted(covered_started_ids),
        "attestation_count": len(positive_by_execution),
    }


def derive_r17_dimension_activity_row(
    *,
    dimension_id: str,
    operation_execution_ids: Sequence[str],
    phases_by_execution: Mapping[str, Sequence[str]],
    attempted_execution_ids: set[str],
    observed_execution_ids: Sequence[str],
    covered_started_execution_ids: set[str],
    no_effect_execution_ids: set[str],
    census_coverage_complete: bool,
) -> dict[str, Any]:
    """Pure R17 activity rule used only after independent evidence validation."""
    started = {
        execution_id for execution_id in operation_execution_ids
        if "STARTED" in phases_by_execution.get(execution_id, ())
    }
    observed = sorted(set(observed_execution_ids))
    unresolved_started = sorted(started - covered_started_execution_ids)
    complete = bool(census_coverage_complete and not unresolved_started)
    blocked_attempts = {
        execution_id for execution_id in operation_execution_ids
        if execution_id in attempted_execution_ids
        and "BLOCKED" in phases_by_execution.get(execution_id, ())
    }
    attempted_nonblocked = any(
        execution_id in attempted_execution_ids
        and "BLOCKED" not in phases_by_execution.get(execution_id, ())
        for execution_id in operation_execution_ids
    )
    if observed:
        status = "POSITIVE_OBSERVED"
    elif (complete and not started and not attempted_nonblocked
          and blocked_attempts.issubset(no_effect_execution_ids)):
        status = "ZERO_WITH_COMPLETE_COVERAGE"
    else:
        status = "UNKNOWN"
    return {
        "dimension_id": dimension_id,
        "status": status,
        "observed_execution_ids": observed,
        "coverage_complete": complete,
        "census_coverage_complete": bool(census_coverage_complete),
        "unclassified_started_execution_ids": unresolved_started,
    }


def derive_r17_repository_activity_state(
    authorization: ParsedRealAuthorization | ConsumedRealAuthorization,
    receipt_validation: Mapping[str, Any],
    coverage_rows: list[dict[str, Any]],
    verifier: RecordSignatureVerifier,
    *,
    effect_attestations: list[dict[str, Any]],
    dimension_activity_attestations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """R17 successor: lifecycle attempts are retained but never prove activity."""
    if (not isinstance(receipt_validation, Mapping)
            or not isinstance(receipt_validation.get("boundary_records"), list)
            or not isinstance(receipt_validation.get("receipt_records"), list)):
        raise GuardFailure("REAL_RECEIPT_LEDGER_RAW_EVIDENCE_MISSING")
    prior_receipt_ids = receipt_validation.get("prior_receipt_ids", [])
    if (not isinstance(prior_receipt_ids, list)
            or any(not isinstance(item, str) or not item for item in prior_receipt_ids)):
        raise GuardFailure("REAL_RECEIPT_LEDGER_PRIOR_IDS_INVALID")
    parsed = authorization.authorization if isinstance(authorization, ConsumedRealAuthorization) else authorization
    # Reconstruct the receipt facts from signed raw evidence.  In particular,
    # do not let caller-supplied summary flags become the R17 zero proposition.
    validated_receipts = validate_real_activity_receipts(
        authorization,
        list(receipt_validation.get("boundary_records", [])),
        list(receipt_validation.get("receipt_records", [])),
        verifier,
        prior_receipt_ids=set(prior_receipt_ids),
    )
    if validated_receipts["ledger_head_sha256"] != receipt_validation.get("ledger_head_sha256"):
        raise GuardFailure("REAL_RECEIPT_LEDGER_AGGREGATE_MISMATCH")
    validated_receipts["authorized_operations"] = parsed.operations
    base = derive_real_repository_activity_state(
        authorization, dict(validated_receipts), coverage_rows, verifier,
        effect_attestations=effect_attestations,
    )
    effects = validate_real_effect_attestations(
        authorization, validated_receipts, effect_attestations, verifier,
    )
    activity = validate_r17_dimension_activity_attestations(
        parsed, validated_receipts, effects, dimension_activity_attestations, verifier,
    )
    phases: dict[str, list[str]] = {}
    for record in validated_receipts["receipt_records"]:
        phases.setdefault(record["execution_id"], []).append(record["phase"])
    positive = activity["positive_by_execution"]
    covered_started = set(activity["covered_started_execution_ids"])
    no_effect_ids = {
        execution_id for execution_id, effect in effects["effects_by_execution"].items()
        if effect.get("effect_kind") == "NO_EFFECT_VERIFIED"
    }
    base_rows = {row["dimension_id"]: row for row in base["repository_dimension_rows"]}
    result_rows: list[dict[str, Any]] = []
    for dimension in sorted(base_rows):
        dimension_ops = {op.execution_id: op for op in parsed.operations if op.dimension_id == dimension}
        observed = sorted(
            execution_id for execution_id, evidence in positive.items()
            if evidence["dimension_id"] == dimension
        )
        result_rows.append(derive_r17_dimension_activity_row(
            dimension_id=dimension,
            operation_execution_ids=sorted(dimension_ops),
            phases_by_execution=phases,
            attempted_execution_ids=set(validated_receipts["attempted_execution_ids"]),
            observed_execution_ids=observed,
            covered_started_execution_ids=covered_started,
            no_effect_execution_ids=no_effect_ids,
            census_coverage_complete=bool(base_rows[dimension]["coverage_complete"]),
        ))
    all_complete = all(row["coverage_complete"] for row in result_rows)
    zero_valid = (all_complete and bool(result_rows)
                  and all(row["status"] == "ZERO_WITH_COMPLETE_COVERAGE" for row in result_rows)
                  and isinstance(authorization, ConsumedRealAuthorization)
                  and parsed.live_authority_qualified
                  and base["repository_coverage_complete"]
                  and validated_receipts["ledger_chain_valid"] is True
                  and validated_receipts["lifecycle_complete"] is True
                  and effects["complete"] is True)
    return {
        **base,
        "schema_version": "r7r17-repository-activity-derivation-1.0.0",
        "repository_dimension_rows": result_rows,
        "repository_coverage_complete": all_complete,
        "repository_scope_zero_authority_valid": zero_valid,
        # Five external dimensions remain deliberately unavailable in R17.
        "FULL_9_DIMENSION_OBSERVABILITY": False,
        "ZERO_ACTIVITY_AUTHORITY_VALID": False,
        "R7_TO_R11_ZERO_ACTIVITY_RETROACTIVE_AUTHORITY": "UNPROVABLE",
        "dimension_activity_attestations": activity,
    }


@dataclass(frozen=True)
class PreparedCampaign:
    """Signed authorization plus a persisted one-time claim, prepared pre-B0."""

    authorization_document: Mapping[str, Any]
    expected_scope: RealAuthorizationScope
    claim: Mapping[str, Any]


@dataclass(frozen=True)
class R17IntervalBCompositionContext:
    """Raw A-side evidence required to revalidate composed interval B at attach."""

    interval_a: PreparedCampaign
    candidate_activity_attestations: Sequence[Mapping[str, Any]]
    interval_a_ledger: SQLiteActivityLedger
    interval_a_effect_attestations: Sequence[Mapping[str, Any]]


def read_campaign_authorization(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardFailure("R17_CAMPAIGN_AUTHORIZATION_UNREADABLE") from exc
    if not isinstance(value, dict):
        raise GuardFailure("R17_CAMPAIGN_AUTHORIZATION_SCHEMA_INVALID")
    return value


def prepare_campaign_authorization(
    authorization_document: Mapping[str, Any],
    expected_scope: RealAuthorizationScope,
    *,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
    now: datetime,
) -> PreparedCampaign:
    """Verify and atomically consume ALLOW before B0; no target can run here."""
    if not isinstance(trust, R17TrustConfiguration):
        raise GuardFailure("R17_TRUST_CONFIGURATION_REQUIRED")
    if not isinstance(reservation_store, SQLiteAuthorizationUseStore):
        raise GuardFailure("R17_RESERVATION_STORE_REQUIRED")
    if trust.test_only:
        raise GuardFailure("R17_TEST_ONLY_TRUST_CANNOT_PREPARE_REAL_CAMPAIGN")
    if not trust.readiness()["real_activation_eligible"]:
        raise GuardFailure("R17_NONTEST_TRUST_REQUIRED_TO_PREPARE_REAL_CAMPAIGN")
    if now.tzinfo is None or now.utcoffset() is None:
        raise GuardFailure("R17_ACTIVATION_CLOCK_MUST_BE_TIMEZONE_AWARE")
    authorization = parse_real_epoch_authorization(
        dict(authorization_document), expected_scope, trust, now=now,
    )
    if now.astimezone(timezone.utc) >= _utc_timestamp(authorization.B0, "B0"):
        raise GuardFailure("R17_RESERVATION_MUST_PRECEDE_B0")
    consumed = consume_real_epoch_authorization(
        authorization, reservation_store, trust, now=now,
    )
    stored = reservation_store.reservation_for_claim(consumed.consumption_claim)
    if stored != dict(consumed.consumption_claim):
        raise GuardFailure("R17_RESERVATION_CLAIM_NOT_DURABLY_STORED")
    return PreparedCampaign(
        authorization_document=dict(authorization_document),
        expected_scope=expected_scope,
        claim=dict(consumed.consumption_claim),
    )


def r17_future_campaign_plan(*, campaign_id: str, source_commit: str, source_tree: str) -> dict[str, Any]:
    """Describe A→B requirements without guessing or authorizing the future PID."""
    if (not campaign_id or not re.fullmatch(r"[0-9a-f]{40,64}", source_commit)
            or not re.fullmatch(r"[0-9a-f]{40,64}", source_tree)):
        raise GuardFailure("R17_CAMPAIGN_COMPOSITION_PLAN_IDENTITY_INVALID")
    launch_pair = ["candidate-launch", "CANDIDATE_STARTED"]
    all_pairs = sorted(
        [surface, dimension]
        for surface, details in R7R13_GUARD_REGISTRY.items()
        for dimension in details["dimension_ids"]
    )
    return {
        "schema_version": R17_CAMPAIGN_COMPOSITION_SCHEMA,
        "campaign_id": campaign_id,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "state": "WAITING_FOR_INTERVAL_A_CANDIDATE_IDENTITY_BINDING",
        "candidate_process_identity": None,
        "intervals": [
            {"phase": "A", "purpose": "CANDIDATE_LAUNCH", "required_surface_pairs": [launch_pair]},
            {"phase": "B", "purpose": "POST_LAUNCH_CAMPAIGN", "required_surface_pairs": [pair for pair in all_pairs if pair != launch_pair],
             "authorization_must_be_issued_after_candidate_binding": True,
             "claim_must_be_created_after_candidate_binding": True,
             "authorization_and_claim_must_be_after_interval_a_closure": True,
             "authorization_and_claim_must_precede_interval_b_b0": True},
        ],
        "required_surface_pairs": all_pairs,
        "prelaunch_pid_guess_used": False,
    }


def _r17_required_surface_pairs() -> set[tuple[str, str]]:
    return {
        (surface, dimension)
        for surface, details in R7R13_GUARD_REGISTRY.items()
        for dimension in details["dimension_ids"]
    }


def _is_r17_interval_b_footprint(authorization: ParsedRealAuthorization) -> bool:
    """Recognize the exact R7–R11 B pair partition, without trusting a caller flag."""
    launch_pair = ("candidate-launch", "CANDIDATE_STARTED")
    actual_pairs = {(op.surface_id, op.dimension_id) for op in authorization.operations}
    return actual_pairs == _r17_required_surface_pairs() - {launch_pair}


def _is_r19_interval_b_footprint(authorization: ParsedRealAuthorization) -> bool:
    launch_pair = ("candidate-launch", "CANDIDATE_STARTED")
    scorer_pair = ("scorer", "SCORER_RERUN")
    actual_pairs = {(op.surface_id, op.dimension_id) for op in authorization.operations}
    return actual_pairs == _r17_required_surface_pairs() - {launch_pair, scorer_pair}


def _is_r19_interval_c_footprint(authorization: ParsedRealAuthorization) -> bool:
    actual_pairs = {(op.surface_id, op.dimension_id) for op in authorization.operations}
    return actual_pairs == {("scorer", "SCORER_RERUN")} and len(authorization.operations) == 1


def _consumed_prepared_interval(
    prepared: PreparedCampaign,
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
) -> tuple[ParsedRealAuthorization, ConsumedRealAuthorization]:
    authorization = parse_real_epoch_authorization(
        dict(prepared.authorization_document), prepared.expected_scope, trust,
        now=_utc_timestamp(prepared.expected_scope.B0, "B0"),
    )
    claim = dict(prepared.claim)
    claimed_at = _utc_timestamp(claim.get("claimed_at"), "CLAIMED_AT")
    if claimed_at >= _utc_timestamp(authorization.B0, "B0"):
        raise GuardFailure("R17_COMPOSITION_CLAIM_NOT_PRE_B0")
    digest = validate_authorization_use_claim(authorization, claim, trust, now=claimed_at)
    if reservation_store.reservation_for_claim(claim) != claim:
        raise GuardFailure("R17_COMPOSITION_RESERVATION_READBACK_MISMATCH")
    return authorization, ConsumedRealAuthorization(
        authorization=authorization,
        claim_id=str(claim["claim_id"]),
        reservation_id=str(claim["reservation_id"]),
        consumption_receipt_sha256=digest,
        store_identity=str(claim["store_identity"]),
        consumption_claim=claim,
    )


def validate_r17_campaign_composition(
    interval_a: PreparedCampaign,
    interval_b: PreparedCampaign,
    *,
    candidate_activity_attestations: Sequence[Mapping[str, Any]],
    interval_a_ledger: SQLiteActivityLedger,
    interval_a_effect_attestations: list[dict[str, Any]],
    trust: R17TrustConfiguration,
    reservation_store: SQLiteAuthorizationUseStore,
) -> dict[str, Any]:
    """Prove ordered launch→observed identity→fresh exact B authorization composition."""
    auth_a, consumed_a = _consumed_prepared_interval(interval_a, trust, reservation_store)
    auth_b, _consumed_b = _consumed_prepared_interval(interval_b, trust, reservation_store)
    if (auth_a.campaign_id != auth_b.campaign_id
            or auth_a.source_commit != auth_b.source_commit
            or auth_a.source_tree != auth_b.source_tree
            or auth_a.interval_id == auth_b.interval_id
            or auth_a.decision_id == auth_b.decision_id
            or auth_a.nonce == auth_b.nonce
            or set(op.execution_id for op in auth_a.operations).intersection(op.execution_id for op in auth_b.operations)
            or _utc_timestamp(auth_a.B1, "A.B1") > _utc_timestamp(auth_b.B0, "B.B0")):
        raise GuardFailure("R17_CAMPAIGN_INTERVAL_COMPOSITION_INVALID")
    expected_pairs = _r17_required_surface_pairs()
    topology_pairs = {
        (surface, details["dimension_id"])
        for surface, details in R17_GOVERNED_CALL_TOPOLOGY["surfaces"].items()
    }
    if (topology_pairs != expected_pairs
            or any(
                details.get("primitive") != R7R13_GUARD_REGISTRY.get(surface, {}).get("primitive")
                for surface, details in R17_GOVERNED_CALL_TOPOLOGY["surfaces"].items()
            )):
        raise GuardFailure("R17_GOVERNED_SURFACE_TOPOLOGY_DIVERGENCE")
    pairs_a = {(op.surface_id, op.dimension_id) for op in auth_a.operations}
    pairs_b = {(op.surface_id, op.dimension_id) for op in auth_b.operations}
    launch_pair = {("candidate-launch", "CANDIDATE_STARTED")}
    if (pairs_a != launch_pair or len(auth_a.operations) != 1
            or pairs_b != expected_pairs - launch_pair
            or len(auth_b.operations) != len(expected_pairs - launch_pair)
            or pairs_a.intersection(pairs_b) or pairs_a.union(pairs_b) != expected_pairs):
        raise GuardFailure("R17_CAMPAIGN_COMPOSITION_SURFACE_COVERAGE_INVALID")
    for operation in (*auth_a.operations, *auth_b.operations):
        contract = R17_GOVERNED_CALL_TOPOLOGY["surfaces"].get(operation.surface_id)
        if (contract is None or operation.dimension_id != contract["dimension_id"]
                or operation.operation_identity != contract["primitive"]):
            raise GuardFailure("R17_COMPOSITION_OPERATION_PRIMITIVE_MISMATCH")
    candidate_operations = [op for op in auth_a.operations if op.surface_id == "candidate-launch"]
    if len(candidate_operations) != 1:
        raise GuardFailure("R17_COMPOSITION_CANDIDATE_LAUNCH_CARDINALITY_INVALID")
    if interval_a_ledger.authorization != auth_a:
        raise GuardFailure("R17_COMPOSITION_INTERVAL_A_LEDGER_AUTHORITY_MISMATCH")
    ledger_state = interval_a_ledger.verify()
    closure = ledger_state.get("closure")
    if (ledger_state.get("integrity") != "PASS" or not ledger_state.get("closed")
            or not isinstance(closure, Mapping)
            or _utc_timestamp(closure.get("closed_at"), "A.CLOSED_AT") < _utc_timestamp(auth_a.B1, "A.B1")
            or closure.get("final_sequence") != ledger_state.get("record_count")
            or closure.get("final_head_sha256") != ledger_state.get("head_sha256")):
        raise GuardFailure("R17_COMPOSITION_INTERVAL_A_NOT_CLOSED_AFTER_B1")
    a_closed_at = _utc_timestamp(closure["closed_at"], "A.CLOSED_AT")
    interval_a_boundaries = interval_a_ledger.boundary_records()
    interval_a_receipts = interval_a_ledger.receipt_records()
    receipt_validation = validate_real_activity_receipts(
        consumed_a, interval_a_boundaries, interval_a_receipts, trust,
    )
    effects = validate_real_effect_attestations(
        consumed_a, receipt_validation, interval_a_effect_attestations, trust,
    )
    activity = validate_r17_dimension_activity_attestations(
        auth_a, receipt_validation, effects, candidate_activity_attestations, trust,
    )
    launch_execution_id = candidate_operations[0].execution_id
    binding = activity["positive_by_execution"].get(launch_execution_id)
    if (binding is None or binding["event_kind"] != "CANDIDATE_PROCESS_STARTED"
            or not binding["candidate_process_identity"]):
        raise GuardFailure("R17_COMPOSITION_CANDIDATE_IDENTITY_BINDING_MISSING")
    candidate_identity = binding["candidate_process_identity"]
    activity_record = next(
        row for row in candidate_activity_attestations if row.get("execution_id") == launch_execution_id
    )
    binding_time = _utc_timestamp(activity_record["timestamp"], "CANDIDATE_BINDING_TIMESTAMP")
    b_claim_time = _utc_timestamp(interval_b.claim.get("claimed_at"), "B.CLAIMED_AT")
    b_issued = _utc_timestamp(interval_b.authorization_document.get("issued_at"), "B.ISSUED_AT")
    b_b0 = _utc_timestamp(auth_b.B0, "B.B0")
    if b_claim_time < a_closed_at or b_issued < a_closed_at:
        raise GuardFailure("R17_INTERVAL_B_AUTHORITY_PRECEDES_A_CLOSURE")
    if b_claim_time >= b_b0 or b_issued >= b_b0:
        raise GuardFailure("R17_INTERVAL_B_AUTHORITY_NOT_PRE_B0")
    if b_claim_time < binding_time or b_issued < binding_time:
        raise GuardFailure("R17_INTERVAL_B_AUTHORITY_PRECEDES_CANDIDATE_BINDING")
    chat_targets = [op for op in auth_b.operations if op.surface_id == "chat-api-request"]
    if not chat_targets or any(op.process_identity != candidate_identity for op in chat_targets):
        raise GuardFailure("R17_INTERVAL_B_CANDIDATE_IDENTITY_MISMATCH")
    return {
        "schema_version": R17_CAMPAIGN_COMPOSITION_SCHEMA,
        "status": "READY_FOR_INTERVAL_B",
        "campaign_id": auth_a.campaign_id,
        "source_identity": {"commit": auth_a.source_commit, "tree": auth_a.source_tree},
        "ordered_intervals": [
            {"interval_id": auth_a.interval_id, "B0": auth_a.B0, "B1": auth_a.B1,
             "decision_id": auth_a.decision_id, "authorization_sha256": auth_a.authorization_sha256,
             "claim_id": consumed_a.claim_id, "reservation_id": consumed_a.reservation_id,
             "reservation_store_identity": consumed_a.store_identity,
             "claim_sha256": hashlib.sha256(_canonical_record_payload(interval_a.claim)).hexdigest(),
             "issued_at": interval_a.authorization_document.get("issued_at"),
             "claimed_at": interval_a.claim.get("claimed_at"),
             "authorized_operations": [
                 {"execution_id": op.execution_id, "surface_id": op.surface_id,
                  "dimension_id": op.dimension_id, "operation_identity": op.operation_identity,
                  "process_identity": op.process_identity}
                 for op in sorted(auth_a.operations, key=lambda item: item.execution_id)
             ],
             "authorized_surface_pairs": sorted([list(pair) for pair in pairs_a])},
             {"interval_id": auth_b.interval_id, "B0": auth_b.B0, "B1": auth_b.B1,
             "decision_id": auth_b.decision_id, "authorization_sha256": auth_b.authorization_sha256,
             "claim_id": _consumed_b.claim_id, "reservation_id": _consumed_b.reservation_id,
             "reservation_store_identity": _consumed_b.store_identity,
             "claim_sha256": hashlib.sha256(_canonical_record_payload(interval_b.claim)).hexdigest(),
             "issued_at": interval_b.authorization_document.get("issued_at"),
             "claimed_at": interval_b.claim.get("claimed_at"),
             "authorized_operations": [
                 {"execution_id": op.execution_id, "surface_id": op.surface_id,
                  "dimension_id": op.dimension_id, "operation_identity": op.operation_identity,
                  "process_identity": op.process_identity}
                 for op in sorted(auth_b.operations, key=lambda item: item.execution_id)
             ],
             "authorized_surface_pairs": sorted([list(pair) for pair in pairs_b])},
        ],
        "interval_a_closure": {
            "closed_at": closure["closed_at"],
            "final_sequence": closure["final_sequence"],
            "final_head_sha256": closure["final_head_sha256"],
        },
        "candidate_binding": {
            "launch_execution_id": launch_execution_id,
            "activity_id": binding["activity_id"],
            "activity_attestation_sha256": binding["activity_sha256"],
            "candidate_process_identity": candidate_identity,
            "binding_timestamp": activity_record["timestamp"],
            "signature_authority_class": "STATE_TRANSITION_OBSERVER_AUTHORITY",
        },
        "required_surface_pairs": sorted([list(pair) for pair in expected_pairs]),
        "all_required_surface_pairs_authorized_once": True,
        "candidate_identity_exactly_bound_in_interval_b": True,
        "prelaunch_pid_guess_used": False,
        "live_activity_performed_by_composition_check": False,
        "live_authority_qualified": auth_a.live_authority_qualified and auth_b.live_authority_qualified,
    }


@dataclass(frozen=True)
class InvocationBinding:
    execution_id: str
    request_identity: str
    parent_execution_id: str | None


class IndependentEffectAuthority(Protocol):
    def attest(self, request: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...


class IndependentNoEffectAuthority(Protocol):
    def prove_absence(self, request: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...


class IndependentCensusAuthority(Protocol):
    def enumerate(self, request: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...


class IndependentDimensionActivityAuthority(Protocol):
    """Attests the dimension-specific positive event, not a runtime attempt."""

    def attest_activity(self, request: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...


class R17CampaignRuntime:
    """Verified dispatcher installed only for one explicitly authorized interval."""

    def __init__(
        self,
        *,
        _activation_token: object,
        authorization: ParsedRealAuthorization,
        authorization_document: Mapping[str, Any],
        expected_scope: RealAuthorizationScope,
        claim: Mapping[str, Any],
        trust: R17TrustConfiguration,
        reservation_store: SQLiteAuthorizationUseStore,
        ledger: SQLiteActivityLedger,
        identity_provider: RuntimeIdentityProvider,
        b0_record: Mapping[str, Any],
        activated_at: datetime,
    ) -> None:
        if _activation_token is not _R17_ACTIVATION_TOKEN:
            raise GuardFailure("R17_RUNTIME_MUST_BE_CREATED_BY_VERIFIED_ACTIVATION")
        self.authorization = authorization
        self.authorization_document = dict(authorization_document)
        self.expected_scope = expected_scope
        self.claim = dict(claim)
        self.trust = trust
        self.reservation_store = reservation_store
        self.ledger = ledger
        self.identity_provider = identity_provider
        self.b0_record = dict(b0_record)
        self.activated_at = activated_at
        self._activated = True
        self._closed = False
        self._operations = {item.execution_id: item for item in authorization.operations}
        self._process_identity = identity_provider.process_identity()
        self._source_identity = identity_provider.source_identity()
        self._validate_configuration()

    @classmethod
    def attach_after_b0(
        cls,
        prepared: PreparedCampaign,
        *,
        composition_context: R17IntervalBCompositionContext | None = None,
        trust: R17TrustConfiguration,
        reservation_store: SQLiteAuthorizationUseStore,
        ledger: SQLiteActivityLedger,
        identity_provider: RuntimeIdentityProvider,
        b0_record: Mapping[str, Any],
        now: datetime,
    ) -> "R17CampaignRuntime":
        if not isinstance(trust, R17TrustConfiguration):
            raise GuardFailure("R17_TRUST_CONFIGURATION_REQUIRED")
        if not isinstance(reservation_store, SQLiteAuthorizationUseStore):
            raise GuardFailure("R17_RESERVATION_STORE_REQUIRED")
        if not isinstance(ledger, SQLiteActivityLedger):
            raise GuardFailure("R17_DURABLE_LEDGER_REQUIRED")
        if not all(callable(getattr(identity_provider, name, None)) for name in (
            "source_identity", "process_identity", "process_identity_alive",
        )):
            raise GuardFailure("R17_RUNTIME_IDENTITY_PROVIDER_REQUIRED")
        if now.tzinfo is None or now.utcoffset() is None:
            raise GuardFailure("R17_ACTIVATION_CLOCK_MUST_BE_TIMEZONE_AWARE")
        authorization = parse_real_epoch_authorization(
            dict(prepared.authorization_document), prepared.expected_scope, trust, now=now,
        )
        if (authorization.campaign_protocol is not None
                or _is_r19_interval_b_footprint(authorization)):
            raise GuardFailure("R19_STAGED_INTERVAL_REQUIRES_PROTOCOL_ATTACHER")
        if _is_r17_interval_b_footprint(authorization):
            if not isinstance(composition_context, R17IntervalBCompositionContext):
                raise GuardFailure("R17_INTERVAL_B_COMPOSITION_REQUIRED")
            if not isinstance(composition_context.interval_a_ledger, SQLiteActivityLedger):
                raise GuardFailure("R17_INTERVAL_A_DURABLE_LEDGER_REQUIRED")
            composition = validate_r17_campaign_composition(
                composition_context.interval_a, prepared,
                candidate_activity_attestations=list(composition_context.candidate_activity_attestations),
                interval_a_ledger=composition_context.interval_a_ledger,
                interval_a_effect_attestations=[dict(row) for row in composition_context.interval_a_effect_attestations],
                trust=trust, reservation_store=reservation_store,
            )
            if (composition.get("status") != "READY_FOR_INTERVAL_B"
                    or composition.get("ordered_intervals", [{}])[-1].get("authorization_sha256")
                    != authorization.authorization_sha256
                    or composition.get("ordered_intervals", [{}])[-1].get("interval_id")
                    != authorization.interval_id):
                raise GuardFailure("R17_INTERVAL_B_COMPOSITION_BINDING_MISMATCH")
        elif composition_context is not None:
            raise GuardFailure("R17_INTERVAL_B_COMPOSITION_CONTEXT_UNEXPECTED")
        return cls._activate_after_b0_verified(
            prepared, authorization, trust=trust, reservation_store=reservation_store,
            ledger=ledger, identity_provider=identity_provider, b0_record=b0_record,
            now=now,
        )

    @classmethod
    def _activate_r19_after_protocol_validation(
        cls,
        prepared: PreparedCampaign,
        authorization: ParsedRealAuthorization,
        *,
        protocol_validation_token: object,
        composition_context: Any,
        trust: R17TrustConfiguration,
        reservation_store: SQLiteAuthorizationUseStore,
        ledger: SQLiteActivityLedger,
        identity_provider: RuntimeIdentityProvider,
        b0_record: Mapping[str, Any],
        now: datetime,
    ) -> "R17CampaignRuntime":
        if protocol_validation_token is not _R19_ACTIVATION_TOKEN:
            raise GuardFailure("R19_PROTOCOL_VALIDATION_REQUIRED")
        reparsed = parse_real_epoch_authorization(
            dict(prepared.authorization_document), prepared.expected_scope,
            trust, now=now,
        )
        if (reparsed != authorization
                or authorization.campaign_protocol != R19_CAMPAIGN_COMPOSITION_SCHEMA
                or authorization.interval_role not in {"B", "C"}
                or (authorization.interval_role == "B" and not _is_r19_interval_b_footprint(authorization))
                or (authorization.interval_role == "C" and not _is_r19_interval_c_footprint(authorization))):
            raise GuardFailure("R19_PROTOCOL_AUTHORIZATION_SCOPE_INVALID")
        from r7r19_activity import (
            R19IntervalBCompositionContext,
            R19IntervalCCompositionContext,
            validate_r19_a_to_b,
            validate_r19_b_to_c,
        )

        realization_sha256 = None
        if authorization.interval_role == "B":
            if not isinstance(composition_context, R19IntervalBCompositionContext):
                raise GuardFailure("R19_INTERVAL_B_COMPOSITION_REQUIRED")
            composition = validate_r19_a_to_b(
                composition_context.interval_a, prepared,
                candidate_activity_attestations=composition_context.candidate_activity_attestations,
                interval_a_ledger=composition_context.interval_a_ledger,
                interval_a_effect_attestations=composition_context.interval_a_effect_attestations,
                trust=trust, reservation_store=reservation_store,
            )
            composition_sha256 = canonical_sha256(composition)
        else:
            if not isinstance(composition_context, R19IntervalCCompositionContext):
                raise GuardFailure("R19_INTERVAL_C_COMPOSITION_REQUIRED")
            composition = validate_r19_b_to_c(
                composition_context.interval_a, composition_context.interval_b, prepared,
                candidate_activity_attestations=composition_context.candidate_activity_attestations,
                interval_a_ledger=composition_context.interval_a_ledger,
                interval_a_effect_attestations=composition_context.interval_a_effect_attestations,
                interval_b_ledger=composition_context.interval_b_ledger,
                interval_b_effect_attestations=composition_context.interval_b_effect_attestations,
                interval_b_activity_attestations=composition_context.interval_b_activity_attestations,
                scorer_input_realization=composition_context.scorer_input_realization,
                b_to_c_authority=composition_context.b_to_c_authority,
                expected_result_artifact_path=composition_context.expected_result_artifact_path,
                trust=trust, reservation_store=reservation_store,
            )
            realization_sha256 = composition["realization_sha256"]
            composition_sha256 = composition["composition_sha256"]
        runtime = cls._activate_after_b0_verified(
            prepared, authorization, protocol_validation_token=protocol_validation_token,
            trust=trust, reservation_store=reservation_store, ledger=ledger,
            identity_provider=identity_provider, b0_record=b0_record, now=now,
        )
        runtime._r19_realization_sha256 = realization_sha256
        runtime._r19_composition_sha256 = composition_sha256
        return runtime

    @classmethod
    def _activate_r20_after_protocol_validation(
        cls,
        prepared: PreparedCampaign,
        authorization: ParsedRealAuthorization,
        *,
        composition_context: Any,
        trust: R17TrustConfiguration,
        reservation_store: SQLiteAuthorizationUseStore,
        ledger: SQLiteActivityLedger,
        identity_provider: RuntimeIdentityProvider,
        b0_record: Mapping[str, Any],
        now: datetime,
    ) -> "R17CampaignRuntime":
        from r7r20_activity import R20IntervalCCompositionContext
        from r7r13_observer import R20_CAMPAIGN_COMPOSITION_SCHEMA

        if (authorization.campaign_protocol != R20_CAMPAIGN_COMPOSITION_SCHEMA
                or authorization.interval_role != "C"):
            raise GuardFailure("R20_PROTOCOL_AUTHORIZATION_REQUIRED")
        if not isinstance(composition_context, R20IntervalCCompositionContext):
            raise GuardFailure("R20_C_COMPOSITION_CONTEXT_REQUIRED")
        return cls._activate_after_b0_verified(
            prepared, authorization, composition_context=composition_context,
            trust=trust, reservation_store=reservation_store, ledger=ledger,
            identity_provider=identity_provider, b0_record=b0_record, now=now,
        )

    @classmethod
    def _activate_after_b0_verified(
        cls,
        prepared: PreparedCampaign,
        authorization: ParsedRealAuthorization,
        *,
        protocol_validation_token: object | None = None,
        composition_context: Any | None = None,
        trust: R17TrustConfiguration,
        reservation_store: SQLiteAuthorizationUseStore,
        ledger: SQLiteActivityLedger,
        identity_provider: RuntimeIdentityProvider,
        b0_record: Mapping[str, Any],
        now: datetime,
    ) -> "R17CampaignRuntime":
        r20_composition: Mapping[str, Any] | None = None
        if authorization.campaign_protocol is not None:
            from r7r13_observer import R20_CAMPAIGN_COMPOSITION_SCHEMA

            if authorization.campaign_protocol == R19_CAMPAIGN_COMPOSITION_SCHEMA:
                if protocol_validation_token is not _R19_ACTIVATION_TOKEN:
                    raise GuardFailure("R17_PROTOCOL_VALIDATION_REQUIRED")
            elif authorization.campaign_protocol == R20_CAMPAIGN_COMPOSITION_SCHEMA:
                from r7r20_activity import R20IntervalCCompositionContext, validate_r20_b_to_c

                if protocol_validation_token is not None:
                    raise GuardFailure("R20_TOKEN_IS_NOT_COMPOSITION_AUTHORITY")
                if not isinstance(composition_context, R20IntervalCCompositionContext):
                    raise GuardFailure("R20_C_COMPOSITION_CONTEXT_REQUIRED")
                r20_composition = validate_r20_b_to_c(
                    composition_context.interval_a,
                    composition_context.interval_b,
                    prepared,
                    candidate_activity_attestations=composition_context.candidate_activity_attestations,
                    interval_a_ledger=composition_context.interval_a_ledger,
                    interval_a_effect_attestations=composition_context.interval_a_effect_attestations,
                    interval_b_ledger=composition_context.interval_b_ledger,
                    interval_b_effect_attestations=composition_context.interval_b_effect_attestations,
                    interval_b_activity_attestations=composition_context.interval_b_activity_attestations,
                    scorer_input_envelope=composition_context.scorer_input_envelope,
                    b_to_c_authority=composition_context.b_to_c_authority,
                    expected_result_artifact_path=composition_context.expected_result_artifact_path,
                    trust=trust,
                    reservation_store=reservation_store,
                )
            else:
                raise GuardFailure("R17_PROTOCOL_VALIDATION_REQUIRED")
        elif protocol_validation_token is not None:
            raise GuardFailure("R19_PROTOCOL_AUTHORIZATION_MISSING")
        elif composition_context is not None:
            raise GuardFailure("R17_UNBOUND_COMPOSITION_CONTEXT")
        if not isinstance(trust, R17TrustConfiguration):
            raise GuardFailure("R17_TRUST_CONFIGURATION_REQUIRED")
        if not isinstance(reservation_store, SQLiteAuthorizationUseStore):
            raise GuardFailure("R17_RESERVATION_STORE_REQUIRED")
        if not isinstance(ledger, SQLiteActivityLedger):
            raise GuardFailure("R17_DURABLE_LEDGER_REQUIRED")
        if not all(callable(getattr(identity_provider, name, None)) for name in (
            "source_identity", "process_identity", "process_identity_alive",
        )):
            raise GuardFailure("R17_RUNTIME_IDENTITY_PROVIDER_REQUIRED")
        if now.tzinfo is None or now.utcoffset() is None:
            raise GuardFailure("R17_ACTIVATION_CLOCK_MUST_BE_TIMEZONE_AWARE")
        if parse_real_epoch_authorization(
            dict(prepared.authorization_document), prepared.expected_scope, trust, now=now,
        ) != authorization:
            raise GuardFailure("R17_AUTHORIZATION_OBJECT_DIFFERS_FROM_SIGNED_DOCUMENT")
        if trust.test_only:
            raise GuardFailure("R17_TEST_ONLY_TRUST_CANNOT_ACTIVATE_REAL_CAMPAIGN")
        if not trust.readiness()["real_activation_eligible"]:
            raise GuardFailure("R17_NONTEST_TRUST_REQUIRED_TO_ACTIVATE_REAL_CAMPAIGN")
        claim = dict(prepared.claim)
        stored = reservation_store.reservation_for_claim(claim)
        if stored is None or stored != claim:
            raise GuardFailure("R17_RESERVATION_CLAIM_NOT_DURABLY_STORED")
        claimed_at = _utc_timestamp(claim.get("claimed_at"), "CLAIMED_AT")
        claim_digest = validate_authorization_use_claim(
            authorization, claim, trust, now=claimed_at,
        )
        _validate_single_boundary(authorization, b0_record, "B0", trust)
        now_utc = now.astimezone(timezone.utc)
        if now_utc < _utc_timestamp(authorization.B0, "B0"):
            raise GuardFailure("R17_CAMPAIGN_NOT_OPEN_AT_B0")
        if now_utc >= _utc_timestamp(authorization.B1, "B1"):
            raise GuardFailure("R17_CAMPAIGN_INTERVAL_ALREADY_CLOSED")
        if claim_digest != hashlib.sha256(_canonical_record_payload(claim)).hexdigest():
            raise GuardFailure("R17_RESERVATION_CLAIM_DIGEST_MISMATCH")
        if ledger.authorization != authorization or (ledger.namespace["epoch_id"], ledger.namespace["authorization_decision_id"], ledger.namespace["campaign_id"], ledger.namespace["interval_id"],
                ledger.namespace["B0"], ledger.namespace["B1"], ledger.namespace["source_commit"], ledger.namespace["source_tree"]) != (
                authorization.epoch_id, authorization.decision_id, authorization.campaign_id, authorization.interval_id,
                authorization.B0, authorization.B1, authorization.source_commit, authorization.source_tree):
            raise GuardFailure("R17_LEDGER_AUTHORIZATION_BINDING_MISMATCH")
        ledger.bind_b0(b0_record, authorization)
        state = ledger.verify()
        if state["closed"]:
            raise GuardFailure("R17_LEDGER_ALREADY_CLOSED")
        runtime = cls(
            _activation_token=_R17_ACTIVATION_TOKEN,
            authorization=authorization,
            authorization_document=prepared.authorization_document,
            expected_scope=prepared.expected_scope,
            claim=claim, trust=trust,
            reservation_store=reservation_store, ledger=ledger,
            identity_provider=identity_provider, b0_record=b0_record, activated_at=now,
        )
        runtime._validate_current_identity()
        runtime._validate_recovered_open_executions(state["open_execution_ids"])
        if r20_composition is not None:
            runtime._r20_realization_sha256 = r20_composition["r20_envelope_sha256"]
            runtime._r20_composition_sha256 = r20_composition["composition_sha256"]
            runtime._r20_result_artifact_path = r20_composition["result_artifact_path"]
        return runtime

    def _validate_configuration(self) -> None:
        trust_state = self.trust.readiness()
        if trust_state["test_only"] or not trust_state["real_activation_eligible"]:
            raise GuardFailure("R17_TRUST_CONFIGURATION_NOT_REAL_AUTHORITY")
        if set(R17_GOVERNED_CALL_TOPOLOGY["surfaces"]) != set(R7R13_GUARD_REGISTRY):
            raise GuardFailure("R17_ACCEPTED_SURFACE_REGISTRY_MISMATCH")
        for execution_id, operation in self._operations.items():
            surface = R17_GOVERNED_CALL_TOPOLOGY["surfaces"].get(operation.surface_id)
            registered = R7R13_GUARD_REGISTRY.get(operation.surface_id)
            if (surface is None or registered is None
                    or operation.dimension_id != surface["dimension_id"]
                    or operation.operation_identity != surface["primitive"]
                    or registered["primitive"] != surface["primitive"]
                    or operation.dimension_id not in registered["dimension_ids"]
                    or not _HEX_256.fullmatch(operation.initiating_record_sha256)
                    or not _HEX_256.fullmatch(operation.frozen_input_sha256)):
                raise GuardFailure(f"R17_OPERATION_SCOPE_NOT_ACCEPTED:{execution_id}")

    def _validate_current_identity(self) -> None:
        try:
            actual_source = self.identity_provider.source_identity()
            actual_process = self.identity_provider.process_identity()
        except Exception as exc:
            if isinstance(exc, GuardFailure):
                raise
            raise GuardFailure("R17_RUNTIME_IDENTITY_UNAVAILABLE") from exc
        if actual_source != (self.authorization.source_commit, self.authorization.source_tree):
            raise GuardFailure("R17_RUNTIME_SOURCE_IDENTITY_MISMATCH")
        if actual_process != self._process_identity:
            raise GuardFailure("R17_RUNTIME_PROCESS_IDENTITY_CHANGED")
        self._source_identity = actual_source

    def _validate_recovered_open_executions(self, execution_ids: Sequence[str]) -> None:
        """Accept open lifecycles only while their authorized process is live."""
        is_alive = getattr(self.identity_provider, "process_identity_alive", None)
        for execution_id in execution_ids:
            operation = self._operations.get(execution_id)
            if operation is None or not callable(is_alive) or not is_alive(operation.process_identity):
                raise GuardFailure("R17_CRASH_RECOVERY_OPEN_EXECUTIONS_UNRESOLVED")

    def assert_activated(self) -> None:
        if not self._activated or self._closed:
            raise GuardFailure("R17_CAMPAIGN_RUNTIME_NOT_ACTIVE")
        self._validate_current_identity()
        current = self.ledger.clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise GuardFailure("R17_CAMPAIGN_CLOCK_MUST_BE_TIMEZONE_AWARE")
        current_utc = current.astimezone(timezone.utc)
        if (current_utc < _utc_timestamp(self.authorization.B0, "B0")
                or current_utc >= _utc_timestamp(self.authorization.B1, "B1")):
            raise GuardFailure("R17_CAMPAIGN_INTERVAL_NOT_OPEN")
        revalidated_authorization = parse_real_epoch_authorization(
            self.authorization_document, self.expected_scope, self.trust, now=current_utc,
        )
        if revalidated_authorization != self.authorization:
            raise GuardFailure("R17_AUTHORIZATION_OBJECT_DIFFERS_FROM_SIGNED_DOCUMENT")
        claim_time = _utc_timestamp(self.claim.get("claimed_at"), "CLAIMED_AT")
        validate_authorization_use_claim(self.authorization, self.claim, self.trust, now=claim_time)
        if self.reservation_store.reservation_for_claim(self.claim) != self.claim:
            raise GuardFailure("R17_RESERVATION_READBACK_MISMATCH")
        state = self.ledger.verify()
        if state["closed"]:
            raise GuardFailure("R17_LEDGER_ALREADY_CLOSED")

    def _assert_closure_eligible(self) -> None:
        """Closure has a post-B1 clock rule, distinct from the half-open operation guard."""
        if not self._activated or self._closed:
            raise GuardFailure("R17_CAMPAIGN_RUNTIME_NOT_ACTIVE")
        current = self.ledger.clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise GuardFailure("R17_CAMPAIGN_CLOCK_MUST_BE_TIMEZONE_AWARE")
        current_utc = current.astimezone(timezone.utc)
        if current_utc < _utc_timestamp(self.authorization.B1, "B1"):
            raise GuardFailure("R17_B1_TEMPORAL_BOUNDARY_NOT_REACHED")
        self._validate_current_identity()
        # Authenticate the signed authorization at its already-valid B0, so a
        # long recovery delay after B1 does not turn into an early-close bypass
        # or make post-B1 closure impossible when the campaign token expires.
        signed_authorization = parse_real_epoch_authorization(
            self.authorization_document, self.expected_scope, self.trust,
            now=_utc_timestamp(self.authorization.B0, "B0"),
        )
        if signed_authorization != self.authorization:
            raise GuardFailure("R17_AUTHORIZATION_OBJECT_DIFFERS_FROM_SIGNED_DOCUMENT")
        claimed_at = _utc_timestamp(self.claim.get("claimed_at"), "CLAIMED_AT")
        validate_authorization_use_claim(self.authorization, self.claim, self.trust, now=claimed_at)
        if self.reservation_store.reservation_for_claim(self.claim) != self.claim:
            raise GuardFailure("R17_CLOSE_RESERVATION_READBACK_MISMATCH")
        state = self.ledger.verify()
        if state["closed"]:
            raise GuardFailure("R17_LEDGER_ALREADY_CLOSED")

    def assert_can_deactivate(self) -> None:
        if not self._closed:
            raise GuardFailure("R17_DEACTIVATION_REQUIRES_B1_CLOSURE")

    @contextmanager
    def bind_authorized_execution(
        self,
        execution_id: str,
        *,
        request_identity: str | None = None,
        parent_execution_id: str | None = None,
    ) -> Iterator[InvocationBinding]:
        self.assert_activated()
        operation = self._operations.get(execution_id)
        if operation is None:
            raise GuardFailure("R17_EXECUTION_ID_NOT_AUTHORIZED")
        if request_identity is not None and request_identity != operation.request_identity:
            raise GuardFailure("R17_REQUEST_IDENTITY_MISMATCH")
        if operation.process_identity != self._process_identity:
            raise GuardFailure("R17_OPERATION_PROCESS_IDENTITY_MISMATCH")
        if parent_execution_id is not None:
            parent = self._operations.get(parent_execution_id)
            edge = R17_GOVERNED_CALL_TOPOLOGY["edges"].get((parent.surface_id, operation.surface_id)) if parent else None
            if (parent is None or edge is None or parent.request_identity != operation.request_identity
                    or parent.execution_id == operation.execution_id
                    or edge["dimension_relation"] != (
                        "SAME" if parent.dimension_id == operation.dimension_id else "DIFFERENT"
                    )):
                raise GuardFailure("R17_NESTED_CALL_CAUSAL_BINDING_INVALID")
            if not self.ledger.execution_is_started(parent_execution_id):
                raise GuardFailure("R17_PARENT_EXECUTION_NOT_STARTED")
        binding = InvocationBinding(execution_id, operation.request_identity, parent_execution_id)
        token = _CURRENT_BINDING.set(binding)
        try:
            yield binding
        finally:
            _CURRENT_BINDING.reset(token)

    @contextmanager
    def bind_from_headers(self, headers: Mapping[str, str]) -> Iterator[InvocationBinding]:
        execution_id = headers.get("X-Project-X-R17-Execution-ID") or ""
        parent_id = headers.get("X-Project-X-R17-Parent-Execution-ID") or None
        request_id = headers.get("X-Request-ID") or ""
        with self.bind_authorized_execution(
            execution_id, request_identity=request_id, parent_execution_id=parent_id,
        ) as binding:
            yield binding

    def child_headers(self, *, surface_id: str, dimension_id: str, operation_identity: str) -> dict[str, str]:
        binding = _CURRENT_BINDING.get()
        if binding is None:
            raise GuardFailure("R17_NESTED_REQUEST_PARENT_BINDING_MISSING")
        parent = self._operations.get(binding.execution_id)
        surface_contract = R17_GOVERNED_CALL_TOPOLOGY["surfaces"].get(surface_id)
        registered = R7R13_GUARD_REGISTRY.get(surface_id)
        edge = R17_GOVERNED_CALL_TOPOLOGY["edges"].get((parent.surface_id, surface_id)) if parent else None
        if (parent is None or surface_contract is None or registered is None or edge is None
                or dimension_id not in registered["dimension_ids"]
                or surface_contract["dimension_id"] != dimension_id
                or surface_contract["operation_identity"] != operation_identity
                or edge["dimension_relation"] != (
                    "SAME" if parent.dimension_id == dimension_id else "DIFFERENT"
                )):
            raise GuardFailure("R17_NESTED_REQUEST_EDGE_NOT_AUTHORIZED")
        matches = [row for row in self.authorization.operations
                   if (row.surface_id, row.dimension_id, row.operation_identity, row.request_identity)
                   == (surface_id, dimension_id, surface_contract["primitive"], binding.request_identity)]
        if len(matches) != 1:
            raise GuardFailure("R17_NESTED_REQUEST_AUTHORITY_AMBIGUOUS_OR_MISSING")
        child = matches[0]
        return {
            "X-Request-ID": child.request_identity,
            "X-Project-X-R17-Execution-ID": child.execution_id,
            "X-Project-X-R17-Parent-Execution-ID": parent.execution_id,
        }

    @staticmethod
    def _initiating_record_hash(
        *, execution_id: str, operation: Any, request_identity: str,
        parent_execution_id: str | None,
    ) -> str:
        return canonical_sha256({
            "schema_version": R17_INVOCATION_BINDING_SCHEMA,
            "execution_id": execution_id,
            "surface_id": operation.surface_id,
            "dimension_id": operation.dimension_id,
            "operation_identity": operation.operation_identity,
            "request_identity": request_identity,
            "parent_execution_id": parent_execution_id,
        })

    def _binding_and_operation(
        self, dimension: str, surface: str, operation_identity: str,
        r17_input: Mapping[str, Any] | None,
    ) -> tuple[InvocationBinding, Any, str, str, bool, str, str]:
        self.assert_activated()
        binding = _CURRENT_BINDING.get()
        if binding is None:
            raise GuardFailure("R17_EXECUTION_BINDING_REQUIRED")
        operation = self._operations.get(binding.execution_id)
        if operation is None:
            raise GuardFailure("R17_EXECUTION_ID_NOT_AUTHORIZED")
        surface_contract = R17_GOVERNED_CALL_TOPOLOGY["surfaces"].get(surface)
        if (surface_contract is None or operation.dimension_id != dimension
                or operation.surface_id != surface
                or operation.operation_identity != surface_contract["primitive"]
                or surface_contract["operation_identity"] != operation_identity):
            raise GuardFailure("R17_SURFACE_DIMENSION_OPERATION_MISMATCH")
        if not isinstance(r17_input, Mapping):
            raise GuardFailure("R17_INVOCATION_INPUT_EVIDENCE_MISSING")
        initiating_hash = self._initiating_record_hash(
            execution_id=binding.execution_id, operation=operation,
            request_identity=binding.request_identity,
            parent_execution_id=binding.parent_execution_id,
        )
        frozen_hash = canonical_sha256(dict(r17_input))
        if initiating_hash != operation.initiating_record_sha256:
            raise GuardFailure("R17_INITIATING_RECORD_BINDING_MISMATCH")
        if frozen_hash != operation.frozen_input_sha256:
            raise GuardFailure("R17_FROZEN_INPUT_BINDING_MISMATCH")
        if operation.process_identity != self._process_identity:
            raise GuardFailure("R17_OPERATION_PROCESS_IDENTITY_MISMATCH")
        edge = None
        if binding.parent_execution_id:
            parent = self._operations[binding.parent_execution_id]
            edge = R17_GOVERNED_CALL_TOPOLOGY["edges"].get((parent.surface_id, surface))
            if (edge is None
                    or edge["dimension_relation"] != (
                        "SAME" if parent.dimension_id == operation.dimension_id else "DIFFERENT"
                    )
                    or not self.ledger.execution_is_started(parent.execution_id)):
                raise GuardFailure("R17_NESTED_CALL_CAUSAL_BINDING_INVALID")
        owns_dimension = binding.parent_execution_id is None or edge["dimension_count_owner"] != "PARENT"
        behavior_hash = canonical_sha256({
            "observation": "PRE_TARGET_INVOCATION_CONTEXT",
            "execution_id": binding.execution_id,
            "request_identity": binding.request_identity,
            "source_commit": operation.source_commit,
            "source_tree": operation.source_tree,
            "frozen_input_sha256": frozen_hash,
        })
        outcome_id = canonical_sha256(["r17-outcome-slot", self.authorization.epoch_id, binding.execution_id])
        return binding, operation, initiating_hash, frozen_hash, owns_dimension, behavior_hash, outcome_id

    def execute(
        self,
        *,
        dimension: str,
        surface: str,
        operation: str,
        target: Callable[[], Any],
        policy_allows: bool,
        r17_input: Mapping[str, Any] | None,
    ) -> Any:
        binding, operation_authority, initiating_hash, frozen_hash, owner, behavior_hash, outcome_id = self._binding_and_operation(
            dimension, surface, operation, r17_input,
        )
        self.ledger.append_lifecycle(
            authorization=self.authorization, operation=operation_authority,
            execution_id=binding.execution_id, request_identity=binding.request_identity,
            initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
            behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
            parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
            phase="ATTEMPTED",
        )
        if not policy_allows:
            self.ledger.append_lifecycle(
                authorization=self.authorization, operation=operation_authority,
                execution_id=binding.execution_id, request_identity=binding.request_identity,
                initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
                behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
                parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
                phase="BLOCKED", reason_code="POLICY_PROHIBITS_OPERATION",
            )
            self._close_lifecycle(binding, operation_authority, initiating_hash, frozen_hash, behavior_hash, outcome_id, owner)
            raise GuardFailure("POLICY_PROHIBITS_OPERATION")
        for phase in ("AUTHORIZED", "STARTED"):
            self.ledger.append_lifecycle(
                authorization=self.authorization, operation=operation_authority,
                execution_id=binding.execution_id, request_identity=binding.request_identity,
                initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
                behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
                parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
                phase=phase,
            )
        try:
            result = target()
        except BaseException as exc:
            self.ledger.append_lifecycle(
                authorization=self.authorization, operation=operation_authority,
                execution_id=binding.execution_id, request_identity=binding.request_identity,
                initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
                behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
                parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
                phase="FAILED", reason_code=f"TARGET_EXCEPTION:{type(exc).__name__}",
            )
            self._close_lifecycle(binding, operation_authority, initiating_hash, frozen_hash, behavior_hash, outcome_id, owner)
            raise
        self.ledger.append_lifecycle(
            authorization=self.authorization, operation=operation_authority,
            execution_id=binding.execution_id, request_identity=binding.request_identity,
            initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
            behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
            parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
            phase="COMPLETED",
        )
        self._close_lifecycle(binding, operation_authority, initiating_hash, frozen_hash, behavior_hash, outcome_id, owner)
        return result

    async def execute_async(
        self,
        *,
        dimension: str,
        surface: str,
        operation: str,
        target: Callable[[], Any],
        policy_allows: bool,
        r17_input: Mapping[str, Any] | None,
    ) -> Any:
        binding, operation_authority, initiating_hash, frozen_hash, owner, behavior_hash, outcome_id = self._binding_and_operation(
            dimension, surface, operation, r17_input,
        )
        self.ledger.append_lifecycle(
            authorization=self.authorization, operation=operation_authority,
            execution_id=binding.execution_id, request_identity=binding.request_identity,
            initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
            behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
            parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
            phase="ATTEMPTED",
        )
        if not policy_allows:
            self.ledger.append_lifecycle(
                authorization=self.authorization, operation=operation_authority,
                execution_id=binding.execution_id, request_identity=binding.request_identity,
                initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
                behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
                parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
                phase="BLOCKED", reason_code="POLICY_PROHIBITS_OPERATION",
            )
            self._close_lifecycle(binding, operation_authority, initiating_hash, frozen_hash, behavior_hash, outcome_id, owner)
            raise GuardFailure("POLICY_PROHIBITS_OPERATION")
        for phase in ("AUTHORIZED", "STARTED"):
            self.ledger.append_lifecycle(
                authorization=self.authorization, operation=operation_authority,
                execution_id=binding.execution_id, request_identity=binding.request_identity,
                initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
                behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
                parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
                phase=phase,
            )
        try:
            result = await target()
        except BaseException as exc:
            self.ledger.append_lifecycle(
                authorization=self.authorization, operation=operation_authority,
                execution_id=binding.execution_id, request_identity=binding.request_identity,
                initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
                behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
                parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
                phase="FAILED", reason_code=f"TARGET_EXCEPTION:{type(exc).__name__}",
            )
            self._close_lifecycle(binding, operation_authority, initiating_hash, frozen_hash, behavior_hash, outcome_id, owner)
            raise
        self.ledger.append_lifecycle(
            authorization=self.authorization, operation=operation_authority,
            execution_id=binding.execution_id, request_identity=binding.request_identity,
            initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
            behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
            parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
            phase="COMPLETED",
        )
        self._close_lifecycle(binding, operation_authority, initiating_hash, frozen_hash, behavior_hash, outcome_id, owner)
        return result

    def _close_lifecycle(self, binding: InvocationBinding, operation: Any, initiating_hash: str,
                         frozen_hash: str, behavior_hash: str, outcome_id: str, owner: bool) -> None:
        for phase in ("CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLOSED"):
            self.ledger.append_lifecycle(
                authorization=self.authorization, operation=operation,
                execution_id=binding.execution_id, request_identity=binding.request_identity,
                initiating_record_sha256=initiating_hash, frozen_input_sha256=frozen_hash,
                behavior_observation_sha256=behavior_hash, outcome_attestation_id=outcome_id,
                parent_execution_id=binding.parent_execution_id, dimension_count_owner=owner,
                phase=phase,
            )

    def preflight(self) -> dict[str, Any]:
        self.assert_activated()
        state = self.ledger.verify()
        operation_pairs = {(row.surface_id, row.dimension_id) for row in self.authorization.operations}
        accepted_pairs = {
            (surface, dimension)
            for surface, details in R7R13_GUARD_REGISTRY.items()
            for dimension in details["dimension_ids"]
        }
        unsupported_pairs = operation_pairs - accepted_pairs
        return {
            "schema_version": "r7r17-campaign-preflight-1.1.0",
            "readiness_scope": "AUTHORIZED_INTERVAL_ONLY",
            "authorization_signature_valid": True,
            "reservation_persisted_and_valid": True,
            "source_identity_matches": self._source_identity == (self.authorization.source_commit, self.authorization.source_tree),
            "process_identity": self._process_identity,
            "B0_open": True,
            "B1_reached": False,
            "ledger_integrity": state["integrity"],
            "ledger_open_executions": state["open_execution_ids"],
            "accepted_surface_pairs": sorted([list(pair) for pair in operation_pairs]),
            "missing_authorized_surface_pairs": [],
            "extra_authorized_surface_pairs": sorted([list(pair) for pair in unsupported_pairs]),
            "campaign_composition_required": True,
            "campaign_composition_verified": False,
            "campaign_ready": False,
            "external_provider_availability_complete": False,
            "FULL_9_DIMENSION_OBSERVABILITY": False,
            "ZERO_ACTIVITY_AUTHORITY_VALID": False,
            "ready": bool(operation_pairs) and not state["open_execution_ids"] and not unsupported_pairs,
        }

    def close_at_b1(self, b1_record: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_closure_eligible()
        result = self.ledger.close_campaign(
            b1_record, self.authorization, self.trust,
            claim=self.claim, reservation_store=self.reservation_store,
            identity_provider=self.identity_provider,
            expected_process_identity=self._process_identity,
        )
        self._closed = True
        return result

    def reconcile_independent_evidence(
        self,
        *,
        effect_authority: IndependentEffectAuthority | None,
        no_effect_authority: IndependentNoEffectAuthority | None,
        census_authority: IndependentCensusAuthority | None,
        dimension_activity_authority: IndependentDimensionActivityAuthority | None = None,
    ) -> dict[str, Any]:
        if not self._closed:
            raise GuardFailure("R17_RECONCILIATION_REQUIRES_B1_CLOSURE")
        if (effect_authority is None or no_effect_authority is None or census_authority is None
                or dimension_activity_authority is None):
            return {
                "status": "UNKNOWN",
                "reason": "INDEPENDENT_EFFECT_ACTIVITY_OR_CENSUS_AUTHORITY_MISSING",
                "FULL_9_DIMENSION_OBSERVABILITY": False,
                "ZERO_ACTIVITY_AUTHORITY_VALID": False,
                "R7_TO_R11_ZERO_ACTIVITY_RETROACTIVE_AUTHORITY": "UNPROVABLE",
            }
        boundaries = self.ledger.boundary_records()
        receipts = self.ledger.receipt_records()
        state = self.ledger.verify()
        request = {
            "schema_version": "r7r17-independent-evidence-request-1.0.0",
            "authorization": {
                "decision_id": self.authorization.decision_id,
                "epoch_id": self.authorization.epoch_id,
                "campaign_id": self.authorization.campaign_id,
                "interval_id": self.authorization.interval_id,
                "B0": self.authorization.B0,
                "B1": self.authorization.B1,
                "source_commit": self.authorization.source_commit,
                "source_tree": self.authorization.source_tree,
            },
            "authorized_execution_ids": sorted(self._operations),
            "boundaries": boundaries,
            "receipts": receipts,
            "ledger_head_sha256": state["head_sha256"],
        }
        try:
            state_attestations = list(effect_authority.attest(request))
            no_effect_attestations = list(no_effect_authority.prove_absence(request))
            attestations = state_attestations + no_effect_attestations
            activity_attestations = list(dimension_activity_authority.attest_activity(request))
            census_request = {
                **request,
                "independent_effect_attestations": attestations,
                "independent_dimension_activity_attestations": activity_attestations,
            }
            coverage_rows = list(census_authority.enumerate(census_request))
            if (any(not isinstance(row, Mapping) or row.get("effect_kind") != "STATE_TRANSITION"
                    for row in state_attestations)
                    or any(not isinstance(row, Mapping) or row.get("effect_kind") != "NO_EFFECT_VERIFIED"
                           for row in no_effect_attestations)):
                raise GuardFailure("R17_INDEPENDENT_EFFECT_CLASSIFICATION_INVALID")
            digest = validate_authorization_use_claim(
                self.authorization, self.claim, self.trust,
                now=_utc_timestamp(self.claim.get("claimed_at"), "CLAIMED_AT"),
            )
            consumed = ConsumedRealAuthorization(
                authorization=self.authorization,
                claim_id=str(self.claim["claim_id"]),
                reservation_id=str(self.claim["reservation_id"]),
                consumption_receipt_sha256=digest,
                store_identity=str(self.claim["store_identity"]),
                consumption_claim=dict(self.claim),
            )
            receipt_validation = validate_real_activity_receipts(
                consumed, boundaries, receipts, self.trust,
            )
            result = derive_r17_repository_activity_state(
                consumed, receipt_validation, coverage_rows, self.trust,
                effect_attestations=attestations,
                dimension_activity_attestations=activity_attestations,
            )
            return {
                "status": "REPOSITORY_EVIDENCE_COMPLETE" if result["repository_coverage_complete"] else "UNKNOWN",
                "reason": None if result["repository_coverage_complete"] else "INDEPENDENT_CENSUS_INCOMPLETE",
                "FULL_9_DIMENSION_OBSERVABILITY": result["FULL_9_DIMENSION_OBSERVABILITY"],
                "ZERO_ACTIVITY_AUTHORITY_VALID": result["ZERO_ACTIVITY_AUTHORITY_VALID"],
                "R7_TO_R11_ZERO_ACTIVITY_RETROACTIVE_AUTHORITY": "UNPROVABLE",
                "repository_activity": result,
                "effect_attestation_count": len(attestations),
                "dimension_activity_attestation_count": len(activity_attestations),
                "coverage_row_count": len(coverage_rows),
            }
        except GuardFailure as exc:
            # Invalid signatures/bindings are not absence; expose a closed
            # non-authoritative result without substituting runtime claims.
            return {
                "status": "INVALID",
                "reason": str(exc),
                "FULL_9_DIMENSION_OBSERVABILITY": False,
                "ZERO_ACTIVITY_AUTHORITY_VALID": False,
                "R7_TO_R11_ZERO_ACTIVITY_RETROACTIVE_AUTHORITY": "UNPROVABLE",
            }
        except (TypeError, ValueError, OSError) as exc:
            return {
                "status": "INVALID",
                "reason": f"R17_INDEPENDENT_ADAPTER_FAILURE:{type(exc).__name__}",
                "FULL_9_DIMENSION_OBSERVABILITY": False,
                "ZERO_ACTIVITY_AUTHORITY_VALID": False,
                "R7_TO_R11_ZERO_ACTIVITY_RETROACTIVE_AUTHORITY": "UNPROVABLE",
            }


def install_campaign_runtime(runtime: R17CampaignRuntime) -> None:
    from r7r13_observer import install_real_activity_dispatcher

    runtime.assert_activated()
    if runtime.trust.test_only:
        raise GuardFailure("R17_TEST_ONLY_RUNTIME_CANNOT_BE_INSTALLED")
    if not runtime.trust.readiness()["real_activation_eligible"]:
        raise GuardFailure("R17_NONTEST_TRUST_REQUIRED_TO_INSTALL_REAL_RUNTIME")
    install_real_activity_dispatcher(runtime)


def uninstall_campaign_runtime(runtime: R17CampaignRuntime) -> None:
    from r7r13_observer import uninstall_real_activity_dispatcher

    runtime.assert_can_deactivate()
    uninstall_real_activity_dispatcher(runtime)


def current_campaign_runtime() -> R17CampaignRuntime | None:
    from r7r13_observer import active_real_activity_dispatcher

    current = active_real_activity_dispatcher()
    return current if type(current) is R17CampaignRuntime else None


def r17_child_headers(*, surface_id: str, dimension_id: str, operation_identity: str) -> dict[str, str]:
    runtime = current_campaign_runtime()
    if runtime is None:
        return {}
    return runtime.child_headers(
        surface_id=surface_id, dimension_id=dimension_id, operation_identity=operation_identity,
    )


def r17_current_request_identity() -> str | None:
    binding = _CURRENT_BINDING.get()
    return binding.request_identity if binding is not None else None


def r17_current_execution_id() -> str | None:
    binding = _CURRENT_BINDING.get()
    return binding.execution_id if binding is not None else None


@contextmanager
def r17_authorized_invocation(
    *,
    dimension: str,
    surface: str,
    operation_identity: str,
    input_factory: Callable[[InvocationBinding], Mapping[str, Any]],
) -> Iterator[Mapping[str, Any] | None]:
    """Bind a unique signed operation by exact frozen input before dispatch.

    When already nested, preserve the explicit caller binding.  For a
    top-level accepted surface, try only same-process signed operations and
    accept exactly one whose complete source-input hash and initiation binding
    match.  No ID/name-only selection or fallback exists.
    """
    runtime = current_campaign_runtime()
    if runtime is None:
        yield None
        return
    runtime.assert_activated()
    current = _CURRENT_BINDING.get()
    if current is not None:
        snapshot = input_factory(current)
        if not isinstance(snapshot, Mapping):
            raise GuardFailure("R17_INVOCATION_INPUT_EVIDENCE_MISSING")
        yield dict(snapshot)
        return

    candidates = [
        operation for operation in runtime.authorization.operations
        if operation.dimension_id == dimension
        and operation.surface_id == surface
        and operation.operation_identity == R17_GOVERNED_CALL_TOPOLOGY["surfaces"].get(surface, {}).get("primitive")
        and operation.process_identity == runtime._process_identity
    ]
    matches: list[tuple[InvocationBinding, Mapping[str, Any]]] = []
    for operation in candidates:
        binding = InvocationBinding(operation.execution_id, operation.request_identity, None)
        token = _CURRENT_BINDING.set(binding)
        try:
            try:
                snapshot = input_factory(binding)
            except GuardFailure:
                continue
            if (isinstance(snapshot, Mapping)
                    and canonical_sha256(dict(snapshot)) == operation.frozen_input_sha256
                    and runtime._initiating_record_hash(
                        execution_id=operation.execution_id,
                        operation=operation,
                        request_identity=operation.request_identity,
                        parent_execution_id=None,
                    ) == operation.initiating_record_sha256):
                matches.append((binding, dict(snapshot)))
        finally:
            _CURRENT_BINDING.reset(token)
    if len(matches) != 1:
        raise GuardFailure(
            "R17_TOP_LEVEL_OPERATION_BINDING_" + ("NOT_FOUND" if not matches else "AMBIGUOUS")
        )
    binding, snapshot = matches[0]
    token = _CURRENT_BINDING.set(binding)
    try:
        yield snapshot
    finally:
        _CURRENT_BINDING.reset(token)


def r17_invocation_active() -> bool:
    return current_campaign_runtime() is not None


def reject_ungoverned_subject_request() -> None:
    """Keep legacy diagnostic-only HTTP paths outside active R17 campaigns."""
    if r17_invocation_active():
        raise GuardFailure("R17_UNGOVERNED_SUBJECT_REQUEST_REJECTED")
