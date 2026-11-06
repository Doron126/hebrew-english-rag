"""Campaign-local Ed25519 authority backed by the host's OpenSSL executable.

Private trust roots must stay outside review archives.  Their separately
exported public documents support evidence-only signature verification.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Iterator, Mapping

from r7r13_observer import GuardFailure, SignatureVerification
from r7r17_activity import (
    AUTHORITY_CLASSES,
    R17TrustConfiguration,
    TrustRoleConfiguration,
    canonical_json_bytes,
)


R22_PRIVATE_SCHEMA = "r7r22-private-trust-root-1.0.0"
R22_PUBLIC_SCHEMA = "r7r22-public-verification-1.0.0"
R22_ALGORITHM = "Ed25519"
_DOMAIN = b"Project-X/R7R22/Ed25519/v1\x00"
_HEX_ID = re.compile(r"[0-9a-f]{40,64}\Z")
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_FIELDS = {
    "schema_version", "algorithm", "campaign_id", "source_commit",
    "source_tree", "created_at", "roles", "public_package_sha256",
}
_PUBLIC_FIELDS = {
    "schema_version", "algorithm", "campaign_id", "source_commit",
    "source_tree", "created_at", "roles",
}
_PUBLIC_ROLE_FIELDS = {
    "authority_class", "authority_id", "signer_id",
    "fingerprint_sha256", "public_key_pem",
}
_APPROVED_OPENSSL = Path("/usr/bin/openssl")
_OPENSSL_CHILD_ENV = {"LANG": "C", "LC_ALL": "C"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_file_bytes(value: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"


def _openssl_binary() -> str:
    """Validate the one reviewed executable and its non-replaceable path."""
    try:
        path = _APPROVED_OPENSSL.resolve(strict=True)
        if path != _APPROVED_OPENSSL:
            raise GuardFailure("R22_OPENSSL_APPROVED_EXECUTABLE_UNAVAILABLE")
        for parent in reversed(path.parents):
            info = parent.lstat()
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0
                    or info.st_mode & 0o022):
                raise GuardFailure("R22_OPENSSL_APPROVED_EXECUTABLE_UNAVAILABLE")
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0
                or info.st_mode & 0o022 or not info.st_mode & 0o111):
            raise GuardFailure("R22_OPENSSL_APPROVED_EXECUTABLE_UNAVAILABLE")
    except (OSError, RuntimeError) as exc:
        raise GuardFailure("R22_OPENSSL_APPROVED_EXECUTABLE_UNAVAILABLE") from exc
    return str(path)


def _openssl(*args: str, input_bytes: bytes | None = None, pass_fds: tuple[int, ...] = ()) -> bytes:
    try:
        result = subprocess.run(
            [_openssl_binary(), *args], input=input_bytes, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, stdin=subprocess.DEVNULL if input_bytes is None else None,
            pass_fds=pass_fds, check=False, timeout=15, env=dict(_OPENSSL_CHILD_ENV),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardFailure("R22_OPENSSL_EXECUTION_FAILED") from exc
    if result.returncode != 0:
        raise GuardFailure("R22_OPENSSL_OPERATION_FAILED")
    return result.stdout


def openssl_runtime_identity() -> dict[str, object]:
    """Describe the actual approved executable; never sign this host evidence."""
    path = Path(_openssl_binary())
    executable_hash = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                executable_hash.update(block)
    except OSError as exc:
        raise GuardFailure("R22_OPENSSL_APPROVED_EXECUTABLE_UNAVAILABLE") from exc
    version = _openssl("version").decode("ascii", errors="replace").strip()
    capable = False
    try:
        private_key = _openssl("genpkey", "-algorithm", "ED25519")
        with _message_fd(private_key) as key_fd:
            public_key = _openssl(
                "pkey", "-in", f"/proc/self/fd/{key_fd}", "-pubout",
                pass_fds=(key_fd,),
            )
            _openssl("pkey", "-pubin", "-inform", "PEM", "-outform", "DER",
                     input_bytes=public_key)
            with _message_fd(b"Project-X/R22/OpenSSL-capability") as message_fd:
                signature = _openssl(
                    "pkeyutl", "-sign", "-rawin", "-inkey", f"/proc/self/fd/{key_fd}",
                    "-in", f"/proc/self/fd/{message_fd}",
                    pass_fds=(key_fd, message_fd),
                )
        with (_message_fd(public_key) as public_fd,
              _message_fd(signature) as signature_fd,
              _message_fd(b"Project-X/R22/OpenSSL-capability") as message_fd):
            _openssl(
                "pkeyutl", "-verify", "-rawin", "-pubin",
                "-inkey", f"/proc/self/fd/{public_fd}",
                "-sigfile", f"/proc/self/fd/{signature_fd}",
                "-in", f"/proc/self/fd/{message_fd}",
                pass_fds=(public_fd, signature_fd, message_fd),
            )
        capable = len(signature) == 64
    except GuardFailure:
        pass
    info = path.stat()
    return {
        "canonical_executable_path": str(path),
        "version_output": version,
        "executable_sha256": executable_hash.hexdigest(),
        "owner_uid": info.st_uid,
        "mode": oct(stat.S_IMODE(info.st_mode)),
        "ed25519_capable": capable,
    }


def _absolute_canonical(path: Path) -> Path:
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise GuardFailure("R22_TRUST_PATH_NONCANONICAL_OR_SYMLINKED")
    return path


def _private_root_outside_repository(root: Path) -> None:
    """Keep private material outside both ordinary and linked Git worktrees."""
    for directory in (root, *root.parents):
        try:
            (directory / ".git").lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise GuardFailure("R22_TRUST_REPOSITORY_BOUNDARY_UNREADABLE") from exc
        raise GuardFailure("R22_PRIVATE_TRUST_ROOT_INSIDE_REPOSITORY")


def _secure_directory(path: Path) -> None:
    _absolute_canonical(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise GuardFailure("R22_TRUST_DIRECTORY_UNREADABLE") from exc
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_mode & 0o077 or not info.st_mode & stat.S_IXUSR):
        raise GuardFailure("R22_TRUST_DIRECTORY_UNSAFE")


def _secure_file(path: Path) -> None:
    _absolute_canonical(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise GuardFailure("R22_TRUST_FILE_UNREADABLE") from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        raise GuardFailure("R22_TRUST_FILE_UNSAFE")


def _read_secure_file(path: Path) -> bytes:
    _secure_file(path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise GuardFailure("R22_TRUST_FILE_OPEN_FAILED") from exc
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise GuardFailure("R22_TRUST_FILE_STATE_CHANGED")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


@contextmanager
def _private_fd(path: Path) -> Iterator[int]:
    _secure_file(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise GuardFailure("R22_PRIVATE_KEY_OPEN_FAILED") from exc
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise GuardFailure("R22_PRIVATE_KEY_STATE_CHANGED")
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _message_fd(message: bytes) -> Iterator[int]:
    # Ed25519 is a one-shot operation; OpenSSL's pkeyutl requires a seekable
    # input with a known size.  An anonymous Linux memfd avoids a disk copy.
    fd = None
    try:
        fd = os.memfd_create(
            "project-x-r22-signature-input", flags=getattr(os, "MFD_CLOEXEC", 0),
        )
        view = memoryview(message)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short memfd write")
            view = view[written:]
        os.lseek(fd, 0, os.SEEK_SET)
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise GuardFailure("R22_MESSAGE_BUFFER_UNAVAILABLE") from exc
    try:
        yield fd
    finally:
        os.close(fd)


def _public_from_private(path: Path) -> bytes:
    with _private_fd(path) as fd:
        return _openssl("pkey", "-in", f"/proc/self/fd/{fd}", "-pubout", pass_fds=(fd,))


def _public_der(pem: bytes) -> bytes:
    return _openssl("pkey", "-pubin", "-inform", "PEM", "-outform", "DER", input_bytes=pem)


def _identity(role: str, fingerprint: str) -> tuple[str, str]:
    return f"r22:{role}:{fingerprint}", f"r22:ed25519:{fingerprint}"


def _message(role: str, signer_id: str, campaign_id: str, source_commit: str,
             source_tree: str, payload: bytes) -> bytes:
    context = canonical_json_bytes({
        "role": role, "signer_id": signer_id, "campaign_id": campaign_id,
        "source_commit": source_commit, "source_tree": source_tree,
    })
    return _DOMAIN + context + b"\x00" + payload


@dataclass(frozen=True)
class OpenSSLRoleVerifier:
    authority_class: str
    authority_id: str
    signer_id: str
    fingerprint_sha256: str
    public_key_pem: str
    campaign_id: str
    source_commit: str
    source_tree: str
    root_path: Path | None = None

    def verify(self, signer_id: str, payload: bytes, signature: str) -> SignatureVerification:
        negative = SignatureVerification(False, self.authority_id, self.authority_class)
        if signer_id != self.signer_id or not isinstance(payload, bytes) or not isinstance(signature, str):
            return negative
        try:
            signature_bytes = base64.b64decode(signature, validate=True)
        except (ValueError, base64.binascii.Error):
            return negative
        if len(signature_bytes) != 64:
            return negative
        message = _message(
            self.authority_class, self.signer_id, self.campaign_id,
            self.source_commit, self.source_tree, payload,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="project-x-r22-verify-") as temporary:
                public_path = Path(temporary) / "public.pem"
                signature_path = Path(temporary) / "signature.bin"
                public_path.write_bytes(self.public_key_pem.encode("ascii"))
                signature_path.write_bytes(signature_bytes)
                with _message_fd(message) as message_fd:
                    _openssl(
                        "pkeyutl", "-verify", "-rawin", "-pubin", "-inkey", str(public_path),
                        "-sigfile", str(signature_path), "-in", f"/proc/self/fd/{message_fd}",
                        pass_fds=(message_fd,),
                    )
        except (GuardFailure, OSError, UnicodeError):
            return negative
        return SignatureVerification(True, self.authority_id, self.authority_class)


@dataclass(frozen=True)
class OpenSSLRoleSigner:
    authority_class: str
    authority_id: str
    signer_id: str
    fingerprint_sha256: str
    public_key_pem: str
    campaign_id: str
    source_commit: str
    source_tree: str
    private_key_path: Path
    root_path: Path

    def sign(self, payload: bytes) -> str:
        if not isinstance(payload, bytes):
            raise GuardFailure("R22_SIGNED_PAYLOAD_NOT_BYTES")
        _private_root_outside_repository(_absolute_canonical(self.root_path))
        _secure_directory(self.root_path)
        _secure_directory(self.root_path / "keys")
        if _public_from_private(self.private_key_path) != self.public_key_pem.encode("ascii"):
            raise GuardFailure("R22_PRIVATE_PUBLIC_KEY_MISMATCH")
        message = _message(
            self.authority_class, self.signer_id, self.campaign_id,
            self.source_commit, self.source_tree, payload,
        )
        with _private_fd(self.private_key_path) as fd, _message_fd(message) as message_fd:
            raw_signature = _openssl(
                "pkeyutl", "-sign", "-rawin", "-inkey", f"/proc/self/fd/{fd}",
                "-in", f"/proc/self/fd/{message_fd}", pass_fds=(fd, message_fd),
            )
        if len(raw_signature) != 64:
            raise GuardFailure("R22_ED25519_SIGNATURE_LENGTH_INVALID")
        return base64.b64encode(raw_signature).decode("ascii")


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise GuardFailure("R22_TRUST_FILE_ALREADY_EXISTS_OR_UNWRITABLE") from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _check_binding(campaign_id: str, source_commit: str, source_tree: str) -> None:
    if (not isinstance(campaign_id, str) or not campaign_id
            or not isinstance(source_commit, str) or not _HEX_ID.fullmatch(source_commit)
            or not isinstance(source_tree, str) or not _HEX_ID.fullmatch(source_tree)):
        raise GuardFailure("R22_TRUST_BINDING_INVALID")


def _validate_public(document: object, campaign_id: str, source_commit: str,
                     source_tree: str) -> list[dict[str, str]]:
    if not isinstance(document, dict) or set(document) != _PUBLIC_FIELDS:
        raise GuardFailure("R22_PUBLIC_PACKAGE_SCHEMA_INVALID")
    if (document["schema_version"] != R22_PUBLIC_SCHEMA
            or document["algorithm"] != R22_ALGORITHM
            or (document["campaign_id"], document["source_commit"], document["source_tree"])
            != (campaign_id, source_commit, source_tree)
            or not isinstance(document["created_at"], str)):
        raise GuardFailure("R22_PUBLIC_PACKAGE_BINDING_INVALID")
    rows = document["roles"]
    if not isinstance(rows, list) or len(rows) != len(AUTHORITY_CLASSES):
        raise GuardFailure("R22_PUBLIC_ROLE_SET_INVALID")
    by_role: dict[str, dict[str, str]] = {}
    fingerprints: set[str] = set()
    signers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _PUBLIC_ROLE_FIELDS:
            raise GuardFailure("R22_PUBLIC_ROLE_SCHEMA_INVALID")
        role = row["authority_class"]
        fingerprint = row["fingerprint_sha256"]
        if (role not in AUTHORITY_CLASSES or role in by_role
                or not isinstance(fingerprint, str) or not _HEX_256.fullmatch(fingerprint)
                or not isinstance(row["public_key_pem"], str)
                or not isinstance(row["signer_id"], str)
                or not isinstance(row["authority_id"], str)):
            raise GuardFailure("R22_PUBLIC_ROLE_IDENTITY_INVALID")
        try:
            public_bytes = row["public_key_pem"].encode("ascii")
            actual_fingerprint = _digest(_public_der(public_bytes))
        except (UnicodeError, GuardFailure) as exc:
            raise GuardFailure("R22_PUBLIC_KEY_INVALID") from exc
        if actual_fingerprint != fingerprint or _identity(role, fingerprint) != (
            row["authority_id"], row["signer_id"],
        ):
            raise GuardFailure("R22_PUBLIC_KEY_IDENTITY_MISMATCH")
        if fingerprint in fingerprints or row["signer_id"] in signers:
            raise GuardFailure("R22_PUBLIC_KEY_REUSED")
        by_role[role] = row
        fingerprints.add(fingerprint)
        signers.add(row["signer_id"])
    if set(by_role) != set(AUTHORITY_CLASSES):
        raise GuardFailure("R22_PUBLIC_ROLE_SET_INVALID")
    return [by_role[role] for role in AUTHORITY_CLASSES]


def generate_real_trust_root(root: Path, *, campaign_id: str, source_commit: str,
                             source_tree: str) -> tuple[Path, Path]:
    """Create one non-overwriting, campaign-bound six-key root and public export."""
    _check_binding(campaign_id, source_commit, source_tree)
    root = _absolute_canonical(Path(root))
    _private_root_outside_repository(root)
    if not root.parent.is_dir():
        raise GuardFailure("R22_TRUST_PARENT_MISSING")
    try:
        os.mkdir(root, 0o700)
        os.mkdir(root / "keys", 0o700)
    except OSError as exc:
        raise GuardFailure("R22_TRUST_ROOT_ALREADY_EXISTS_OR_UNWRITABLE") from exc
    _secure_directory(root)
    _secure_directory(root / "keys")
    rows = []
    private_rows = []
    for role in AUTHORITY_CLASSES:
        private_path = root / "keys" / f"{role.lower()}.pem"
        private_bytes = _openssl("genpkey", "-algorithm", "ED25519")
        _write_exclusive(private_path, private_bytes)
        public_pem = _public_from_private(private_path)
        fingerprint = _digest(_public_der(public_pem))
        authority_id, signer_id = _identity(role, fingerprint)
        row = {
            "authority_class": role, "authority_id": authority_id,
            "signer_id": signer_id, "fingerprint_sha256": fingerprint,
            "public_key_pem": public_pem.decode("ascii"),
        }
        rows.append(row)
        private_rows.append({**row, "private_key_ref": f"keys/{role.lower()}.pem"})
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    public = {
        "schema_version": R22_PUBLIC_SCHEMA, "algorithm": R22_ALGORITHM,
        "campaign_id": campaign_id, "source_commit": source_commit,
        "source_tree": source_tree, "created_at": created_at, "roles": rows,
    }
    public_bytes = _canonical_file_bytes(public)
    private = {
        "schema_version": R22_PRIVATE_SCHEMA, "algorithm": R22_ALGORITHM,
        "campaign_id": campaign_id, "source_commit": source_commit,
        "source_tree": source_tree, "created_at": created_at,
        "roles": private_rows, "public_package_sha256": _digest(public_bytes),
    }
    _write_exclusive(root / "public-verification.json", public_bytes)
    _write_exclusive(root / "trust-root.json", _canonical_file_bytes(private))
    _fsync_directory(root / "keys")
    _fsync_directory(root)
    _fsync_directory(root.parent)
    _read_real_root(root, campaign_id, source_commit, source_tree)
    return root / "trust-root.json", root / "public-verification.json"


def _read_real_root(root: Path, campaign_id: str, source_commit: str,
                    source_tree: str) -> tuple[list[dict[str, str]], str]:
    root = _absolute_canonical(Path(root))
    _private_root_outside_repository(root)
    _secure_directory(root)
    _secure_directory(root / "keys")
    private_path = root / "trust-root.json"
    public_path = root / "public-verification.json"
    _secure_file(private_path)
    _secure_file(public_path)
    private_bytes = _read_secure_file(private_path)
    public_bytes = _read_secure_file(public_path)
    try:
        private = json.loads(private_bytes)
        public = json.loads(public_bytes)
    except (ValueError, UnicodeError) as exc:
        raise GuardFailure("R22_TRUST_MANIFEST_MALFORMED") from exc
    if (not isinstance(private, dict) or set(private) != _ROOT_FIELDS
            or private["schema_version"] != R22_PRIVATE_SCHEMA
            or private["algorithm"] != R22_ALGORITHM
            or (private["campaign_id"], private["source_commit"], private["source_tree"])
            != (campaign_id, source_commit, source_tree)
            or _canonical_file_bytes(private) != private_bytes
            or _canonical_file_bytes(public) != public_bytes
            or private["public_package_sha256"] != _digest(public_bytes)):
        raise GuardFailure("R22_TRUST_ROOT_BINDING_INVALID")
    public_rows = _validate_public(public, campaign_id, source_commit, source_tree)
    if private["created_at"] != public["created_at"]:
        raise GuardFailure("R22_TRUST_CREATION_IDENTITY_MISMATCH")
    rows = private["roles"]
    if not isinstance(rows, list) or len(rows) != len(AUTHORITY_CLASSES):
        raise GuardFailure("R22_PRIVATE_ROLE_SET_INVALID")
    by_role = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _PUBLIC_ROLE_FIELDS | {"private_key_ref"}:
            raise GuardFailure("R22_PRIVATE_ROLE_SCHEMA_INVALID")
        role = row["authority_class"]
        if role not in AUTHORITY_CLASSES or role in by_role:
            raise GuardFailure("R22_PRIVATE_ROLE_SET_INVALID")
        if row["private_key_ref"] != f"keys/{role.lower()}.pem":
            raise GuardFailure("R22_PRIVATE_KEY_REFERENCE_INVALID")
        if {key: row[key] for key in _PUBLIC_ROLE_FIELDS} != public_rows[AUTHORITY_CLASSES.index(role)]:
            raise GuardFailure("R22_PRIVATE_PUBLIC_MANIFEST_MISMATCH")
        key_path = root / row["private_key_ref"]
        _secure_file(key_path)
        try:
            derived = _public_from_private(key_path)
        except GuardFailure as exc:
            raise GuardFailure("R22_PRIVATE_KEY_MALFORMED") from exc
        if derived != row["public_key_pem"].encode("ascii"):
            raise GuardFailure("R22_PRIVATE_PUBLIC_KEY_MISMATCH")
        by_role[role] = row
    if set(by_role) != set(AUTHORITY_CLASSES):
        raise GuardFailure("R22_PRIVATE_ROLE_SET_INVALID")
    return [by_role[role] for role in AUTHORITY_CLASSES], _digest(private_bytes)


def load_real_trust_root(root: Path, *, expected_campaign_id: str,
                         expected_source_commit: str, expected_source_tree: str) -> R17TrustConfiguration:
    _check_binding(expected_campaign_id, expected_source_commit, expected_source_tree)
    root = Path(root)
    rows, _ = _read_real_root(
        root, expected_campaign_id, expected_source_commit, expected_source_tree,
    )
    roles = {}
    for row in rows:
        role = row["authority_class"]
        common = {
            "authority_class": role, "authority_id": row["authority_id"],
            "signer_id": row["signer_id"], "fingerprint_sha256": row["fingerprint_sha256"],
            "public_key_pem": row["public_key_pem"], "campaign_id": expected_campaign_id,
            "source_commit": expected_source_commit, "source_tree": expected_source_tree,
        }
        verifier = OpenSSLRoleVerifier(**common, root_path=root)
        signer = OpenSSLRoleSigner(
            **common, private_key_path=root / row["private_key_ref"], root_path=root,
        )
        roles[role] = TrustRoleConfiguration(role, row["authority_id"], (row["signer_id"],), verifier, signer)
    return R17TrustConfiguration(roles, test_only=False)


def load_public_verification_package(path: Path, *, expected_sha256: str,
                                     expected_campaign_id: str, expected_source_commit: str,
                                     expected_source_tree: str) -> R17TrustConfiguration:
    _check_binding(expected_campaign_id, expected_source_commit, expected_source_tree)
    if not isinstance(expected_sha256, str) or not _HEX_256.fullmatch(expected_sha256):
        raise GuardFailure("R22_PUBLIC_PACKAGE_HASH_REQUIRED")
    path = _absolute_canonical(Path(path))
    if path.is_symlink() or not path.is_file():
        raise GuardFailure("R22_PUBLIC_PACKAGE_FILE_INVALID")
    content = path.read_bytes()
    if _digest(content) != expected_sha256:
        raise GuardFailure("R22_PUBLIC_PACKAGE_HASH_MISMATCH")
    try:
        document = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise GuardFailure("R22_PUBLIC_PACKAGE_MALFORMED") from exc
    if _canonical_file_bytes(document) != content:
        raise GuardFailure("R22_PUBLIC_PACKAGE_NONCANONICAL")
    rows = _validate_public(
        document, expected_campaign_id, expected_source_commit, expected_source_tree,
    )
    roles = {}
    for row in rows:
        role = row["authority_class"]
        verifier = OpenSSLRoleVerifier(
            authority_class=role, authority_id=row["authority_id"],
            signer_id=row["signer_id"], fingerprint_sha256=row["fingerprint_sha256"],
            public_key_pem=row["public_key_pem"], campaign_id=expected_campaign_id,
            source_commit=expected_source_commit, source_tree=expected_source_tree,
        )
        roles[role] = TrustRoleConfiguration(role, row["authority_id"], (row["signer_id"],), verifier)
    return R17TrustConfiguration(roles, test_only=True)


class R26CandidateTrustConfiguration(R17TrustConfiguration):
    """Six public verifiers, but only the observer's private signing capability."""

    def __init__(self, roles: Mapping[str, TrustRoleConfiguration], *, root: Path,
                 public_sha256: str, campaign_id: str, source_commit: str,
                 source_tree: str) -> None:
        super().__init__(roles, test_only=False)
        self._candidate_root = root
        self._candidate_public_sha256 = public_sha256
        self._candidate_binding = (campaign_id, source_commit, source_tree)

    @property
    def test_only(self) -> bool:
        if self._test_only_requested:
            return True
        try:
            root = self._candidate_root
            _secure_directory(root)
            _secure_directory(root / "keys")
            keys = list((root / "keys").iterdir())
            observer_path = root / "keys" / "repository_observer_authority.pem"
            if keys != [observer_path]:
                return True
            _secure_file(observer_path)
            campaign, commit, tree = self._candidate_binding
            public = load_public_verification_package(
                root / "public-verification.json",
                expected_sha256=self._candidate_public_sha256,
                expected_campaign_id=campaign, expected_source_commit=commit,
                expected_source_tree=tree,
            )
            observer = self._roles["REPOSITORY_OBSERVER_AUTHORITY"].signer
            if type(observer) is not OpenSSLRoleSigner or observer.root_path != root or observer.private_key_path != observer_path:
                return True
            if _public_from_private(observer_path) != observer.public_key_pem.encode("ascii"):
                return True
            for role in AUTHORITY_CLASSES:
                ours, reference = self._roles[role], public._roles[role]
                if (type(ours.backend) is not OpenSSLRoleVerifier
                        or ours.backend != reference.backend
                        or ours.authority_id != reference.authority_id
                        or ours.signer_ids != reference.signer_ids
                        or (role != "REPOSITORY_OBSERVER_AUTHORITY" and ours.signer is not None)):
                    return True
            return False
        except (GuardFailure, OSError, ValueError, TypeError):
            return True

    @test_only.setter
    def test_only(self, value: bool) -> None:
        self._test_only_requested = bool(value)


def load_candidate_observer_trust(root: Path, *, expected_public_sha256: str,
                                  expected_campaign_id: str,
                                  expected_source_commit: str,
                                  expected_source_tree: str) -> R26CandidateTrustConfiguration:
    """Load a candidate-owned, observer-only key root; never load the six-key root."""
    root = _absolute_canonical(Path(root))
    _private_root_outside_repository(root)
    public = load_public_verification_package(
        root / "public-verification.json", expected_sha256=expected_public_sha256,
        expected_campaign_id=expected_campaign_id,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
    )
    role = public._roles["REPOSITORY_OBSERVER_AUTHORITY"]
    verifier = role.backend
    signer = OpenSSLRoleSigner(
        authority_class=role.authority_class, authority_id=role.authority_id,
        signer_id=role.signer_ids[0], fingerprint_sha256=verifier.fingerprint_sha256,
        public_key_pem=verifier.public_key_pem,
        campaign_id=expected_campaign_id, source_commit=expected_source_commit,
        source_tree=expected_source_tree,
        private_key_path=root / "keys" / "repository_observer_authority.pem",
        root_path=root,
    )
    roles = dict(public._roles)
    roles["REPOSITORY_OBSERVER_AUTHORITY"] = TrustRoleConfiguration(
        role.authority_class, role.authority_id, role.signer_ids, role.backend, signer,
    )
    candidate = R26CandidateTrustConfiguration(
        roles, root=root, public_sha256=expected_public_sha256,
        campaign_id=expected_campaign_id, source_commit=expected_source_commit,
        source_tree=expected_source_tree,
    )
    if candidate.test_only:
        raise GuardFailure("R26_CANDIDATE_TRUST_NOT_OBSERVER_ONLY")
    return candidate


def stage_candidate_observer_trust(source_root: Path, destination: Path, *,
                                   campaign_id: str, source_commit: str,
                                   source_tree: str, owner_uid: int | None = None,
                                   owner_gid: int | None = None) -> str:
    """Provision only the observer key into a new private candidate root.

    This is controller-side staging, before the candidate is launched.  It
    never exports another role's private signing material.
    """
    source_root = _absolute_canonical(Path(source_root))
    destination = _absolute_canonical(Path(destination))
    _private_root_outside_repository(destination)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise GuardFailure("R26_CANDIDATE_TRUST_DESTINATION_NOT_NEW")
    rows, _ = _read_real_root(source_root, campaign_id, source_commit, source_tree)
    observer = next(row for row in rows if row["authority_class"] == "REPOSITORY_OBSERVER_AUTHORITY")
    public_bytes = _read_secure_file(source_root / "public-verification.json")
    private_bytes = _read_secure_file(source_root / observer["private_key_ref"])
    os.mkdir(destination, 0o700)
    os.mkdir(destination / "keys", 0o700)
    observer_path = destination / "keys" / "repository_observer_authority.pem"
    _write_exclusive(destination / "public-verification.json", public_bytes)
    _write_exclusive(observer_path, private_bytes)
    if _public_from_private(observer_path) != observer["public_key_pem"].encode("ascii"):
        raise GuardFailure("R26_CANDIDATE_OBSERVER_KEY_MISMATCH")
    if owner_uid is not None or owner_gid is not None:
        if owner_uid is None or owner_gid is None:
            raise GuardFailure("R26_CANDIDATE_TRUST_OWNER_INCOMPLETE")
        for path in (destination / "public-verification.json", observer_path,
                     destination / "keys", destination):
            os.chown(path, owner_uid, owner_gid)
    _fsync_directory(destination / "keys")
    _fsync_directory(destination)
    _fsync_directory(destination.parent)
    return _digest(public_bytes)


def real_trust_configuration_verified(roles: Mapping[str, TrustRoleConfiguration]) -> bool:
    """Structural real admission; a caller's test_only flag cannot grant it."""
    if set(roles) != set(AUTHORITY_CLASSES):
        return False
    roots = set()
    bindings = set()
    for role in AUTHORITY_CLASSES:
        configured = roles[role]
        backend, signer = configured.backend, configured.signer
        if type(backend) is not OpenSSLRoleVerifier or type(signer) is not OpenSSLRoleSigner:
            return False
        if (backend.authority_class != role or signer.authority_class != role
                or backend.authority_id != configured.authority_id
                or signer.authority_id != configured.authority_id
                or configured.signer_ids != (backend.signer_id,)
                or signer.signer_id != backend.signer_id
                or signer.public_key_pem != backend.public_key_pem
                or signer.fingerprint_sha256 != backend.fingerprint_sha256
                or signer.root_path != backend.root_path):
            return False
        roots.add(signer.root_path)
        bindings.add((signer.campaign_id, signer.source_commit, signer.source_tree))
    if len(roots) != 1 or len(bindings) != 1:
        return False
    root = next(iter(roots))
    if root is None:
        return False
    campaign_id, source_commit, source_tree = next(iter(bindings))
    try:
        rows, _ = _read_real_root(root, campaign_id, source_commit, source_tree)
    except (GuardFailure, OSError, TypeError, ValueError):
        return False
    for row in rows:
        role = row["authority_class"]
        configured = roles[role]
        backend = roles[role].backend
        signer = roles[role].signer
        if (configured.authority_id != row["authority_id"]
                or configured.signer_ids != (row["signer_id"],)
                or backend.authority_id != row["authority_id"]
                or backend.signer_id != row["signer_id"]
                or signer.authority_id != row["authority_id"]
                or signer.signer_id != row["signer_id"]
                or backend.fingerprint_sha256 != row["fingerprint_sha256"]
                or backend.public_key_pem != row["public_key_pem"]
                or backend.campaign_id != campaign_id
                or backend.source_commit != source_commit
                or backend.source_tree != source_tree
                or signer.private_key_path != root / row["private_key_ref"]):
            return False
    return True
