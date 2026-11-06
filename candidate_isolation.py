"""Fail-closed startup isolation checks for evaluation candidates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


PRODUCTION_CHROMA_PATH = Path("/var/lib/project-x/chroma")
PRODUCTION_MODEL_TREE_PATH = Path("/var/lib/project-x/models")
PRODUCTION_RELEASE_PATH = Path("/opt/project-x/current")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def candidate_execution_environment(
    *,
    chroma_path: str | Path,
    port: int | str,
    production_chroma_path: str | Path,
    production_release_path: str | Path,
    trace_directory: str | Path,
) -> tuple[str, ...]:
    """Return candidate overrides for the final ``/usr/bin/env`` exec layer.

    A candidate inherits the standard service environment file for ordinary
    configuration.  These values must be applied by the executable itself,
    after that file, because systemd manager environment values can be
    overwritten by ``EnvironmentFile=``.
    """
    values: Mapping[str, str] = {
        "PROJECT_X_CANDIDATE_MODE": "1",
        "PROJECT_X_CANDIDATE_PORT": str(port),
        "PROJECT_X_EVAL_TRACE": "1",
        # The official isolated-launch path cannot silently start an
        # unobserved future qualification candidate.
        "PROJECT_X_R7R13_GUARD_REQUIRED": "1",
        "PROJECT_X_PRODUCTION_CHROMA_PATH": str(production_chroma_path),
        "PROJECT_X_PRODUCTION_RELEASE_PATH": str(production_release_path),
        "CHROMA_PATH": str(chroma_path),
        "PROJECT_X_EVAL_TRACE_DIR": str(trace_directory),
    }
    for name, value in values.items():
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"invalid candidate execution environment value: {name}")
    return tuple(f"{name}={value}" for name, value in values.items())


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def chroma_storage_aliases(
    candidate_chroma: Path,
    production_chroma: Path,
) -> bool:
    """Detect aliases that retain distinct path spellings, including bind mounts."""
    try:
        if os.path.samefile(candidate_chroma, production_chroma):
            return True
    except OSError:
        pass
    candidate_sqlite = candidate_chroma / "chroma.sqlite3"
    production_sqlite = production_chroma / "chroma.sqlite3"
    try:
        return os.path.samefile(candidate_sqlite, production_sqlite)
    except OSError:
        return False


def _aliases_same_chroma_storage(
    candidate_chroma: Path,
    production_chroma: Path,
) -> bool:
    """Backward-compatible private name for the canonical alias probe."""
    return chroma_storage_aliases(candidate_chroma, production_chroma)


def validate_candidate_isolation(*, chroma_path: str, bind_host: str) -> None:
    """Validate candidate storage, binding and production immutability."""
    if os.getenv("PROJECT_X_CANDIDATE_MODE", "0").strip().casefold() not in {
        "1", "true", "yes", "on",
    }:
        return
    candidate_chroma = _resolved(chroma_path)
    production_chroma = _resolved(
        os.getenv("PROJECT_X_PRODUCTION_CHROMA_PATH", str(PRODUCTION_CHROMA_PATH))
    )
    if candidate_chroma == production_chroma:
        raise RuntimeError("candidate Chroma path resolves to production Chroma")
    if chroma_storage_aliases(candidate_chroma, production_chroma):
        raise RuntimeError("candidate Chroma storage aliases production Chroma")
    if str(bind_host or "").strip().casefold() not in LOOPBACK_HOSTS:
        raise RuntimeError("candidate API must bind to a loopback address")
    raw_port = os.getenv("PROJECT_X_CANDIDATE_PORT", "").strip()
    try:
        candidate_port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PROJECT_X_CANDIDATE_PORT must be explicit") from exc
    if not 1 <= candidate_port <= 65535 or candidate_port == 8000:
        raise RuntimeError("candidate API port must be valid and must not be 8000")
    production_release = _resolved(
        os.getenv(
            "PROJECT_X_PRODUCTION_RELEASE_PATH",
            str(PRODUCTION_RELEASE_PATH),
        )
    )
    if not production_release.exists():
        raise RuntimeError("production release path cannot be verified")
    if os.access(production_release, os.W_OK):
        raise RuntimeError("production release path is writable by candidate")
