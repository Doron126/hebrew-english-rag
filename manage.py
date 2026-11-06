#!/usr/bin/env python3
"""Set up and run a local Hebrew-English RAG workspace."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from config_env import DEFAULT_EMBED_MODEL, _read_env_file, validate_tenant_id


ROOT = Path(__file__).resolve().parent


class SetupError(Exception):
    """An actionable configuration error, safe to show in the terminal."""


def absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def read_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SetupError("Run `python manage.py init --help` to configure this checkout first.")
    return _read_env_file(path)


def write_config(path: Path, values: dict[str, str]) -> None:
    for key, value in values.items():
        if any(character in str(value) for character in "\r\n\0"):
            raise SetupError(f"{key} must fit on a single line.")
    text = "# Local installation settings. Keep this file out of Git.\n"
    text += "".join(f"{key}={value}\n" for key, value in values.items())
    atomic_write(path, text)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".rag-", delete=False) as handle:
            name = handle.name
            os.chmod(name, 0o600)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


@contextmanager
def exclusive_lock(path: Path):
    import fcntl

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SetupError("Another command is already using this installation.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_binary(value: str | None) -> Path:
    found = shutil.which(value or "llama-server")
    binary = absolute_path(found or value or "llama-server")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SetupError("Set --llama-server to an executable llama-server binary.")
    return binary


def validate_model(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        raise SetupError(f"Cannot read the GGUF model at {path}.") from exc
    if magic != b"GGUF":
        raise SetupError("The model must be a GGUF file, not a download page or Git LFS pointer.")


def checked_integer(value: str | int, name: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SetupError(f"{name} must be an integer between {low} and {high}.") from exc
    if not low <= number <= high:
        raise SetupError(f"{name} must be between {low} and {high}.")
    return number


def model_origin(values: dict[str, str]) -> tuple[str, int]:
    value = values.get("LLAMA_BASE_URL", "http://127.0.0.1:18081")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise SetupError("LLAMA_BASE_URL has an invalid port.") from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
            parsed.username or parsed.password or parsed.query or parsed.fragment or
            parsed.path not in {"", "/"} or port is None):
        raise SetupError("LLAMA_BASE_URL must be http://127.0.0.1:<port> for the local launcher.")
    return value.rstrip("/"), checked_integer(port, "Model port", 1024, 65535)


def tenant_ids(values: dict[str, str]) -> list[str]:
    try:
        return list(dict.fromkeys(validate_tenant_id(item) for item in
                                 values.get("LOCAL_ALLOWED_TENANTS", "").split(",") if item.strip()))
    except ValueError as exc:
        raise SetupError(str(exc)) from exc


def data_path(values: dict[str, str], name: str) -> Path:
    raw = values.get(name, "").strip()
    if not raw:
        raise SetupError(f"Set {name} in the installation configuration.")
    return absolute_path(raw)


def tenant_documents(values: dict[str, str], tenant: str) -> Path:
    root = data_path(values, "DOCS_BASE_DIR")
    candidate = root / tenant
    if candidate.is_symlink() or candidate.resolve().parent != root:
        raise SetupError("The tenant document folder must stay inside DOCS_BASE_DIR.")
    return candidate


def runtime_environment(values: dict[str, str]) -> dict[str, str]:
    # The generated installation is authoritative for child processes. Ambient
    # settings from another checkout must not redirect its data or model.
    environment = dict(os.environ)
    environment.update(values)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def command_init(args: argparse.Namespace) -> int:
    config = absolute_path(args.config)
    if config.exists():
        raise SetupError(f"Configuration already exists at {config}. Edit it to change model settings.")
    model = absolute_path(args.model)
    validate_model(model)
    binary = resolve_binary(args.llama_server)
    model_id = args.model_id or model.stem
    if not model_id.strip() or len(model_id) > 128 or any(c in model_id for c in "\r\n\0"):
        raise SetupError("The model ID must be a nonempty single line, up to 128 characters.")
    model_port = checked_integer(args.model_port, "Model port", 1024, 65535)
    app_port = checked_integer(args.port, "Application port", 1024, 65535)
    if model_port == app_port:
        raise SetupError("The application and model need different ports.")
    data = absolute_path(args.data_dir)
    if data == ROOT or ROOT in data.parents:
        raise SetupError("Choose a data directory outside the source checkout.")
    values = {
        "APP_ENV": "local", "APP_BIND_HOST": "127.0.0.1", "APP_PORT": str(app_port),
        "TENANT_AUTH_MODE": "client", "LOCAL_ALLOWED_TENANTS": "", "WARMUP_TENANTS": "",
        "RAG_MODEL_RUNTIME": "llama_cpp", "LLM_BACKEND": "llama_cpp",
        "BASE_MODEL": model_id, "LLAMA_MODEL_ID": model_id,
        "LLAMA_SERVER_BIN": str(binary), "LLAMA_MODEL_PATH": str(model),
        "LLAMA_BASE_URL": f"http://127.0.0.1:{model_port}",
        "LLAMA_API_KEY_FILE": str(data / "run" / "model-api.key"),
        "LLAMA_CONTEXT_SIZE_TOKENS": "4096", "LLAMA_CONTEXT_RESERVE_TOKENS": "256",
        "LLAMA_N_GPU_LAYERS": str(checked_integer(args.gpu_layers, "GPU layers", 0, 999)),
        "LLAMA_THREADS": str(max(1, min(8, os.cpu_count() or 1))),
        "LLAMA_STARTUP_TIMEOUT_SECONDS": "600", "LLAMA_REQUEST_TIMEOUT_SECONDS": "1200",
        "RAG_DATA_DIR": str(data), "DOCS_BASE_DIR": str(data / "documents"),
        "TENANT_CONFIG_DIR": str(data / "config" / "tenants"),
        "CHROMA_PATH": str(data / "chroma"), "INGEST_EXTRACTION_AUDIT_DIR": str(data / "extraction-audit"),
        "PROJECT_X_CONVERSATION_STATE_PATH": str(data / "conversations.sqlite3"),
        "HF_HOME": str(data / "cache" / "huggingface"),
        "EMBED_MODEL": args.embedding_model, "EMBED_DEVICE": "cpu",
        "USE_RERANKER": "0", "CHAT_USE_RERANKER": "0", "ANONYMIZED_TELEMETRY": "False",
    }
    # Validate all text before creating anything, including paths from the CLI.
    if any(any(c in value for c in "\r\n\0") for value in values.values()):
        raise SetupError("Configuration values and paths must fit on one line.")
    with exclusive_lock(config.with_suffix(config.suffix + ".lock")):
        if config.exists():
            raise SetupError("Configuration was created by another command; it was left unchanged.")
        for directory in (data, data / "run", data / "logs", data / "documents",
                          data / "config" / "tenants", data / "cache" / "huggingface"):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = Path(values["LLAMA_API_KEY_FILE"])
        if key.exists():
            if not key.is_file() or key.is_symlink() or key.stat().st_mode & 0o077:
                raise SetupError("The existing model key must be a private regular file (chmod 600).")
        else:
            descriptor = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(secrets.token_urlsafe(32) + "\n")
        write_config(config, values)
    print(f"Configuration: {config}\nData directory: {data}\nModel: {model_id}")
    print("Create your first tenant: python manage.py tenant add my-documents")
    return 0


def command_tenant_add(args: argparse.Namespace) -> int:
    config = absolute_path(args.config)
    tenant = validate_tenant_id(args.tenant)
    if args.name and (len(args.name) > 128 or any(c in args.name for c in "\r\n\0")):
        raise SetupError("Choose a single-line display name of up to 128 characters.")
    with exclusive_lock(config.with_suffix(config.suffix + ".lock")):
        values = read_config(config)
        allowed = tenant_ids(values)
        profile = data_path(values, "TENANT_CONFIG_DIR") / f"{tenant}.json"
        documents = tenant_documents(values, tenant)
        if tenant in allowed or profile.exists():
            raise SetupError(f"Tenant {tenant} already exists; its files and configuration were left unchanged.")
        documents.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_write(profile, json.dumps({"display_name": args.name or tenant}, ensure_ascii=False, indent=2) + "\n")
        values["LOCAL_ALLOWED_TENANTS"] = ",".join([*allowed, tenant])
        write_config(config, values)
    print(f"Tenant: {tenant}\nCopy your documents into: {documents}")
    print(f"Then index them: python manage.py ingest --tenant {tenant}")
    print("If the app is running, restart it to load the new tenant.")
    return 0


def command_tenant_list(args: argparse.Namespace) -> int:
    values = read_config(absolute_path(args.config))
    allowed = tenant_ids(values)
    if not allowed:
        print("No tenants yet. Create one with: python manage.py tenant add my-documents")
    for tenant in allowed:
        print(f"{tenant}\t{tenant_documents(values, tenant)}")
    return 0


def command_ingest(args: argparse.Namespace) -> int:
    values = read_config(absolute_path(args.config))
    tenant = validate_tenant_id(args.tenant)
    if tenant not in tenant_ids(values):
        raise SetupError(f"Create tenant {tenant} first with `python manage.py tenant add {tenant}`.")
    documents = tenant_documents(values, tenant)
    if not documents.is_dir() or not any(documents.iterdir()):
        raise SetupError(f"Copy documents into {documents} before indexing.")
    report = data_path(values, "RAG_DATA_DIR") / "reports" / f"{tenant}-ingest.json"
    command = [sys.executable, str(ROOT / "ingest.py"), "--tenant", tenant,
               "--docs-dir", str(documents), "--report-json", str(report), "--continue-on-error"]
    if args.force:
        command.append("--force")
    result = subprocess.run(command, cwd=ROOT, env=runtime_environment(values), check=False)
    print(f"Ingestion report: {report}")
    return result.returncode


def model_command(values: dict[str, str]) -> list[str]:
    binary = resolve_binary(values.get("LLAMA_SERVER_BIN"))
    model = data_path(values, "LLAMA_MODEL_PATH")
    validate_model(model)
    _, port = model_origin(values)
    context = checked_integer(values.get("LLAMA_CONTEXT_SIZE_TOKENS", "4096"), "Context size", 1024, 4096)
    layers = checked_integer(values.get("LLAMA_N_GPU_LAYERS", "0"), "GPU layers", 0, 999)
    threads = checked_integer(values.get("LLAMA_THREADS", "4"), "Threads", 1, 256)
    key = data_path(values, "LLAMA_API_KEY_FILE")
    if not key.is_file() or key.is_symlink() or key.stat().st_mode & 0o077:
        raise SetupError("The model API key must exist and have owner-only permissions (chmod 600).")
    from llm_client import _read_api_key
    if not _read_api_key(str(key)):
        raise SetupError("The model API key file must contain one key.")
    return [str(binary), "--model", str(model), "--alias", values["LLAMA_MODEL_ID"],
            "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(context),
            "--parallel", "1", "--threads", str(threads), "--n-gpu-layers", str(layers),
            "--api-key-file", str(key), "--jinja", "--no-webui",
            "--chat-template-kwargs", '{"enable_thinking":false}', "--no-context-shift"]


def dependency_errors() -> list[str]:
    required = {"fastapi": "fastapi", "uvicorn": "uvicorn", "chromadb": "chromadb",
                "sentence_transformers": "sentence-transformers", "torch": "torch",
                "pydantic": "pydantic", "fitz": "PyMuPDF", "pypdf": "pypdf",
                "jsonschema": "jsonschema"}
    return [package for module, package in required.items() if importlib.util.find_spec(module) is None]


def command_doctor(args: argparse.Namespace) -> int:
    failures = []
    print(f"Python: {sys.version.split()[0]}")
    missing = dependency_errors()
    if missing:
        failures.append("Install the application dependencies: " + ", ".join(missing))
    else:
        print("Application dependencies: available")
    try:
        values = read_config(absolute_path(args.config))
        model_command(values)
        print(f"Model: {values['LLAMA_MODEL_ID']}\nModel server: {values['LLAMA_SERVER_BIN']}")
        print("Tenants: " + (", ".join(tenant_ids(values)) or "none yet"))
        for tenant in tenant_ids(values):
            print(f"Documents ({tenant}): {tenant_documents(values, tenant)}")
        if shutil.which("tesseract"):
            print("Optional OCR: Tesseract available")
        else:
            print("Optional OCR: install Tesseract with Hebrew and English language data for scanned files")
    except (SetupError, ValueError) as exc:
        failures.append(str(exc))
    for failure in failures:
        print(f"Action needed: {failure}", file=sys.stderr)
    if not failures:
        print("Configuration is ready. Start the application with: python manage.py serve")
    return 1 if failures else 0


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The process may exit after poll(). Reap it and continue cleaning up
        # the other child instead of aborting the launcher's finally block.
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class _NoLocalRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The startup key belongs only to the explicitly configured local server.
        return None


def wait_for_model(process: subprocess.Popen, values: dict[str, str]) -> None:
    origin, _ = model_origin(values)
    key = data_path(values, "LLAMA_API_KEY_FILE").read_text(encoding="utf-8").strip()
    opener = build_opener(ProxyHandler({}), _NoLocalRedirects())
    deadline = time.monotonic() + checked_integer(values.get("LLAMA_STARTUP_TIMEOUT_SECONDS", "600"),
                                                 "Model startup timeout", 1, 7200)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SetupError("The model process stopped during startup. See the local model log.")
        try:
            request = Request(origin + "/health", headers={"Authorization": f"Bearer {key}"})
            with opener.open(request, timeout=2) as response:
                result = json.loads(response.read(65536))
                if response.status == 200 and result.get("status") == "ok":
                    return
        except (URLError, HTTPError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(0.25)
    raise SetupError("The model did not finish loading. Check the model log and LLAMA_STARTUP_TIMEOUT_SECONDS.")


def command_serve(args: argparse.Namespace) -> int:
    values = read_config(absolute_path(args.config))
    if values.get("RAG_MODEL_RUNTIME") != "llama_cpp":
        raise SetupError("This launcher expects RAG_MODEL_RUNTIME=llama_cpp. Use init for a local installation.")
    if values.get("APP_BIND_HOST") != "127.0.0.1" or values.get("TENANT_AUTH_MODE") != "client":
        raise SetupError("The local launcher requires APP_BIND_HOST=127.0.0.1 and TENANT_AUTH_MODE=client.")
    if not tenant_ids(values):
        raise SetupError("Create a tenant with `python manage.py tenant add my-documents` before starting.")
    missing = dependency_errors()
    if missing:
        raise SetupError("Install the application dependencies first; missing: " + ", ".join(missing))
    command = model_command(values)
    _, model_port = model_origin(values)
    app_port = checked_integer(values.get("APP_PORT", "18080"), "Application port", 1024, 65535)
    if app_port == model_port:
        raise SetupError("The application and model ports must be different.")
    data = data_path(values, "RAG_DATA_DIR")
    processes = []
    def terminate_requested(signum, frame):
        raise KeyboardInterrupt
    log_path = data / "logs" / "model.log"
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_lock(data / "run" / "application.lock"):
        for port in (app_port, model_port):
            if not port_available(port):
                raise SetupError(f"Port {port} is already in use. Choose a free port in the configuration.")
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as log:
            previous_sigterm = signal.signal(signal.SIGTERM, terminate_requested)
            try:
                environment = runtime_environment(values)
                print(f"Loading {values['LLAMA_MODEL_ID']} ...\nModel log: {log_path}", flush=True)
                model_process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(model_process)
                wait_for_model(model_process, values)
                api = subprocess.Popen([sys.executable, "-m", "uvicorn", "chat_api:app",
                                        "--host", "127.0.0.1", "--port", str(app_port), "--workers", "1"],
                                       cwd=ROOT, env=environment, start_new_session=True)
                processes.append(api)
                print(f"Workspace: http://127.0.0.1:{app_port}\nPress Ctrl+C to stop both processes.", flush=True)
                while all(process.poll() is None for process in processes):
                    time.sleep(0.25)
                failed = next(process for process in processes if process.poll() is not None)
                return failed.returncode or 0
            except KeyboardInterrupt:
                print("\nStopping the application and model ...", flush=True)
                return 0
            finally:
                try:
                    for process in reversed(processes):
                        stop_child(process)
                finally:
                    signal.signal(signal.SIGTERM, previous_sigterm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / ".env"), help="Installation configuration (default: this checkout's .env).")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="Create local configuration and data folders.")
    initialize.add_argument("--model", required=True, help="Path to your instruction-tuned GGUF model.")
    initialize.add_argument("--llama-server", help="Path to llama-server, or install it on PATH.")
    initialize.add_argument("--model-id", help="Display name and llama.cpp model alias (default: GGUF filename).")
    initialize.add_argument("--data-dir", default=str(Path.home() / ".local" / "share" / "hebrew-english-rag"))
    initialize.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    initialize.add_argument("--gpu-layers", default=0, type=int, help="Model layers to offload to a GPU (default: CPU).")
    initialize.add_argument("--port", default=18080, type=int)
    initialize.add_argument("--model-port", default=18081, type=int)
    initialize.set_defaults(handler=command_init)
    tenants = commands.add_parser("tenant", help="Create or list your document workspaces.")
    tenant_commands = tenants.add_subparsers(dest="tenant_command", required=True)
    add = tenant_commands.add_parser("add")
    add.add_argument("tenant")
    add.add_argument("--name", help="Display name, in Hebrew or English.")
    add.set_defaults(handler=command_tenant_add)
    tenant_commands.add_parser("list").set_defaults(handler=command_tenant_list)
    ingest = commands.add_parser("ingest", help="Index a tenant's document folder.")
    ingest.add_argument("--tenant", required=True)
    ingest.add_argument("--force", action="store_true", help="Re-extract and re-embed unchanged documents.")
    ingest.set_defaults(handler=command_ingest)
    commands.add_parser("doctor", help="Check dependencies and local configuration.").set_defaults(handler=command_doctor)
    commands.add_parser("serve", help="Run the model and web application; Ctrl+C stops both.").set_defaults(handler=command_serve)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (SetupError, ValueError, OSError) as exc:
        print(f"Setup: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
