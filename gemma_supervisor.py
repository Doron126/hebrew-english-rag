"""Narrow root helper for one authenticated application's Gemma generation.

The service accepts only root/projectx Unix peers and operator-authorized tenants. Requests
cannot choose commands, model files, ports, configuration, or service names.
No queries, passages, answers or authentication payloads are logged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pwd
import queue
import re
import select
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen
import uuid

from gemma_output_policy import MAX_OUTPUT_TOKENS

GIB = 1024 ** 3
CONFIG = Path("/etc/project-x/gemma.json")
STATE_DIR = Path("/var/lib/project-x-gemma")
SOCKET_PATH = "/run/project-x-gemma/supervisor.sock"
ORIGIN = "http://127.0.0.1:18081"
MODEL_PORT = 18081
MODEL_ID = "gemma-4-31b"
API_KEY_FILE = Path("/etc/project-x/llama-api-key")
INTEROP_ROOT = Path("/run/WSL")
PROC_ROOT = Path("/proc")
MODEL = "/nonexistent/hebrew-english-rag-configure-runtime/models/gemma-4-31B_q4_0-it.gguf"
BINARY = "/nonexistent/hebrew-english-rag-configure-runtime/runtime/llama/llama-b11669/llama-server"
MODEL_BYTES = 17651001568
EXPECTED_ARGV = [
    BINARY, "--model", MODEL, "--host", "127.0.0.1", "--port", "18081",
    "--ctx-size", "4096", "--parallel", "1", "--n-gpu-layers", "32",
    "--batch-size", "256", "--ubatch-size", "128", "--threads", "6",
    "--threads-batch", "6", "--flash-attn", "auto", "--cache-type-k", "f16",
    "--cache-type-v", "f16", "--cache-ram", "0", "--jinja", "--reasoning",
    "off", "--reasoning-format", "deepseek", "--offline", "--no-warmup",
    "--no-webui", "--slots", "--load-mode", "dio",
]
EXPECTED_ENV = {
    "LD_LIBRARY_PATH": "/nonexistent/hebrew-english-rag-configure-runtime/runtime/cudart/cudart-llama-b11669-bin-ubuntu-cuda-12.8-x64:/nonexistent/hebrew-english-rag-configure-runtime/runtime/llama/llama-b11669",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
FLOORS = {"host_physical_available_bytes": int(1.0 * GIB),
          "host_commit_headroom_bytes": int(1.5 * GIB),
          "guest_memavailable_bytes": int(6.0 * GIB)}
MAX_REQUEST = 128 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_STARTUP_LOG_BYTES = 256 * 1024
_STOP = threading.Event()


class SupervisorError(RuntimeError):
    pass


class RequestRejected(SupervisorError):
    """A deterministic request failure that does not invalidate a ready model."""
    def __init__(self, code, **details):
        super().__init__(code)
        self.details = details


class StartupDiagnostics:
    """Bounded startup stderr only; drain and discard all later runtime output."""
    def __init__(self):
        self.path = STATE_DIR / ("startup-" + uuid.uuid4().hex + ".stderr.log")
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                          os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        self.lock = threading.Lock()
        self.remaining = MAX_STARTUP_LOG_BYTES
        self.reader = None

    def start(self, pipe):
        def drain():
            try:
                while True:
                    chunk = os.read(pipe.fileno(), 8192)
                    if not chunk:
                        break
                    with self.lock:
                        if self.fd is not None and self.remaining:
                            kept = chunk[:self.remaining]
                            try:
                                while kept:
                                    written = os.write(self.fd, kept)
                                    self.remaining -= written
                                    kept = kept[written:]
                            except OSError:
                                os.close(self.fd)
                                self.fd = None
            except OSError:
                # A closed process pipe ends diagnostics; no exception message
                # or raw runtime output is ever copied into the journal.
                pass
            finally:
                pipe.close()
                self.stop_capture()
        self.reader = threading.Thread(target=drain, daemon=True)
        self.reader.start()

    def stop_capture(self):
        # This lock is also held during writes. Once it returns, a later query,
        # passage, response or request header cannot be written to the log.
        with self.lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    def finish(self):
        if self.reader is not None:
            self.reader.join(2)
        self.stop_capture()


def command(argv, timeout=5):
    return subprocess.check_output(argv, text=True, timeout=timeout, stderr=subprocess.DEVNULL).strip()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(value, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def load_config(path=CONFIG):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise SupervisorError("unsafe_configuration_permissions")
    if info.st_size > 16384:
        raise SupervisorError("configuration_too_large")
    config = json.loads(path.read_text())
    if config.get("server_argv") != EXPECTED_ARGV or config.get("server_env") != EXPECTED_ENV:
        raise SupervisorError("configuration_not_proven_profile")
    admission = config.get("admission_min_bytes")
    required = {"host_physical_available_bytes": 12 * GIB,
                "host_commit_headroom_bytes": 12 * GIB,
                "guest_memavailable_bytes": 19 * GIB}
    if not isinstance(admission, dict) or any(type(admission.get(k)) is not int or admission[k] < v for k, v in required.items()):
        raise SupervisorError("unsafe_admission_threshold")
    model = Path(MODEL).stat()
    if model.st_size != MODEL_BYTES or not stat.S_ISREG(model.st_mode):
        raise SupervisorError("model_size_changed")
    if not os.access(BINARY, os.X_OK):
        raise SupervisorError("runtime_unavailable")
    return config


def read_api_key():
    """Read the existing root-owned credential without accepting caller paths."""
    fd = os.open(API_KEY_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or
                info.st_mode & 0o027 or not 1 <= info.st_size <= 4096):
            raise SupervisorError("unsafe_model_api_key_file")
        raw = os.read(fd, 4097)
    finally:
        os.close(fd)
    if not 1 <= len(raw) <= 4096:
        raise SupervisorError("invalid_model_api_key")
    try:
        key = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise SupervisorError("invalid_model_api_key") from exc
    if not key or any(ord(character) <= 32 or ord(character) >= 127 for character in key):
        raise SupervisorError("invalid_model_api_key")
    return key


def server_environment(config):
    # llama.cpp reads controls outside LLAMA_ARG_* too (e.g. LLAMA_API_KEY,
    # LLAMA_CACHE, GGML_* and CUDA_*). Build from system essentials so none of
    # the API service's model controls or application credentials can leak in.
    environment = {key: value for key, value in os.environ.items()
                   if key in {"PATH", "LANG", "LANGUAGE", "TZ"} or key.startswith("LC_")}
    environment.update(config["server_env"])
    read_api_key()  # Fail before spawning if the shared credential is unsafe.
    environment.update(HOME=str(STATE_DIR), TMPDIR="/tmp", HF_HUB_OFFLINE="1",
                       LLAMA_ARG_ALIAS=MODEL_ID, LLAMA_ARG_API_KEY_FILE=str(API_KEY_FILE))
    return environment


class _NoModelRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise SupervisorError("model_http_redirect_rejected")


def json_http(path, payload=None, *, timeout=3, origin=ORIGIN, headers=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json"} if data is not None else {}
    request_headers.update(headers or {})
    if origin == ORIGIN:
        if not any(key.lower() == "authorization" for key in request_headers):
            request_headers["Authorization"] = "Bearer " + read_api_key()
        # Never let a redirect or inherited proxy forward this local credential.
        open_request = build_opener(ProxyHandler({}), _NoModelRedirect()).open
    else:
        open_request = urlopen
    with open_request(Request(origin + path, data=data, headers=request_headers), timeout=timeout) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise SupervisorError("model_response_too_large")
        return json.loads(raw)


def service_state(unit):
    state = dict(line.split("=", 1) for line in command(["systemctl", "show", unit, "-p", "ActiveState", "-p", "SubState"]).splitlines())
    return state == {"ActiveState": "active", "SubState": "running"}


def shutdown_in_progress():
    """Never start units against systemd's shutdown transaction."""
    try:
        state = subprocess.run(["systemctl", "is-system-running"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        if state == "stopping":
            return True
        jobs = command(["systemctl", "list-jobs", "--no-legend", "--no-pager"])
        return any(target in jobs for target in ("shutdown.target", "poweroff.target",
                                                  "reboot.target", "halt.target"))
    except Exception:
        # Cannot establish that service starts are safe.
        return True


def require_legacy_runtime_inactive():
    """Reject overlapping legacy residency; a retired/missing unit is allowed."""
    result = subprocess.run([
        "systemctl", "show", "project-x-llama-wsl.service", "-p", "LoadState",
        "-p", "ActiveState", "-p", "MainPID"],
        timeout=5, capture_output=True, text=True)
    details = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    absent = details.get("LoadState") == "not-found"
    if ((result.returncode == 0 or (result.returncode == 1 and absent)) and details.get("ActiveState") == "inactive"
            and details.get("MainPID") == "0"):
        return
    raise SupervisorError("legacy_model_not_inactive")


# This is the already exercised Windows API metrics probe, kept in one owned
# relay process rather than repeatedly launching Windows helpers.
WINDOWS_METRICS_PS = r"""
$ErrorActionPreference='Stop';$pxKeep=@('SystemRoot','WINDIR','TEMP','TMP','PATH','PSModulePath','USERPROFILE','LOCALAPPDATA','APPDATA','ProgramFiles','ProgramFiles(x86)','ComSpec','HOMEDRIVE','HOMEPATH','NUMBER_OF_PROCESSORS','PROCESSOR_ARCHITECTURE');Get-ChildItem Env: | Where-Object {$pxKeep -notcontains $_.Name} | ForEach-Object {[Environment]::SetEnvironmentVariable($_.Name,$null,'Process')};
$pxStart=(Get-Process -Id $PID).StartTime.ToFileTimeUtc();[Console]::WriteLine((@{windows_pid=$PID;windows_start_ticks=$pxStart}|ConvertTo-Json -Compress));[Console]::Out.Flush();
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class PxMem { [StructLayout(LayoutKind.Sequential)] public class M { public uint dwLength=(uint)Marshal.SizeOf(typeof(M)); public uint dwMemoryLoad; public ulong totalPhys,availPhys,totalPageFile,availPageFile,totalVirtual,availVirtual,availExtendedVirtual; } [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GlobalMemoryStatusEx([In,Out] M m); [StructLayout(LayoutKind.Sequential)] public class P { public uint cb=(uint)Marshal.SizeOf(typeof(P)); public UIntPtr CommitTotal,CommitLimit,CommitPeak,PhysicalTotal,PhysicalAvailable,SystemCache,KernelTotal,KernelPaged,KernelNonpaged,PageSize; public uint HandleCount,ProcessCount,ThreadCount; } [DllImport("psapi.dll",SetLastError=true)] public static extern bool GetPerformanceInfo([In,Out] P p,uint size); }'
$pxStart=(Get-Process -Id $PID).StartTime.ToFileTimeUtc();while($true){$m=[PxMem+M]::new();if(-not [PxMem]::GlobalMemoryStatusEx($m)){exit 2};$perf=[PxMem+P]::new();if(-not [PxMem]::GetPerformanceInfo($perf,$perf.cb)){exit 3};$head=($perf.CommitLimit.ToUInt64()-$perf.CommitTotal.ToUInt64())*$perf.PageSize.ToUInt64();$v=@{windows_pid=$PID;windows_start_ticks=$pxStart;captured_unix=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0;host_physical_available_bytes=$m.availPhys;host_commit_headroom_bytes=$head};[Console]::WriteLine(($v|ConvertTo-Json -Compress));[Console]::Out.Flush();Start-Sleep -Milliseconds 700}
"""


def _interop_init_identity(pid):
    process = PROC_ROOT / str(pid)
    if (process.stat().st_uid != 0 or os.readlink(process / "exe") != "/init" or
            process.joinpath("cmdline").read_bytes().split(b"\0", 1)[0] != b"/init"):
        raise SupervisorError("windows_metrics_interop_unavailable")
    fields = process.joinpath("stat").read_text().rsplit(") ", 1)[1].split()
    if int(fields[1]) != 1:
        raise SupervisorError("windows_metrics_interop_unavailable")
    return pid, int(fields[19])


def interop_environment():
    # Client-specific WSL_INTEROP endpoints die with their wsl.exe session and
    # take its Windows children with them. Only the distro-init endpoint may
    # own this long-lived metrics relay; never select by environment or mtime.
    try:
        directory = INTEROP_ROOT.resolve(strict=True)
        info = directory.stat()
        if directory != INTEROP_ROOT or info.st_uid != 0 or info.st_mode & 0o022:
            raise SupervisorError("windows_metrics_interop_unavailable")
        candidate = (directory / "1_interop").resolve(strict=True)
        matched = re.fullmatch(r"([1-9][0-9]*)_interop", candidate.name)
        info = candidate.lstat()
        if (candidate.parent != directory or matched is None or
                not stat.S_ISSOCK(info.st_mode) or info.st_uid != 0):
            raise SupervisorError("windows_metrics_interop_unavailable")
        identity = _interop_init_identity(int(matched[1]))
        if identity != _interop_init_identity(int(matched[1])):
            raise SupervisorError("windows_metrics_interop_unavailable")
    except (OSError, ValueError, IndexError) as exc:
        raise SupervisorError("windows_metrics_interop_unavailable") from exc
    environment = os.environ.copy()
    environment["WSL_INTEROP"] = str(candidate)
    return environment


class Metrics:
    def __init__(self):
        self.values = queue.Queue(maxsize=8)
        self.windows_identity = None
        self.swap_start = None
        self.last_gpu = 0.0
        self.gpu = None
        self.reader_error = None
        self.last_captured_unix = None
        self.process = subprocess.Popen(
            ["/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe",
             "-NoProfile", "-NonInteractive", "-Command", WINDOWS_METRICS_PS],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            bufsize=1, start_new_session=True, env=interop_environment())
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if len(line) > 4096:
                    raise SupervisorError("metrics_line_too_large")
                value = json.loads(line)
                if type(value.get("windows_pid")) is int and type(value.get("windows_start_ticks")) is int:
                    self.windows_identity = (value["windows_pid"], value["windows_start_ticks"])
                if "captured_unix" not in value:
                    continue
                if type(value["captured_unix"]) in (int, float):
                    self.last_captured_unix = value["captured_unix"]
                try:
                    self.values.put_nowait(value)
                except queue.Full:
                    # A consumer may empty the queue after put reported Full.
                    # That is normal progress, not a terminal reader failure.
                    try:
                        self.values.get_nowait()
                    except queue.Empty:
                        pass
                    self.values.put_nowait(value)
        except Exception as exc:
            self.reader_error = type(exc).__name__
            print(json.dumps({"event": "windows_metrics_reader_stopped",
                              "exception_type": self.reader_error}), file=sys.stderr, flush=True)

    def diagnostics(self):
        return {"reader_alive": self.reader.is_alive(), "reader_error": self.reader_error,
                "relay_returncode": self.process.poll(), "pending_samples": self.values.qsize(),
                "last_captured_unix": self.last_captured_unix}

    def sample(self, first=False, force_gpu=False):
        try:
            value = self.values.get(timeout=12 if first else 2)
        except queue.Empty as exc:
            error = SupervisorError("host_metrics_unavailable")
            error.telemetry = self.diagnostics()
            raise error from exc
        while True:
            try:
                value = self.values.get_nowait()
            except queue.Empty:
                break
        if not 0 <= time.time() - value.get("captured_unix", 0) <= 5:
            raise SupervisorError("host_metrics_stale")
        for key in ("host_physical_available_bytes", "host_commit_headroom_bytes", "windows_pid", "windows_start_ticks"):
            if type(value.get(key)) is not int or value[key] < 0:
                raise SupervisorError("host_metrics_invalid")
        self.windows_identity = (value["windows_pid"], value["windows_start_ticks"])
        memory = {line.split(":")[0]: int(line.split()[1]) * 1024
                  for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith(("MemAvailable:", "SwapTotal:", "SwapFree:"))}
        value["guest_memavailable_bytes"] = memory["MemAvailable"]
        swap = {line.split()[0]: int(line.split()[1])
                for line in Path("/proc/vmstat").read_text().splitlines()
                if line.startswith(("pswpin ", "pswpout "))}
        if self.swap_start is None:
            self.swap_start = swap
        if any(swap[key] - self.swap_start[key] > 65536 for key in swap):
            error = SupervisorError("swap_activity_256MiB_stop")
            error.last_sample = dict(value)
            raise error
        if any(value.get(key, 0) < floor for key, floor in FLOORS.items()):
            error = SupervisorError("resource_floor")
            error.last_sample = dict(value)
            raise error
        if force_gpu or time.monotonic() - self.last_gpu >= 5:
            try:
                raw = command(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used,memory.free",
                               "--format=csv,noheader,nounits"], timeout=3)
                self.gpu = [int(part.strip()) for part in raw.split(",")]
                if len(self.gpu) != 2:
                    raise ValueError()
            except Exception as exc:
                raise SupervisorError("gpu_metrics_unavailable") from exc
            self.last_gpu = time.monotonic()
        value["gpu_used_free_MiB"] = self.gpu
        return value

    def close(self):
        try:
            if self.windows_identity:
                pid, started = self.windows_identity
                script = (
                    f"$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue;"
                    f"if($p -and $p.StartTime.ToFileTimeUtc() -eq {started})"
                    "{Stop-Process -Id " + str(pid) + " -Force -ErrorAction Stop;"
                    "$p.WaitForExit(5000)|Out-Null};"
                    "$pxDeadline=[DateTime]::UtcNow.AddSeconds(5);do{"
                    f"$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue;"
                    f"if(-not $p -or $p.StartTime.ToFileTimeUtc() -ne {started})" + "{exit 0};"
                    "Start-Sleep -Milliseconds 100}while([DateTime]::UtcNow -lt $pxDeadline);exit 4")
                subprocess.run(
                    ["/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe",
                     "-NoProfile", "-NonInteractive", "-Command", script],
                    check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env=interop_environment())
            elif self.process.poll() is None:
                raise SupervisorError("windows_metrics_identity_missing")
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(5)
            self.reader.join(2)


def await_admission(metrics, config, after_unix, *, cancel_check=lambda: False, observe=None):
    """Observe fresh pre-start counters for at most 30s; retain all hard floors."""
    deadline = time.monotonic() + 30
    while True:
        if cancel_check():
            raise SupervisorError("cancelled")
        sample = metrics.sample(force_gpu=True)
        if observe is not None:
            observe(sample)
        fresh = sample.get("captured_unix", 0) > after_unix
        admitted = fresh and all(sample.get(key, 0) >= floor
                                 for key, floor in config["admission_min_bytes"].items())
        if admitted and sample["gpu_used_free_MiB"][1] >= 8192:
            return sample
        if time.monotonic() >= deadline:
            error = SupervisorError("resource_admission")
            error.last_sample = sample
            error.fresh_after_boundary = fresh
            raise error


def _configured_tenants(key, *, required=True):
    raw = os.environ.get(key, "")
    if len(raw) > 4096:
        raise SupervisorError("tenant_authorization_invalid")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if ((required and not values) or len(values) > 64
            or any(not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,62}[A-Za-z0-9])?", item) for item in values)):
        raise SupervisorError("tenant_authorization_invalid")
    return set(values)


def authorize_tenant(value):
    enabled = _configured_tenants("PROJECTX_GEMMA_TENANTS")
    local = _configured_tenants("LOCAL_ALLOWED_TENANTS", required=False)
    if local and not enabled.issubset(local):
        raise SupervisorError("tenant_authorization_invalid")
    # Local tenants may observe readiness; generation requires explicit enablement.
    allowed = (local or enabled) if value.get("action") == "status" else enabled
    if value.get("tenant_id") not in allowed:
        raise SupervisorError("tenant_not_authorized")


def validate_request(value):
    if not isinstance(value, dict):
        raise SupervisorError("invalid_request")
    authorize_tenant(value)
    if value.get("action") in {"status", "select"}:
        allowed = {"tenant_id", "action"} if value["action"] == "status" else {"tenant_id", "action", "model"}
        if set(value) != allowed or (value["action"] == "select" and value.get("model") != MODEL_ID):
            raise SupervisorError("invalid_model_selection")
        return value
    if value.get("action") in {"completion", "measure"}:
        if set(value) - {"tenant_id", "action", "messages", "settings", "request_id"}:
            raise RequestRejected("invalid_request")
        if "request_id" in value and (not isinstance(value["request_id"], str) or
                not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value["request_id"])):
            raise RequestRejected("invalid_request")
        messages = value.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 6:
            raise RequestRejected("invalid_messages")
        chars = 0
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"} or
                    message["role"] not in {"system", "user", "assistant"} or
                    not isinstance(message["content"], str)):
                raise RequestRejected("invalid_messages")
            chars += len(message["content"])
        if not 1 <= chars <= 90000:
            raise RequestRejected("invalid_messages")
        completion_settings(value)
        return value
    if set(value) - {"tenant_id", "messages", "context_blocks", "query"}:
        raise SupervisorError("invalid_request")
    messages = value.get("messages")
    if not isinstance(messages, list) or not 2 <= len(messages) <= 6:
        raise SupervisorError("invalid_messages")
    chars = 0
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise SupervisorError("invalid_messages")
        if message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
            raise SupervisorError("invalid_messages")
        chars += len(message["content"])
    if chars > 90000 or messages[0]["role"] != "system" or messages[-1]["role"] != "user":
        raise SupervisorError("invalid_messages")
    blocks = value.get("context_blocks")
    if blocks is not None:
        query = value.get("query")
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 8000:
            raise SupervisorError("invalid_query")
        if not isinstance(blocks, list) or not 1 <= len(blocks) <= 6:
            raise SupervisorError("invalid_evidence_blocks")
        identifiers = set()
        for block in blocks:
            if not isinstance(block, dict) or set(block) != {"id", "text"}:
                raise SupervisorError("invalid_evidence_blocks")
            identifier, text = block["id"], block["text"]
            if not isinstance(identifier, str) or not re.fullmatch(r"[1-9][0-9]{0,2}", identifier) or identifier in identifiers:
                raise SupervisorError("invalid_evidence_ids")
            if not isinstance(text, str) or not text.startswith("[" + identifier + "]") or not 1 <= len(text) <= 30000:
                raise SupervisorError("invalid_evidence_blocks")
            identifiers.add(identifier)
        if sum(len(block["text"]) for block in blocks) > 80000:
            raise SupervisorError("invalid_evidence_blocks")
    return value


def completion_settings(request):
    settings = request.get("settings", {})
    allowed = {"temperature", "top_p", "top_k", "seed", "max_tokens", "timeout_seconds",
               "repeat_penalty", "stop", "context_reserve_tokens"}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise RequestRejected("invalid_generation_settings")
    values = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 42,
              "max_tokens": 768, "timeout_seconds": 2400.0, "repeat_penalty": 1.0,
              "stop": [], "context_reserve_tokens": 256}
    values.update(settings)
    # Admission below still enforces the exact prompt + output + reserve
    # envelope. This is a request limit, not a larger model context.
    for key, low, high in (("max_tokens", 1, MAX_OUTPUT_TOKENS), ("top_k", 0, 1000),
                           ("seed", -1, 2**31 - 1), ("context_reserve_tokens", 128, 1024)):
        if type(values[key]) is not int or not low <= values[key] <= high:
            raise RequestRejected("invalid_generation_settings")
    for key, low, high in (("temperature", 0, 5), ("top_p", 0, 1),
                           ("repeat_penalty", 0, 10), ("timeout_seconds", 1, 2400)):
        if (type(values[key]) not in (int, float) or not math.isfinite(values[key]) or
                not low <= values[key] <= high):
            raise RequestRejected("invalid_generation_settings")
    if (not isinstance(values["stop"], list) or len(values["stop"]) > 8 or
            any(not isinstance(item, str) or not 1 <= len(item) <= 256 for item in values["stop"])):
        raise RequestRejected("invalid_generation_settings")
    return values


def read_request(connection):
    raw = bytearray()
    connection.settimeout(5)
    while True:
        preview = connection.recv(65536, socket.MSG_PEEK)
        if not preview:
            raise SupervisorError("request_disconnected")
        end = preview.find(b"\n")
        count = end + 1 if end >= 0 else len(preview)
        raw.extend(connection.recv(count))
        if len(raw) > MAX_REQUEST:
            raise SupervisorError("request_too_large")
        if raw.endswith(b"\n"):
            return validate_request(json.loads(bytes(raw)))


def cancelled(connection):
    if _STOP.is_set():
        return True
    readable, _, _ = select.select([connection], [], [], 0)
    if readable:
        # Only the same authenticated connection can cancel its current call.
        # Any post-request data aborts that call; it is never another command.
        connection.recv(64, socket.MSG_DONTWAIT)
        return True
    return False


def count_tokens(messages):
    templated = json_http("/apply-template", {
        "messages": messages, "chat_template_kwargs": {"enable_thinking": False}})
    prompt = templated.get("prompt")
    if not isinstance(prompt, str):
        raise SupervisorError("invalid_template_response")
    result = json_http("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise SupervisorError("invalid_tokenizer_response")
    return len(tokens)


def measure_completion(request):
    """Use the same model template and arithmetic for planning and admission."""
    settings = completion_settings(request)
    tokens = count_tokens(request["messages"])
    return {"prompt_tokens": tokens, "max_tokens": settings["max_tokens"],
            "context_reserve_tokens": settings["context_reserve_tokens"],
            "context_size_tokens": 4096,
            "fits": tokens + settings["max_tokens"] + settings["context_reserve_tokens"] <= 4096}


def pack_request(request):
    messages = [dict(item) for item in request["messages"]]
    if request.get("action") == "completion":
        measured = measure_completion(request)
        tokens = measured["prompt_tokens"]
        if not measured["fits"]:
            raise RequestRejected("evidence_context_too_large", prompt_tokens=tokens,
                                  max_tokens=measured["max_tokens"], context_size=4096)
        return messages, tokens, []
    blocks = request.get("context_blocks")
    if blocks is None:
        tokens = count_tokens(messages)
        if tokens + 768 + 128 > 4096:
            raise RequestRejected("evidence_context_too_large", prompt_tokens=tokens,
                                  max_tokens=768, context_size=4096)
        return messages, tokens, []
    selected = []
    included = []
    # History is optional. Keep complete evidence blocks within the exact model
    # template budget; never truncate a paragraph/column or renumber handles.
    prefix = messages[:-1]
    def rendered(items):
        return prefix + [{"role": "user", "content":
            "Source passages:\n" + "\n\n".join(item["text"] for item in items) +
            "\n\nQuestion: " + request["query"].strip()}]
    for block in blocks:
        candidate = selected + [block]
        attempt = rendered(candidate)
        tokens = count_tokens(attempt)
        while not selected and tokens + 768 + 128 > 4096 and len(prefix) > 1:
            prefix.pop(1)
            if len(prefix) > 1 and prefix[1]["role"] == "assistant":
                prefix.pop(1)
            attempt = rendered(candidate)
            tokens = count_tokens(attempt)
        if tokens + 768 + 128 <= 4096:
            selected.append(block)
            included.append(block["id"])
    if not selected:
        raise RequestRejected("no_complete_evidence_block_fits")
    final = rendered(selected)
    return final, count_tokens(final), included


def stop_owned_server(server, pidfd):
    if server is None:
        return
    try:
        if server.poll() is None:
            if pidfd is None:
                # Popen still owns this unreaped child if pidfd creation failed.
                server.terminate()
            else:
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                if pidfd is None:
                    server.kill()
                else:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                server.wait(5)
        else:
            server.wait()
    finally:
        if pidfd is not None:
            os.close(pidfd)


def process_identity(server):
    if server is None:
        return {}
    try:
        text = Path(f"/proc/{server.pid}/stat").read_text()
        ticks = int(text.rsplit(") ", 1)[1].split()[19])
        return {"model_pid": server.pid, "model_start_ticks": ticks,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, ValueError, IndexError):
        return {"model_pid": server.pid}


class Session:
    """One model owner; selection, process ownership and resource monitoring agree."""
    def __init__(self, config, *, metrics_factory=Metrics, start_watchdog=True):
        self.config = config
        self.metrics_factory = metrics_factory
        self.operation = threading.Lock()
        self.state_lock = threading.RLock()
        self.server = self.pidfd = self.metrics = None
        self.identity = {}
        self.model_id = None
        self.last_sample = None
        self.resource_minima = {}
        self.startup_capture = None
        self.startup_stage = "idle"
        self.diagnostics = {}
        self.last_persist = time.monotonic()
        self.admission = {}
        self.abort = threading.Event()
        self.closed = threading.Event()
        self.selected = MODEL_ID
        self.state = "error"
        self.error = "gemma_not_started_select_gemma"
        state_path = STATE_DIR / "state.json"
        if state_path.exists():
            self.error = "supervisor_restarted_select_gemma_to_retry"
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        if start_watchdog:
            self.watchdog.start()

    def snapshot(self):
        with self.state_lock:
            loaded = self.server is not None and self.server.poll() is None
            return {"ok": True, "selected": self.selected, "state": self.state,
                    "ready": self.state == "ready" and loaded, "loaded": loaded,
                    "error": self.error, **self.identity,
                    "resources": self.last_sample, "resource_minima": dict(self.resource_minima),
                    "diagnostics": dict(self.diagnostics)}

    def _publish(self, state, error=None):
        with self.state_lock:
            self.state, self.error = state, error
            atomic_json(STATE_DIR / "state.json", {
                "selected": self.selected, "state": state, "error": error,
                **self.identity, "resources": self.last_sample,
                "resource_minima": dict(self.resource_minima), "diagnostics": dict(self.diagnostics),
                "updated_unix": time.time()})
            self.last_persist = time.monotonic()

    def _close_owned(self):
        # Keep ownership if reaping fails: never start a second large model.
        if self.server is not None:
            stop_owned_server(self.server, self.pidfd)
            self.server = self.pidfd = None
        if self.startup_capture is not None:
            self.startup_capture.finish()
            self.startup_capture = None
        if self.metrics is not None:
            self.metrics.close()
            self.metrics = None
        self.model_id = None

    def _record_failure(self, exc):
        kind = type(exc).__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", kind):
            kind = "Exception"
        details = {"stage": self.startup_stage, "exception_type": kind}
        for key in ("errno", "returncode"):
            value = getattr(exc, key, None)
            if type(value) is int:
                details[key] = value
        if isinstance(exc, HTTPError) and type(exc.code) is int:
            details["http_status"] = exc.code
        if isinstance(getattr(exc, "telemetry", None), dict):
            details["telemetry"] = dict(exc.telemetry)
        self.diagnostics.update(details)
        # Deliberately omit str(exc), traceback, argv, HTTP bodies and headers:
        # those may carry passages, generated answers or credentials.
        print(json.dumps({"event": "gemma_operation_failed", **details}), file=sys.stderr, flush=True)

    def _check_cancel(self, connection):
        if _STOP.is_set():
            raise SupervisorError("supervisor_stopping")
        if self.abort.is_set():
            raise SupervisorError(self.error or "gemma_session_failed")
        if connection is not None and cancelled(connection):
            raise SupervisorError("cancelled")

    def _observe(self, sample):
        with self.state_lock:
            self.last_sample = sample
            for key in FLOORS:
                if type(sample.get(key)) is int:
                    self.resource_minima[key] = min(self.resource_minima.get(key, sample[key]), sample[key])
            gpu = sample.get("gpu_used_free_MiB")
            if isinstance(gpu, list) and len(gpu) == 2:
                self.resource_minima["gpu_free_MiB"] = min(self.resource_minima.get("gpu_free_MiB", gpu[1]), gpu[1])

    def _sample(self, **kwargs):
        sample = self.metrics.sample(**kwargs)
        self._observe(sample)
        return sample

    def _watch(self):
        while not self.closed.wait(0.7):
            with self.state_lock:
                metrics, server = self.metrics, self.server
                # An unconfirmed measurement health probe retains ownership
                # but sets non-ready. Keep its resource watchdog active.
                watch = (self.state in {"ready", "busy"} or
                         (self.state == "error" and not self.abort.is_set())) and self.selected == "gemma-4-31b"
            if not watch or metrics is None or server is None:
                continue
            try:
                if server.poll() is not None:
                    raise SupervisorError("model_exited_select_gemma_to_retry")
                sample = metrics.sample()
                with self.state_lock:
                    if metrics is self.metrics:
                        self._observe(sample)
                        if time.monotonic() - self.last_persist >= 3:
                            self._publish(self.state, self.error)
            except Exception as exc:
                with self.state_lock:
                    if (metrics is not self.metrics or
                            not (self.state in {"ready", "busy"} or
                                 (self.state == "error" and not self.abort.is_set()))):
                        continue
                    code = str(exc) if isinstance(exc, SupervisorError) else "resource_monitor_failed"
                    self._record_failure(exc)
                    if isinstance(getattr(exc, "last_sample", None), dict):
                        self._observe(exc.last_sample)
                    self.abort.set()
                    self._publish("error", code)
                if self.operation.acquire(blocking=False):
                    try:
                        self._close_owned()
                    except Exception:
                        self._publish("error", "owned_process_cleanup_failed")
                    finally:
                        self.operation.release()

    def _initialize(self, connection):
        self.startup_stage = "cleanup"
        self.diagnostics = {}
        self.selected = MODEL_ID
        self.abort.clear()
        self.resource_minima = {}
        self.last_sample = None
        self.admission = {"passed": False}
        self._publish("initializing")
        self._close_owned()
        self._check_cancel(connection)
        self.startup_stage = "service_preflight"
        if shutdown_in_progress():
            raise SupervisorError("system_shutdown_in_progress")
        if not service_state("project-x-wsl.service"):
            raise SupervisorError("api_not_ready")
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", MODEL_PORT))
        require_legacy_runtime_inactive()
        self.startup_stage = "resource_monitor_start"
        self.metrics = self.metrics_factory()
        self._sample(first=True)
        self._check_cancel(connection)
        if not service_state("project-x-wsl.service"):
            raise SupervisorError("api_not_preserved")
        after = time.time()
        samples = []
        self.admission = {"passed": False, "after_unix": after, "samples": samples}
        self.startup_stage = "resource_admission"
        await_admission(self.metrics, self.config, after,
                        cancel_check=lambda: _STOP.is_set() or (connection is not None and cancelled(connection)),
                        observe=lambda value: samples.append(dict(value)))
        self.admission = {"passed": True, "after_unix": after, "samples": samples}
        self.startup_stage = "model_spawn"
        environment = server_environment(self.config)
        self.startup_capture = StartupDiagnostics()
        self.diagnostics["startup_stderr_path"] = str(self.startup_capture.path)
        try:
            self.server = subprocess.Popen(self.config["server_argv"], env=environment,
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, start_new_session=True)
        except OSError as exc:
            raise SupervisorError("model_start_failed_select_gemma_to_retry") from exc
        self.startup_capture.start(self.server.stderr)
        self.pidfd = os.pidfd_open(self.server.pid)
        self.identity = process_identity(self.server)
        self.startup_stage = "model_readiness"
        deadline = time.monotonic() + 600
        while True:
            self._check_cancel(connection)
            self._sample()
            if self.server.poll() is not None:
                raise SupervisorError("model_exited_before_ready")
            try:
                if json_http("/health", timeout=1).get("status") == "ok":
                    break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise SupervisorError("model_readiness_timeout")
        self.startup_stage = "listener_ownership"
        listeners = command(["ss", "-H", "-ltnp", "sport", "=", ":" + str(MODEL_PORT)]).splitlines()
        if not listeners or any(row.split()[3] != "127.0.0.1:" + str(MODEL_PORT) for row in listeners):
            raise SupervisorError("model_listener_not_loopback")
        if any(re.findall(r"pid=([0-9]+),", row) != [str(self.server.pid)] for row in listeners):
            raise SupervisorError("model_listener_not_owned")
        self.startup_stage = "model_identity"
        models = json_http("/v1/models").get("data")
        if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
            raise SupervisorError("model_identity_invalid")
        self.model_id = models[0].get("id")
        if self.model_id != MODEL_ID:
            raise SupervisorError("model_identity_invalid")
        for field in ("path", "model_path"):
            if field in models[0] and models[0][field] != MODEL:
                raise SupervisorError("model_identity_invalid")
        self.startup_capture.stop_capture()
        self.startup_stage = "ready"
        self._publish("ready")

    def _complete(self, request, connection, operation_id):
        generic = request.get("action") == "completion"
        settings = completion_settings(request) if generic else {
            "temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 42,
            "max_tokens": 768, "timeout_seconds": 2400.0}
        messages, prompt_tokens, included = pack_request(request)
        self._check_cancel(connection)
        payload = {"model": self.model_id, "messages": messages, "stream": False,
                   **{key: value for key, value in settings.items()
                      if key not in {"timeout_seconds", "context_reserve_tokens"}},
                   "chat_template_kwargs": {"enable_thinking": False},
                   "cache_prompt": False}
        if payload["seed"] < 0:
            del payload["seed"]
        # A content-free record of the exact allowance passed to the model,
        # correlated with the application request and the supervised worker.
        audit = {"request_id": request.get("request_id"), "operation_id": operation_id,
                 "model_id": self.model_id, "model_pid": self.server.pid,
                 "model_start_ticks": self.identity.get("model_start_ticks"),
                 "prompt_tokens": prompt_tokens, "max_tokens": payload["max_tokens"],
                 "context_reserve_tokens": settings.get("context_reserve_tokens", 128)}
        print(json.dumps({"event": "gemma_worker_request", **audit}), file=sys.stderr, flush=True)
        timeout = settings["timeout_seconds"]
        outcomes = queue.Queue(maxsize=1)
        def complete():
            try:
                outcomes.put((json_http("/v1/chat/completions", payload, timeout=timeout), None))
            except Exception as exc:
                outcomes.put((None, exc))
        worker = threading.Thread(target=complete, daemon=True)
        worker.start()
        deadline = time.monotonic() + timeout
        try:
            while worker.is_alive():
                self._check_cancel(connection)
                if self.server.poll() is not None:
                    raise SupervisorError("model_exited_during_generation")
                if time.monotonic() > deadline:
                    raise SupervisorError("model_generation_timeout")
                worker.join(0.2)
            self._check_cancel(connection)
            response, failure = outcomes.get_nowait()
            if failure is not None:
                if isinstance(failure, (TimeoutError, socket.timeout)) or isinstance(
                        getattr(failure, "reason", None), (TimeoutError, socket.timeout)):
                    raise SupervisorError("model_generation_timeout") from failure
                raise SupervisorError("model_request_failed")
            if response.get("model") != self.model_id:
                raise SupervisorError("model_response_identity_mismatch")
            choice = response.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content")
            reason = choice.get("finish_reason")
            if (not isinstance(content, str) or reason not in {"stop", "length"} or
                    (reason == "stop" and not content.strip())):
                raise SupervisorError("model_response_invalid")
            completion_tokens = (response.get("usage") or {}).get("completion_tokens")
            print(json.dumps({"event": "gemma_worker_response", **audit,
                              "finish_reason": reason, "completion_tokens": completion_tokens}),
                  file=sys.stderr, flush=True)
            return {"ok": True, "answer": content.strip(), "backend": "gemma-4-31b",
                    "model_id": self.model_id, "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": reason, "unverified": True, "included_evidence_ids": included}
        finally:
            if worker.is_alive():
                self._close_owned()
                worker.join(10)
                if worker.is_alive():
                    raise SupervisorError("owned_request_cleanup_failed")

    def _retain_after_measure_error(self):
        """A failed tokenizer-only request need not destroy a live model.

        A failed health probe leaves the owner loaded but not ready; the
        watchdog still enforces the resource floor on that retained process.
        Never override a concurrent resource abort, shutdown, or exited child.
        """
        with self.state_lock:
            server = self.server
            if (self.state not in {"ready", "busy"} or server is None or
                    server.poll() is not None or self.abort.is_set() or _STOP.is_set()):
                return False
        try:
            healthy = json_http("/health", timeout=1).get("status") == "ok"
        except Exception:
            healthy = False
        with self.state_lock:
            if (self.server is not server or server.poll() is not None or
                    self.state not in {"ready", "busy"} or self.abort.is_set() or _STOP.is_set()):
                return False
            self._publish("ready" if healthy else "error",
                          None if healthy else "context_measurement_health_unconfirmed")
            return True

    def handle(self, request, connection=None):
        if request.get("action") == "status":
            return self.snapshot()
        if not self.operation.acquire(blocking=False):
            return {**self.snapshot(), "ok": False, "error": "model_busy"}
        identifier = uuid.uuid4().hex
        operation_error = None
        failure_prompt_tokens = None
        try:
            explicit = request.get("action") == "select"
            if explicit and request.get("model") != MODEL_ID:
                raise RequestRejected("invalid_model_selection")
            if explicit and (self.state == "error" or self.server is None or self.server.poll() is not None):
                self._initialize(connection)
            elif self.state != "ready":
                raise SupervisorError(self.error or "gemma_not_ready")
            if explicit:
                return self.snapshot()
            self._publish("busy")
            if request.get("action") == "measure":
                self.startup_stage = "context_measurement"
                self._check_cancel(connection)
                result = {"ok": True, **measure_completion(request)}
            else:
                self.startup_stage = "generation"
                result = self._complete(request, connection, identifier)
            with self.state_lock:
                self._check_cancel(connection)
                self._publish("ready")
            result.update(self.snapshot())
            return result
        except Exception as exc:
            self._record_failure(exc)
            code = str(exc) if isinstance(exc, SupervisorError) else "supervisor_internal_error"
            operation_error = code
            if isinstance(exc, RequestRejected) and type(exc.details.get("prompt_tokens")) is int:
                failure_prompt_tokens = exc.details["prompt_tokens"]
            if isinstance(exc, RequestRejected):
                with self.state_lock:
                    if (self.state in {"ready", "busy"} and self.server is not None and
                            self.server.poll() is None and not self.abort.is_set() and not _STOP.is_set()):
                        self._publish("ready")
                        return {**self.snapshot(), "ok": False, "error": code, **exc.details}
                    if self.abort.is_set() and self.error:
                        code = operation_error = self.error
            if request.get("action") == "measure" and self._retain_after_measure_error():
                code = operation_error = "cancelled" if code == "cancelled" else "context_measurement_failed"
                return {**self.snapshot(), "ok": False, "error": code}
            if self.abort.is_set() and self.error:
                code = operation_error = self.error
            if isinstance(getattr(exc, "last_sample", None), dict):
                self._observe(exc.last_sample)
            try:
                self._close_owned()
            except Exception:
                code = "owned_process_cleanup_failed"
            self._publish("error", code)
            return {**self.snapshot(), "ok": False, "error": code}
        finally:
            try:
                if operation_error and request.get("action") in {"completion", "measure"}:
                    safe_error = operation_error if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", operation_error) else "operation_failed"
                    settings = request.get("settings") if isinstance(request.get("settings"), dict) else {}
                    print(json.dumps({"event": "gemma_operation_result", "request_id": request.get("request_id"),
                                      "operation_id": identifier, "error": safe_error,
                                      "prompt_tokens": failure_prompt_tokens,
                                      "max_tokens": settings.get("max_tokens", 768),
                                      "context_reserve_tokens": settings.get("context_reserve_tokens", 256)}),
                          file=sys.stderr, flush=True)
                snapshot = self.snapshot()
                atomic_json(STATE_DIR / (identifier + ".json"), {
                    "request_id": identifier, "finished_unix": time.time(),
                    "operation": request.get("action", "generate"), **snapshot,
                    "operation_error": operation_error,
                    "admission": self.admission})
            finally:
                self.operation.release()

    def close(self):
        self.closed.set()
        self.abort.set()
        # Called after socket workers finish; cgroup kill is the last systemd guard.
        with self.operation:
            self._close_owned()
            self._publish("error", "supervisor_stopped")


def allowed_peer(connection):
    _pid, uid, _gid = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    return uid in {0, pwd.getpwnam("projectx").pw_uid}


def serve():
    config = load_config()
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    info = STATE_DIR.lstat()
    if info.st_uid != 0 or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise SupervisorError("unsafe_state_directory")
    if int(os.environ.get("LISTEN_PID", "0")) != os.getpid() or os.environ.get("LISTEN_FDS") != "1":
        raise SupervisorError("systemd_socket_required")
    session = Session(config)
    listener = socket.socket(fileno=3)
    listener.settimeout(1)
    clients = set()
    clients_lock = threading.Lock()
    def handle(connection):
        try:
            with connection:
                try:
                    if not allowed_peer(connection):
                        raise SupervisorError("peer_not_authorized")
                    request = read_request(connection)
                    response = session.handle(request, connection)
                    response["tenant_id"] = request["tenant_id"]
                except Exception as exc:
                    code = str(exc) if isinstance(exc, SupervisorError) else "invalid_request"
                    response = {"ok": False, "error": code}
                try:
                    raw = json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n"
                    if len(raw) > MAX_RESPONSE:
                        raw = b'{"ok":false,"error":"response_too_large"}\n'
                    connection.sendall(raw)
                except OSError:
                    pass
        finally:
            with clients_lock:
                clients.discard(threading.current_thread())
    try:
        while not _STOP.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            with clients_lock:
                if len(clients) >= 8:
                    connection.close()
                    continue
                thread = threading.Thread(target=handle, args=(connection,), daemon=True)
                clients.add(thread)
                thread.start()
    finally:
        listener.close()
        with clients_lock:
            workers = list(clients)
        for thread in workers:
            thread.join(30)
        session.close()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: _STOP.set())
    serve()


if __name__ == "__main__":
    main()

