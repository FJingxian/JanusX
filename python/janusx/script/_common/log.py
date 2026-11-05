from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Iterator

import psutil
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def _console_symbol(preferred: str, fallback: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        str(preferred).encode(enc)
        return str(preferred)
    except Exception:
        return str(fallback)


class _LevelPrefixFormatter(logging.Formatter):
    """Add fixed prefix for warning/error log records."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.ERROR:
            return msg if msg.startswith("Error: ") else f"Error: {msg}"
        if record.levelno == logging.WARNING:
            return msg if msg.startswith("Warning: ") else f"Warning: {msg}"
        return msg


class _ColorLevelPrefixFormatter(_LevelPrefixFormatter):
    """Prefix warning/error and optionally colorize console output."""

    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _RESET = "\033[0m"
    _FAIL = _console_symbol("\u2718\ufe0e", "[X]")
    _WARN = "!"

    def __init__(self, *, enable_color: bool) -> None:
        super().__init__()
        self.enable_color = bool(enable_color)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.ERROR:
            msg = f"{self._FAIL} {msg}"
        elif record.levelno == logging.WARNING:
            msg = f"{self._WARN} {msg}"
        if not self.enable_color:
            return msg
        if record.levelno >= logging.ERROR:
            return f"{self._RED}{msg}{self._RESET}"
        if record.levelno == logging.WARNING:
            return f"{self._YELLOW}{msg}{self._RESET}"
        return msg


def _enable_windows_ansi(stream) -> bool:
    """Enable ANSI escape on Windows consoles; no-op on non-Windows."""
    if os.name != "nt":
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if mode.value & enable_vt:
            return True
        return kernel32.SetConsoleMode(handle, mode.value | enable_vt) != 0
    except Exception:
        return False


def _supports_color(stream) -> bool:
    """Best-effort cross-platform color support detection."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":
        return _enable_windows_ansi(stream)
    return True


def _fallback_log_path(log_file_path: str) -> str:
    base, ext = os.path.splitext(str(log_file_path))
    suffix = ext if ext else ".log"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    cand = f"{base}.{stamp}.{pid}{suffix}"
    idx = 1
    while os.path.exists(cand):
        idx += 1
        cand = f"{base}.{stamp}.{pid}.{idx}{suffix}"
    return cand


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _process_start_time(pid: int) -> float | None:
    try:
        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def _classify_owner(record: dict[str, Any]) -> tuple[str, str]:
    host = str(record.get("hostname", ""))
    if not host or host != socket.gethostname():
        return "unverifiable", "The lock owner is on another host."
    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        return "unverifiable", "The lock does not contain a valid PID."
    try:
        current_start = float(psutil.Process(pid).create_time())
    except psutil.NoSuchProcess:
        return "stale", "The recorded process no longer exists."
    except (psutil.AccessDenied, OSError, ValueError):
        return "unverifiable", "The recorded process state cannot be inspected."
    try:
        recorded_start = float(record.get("process_start_time"))
    except (TypeError, ValueError):
        return "unverifiable", "The lock has no process start identity."
    if abs(current_start - recorded_start) > 0.01:
        return "stale", "The recorded PID now belongs to another process."
    return "active", "The recorded process is still running."


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while recording JanusX log metadata")
        view = view[written:]


def _write_json_atomic(path: str, record: dict[str, Any]) -> None:
    parent = os.path.dirname(path) or os.curdir
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.update.", dir=parent
    )
    try:
        try:
            original_mode = stat.S_IMODE(os.stat(path).st_mode)
        except FileNotFoundError:
            original_mode = None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        if original_mode is not None:
            os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _create_publish_temp(destination: str) -> tuple[int, str]:
    parent = os.path.dirname(destination) or os.curdir
    base = os.path.basename(destination)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    for _ in range(10):
        path = os.path.join(parent, f"{base}.tmp.{uuid.uuid4().hex}")
        try:
            return os.open(path, flags, 0o666), path
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate a temporary file beside {destination}")


def _local_temp_directory() -> str:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        candidates = ([os.path.join(local_app_data, "Temp")] if local_app_data else [])
        candidates.append(tempfile.gettempdir())
    else:
        candidates = ["/tmp", "/var/tmp", tempfile.gettempdir()]

    remote_tokens = ("nfs", "cifs", "smb", "afp", "sshfs", "9p")
    try:
        partitions = psutil.disk_partitions(all=True)
    except Exception:
        partitions = []

    def is_remote(path: str) -> bool:
        target = os.path.realpath(path)
        matches = []
        for part in partitions:
            mount = os.path.realpath(str(part.mountpoint))
            try:
                if os.path.commonpath((target, mount)) == mount:
                    matches.append((len(mount), part))
            except (OSError, ValueError):
                continue
        if not matches:
            return False
        part = max(matches, key=lambda item: item[0])[1]
        fs_type = str(getattr(part, "fstype", "")).lower()
        device = str(getattr(part, "device", "")).lower()
        if any(token in fs_type for token in remote_tokens):
            return True
        if os.name == "nt":
            return device.startswith("\\\\")
        return ":" in device and not device.startswith("/dev/")

    seen: set[str] = set()
    for candidate in candidates:
        path = os.path.realpath(os.path.expanduser(candidate))
        if path in seen or not os.path.isdir(path) or not os.access(path, os.W_OK):
            continue
        seen.add(path)
        if not is_remote(path):
            return path
    raise OSError(
        "JanusX logging requires a writable local temporary directory; "
        "TMPDIR and the platform temporary directories appear to be unavailable "
        "or network-mounted."
    )


def create_local_log_spool(run_id: str, *, prefix: str = "janusx-log") -> str:
    """Create a unique log spool on a writable local filesystem."""
    fd, path = tempfile.mkstemp(
        prefix=f"{_safe_component(prefix)}-{_safe_component(run_id)}-",
        suffix=".log",
        dir=_local_temp_directory(),
    )
    os.close(fd)
    return path


def publish_log_spool(spool_path: str, log_path: str) -> None:
    """Atomically publish a closed local spool to its requested destination."""
    src = os.path.abspath(os.path.expanduser(str(spool_path)))
    dst = os.path.abspath(os.path.expanduser(str(log_path)))
    if src == dst:
        return
    dst_dir = os.path.dirname(dst) or os.curdir
    os.makedirs(dst_dir, mode=0o755, exist_ok=True)
    try:
        dst_mode = stat.S_IMODE(os.stat(dst).st_mode)
    except FileNotFoundError:
        dst_mode = None
    fd, tmp_path = _create_publish_temp(dst)
    try:
        with os.fdopen(fd, "wb") as out_fh, open(src, "rb") as in_fh:
            shutil.copyfileobj(in_fh, out_fh, length=1024 * 1024)
            out_fh.flush()
            os.fsync(out_fh.fileno())
        if dst_mode is not None:
            os.chmod(tmp_path, dst_mode)
        os.replace(tmp_path, dst)
        try:
            dir_fd = os.open(dst_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _files_equal(path_a: str, path_b: str) -> bool:
    try:
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
        with open(path_a, "rb") as fh_a, open(path_b, "rb") as fh_b:
            while True:
                block_a = fh_a.read(1024 * 1024)
                block_b = fh_b.read(1024 * 1024)
                if block_a != block_b:
                    return False
                if not block_a:
                    return True
    except OSError:
        return False


def _has_completed_log_footer(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 4096), os.SEEK_SET)
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    last_line = tail.rstrip().splitlines()[-1] if tail.rstrip() else ""
    return (
        last_line.startswith("JanusX log status: complete; run_id: ")
        or last_line.startswith("GWAS run status: done; run_id: ")
    )


def _log_has_run_footer(path: str, run_id: str, status: str) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 4096), os.SEEK_SET)
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    last_line = tail.rstrip().splitlines()[-1] if tail.rstrip() else ""
    expected = f"GWAS run status: {status}; run_id: {_safe_component(run_id)}"
    return last_line.endswith(expected)


class LogFileBusy(RuntimeError):
    """Raised when another process owns a shared log destination."""


class LogFileLease:
    """Serialize writers for one log path and recover confirmed stale runs."""

    def __init__(self, log_path: str, run_id: str) -> None:
        self.log_path = os.path.abspath(os.path.expanduser(str(log_path)))
        self.run_id = str(run_id)
        self.lock_path = f"{self.log_path}.lock"
        self.recovery_path = f"{self.lock_path}.recover"
        self.spool_path = ""
        self.acquired = False

    def acquire(self, spool_path: str) -> None:
        self.spool_path = os.path.abspath(str(spool_path))
        os.makedirs(os.path.dirname(self.lock_path) or os.curdir, mode=0o755, exist_ok=True)
        if os.path.exists(self.recovery_path):
            raise LogFileBusy(f"Log recovery is already in progress: {self.recovery_path}")
        try:
            self._create_lock()
            return
        except FileExistsError:
            pass
        existing = self._read_lock()
        state, detail = _classify_owner(existing)
        if state != "stale":
            raise LogFileBusy(
                f"Log destination is owned by an active or unverifiable run "
                f"(host={existing.get('hostname', 'unknown')}, "
                f"pid={existing.get('pid', 'unknown')}, "
                f"run_id={existing.get('run_id', 'unknown')}). {detail}"
            )
        self._recover_stale_and_acquire(existing)

    def _create_lock(self) -> None:
        record = {
            "format": 1,
            "run_id": self.run_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "process_start_time": _process_start_time(os.getpid()),
            "spool_path": self.spool_path,
            "started_at": _utc_now(),
            "status": "running",
        }
        fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            _write_all(fd, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(fd)
            self.acquired = True
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(self.lock_path)
            except OSError:
                pass
            raise
        else:
            os.close(fd)

    def _recover_stale_and_acquire(self, observed: dict[str, Any]) -> None:
        recovery_id = uuid.uuid4().hex
        marker = {
            "recovery_id": recovery_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": _utc_now(),
        }
        try:
            fd = os.open(self.recovery_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise LogFileBusy(f"Log recovery is already in progress: {self.recovery_path}") from exc
        try:
            _write_all(fd, (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            current = self._read_lock()
            if current.get("run_id") != observed.get("run_id"):
                raise LogFileBusy(f"Log lock changed during recovery: {self.lock_path}")
            state, detail = _classify_owner(current)
            if state != "stale":
                raise LogFileBusy(f"Log lock is no longer stale: {detail}")
            self._archive_stale_run(current)
            latest = self._read_lock()
            if latest.get("run_id") != current.get("run_id"):
                raise LogFileBusy(f"Log lock changed during recovery: {self.lock_path}")
            os.remove(self.lock_path)
            self._create_lock()
        finally:
            try:
                with open(self.recovery_path, "r", encoding="utf-8") as fh:
                    current_marker = json.load(fh)
                if current_marker.get("recovery_id") == recovery_id:
                    os.remove(self.recovery_path)
            except FileNotFoundError:
                pass

    def _archive_stale_run(self, record: dict[str, Any]) -> None:
        old_run_id = _safe_component(str(record.get("run_id", "")))
        if not old_run_id:
            raise LogFileBusy(f"Invalid run ID in log lock: {self.lock_path}")
        spool = str(record.get("spool_path", "")).strip()
        running_path = f"{self.log_path}.running.{old_run_id}.log"
        incomplete_path = f"{self.log_path}.incomplete.{old_run_id}.log"
        if str(record.get("status", "running")) == "done":
            for candidate in (spool, running_path):
                if candidate:
                    try:
                        os.remove(candidate)
                    except OSError:
                        pass
            return
        recovered_spool = False
        if spool:
            src = os.path.abspath(os.path.expanduser(spool))
            expected = f"janusx-log-{old_run_id}-"
            if not os.path.basename(src).startswith(expected):
                raise LogFileBusy(f"Unexpected local spool path in {self.lock_path}: {src}")
            if os.path.isfile(src):
                recovered_spool = True
                if os.path.isfile(incomplete_path):
                    if not _files_equal(src, incomplete_path):
                        raise LogFileBusy(f"Incomplete log archive already exists: {incomplete_path}")
                    os.remove(src)
                else:
                    publish_log_spool(src, incomplete_path)
                    os.remove(src)
        if not recovered_spool and os.path.isfile(running_path):
            if os.path.isfile(incomplete_path):
                if not _files_equal(running_path, incomplete_path):
                    raise LogFileBusy(f"Incomplete log archive already exists: {incomplete_path}")
                os.remove(running_path)
            else:
                os.replace(running_path, incomplete_path)
        try:
            os.remove(running_path)
        except OSError:
            pass

    def mark_terminal(self, status: str) -> bool:
        try:
            record = self._read_lock()
            if record.get("run_id") != self.run_id:
                return False
            record["status"] = str(status)
            record["updated_at"] = _utc_now()
            _write_json_atomic(self.lock_path, record)
            return True
        except (OSError, LogFileBusy):
            return False

    def release(self) -> bool:
        try:
            record = self._read_lock()
        except FileNotFoundError:
            self.acquired = False
            return True
        except (OSError, LogFileBusy):
            return False
        if record.get("run_id") != self.run_id:
            return False
        try:
            os.remove(self.lock_path)
            self.acquired = False
            return True
        except FileNotFoundError:
            self.acquired = False
            return True
        except OSError:
            return False

    def _read_lock(self) -> dict[str, Any]:
        try:
            with open(self.lock_path, "r", encoding="utf-8") as fh:
                value = json.load(fh)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise LogFileBusy(f"Cannot read log lock; inspect {self.lock_path}: {exc}") from exc
        if not isinstance(value, dict):
            raise LogFileBusy(f"Invalid log lock metadata: {self.lock_path}")
        return value


@contextlib.contextmanager
def _process_file_lock(path: str) -> Iterator[None]:
    """Lock a local spool across processes, including spawned workers."""
    os.makedirs(os.path.dirname(path) or os.curdir, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


_LOG_SESSIONS: dict[str, "NfsSafeFileHandler"] = {}
_LOG_SESSIONS_LOCK = threading.RLock()
_ATEXIT_REGISTERED = False


class NfsSafeFileHandler(logging.FileHandler):
    """Write to local storage and atomically publish snapshots to the target."""

    def __init__(
        self,
        log_path: str,
        *,
        run_id: str,
        spool_path: str | None = None,
        lease: LogFileLease | None = None,
        managed: bool = True,
        publish_enabled: bool = True,
        publish_interval: float | None = 5.0,
        process_lock_path: str | None = None,
    ) -> None:
        self.log_path = os.path.abspath(os.path.expanduser(str(log_path)))
        self.run_id = _safe_component(run_id)
        self.spool_path = (
            os.path.abspath(str(spool_path))
            if spool_path is not None
            else create_local_log_spool(self.run_id)
        )
        self.process_lock_path = process_lock_path or f"{self.spool_path}.lock"
        self.lease = lease
        self.managed = bool(managed)
        self.publish_enabled = bool(publish_enabled)
        self.snapshot_enabled = bool(
            self.publish_enabled and publish_interval is not None
        )
        self.publish_interval = (
            None if publish_interval is None else max(0.0, float(publish_interval))
        )
        self.live_path = f"{self.log_path}.running.{self.run_id}.log"
        self.janusx_io_failed = False
        self.janusx_io_error = ""
        self.janusx_publish_failed = False
        self._warned_failure = False
        self._last_publish = 0.0
        self._inside_emit = False
        self._finalizing = False
        super().__init__(self.spool_path, mode="a", encoding="utf-8", delay=False)
        self._janusx_managed = self.managed

    def emit(self, record: logging.LogRecord) -> None:
        if self.janusx_io_failed:
            return
        try:
            with _process_file_lock(self.process_lock_path):
                self._inside_emit = True
                super().emit(record)
                self._inside_emit = False
                if not self.janusx_io_failed:
                    self._publish_if_due_locked(force=record.levelno >= logging.ERROR)
        except Exception:
            self._inside_emit = False
            self.handleError(record)

    def flush(self) -> None:
        if self.stream is not None:
            super().flush()
        if self._inside_emit or self._finalizing or not self.snapshot_enabled:
            return
        self._publish_if_due()

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        if self.janusx_io_failed:
            return
        self.janusx_io_failed = True
        exc = sys.exc_info()[1]
        self.janusx_io_error = str(exc or "unknown local log spool error")
        self._warn_once(
            "JanusX local log spool failed; file logging is disabled. "
            f"Preserve or inspect {self.spool_path} ({self.janusx_io_error})."
        )

    def _warn_once(self, message: str) -> None:
        if self._warned_failure:
            return
        self._warned_failure = True
        try:
            sys.stderr.write(f"Warning: {message}\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _publish_if_due(self) -> None:
        if self.janusx_publish_failed or not self.snapshot_enabled:
            return
        try:
            with _process_file_lock(self.process_lock_path):
                super().flush()
                self._publish_if_due_locked()
        except Exception as exc:
            self.janusx_publish_failed = True
            self._warn_once(
                "NFS log snapshot failed; further publishing is disabled and "
                f"the local log is preserved at {self.spool_path} ({exc})."
            )

    def _publish_if_due_locked(self, *, force: bool = False) -> None:
        if (
            not self.snapshot_enabled
            or self.publish_interval is None
            or self.janusx_publish_failed
            or self.janusx_io_failed
        ):
            return
        now = time.monotonic()
        if not force and now - self._last_publish < self.publish_interval:
            return
        try:
            super().flush()
            publish_log_spool(self.spool_path, self.live_path)
            self._last_publish = now
        except Exception as exc:
            self.janusx_publish_failed = True
            self._warn_once(
                "NFS log snapshot failed; further publishing is disabled and "
                f"the local log is preserved at {self.spool_path} ({exc})."
            )

    def finalize(self, *, success: bool) -> str:
        """Close the spool, publish once, and release its destination lease."""
        if not self.managed:
            self._finalizing = True
            try:
                super().close()
            except Exception:
                pass
            return self.spool_path
        if self._finalizing:
            return self.spool_path
        self._finalizing = True
        status = "done" if bool(success) else "failed"
        published_path = self.spool_path
        log_published = False
        try:
            if self.stream is not None and not self.janusx_io_failed:
                with _process_file_lock(self.process_lock_path):
                    super().flush()
                    self.stream.write(
                        f"JanusX log status: {'complete' if success else 'incomplete'}; "
                        f"run_id: {self.run_id}\n"
                    )
                    super().flush()
            super().close()
        except Exception as exc:
            self.janusx_io_failed = True
            self.janusx_io_error = str(exc)
            self._warn_once(
                f"Could not close the local log spool; preserve {self.spool_path} ({exc})."
            )

        if (
            self.publish_enabled
            and not self.janusx_io_failed
            and not self.janusx_publish_failed
        ):
            destination = (
                self.log_path
                if success
                else f"{self.log_path}.incomplete.{self.run_id}.log"
            )
            try:
                publish_log_spool(self.spool_path, destination)
                published_path = destination
                log_published = True
                try:
                    os.remove(self.spool_path)
                except OSError:
                    pass
                try:
                    os.remove(self.live_path)
                except OSError:
                    pass
            except Exception as exc:
                self.janusx_publish_failed = True
                self._warn_once(
                    "Final log publish failed; the local spool is preserved at "
                    f"{self.spool_path} ({exc})."
                )
        elif not self.publish_enabled:
            self._warn_once(
                "Log destination is unavailable; the local spool is preserved at "
                f"{self.spool_path}."
            )

        if self.lease is not None:
            terminal_status = (
                status
                if self.publish_enabled and not self.janusx_io_failed and log_published
                else "failed"
            )
            self.lease.mark_terminal(terminal_status)
            if not self.lease.release():
                self._warn_once(
                    f"Could not release the log lock; inspect {self.lease.lock_path}."
                )
        with _LOG_SESSIONS_LOCK:
            if _LOG_SESSIONS.get(self.log_path) is self:
                _LOG_SESSIONS.pop(self.log_path, None)
        return published_path

    def close(self) -> None:
        """Close the local stream; lifecycle publication is handled separately."""
        self._finalizing = True
        super().close()

    def discard_live_snapshot(self) -> None:
        try:
            os.remove(self.live_path)
        except OSError:
            pass


def finalize_logging(*, success: bool) -> None:
    """Publish and close every log session created by setup_logging."""
    with _LOG_SESSIONS_LOCK:
        handlers = list(_LOG_SESSIONS.values())
    for handler in handlers:
        try:
            handler.finalize(success=bool(success))
        except Exception as exc:
            handler._warn_once(
                f"Log finalization failed; the local spool is preserved at "
                f"{handler.spool_path} ({exc})."
            )


def _finalize_logging_at_exit() -> None:
    finalize_logging(success=True)


def get_log_spool_path(logger: logging.Logger) -> str:
    handler = getattr(logger, "_janusx_log_file_handler", None)
    if isinstance(handler, NfsSafeFileHandler):
        return str(handler.spool_path)
    for candidate in getattr(logger, "handlers", []):
        if isinstance(candidate, NfsSafeFileHandler):
            return str(candidate.spool_path)
    return ""


def get_log_process_lock_path(logger: logging.Logger) -> str:
    handler = getattr(logger, "_janusx_log_file_handler", None)
    if isinstance(handler, NfsSafeFileHandler):
        return str(handler.process_lock_path)
    return ""


def attach_worker_log_handler(
    logger: logging.Logger,
    spool_path: str,
    process_lock_path: str,
) -> logging.Handler:
    """Attach a process-safe local spool writer in a spawned worker."""
    target = os.path.abspath(str(spool_path))
    for handler in getattr(logger, "handlers", []):
        if isinstance(handler, NfsSafeFileHandler) and handler.spool_path == target:
            return handler
    handler = NfsSafeFileHandler(
        target,
        run_id="worker",
        spool_path=target,
        managed=False,
        publish_enabled=False,
        publish_interval=None,
        process_lock_path=str(process_lock_path),
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(_LevelPrefixFormatter())
    logger.addHandler(handler)
    return handler


@contextlib.contextmanager
def local_log_capture(
    log_path: str | os.PathLike[str],
    *,
    mode: str = "w",
    publish_interval: float | None = None,
) -> Iterator[Any]:
    """Capture subprocess output locally, then atomically publish the log."""
    if str(mode) not in {"w", "a"}:
        raise ValueError("local_log_capture mode must be 'w' or 'a'")
    target = os.path.abspath(os.path.expanduser(os.fspath(log_path)))
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    spool = create_local_log_spool(run_id, prefix="janusx-capture")
    if str(mode) == "a" and os.path.isfile(target):
        try:
            shutil.copyfile(target, spool)
        except OSError as exc:
            try:
                sys.stderr.write(
                    f"Warning: could not copy the prior log before append; "
                    f"a fresh capture will be used ({exc}).\n"
                )
            except Exception:
                pass
    stop_snapshot = threading.Event()
    snapshot_failure: list[str] = []

    def publish_live_snapshots() -> None:
        interval = max(0.25, float(publish_interval or 0.0))
        while not stop_snapshot.wait(interval):
            if snapshot_failure:
                return
            try:
                publish_log_spool(spool, target)
            except Exception as exc:
                snapshot_failure.append(str(exc))
                try:
                    sys.stderr.write(
                        f"Warning: live log snapshots stopped; local copy is preserved "
                        f"at {spool} ({exc}).\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                return

    snapshot_thread: threading.Thread | None = None

    def stop_live_snapshots() -> bool:
        stop_snapshot.set()
        if snapshot_thread is None:
            return True
        snapshot_thread.join(timeout=2.0)
        return not snapshot_thread.is_alive()

    try:
        with open(spool, "a", encoding="utf-8") as fh:
            if publish_interval is not None and float(publish_interval) > 0.0:
                snapshot_thread = threading.Thread(
                    target=publish_live_snapshots,
                    name=f"janusx-log-snapshot-{run_id}",
                    daemon=True,
                )
                snapshot_thread.start()
            yield fh
    except BaseException:
        snapshots_stopped = stop_live_snapshots()
        incomplete = f"{target}.incomplete.{_safe_component(run_id)}.log"
        try:
            if snapshots_stopped and os.path.isfile(spool):
                publish_log_spool(spool, incomplete)
                os.remove(spool)
            elif not snapshots_stopped:
                raise OSError("live log snapshot is still blocked on the destination")
        except OSError as exc:
            try:
                sys.stderr.write(
                    f"Warning: subprocess log is preserved at {spool} ({exc}).\n"
                )
            except Exception:
                pass
        raise
    else:
        snapshots_stopped = stop_live_snapshots()
        if not snapshots_stopped:
            try:
                sys.stderr.write(
                    f"Warning: log snapshot did not stop cleanly; local copy is preserved "
                    f"at {spool}.\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            return
        if snapshot_failure:
            return
        try:
            publish_log_spool(spool, target)
        except Exception as exc:
            try:
                sys.stderr.write(
                    f"Warning: could not publish captured log; local copy is preserved "
                    f"at {spool} ({exc}).\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            return
        try:
            os.remove(spool)
        except OSError:
            pass


def _archive_previous_incomplete_log(log_path: str, run_id: str) -> str | None:
    if not os.path.isfile(log_path) or _has_completed_log_footer(log_path):
        return None
    archive_path = f"{log_path}.previous.{_safe_component(run_id)}.log"
    if os.path.exists(archive_path):
        raise FileExistsError(f"Previous incomplete log archive already exists: {archive_path}")
    os.replace(log_path, archive_path)
    return archive_path


def _make_log_handler(
    log_path: str,
    *,
    run_id: str,
    spool_path: str | None,
    lease: LogFileLease | None,
    managed: bool,
    publish_enabled: bool,
    publish_interval: float | None,
) -> NfsSafeFileHandler:
    handler = NfsSafeFileHandler(
        log_path,
        run_id=run_id,
        spool_path=spool_path,
        lease=lease,
        managed=managed,
        publish_enabled=publish_enabled,
        publish_interval=publish_interval,
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(_LevelPrefixFormatter())
    return handler


def setup_logging(
    log_file_path: str | os.PathLike[str],
    *,
    spool_path: str | None = None,
    manage_lifecycle: bool = True,
    acquire_lock: bool = True,
    publish_interval: float | None = 5.0,
) -> logging.Logger:
    """Configure console logging and a local-spooled, atomic file log."""
    global _ATEXIT_REGISTERED
    requested_path = os.path.abspath(os.path.expanduser(os.fspath(log_file_path)))
    chosen_log_path = requested_path
    fallback_note: str | None = None
    handler: NfsSafeFileHandler | None = None
    lease: LogFileLease | None = None
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    if manage_lifecycle:
        with _LOG_SESSIONS_LOCK:
            handler = _LOG_SESSIONS.get(requested_path)
            if handler is not None and handler._finalizing:
                _LOG_SESSIONS.pop(requested_path, None)
                handler = None

    if handler is None:
        if manage_lifecycle and not _ATEXIT_REGISTERED:
            atexit.register(_finalize_logging_at_exit)
            _ATEXIT_REGISTERED = True
        local_spool = spool_path
        try:
            if local_spool is None:
                local_spool = create_local_log_spool(
                    run_id,
                    prefix="janusx-log" if manage_lifecycle else "janusx-gwas",
                )
        except OSError as exc:
            local_spool = None
            fallback_note = (
                "A writable local log spool could not be created; computation will "
                f"continue without a file log ({exc})."
            )

        publish_enabled = local_spool is not None
        if manage_lifecycle and acquire_lock and local_spool is not None:
            try:
                lease = LogFileLease(chosen_log_path, run_id)
                lease.acquire(local_spool)
            except LogFileBusy as exc:
                lease = None
                alt = _fallback_log_path(requested_path)
                while os.path.exists(f"{alt}.lock") or os.path.exists(alt):
                    alt = _fallback_log_path(requested_path)
                chosen_log_path = alt
                try:
                    lease = LogFileLease(chosen_log_path, run_id)
                    lease.acquire(local_spool)
                    fallback_note = (
                        f"Log destination is already in use or cannot be verified: "
                        f"{requested_path}. Writing this run to {chosen_log_path}. ({exc})"
                    )
                except (OSError, LogFileBusy) as lock_exc:
                    lease = None
                    publish_enabled = False
                    fallback_note = (
                        f"Could not acquire a log destination lock; the local spool "
                        f"will be kept at {local_spool} ({lock_exc})."
                    )
            except OSError as exc:
                lease = None
                publish_enabled = False
                fallback_note = (
                    f"Could not access the log destination; the local spool will be "
                    f"kept at {local_spool} ({exc})."
                )

            if lease is not None:
                try:
                    previous = _archive_previous_incomplete_log(chosen_log_path, run_id)
                    if previous is not None:
                        fallback_note = (
                            f"A prior log without a completion marker was preserved at {previous}."
                        )
                except OSError as exc:
                    lease.release()
                    lease = None
                    alt = _fallback_log_path(requested_path)
                    chosen_log_path = alt
                    try:
                        lease = LogFileLease(chosen_log_path, run_id)
                        lease.acquire(local_spool)
                        fallback_note = (
                            f"Could not archive the previous log; preserving it and writing "
                            f"this run to {chosen_log_path} ({exc})."
                        )
                    except (OSError, LogFileBusy) as lock_exc:
                        lease = None
                        publish_enabled = False
                        fallback_note = (
                            f"Could not safely publish the log; the local spool is at "
                            f"{local_spool} ({lock_exc})."
                        )

        try:
            handler = _make_log_handler(
                chosen_log_path,
                run_id=run_id,
                spool_path=local_spool,
                lease=lease,
                managed=bool(manage_lifecycle),
                publish_enabled=bool(publish_enabled),
                publish_interval=publish_interval,
            )
        except OSError as exc:
            handler = None
            if lease is not None:
                lease.mark_terminal("failed")
                lease.release()
            fallback_note = (
                "Could not open the local log spool; computation will continue "
                f"without a file log ({exc})."
            )
        if handler is not None and manage_lifecycle:
            with _LOG_SESSIONS_LOCK:
                _LOG_SESSIONS[chosen_log_path] = handler

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if handler is not None:
        logger.addHandler(handler)
        setattr(logger, "_janusx_log_file_handler", handler)
        setattr(logger, "_janusx_log_path", chosen_log_path)
        setattr(logger, "_janusx_log_spool_path", handler.spool_path)
        setattr(logger, "_janusx_log_process_lock_path", handler.process_lock_path)
    else:
        logger.addHandler(logging.NullHandler())
        for name in (
            "_janusx_log_file_handler",
            "_janusx_log_path",
            "_janusx_log_spool_path",
            "_janusx_log_process_lock_path",
        ):
            try:
                delattr(logger, name)
            except AttributeError:
                pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        _ColorLevelPrefixFormatter(enable_color=_supports_color(sys.stdout))
    )
    logger.addHandler(console_handler)
    logging.captureWarnings(True)
    pywarn_logger = logging.getLogger("py.warnings")
    pywarn_logger.handlers.clear()
    pywarn_logger.propagate = True
    warnings.simplefilter("default")
    if fallback_note:
        logger.warning(fallback_note)
    return logger
