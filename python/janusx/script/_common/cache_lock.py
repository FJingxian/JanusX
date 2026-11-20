"""Process-safe cache generation locks with diagnostics.

Cache generation can legitimately take a long time, especially when the input
is on NFS or another mounted volume.  A silent ``.lock`` wait is difficult to
distinguish from a stalled job, so this module keeps a small owner record and
reports wait progress without changing the cross-host locking semantics.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator


_LOG = logging.getLogger("janusx.cache")
_LOCAL_HOST = socket.gethostname()


def _owner_record(pid: int, token: str) -> dict[str, Any]:
    return {
        "version": 1,
        "pid": int(pid),
        "host": _LOCAL_HOST,
        "started_at": time.time(),
        "token": str(token),
    }


def _read_owner(lock_path: str) -> dict[str, Any] | None:
    """Read a current or legacy lock owner record.

    Older JanusX versions wrote only the PID.  Keep accepting that format so
    an upgrade does not turn an existing cache lock into an opaque timeout.
    """

    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        value = None
    if isinstance(value, dict):
        return value
    try:
        return {"version": 0, "pid": int(raw.splitlines()[0].strip())}
    except (TypeError, ValueError, IndexError):
        return {"version": 0}


def _owner_text(owner: dict[str, Any] | None) -> str:
    if not owner:
        return "owner=unknown"
    host = str(owner.get("host") or "unknown")
    pid = owner.get("pid")
    pid_text = str(pid) if pid is not None else "?"
    started = owner.get("started_at")
    if isinstance(started, (int, float)):
        age = max(0.0, time.time() - float(started))
        return f"owner={host}:{pid_text}, age={age:.0f}s"
    return f"owner={host}:{pid_text}"


def _same_host_pid_state(owner: dict[str, Any] | None) -> str:
    """Return ``alive``, ``dead``, ``remote`` or ``unverifiable``.

    A PID is only used for immediate stale-lock recovery when it belongs to
    this host.  Remote owners remain conservative because a shared filesystem
    may expose a different PID namespace.
    """

    if not owner:
        return "unverifiable"
    host = owner.get("host")
    # Legacy PID-only locks do not identify the machine.  Treat them as
    # unverifiable rather than guessing that a matching local PID owns them;
    # this is safer on shared filesystems with independent PID namespaces.
    if not host:
        return "unverifiable"
    if str(host) != _LOCAL_HOST:
        return "remote"
    try:
        pid = int(owner.get("pid"))
    except (TypeError, ValueError):
        return "unverifiable"
    if pid <= 0:
        return "unverifiable"

    # Prefer psutil when available: it lets us detect PID reuse using the
    # process creation time recorded by newer locks.  Fall back to kill(0) so
    # the lock remains usable in minimal installations.
    try:
        import psutil  # type: ignore

        process = psutil.Process(pid)
        recorded = owner.get("process_started_at")
        if recorded is not None:
            try:
                if abs(float(process.create_time()) - float(recorded)) > 5.0:
                    return "dead"
            except (TypeError, ValueError, OSError, psutil.Error):
                return "unverifiable"
        return "alive"
    except ImportError:
        pass
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "unverifiable"
    except Exception:
        # psutil raises platform-specific subclasses; do not make lock
        # acquisition fail merely because process inspection is unavailable.
        pass

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "unverifiable"
    except OSError:
        return "unverifiable"
    return "alive"


def _owner_record_with_process_start(pid: int, token: str) -> dict[str, Any]:
    record = _owner_record(pid, token)
    try:
        import psutil  # type: ignore

        record["process_started_at"] = float(psutil.Process(pid).create_time())
    except Exception:
        pass
    return record


def _write_owner(fd: int, record: dict[str, Any]) -> None:
    payload = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])
    # Do not fsync here.  Some mounted filesystems report ENOTSUP for fsync,
    # and lock creation only needs O_EXCL ownership; the lock is advisory.


def _try_remove(lock_path: str) -> bool:
    try:
        os.remove(lock_path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


@contextmanager
def cache_generation_lock(
    lock_key: str,
    *,
    timeout_s: float = 7200.0,
    poll_s: float = 0.2,
    label: str = "cache",
    report_after_s: float = 3.0,
    report_every_s: float = 30.0,
) -> Iterator[None]:
    """Acquire a cache-generation lock and explain any wait to the user.

    Same-host locks whose owner process has exited are reclaimed immediately.
    Remote or otherwise unverifiable locks are never removed based only on a
    PID; they follow the existing age/timeout policy instead.
    """

    lock_path = f"{lock_key}.lock"
    timeout_s = max(0.0, float(timeout_s))
    poll_s = max(0.01, float(poll_s))
    report_after_s = max(0.0, float(report_after_s))
    report_every_s = max(1.0, float(report_every_s))
    wait_started = time.monotonic()
    next_report = wait_started + report_after_s
    fd: int | None = None
    token = uuid.uuid4().hex

    while True:
        try:
            candidate_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            owner = _read_owner(lock_path)
            state = _same_host_pid_state(owner)
            if state == "dead":
                if _try_remove(lock_path):
                    _LOG.warning(
                        "Recovered stale %s generation lock %s (%s process is no longer running).",
                        label,
                        lock_path,
                        _owner_text(owner),
                    )
                    continue

            now = time.monotonic()
            waited = now - wait_started
            if now >= next_report:
                _LOG.warning(
                    "Waiting for %s generation lock: %s; %s, state=%s, waited=%s. "
                    "Another process may be building this cache; same-host dead locks "
                    "are reclaimed automatically, while remote/unverifiable locks "
                    "remain protected until timeout.",
                    label,
                    lock_path,
                    _owner_text(owner),
                    state,
                    _format_duration(waited),
                )
                next_report = now + report_every_s

            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > timeout_s:
                    if _try_remove(lock_path):
                        _LOG.warning(
                            "Recovered expired %s generation lock %s (%s, lock age=%s).",
                            label,
                            lock_path,
                            _owner_text(owner),
                            _format_duration(age),
                        )
                        continue
            except OSError:
                pass

            if waited > timeout_s:
                raise TimeoutError(
                    f"Timeout waiting for {label} generation lock: {lock_path}; "
                    f"{_owner_text(owner)}, state={state}, waited={_format_duration(waited)}"
                )
            time.sleep(poll_s)
        else:
            fd = candidate_fd
            try:
                _write_owner(fd, _owner_record_with_process_start(os.getpid(), token))
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = None
                _try_remove(lock_path)
                raise
            break

    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        # Remove only our own lock.  This avoids deleting a newer owner's
        # lock if a stale-lock recovery raced with the end of this context.
        owner = _read_owner(lock_path)
        if owner is not None and str(owner.get("token") or "") == token:
            _try_remove(lock_path)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{remainder:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m"
