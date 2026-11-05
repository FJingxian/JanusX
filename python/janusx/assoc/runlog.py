"""GWAS run locking and NFS-safe log spooling."""

from __future__ import annotations

import json
import os
import re
import socket
import uuid
from typing import Any

from janusx.script._common.log import (
    _classify_owner,
    _files_equal,
    _log_has_run_footer,
    _process_start_time,
    _safe_component,
    _utc_now,
    _write_all,
    _write_json_atomic,
    create_local_log_spool,
    publish_log_spool,
)


class GwasRunLockError(RuntimeError):
    """Raised when a GWAS output prefix is already in use or unrecoverable."""


class GwasRunLease:
    """Exclusive output-prefix lease with conservative stale-run recovery."""

    def __init__(self, outprefix: str, run_id: str) -> None:
        self.outprefix = os.path.abspath(os.path.expanduser(str(outprefix)))
        self.run_id = str(run_id)
        self.lock_path = f"{self.outprefix}.gwas.lock"
        self.recovery_path = f"{self.lock_path}.recover"
        self.spool_path = ""
        self.acquired = False

    def acquire(self, spool_path: str) -> None:
        self.spool_path = os.path.abspath(str(spool_path))
        lock_dir = os.path.dirname(self.lock_path) or os.curdir
        os.makedirs(lock_dir, mode=0o755, exist_ok=True)
        if os.path.exists(self.recovery_path):
            raise self._recovery_error()
        try:
            self._create_lock()
            return
        except FileExistsError:
            pass

        existing = self._read_lock(self.lock_path)
        state, detail = _classify_owner(existing)
        if state != "stale":
            raise GwasRunLockError(self._owner_error(existing, state, detail))
        self._recover_stale_and_acquire(existing)

    def archive_stale_run(self, record: dict[str, Any]) -> None:
        """Preserve a stale run's spool and unfinished result files."""
        old_run_id = _safe_component(str(record.get("run_id", "")))
        if not old_run_id:
            raise GwasRunLockError(f"Invalid run ID in lock file: {self.lock_path}")
        archive_log = f"{self.outprefix}.gwas.incomplete.{old_run_id}.log"
        old_spool = str(record.get("spool_path", "")).strip()
        completed_log = f"{self.outprefix}.gwas.log"
        if _log_has_run_footer(completed_log, old_run_id, "done"):
            if old_spool:
                src = os.path.abspath(os.path.expanduser(old_spool))
                if not os.path.basename(src).startswith(f"janusx-gwas-{old_run_id}-"):
                    raise GwasRunLockError(
                        "The completed run lock points to an unexpected local log path; "
                        f"inspect {self.lock_path}: {src}"
                    )
                try:
                    os.remove(src)
                except FileNotFoundError:
                    pass
        elif old_spool:
            self._archive_spool(old_spool, archive_log)
        elif not os.path.isfile(archive_log):
            raise GwasRunLockError(
                "The interrupted GWAS run has no recoverable log spool. "
                f"Inspect its lock and output files before removing {self.lock_path}."
            )

        output_dir = os.path.dirname(self.outprefix) or os.curdir
        output_stem = os.path.basename(self.outprefix)
        tmp_name_pattern = re.compile(
            rf"^{re.escape(output_stem)}(?:\..*)?\.tmp\..+$"
        )
        try:
            with os.scandir(output_dir) as entries:
                tmp_paths = sorted(
                    entry.path
                    for entry in entries
                    if entry.is_file(follow_symlinks=False)
                    and tmp_name_pattern.match(entry.name)
                )
        except OSError as exc:
            raise GwasRunLockError(
                "Cannot scan the GWAS output directory while recovering an "
                f"interrupted run; the old lock and files were left in place: {exc}"
            ) from exc
        for tmp_path in tmp_paths:
            archived_path = f"{tmp_path}.incomplete.{old_run_id}"
            if os.path.abspath(tmp_path) == os.path.abspath(archived_path):
                continue
            if os.path.exists(archived_path):
                if _files_equal(tmp_path, archived_path):
                    os.remove(tmp_path)
                    continue
                raise GwasRunLockError(
                    "Cannot archive an interrupted GWAS result because the "
                    f"archive target already exists: {archived_path}"
                )
            os.replace(tmp_path, archived_path)

    def mark_terminal(self, status: str) -> bool:
        """Update this run's lock status without replacing another owner's file."""
        try:
            record = self._read_lock(self.lock_path)
            if record.get("run_id") != self.run_id:
                return False
            record["status"] = str(status)
            record["updated_at"] = _utc_now()
            _write_json_atomic(self.lock_path, record)
            return True
        except (OSError, GwasRunLockError):
            return False

    def set_spool_path(self, spool_path: str) -> bool:
        """Record a logger fallback path in the active lock."""
        self.spool_path = os.path.abspath(str(spool_path))
        try:
            record = self._read_lock(self.lock_path)
            if record.get("run_id") != self.run_id:
                return False
            record["spool_path"] = self.spool_path
            _write_json_atomic(self.lock_path, record)
            return True
        except (OSError, GwasRunLockError):
            return False

    def release(self) -> bool:
        """Remove the lock only if it still belongs to this run."""
        try:
            record = self._read_lock(self.lock_path)
        except FileNotFoundError:
            self.acquired = False
            return True
        except (OSError, GwasRunLockError):
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
            payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
            _write_all(fd, payload)
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
        recovery_record = {
            "recovery_id": recovery_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": _utc_now(),
        }
        try:
            recovery_fd = os.open(
                self.recovery_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as exc:
            raise self._recovery_error() from exc
        try:
            _write_all(
                recovery_fd,
                (json.dumps(recovery_record, sort_keys=True) + "\n").encode("utf-8"),
            )
            os.fsync(recovery_fd)
        except BaseException:
            os.close(recovery_fd)
            try:
                os.remove(self.recovery_path)
            except OSError:
                pass
            raise
        else:
            os.close(recovery_fd)

        try:
            current = self._read_lock(self.lock_path)
            if current.get("run_id") != observed.get("run_id"):
                raise GwasRunLockError(
                    f"GWAS lock changed during recovery; inspect {self.lock_path}."
                )
            state, detail = _classify_owner(current)
            if state != "stale":
                raise GwasRunLockError(self._owner_error(current, state, detail))
            self.archive_stale_run(current)
            latest = self._read_lock(self.lock_path)
            if latest.get("run_id") != current.get("run_id"):
                raise GwasRunLockError(
                    f"GWAS lock changed during recovery; inspect {self.lock_path}."
                )
            os.remove(self.lock_path)
            try:
                self._create_lock()
            except FileExistsError as exc:
                raise GwasRunLockError(
                    "Another GWAS run acquired this output prefix during stale-run "
                    f"recovery: {self.lock_path}"
                ) from exc
        finally:
            marker_removed = False
            try:
                current_marker = self._read_lock(self.recovery_path)
                if current_marker.get("recovery_id") == recovery_id:
                    os.remove(self.recovery_path)
                    marker_removed = True
            except FileNotFoundError:
                marker_removed = True
            except (OSError, GwasRunLockError):
                marker_removed = False
            if not marker_removed:
                if self.acquired:
                    self.release()
                raise GwasRunLockError(
                    "GWAS recovery finished but could not clear its recovery marker; "
                    f"inspect {self.recovery_path} before retrying."
                )

    def _archive_spool(self, source_path: str, archive_path: str) -> None:
        src = os.path.abspath(os.path.expanduser(source_path))
        safe_id = _safe_component(os.path.basename(archive_path).split(".incomplete.")[-1].removesuffix(".log"))
        if not os.path.basename(src).startswith(f"janusx-gwas-{safe_id}-"):
            raise GwasRunLockError(
                "The interrupted run lock points to an unexpected local log path; "
                f"inspect {self.lock_path} before continuing: {src}"
            )
        if os.path.isfile(archive_path):
            if os.path.isfile(src):
                if _files_equal(src, archive_path):
                    os.remove(src)
                else:
                    raise GwasRunLockError(
                        f"Interrupted log archive already exists and differs: {archive_path}"
                    )
            return
        if not os.path.isfile(src):
            raise GwasRunLockError(
                "The previous GWAS log spool is missing and no archived copy exists: "
                f"{src}. Inspect {self.lock_path} before continuing."
            )
        publish_log_spool(src, archive_path)
        os.remove(src)

    def _read_lock(self, path: str) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                value = json.load(fh)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise GwasRunLockError(
                f"Cannot read GWAS lock metadata; inspect it before recovery: {path} ({exc})"
            ) from exc
        if not isinstance(value, dict):
            raise GwasRunLockError(f"Invalid GWAS lock metadata; inspect {path}.")
        return value

    def _owner_error(self, record: dict[str, Any], state: str, detail: str) -> str:
        return (
            "GWAS output prefix is locked by an active or unverifiable run "
            f"(state={state}, host={record.get('hostname', 'unknown')}, "
            f"pid={record.get('pid', 'unknown')}, run_id={record.get('run_id', 'unknown')}). "
            f"{detail} Wait for that run or choose another output prefix. "
            "For an unverifiable owner, confirm the job has stopped before "
            f"removing the lock manually: {self.lock_path}"
        )

    def _recovery_error(self) -> GwasRunLockError:
        return GwasRunLockError(
            "GWAS stale-run recovery is already in progress or was interrupted. "
            "Check that no recovery or GWAS process is active before removing "
            f"the recovery marker manually: {self.recovery_path}"
        )
