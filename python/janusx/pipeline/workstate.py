from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def _read_json_or_empty(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@contextmanager
def _state_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _refresh_summary(state: dict) -> None:
    steps = state.setdefault("steps", {})
    total_items = 0
    done_items = 0
    done_steps = 0
    total_steps = 0

    for sid, entry in list(steps.items()):
        if not isinstance(entry, dict):
            continue
        total_steps += 1
        items = dict(entry.get("items", {}))
        done = sum(1 for v in items.values() if bool(v))
        total = int(entry.get("total", max(len(items), done)))
        if total < done:
            total = done
        entry["done"] = int(done)
        entry["total"] = int(total)
        steps[sid] = entry
        total_items += total
        done_items += done
        if total > 0 and done >= total:
            done_steps += 1

    state["summary"] = {
        "done_items": int(done_items),
        "total_items": int(total_items),
        "done_steps": int(done_steps),
        "total_steps": int(total_steps),
    }
    state["updated_at"] = _utc_now()


def _wait_outputs(outputs: list[Path], wait_secs: float) -> bool:
    if len(outputs) == 0:
        return True
    wait_secs = max(0.0, float(wait_secs))
    deadline = time.monotonic() + wait_secs
    while True:
        if all(p.exists() for p in outputs):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def mark_item_done(
    *,
    state_path: Path,
    step_id: str,
    item_id: str,
    outputs: list[Path],
    wait_secs: float = 30.0,
) -> bool:
    outputs_ok = _wait_outputs(outputs, wait_secs)
    if not outputs_ok:
        return False

    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with _state_lock(lock_path):
        state = _read_json_or_empty(state_path)
        if len(state) == 0:
            state = {
                "version": 1,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "status": "running",
                "steps": {},
                "summary": {},
            }

        steps = state.setdefault("steps", {})
        entry = dict(steps.get(step_id, {}))
        entry["name"] = str(entry.get("name", step_id))
        items = dict(entry.get("items", {}))
        items_done_at = dict(entry.get("items_completed_at", {}))

        items[item_id] = True
        items_done_at.setdefault(item_id, _utc_now())
        entry["items"] = items
        entry["items_completed_at"] = items_done_at

        prev_total = int(entry.get("total", 0))
        entry["total"] = int(max(prev_total, len(items)))
        entry["done"] = int(sum(1 for v in items.values() if bool(v)))
        steps[step_id] = entry

        _refresh_summary(state)
        if str(state.get("status", "")) != "failed":
            state["status"] = "running"
        _safe_write_json(state_path, state)
    return True


def can_skip_item(
    *,
    state_path: Path,
    step_id: str,
    item_id: str,
    outputs: list[Path],
) -> bool:
    state = _read_json_or_empty(state_path)
    if len(state) == 0:
        return False
    steps = state.get("steps", {})
    if not isinstance(steps, dict):
        return False
    entry = steps.get(step_id, {})
    if not isinstance(entry, dict):
        return False
    items = entry.get("items", {})
    if not isinstance(items, dict):
        return False
    has_record = bool(items.get(item_id, False))
    outputs_ok = all(Path(p).exists() for p in outputs)
    return bool(has_record and outputs_ok)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Update JanusX fastq2vcf work-state json for a completed subtask.",
    )
    p.add_argument("--state", required=True, help="Path to .work.json")
    p.add_argument("--step", required=True, help="Step id, e.g. step4_bam2gvcf")
    p.add_argument("--item", required=True, help="Item id, e.g. bam2gvcf.S1.Chr01")
    p.add_argument("--outputs", nargs="*", default=[], help="Expected output files")
    p.add_argument(
        "--wait-secs",
        type=float,
        default=30.0,
        help="Max seconds to wait for outputs before skipping update.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Check skip condition only: return code 0 if skip is allowed, else 1.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    state_path = Path(args.state)
    outputs = [Path(x) for x in list(args.outputs or [])]
    if bool(args.check):
        ok = can_skip_item(
            state_path=state_path,
            step_id=str(args.step),
            item_id=str(args.item),
            outputs=outputs,
        )
        raise SystemExit(0 if ok else 1)
    ok = mark_item_done(
        state_path=state_path,
        step_id=str(args.step),
        item_id=str(args.item),
        outputs=outputs,
        wait_secs=float(args.wait_secs),
    )
    if not ok:
        msg = (
            f"[workstate] outputs not ready in time; skip update: "
            f"step={args.step}, item={args.item}\n"
        )
        try:
            os.write(2, msg.encode("utf-8"))
        except Exception:
            pass
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
