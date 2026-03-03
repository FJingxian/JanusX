import subprocess
import time
import sys
from pathlib import Path
from typing import List, Literal, Optional, Union, Sequence, Any

PathLike = Union[Path, str]

try:
    from janusx.script._common.status import (
        CliStatus,
        print_success,
        print_failure,
        format_elapsed,
        get_rich_spinner_name,
    )

    _HAS_CLI_STATUS = True
except Exception:
    CliStatus = None  # type: ignore[assignment]
    print_success = None  # type: ignore[assignment]
    print_failure = None  # type: ignore[assignment]
    format_elapsed = None  # type: ignore[assignment]
    get_rich_spinner_name = None  # type: ignore[assignment]
    _HAS_CLI_STATUS = False

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False


def _step_label(step_idx: int, total_steps: int, step_names: Optional[List[str]]) -> str:
    if step_names is not None and step_idx < len(step_names):
        name = str(step_names[step_idx])
    else:
        name = f"Step-{step_idx}"
    return f"Step {step_idx + 1}/{total_steps}: {name}"


def _all_exists(paths: Sequence[PathLike]) -> bool:
    return all(Path(p).exists() for p in paths)


def _safe_job_label(job: str) -> str:
    s = str(job)
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


def _find_failed_item_logs(
    items: List[dict[str, Any]],
    *,
    max_items: int = 3,
    max_lines: int = 4,
) -> List[str]:
    snippets: List[str] = []
    log_dir = Path("log")
    if not log_dir.exists():
        return snippets

    for it in items:
        outputs = list(it.get("outputs", []))
        if _all_exists(outputs):
            continue
        item_id = str(it.get("id", "")).strip()
        if not item_id:
            continue
        for ef in sorted(log_dir.glob(f"{_safe_job_label(item_id)}.*.e")):
            try:
                if ef.stat().st_size <= 0:
                    continue
                txt = ef.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            if len(lines) == 0:
                continue
            head = " | ".join(lines[:max_lines])
            snippets.append(f"{item_id}: {head}")
            if len(snippets) >= max_items:
                return snippets
    return snippets


def _format_step_done(step_text: str, tstart: float) -> str:
    if format_elapsed is None:
        elapsed = f"{max(0.0, time.time() - tstart):.1f}s"
    else:
        elapsed = format_elapsed(time.time() - tstart)
    return f"{step_text} ...Finished [{elapsed}]"


def _normalize_step_items(
    step_idx: int,
    ofiles: List[PathLike],
    step_items: Optional[List[List[dict[str, Any]]]],
) -> List[dict[str, Any]]:
    if step_items is not None and step_idx < len(step_items):
        raw = step_items[step_idx]
        out: List[dict[str, Any]] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            outputs = item.get("outputs", [])
            if not isinstance(outputs, list) or len(outputs) == 0:
                continue
            item_id = str(item.get("id", f"item-{i+1}"))
            out.append({"id": item_id, "outputs": list(outputs)})
        if len(out) > 0:
            return out
    # Fallback: use one subtask per expected output file.
    return [
        {"id": f"output-{i + 1}", "outputs": [p]}
        for i, p in enumerate(ofiles)
    ]


def _run_step_script(sh: Path, scheduler: str) -> None:
    if scheduler != "csub":
        subprocess.run(["bash", str(sh)], check=True)
        return

    proc = subprocess.run(
        ["bash", str(sh)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        output = proc.stdout or ""
        if len(output.strip()) > 0:
            print(output, flush=True)
        raise subprocess.CalledProcessError(
            returncode=int(proc.returncode),
            cmd=["bash", str(sh)],
            output=proc.stdout,
        )


def _wait_outputs_with_rich_subtasks(
    *,
    step_text: str,
    items: List[dict[str, Any]],
    poll_sec: float = 2.0,
    show_time_remaining: bool = True,
) -> None:
    total = len(items)
    if total == 0:
        return
    t0 = time.time()

    if not (_HAS_RICH_PROGRESS and _HAS_CLI_STATUS and sys.stdout.isatty()):
        done = 0
        while done < total:
            done = sum(
                1 for it in items
                if _all_exists(list(it.get("outputs", [])))
            )
            elapsed = max(0.0, time.time() - t0)
            elapsed_txt = format_elapsed(elapsed) if format_elapsed is not None else f"{elapsed:.1f}s"
            if done >= total:
                break
            print(
                f"\r{step_text} [{done}/{total}] [{elapsed_txt}]",
                end="",
                flush=True,
            )
            time.sleep(max(0.5, poll_sec))
        if done < total:
            done = total
        elapsed = max(0.0, time.time() - t0)
        elapsed_txt = format_elapsed(elapsed) if format_elapsed is not None else f"{elapsed:.1f}s"
        print(f"\r{step_text} [{done}/{total}] [{elapsed_txt}]")
        return

    assert Progress is not None
    assert SpinnerColumn is not None
    assert BarColumn is not None
    assert TextColumn is not None
    assert TimeElapsedColumn is not None
    if show_time_remaining:
        assert TimeRemainingColumn is not None

    done_count = sum(
        1 for it in items
        if _all_exists(list(it.get("outputs", [])))
    )
    columns: list[Any] = [
        SpinnerColumn(
            spinner_name=get_rich_spinner_name(),
            style="cyan",
            finished_text=" ",
        ),
        TextColumn("[green]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ]
    if show_time_remaining and TimeRemainingColumn is not None:
        columns.append(TimeRemainingColumn())
    progress = Progress(*columns, transient=True)

    with progress:
        task_id = progress.add_task(step_text, total=total, completed=done_count)
        last_error_scan = 0.0
        while done_count < total:
            new_done = sum(
                1 for it in items
                if _all_exists(list(it.get("outputs", [])))
            )
            if new_done > done_count:
                done_count = new_done
                progress.update(task_id, completed=done_count)
            if done_count >= total:
                break
            now = time.time()
            if now - last_error_scan >= max(5.0, poll_sec * 3.0):
                last_error_scan = now
                failed = _find_failed_item_logs(items)
                if len(failed) > 0:
                    details = "\n".join(f"- {x}" for x in failed)
                    raise RuntimeError(
                        f"{step_text} detected failed subtasks. Example stderr:\n{details}"
                    )
            time.sleep(max(0.5, poll_sec))
        progress.update(task_id, completed=total)


def wrap_cmd(
    cmd: str,
    job: str,
    threads: int,
    scheduler: Literal["nohup", "csub"] = "nohup",
    queue: str = "c01",
    singularity: str = "",
) -> str:
    if scheduler == "nohup":
        safe_cmd = cmd.replace('"', '\\"')
        return (
            f'nohup bash -c "{singularity} {safe_cmd}" '
            f'> ./log/{job}.o 2> ./log/{job}.e'
        )
    elif scheduler == "csub":
        safe_job = _safe_job_label(job)
        return (
            f'csub -J {job} -o ./log/{safe_job}.%J.o -e ./log/{safe_job}.%J.e -q {queue} '
            f'-n {threads} "{singularity} {cmd}"'
        )
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def node(ifiles: List[PathLike], ofiles: List[PathLike], *, allow_skip: bool = True) -> Literal[-1, 0, 1]:
    outexists = all(Path(p).exists() for p in ofiles)
    if allow_skip and outexists:
        return 1  # pass

    inexists = all(Path(p).exists() for p in ifiles)
    if inexists:
        return 0  # run
    return -1  # back


def pipeline(
    AllCMD: List[str],
    Alli: List[List[PathLike]],
    Allo: List[List[PathLike]],
    scheduler: str = "csub",
    nohup_max_jobs: int = 0,
    skip_if_outputs_exist: bool = True,
    step_names: Optional[List[str]] = None,
    use_rich: bool = True,
    step_items: Optional[List[List[dict[str, Any]]]] = None,
    show_time_remaining: bool = True,
) -> None:
    Path("tmp").mkdir(mode=0o755, parents=True, exist_ok=True)
    total_steps = len(AllCMD)
    rich_enabled = bool(use_rich and _HAS_CLI_STATUS)

    for num, cmd in enumerate(AllCMD):
        ifiles = Alli[num]
        ofiles = Allo[num]
        step_text = _step_label(num, total_steps, step_names)
        status = node(ifiles, ofiles, allow_skip=skip_if_outputs_exist)

        sh = Path(f"./tmp/{num}.sh")
        if scheduler == "nohup":
            lines = [line.strip() for line in cmd.splitlines() if line.strip()]
            if nohup_max_jobs > 0:
                parts = [f"MAX_JOBS={nohup_max_jobs}"]
                for line in lines:
                    parts.append('while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do sleep 2; done')
                    parts.append(f"{line} &")
                parts.append("wait")
                text = "\n".join(parts) + "\n"
            else:
                text = "\n".join([f"{line} &" for line in lines] + ["wait"]) + "\n"
        else:
            text = cmd if cmd.endswith("\n") else (cmd + "\n")
        sh.write_text(text, encoding="utf-8")
        sh.chmod(0o755)

        tstart = time.time()

        if status == 1:
            if rich_enabled:
                print_success(f"{step_text} ...Skipped")
            else:
                print(f"{step_text} completed! (skip)")
            continue

        if status == -1:
            raise ValueError(f"{step_text} inputs not exists: {ifiles}")

        if rich_enabled and CliStatus is not None:
            try:
                _run_step_script(sh, scheduler)
                if scheduler == "csub":
                    items = _normalize_step_items(num, ofiles, step_items)
                    _wait_outputs_with_rich_subtasks(
                        step_text=step_text,
                        items=items,
                        poll_sec=2.0,
                        show_time_remaining=show_time_remaining,
                    )
                else:
                    with CliStatus(step_text, enabled=True, show_elapsed=True):
                        while True:
                            if _all_exists(ofiles):
                                break
                            time.sleep(5)
            except Exception:
                if print_failure is not None:
                    print_failure(f"{step_text} ...Failed")
                raise
            if print_success is not None:
                print_success(_format_step_done(step_text, tstart))
            continue

        print(f"Run {step_text}: {sh}")
        _run_step_script(sh, scheduler)
        while True:
            if _all_exists(ofiles):
                print(f"\n{_format_step_done(step_text, tstart)}")
                break
            print(f"\rWaiting outputs... time={time.time() - tstart:.1f}s", end="")
            time.sleep(60)
