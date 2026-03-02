import subprocess
import time
import re
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
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

_CSUB_SUBMIT_LINE_RE = re.compile(
    r"^Job\s+\d+\s+has\s+been\s+submitted\s+to\s+the\s+queue\s+\[[^\]]+\]\.?$"
)


def _step_label(step_idx: int, total_steps: int, step_names: Optional[List[str]]) -> str:
    if step_names is not None and step_idx < len(step_names):
        name = str(step_names[step_idx])
    else:
        name = f"Step-{step_idx}"
    return f"Step {step_idx + 1}/{total_steps}: {name}"


def _all_exists(paths: Sequence[PathLike]) -> bool:
    return all(Path(p).exists() for p in paths)


def _format_step_done(step_text: str, tstart: float) -> str:
    if format_elapsed is None:
        elapsed = f"{max(0.0, time.time() - tstart):.1f}s"
    else:
        elapsed = format_elapsed(time.time() - tstart)
    return f"{step_text} ...Step-Finished [{elapsed}]"


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
    output = proc.stdout or ""
    kept_lines: List[str] = []
    for line in output.splitlines():
        s = line.strip()
        if len(s) == 0:
            continue
        if _CSUB_SUBMIT_LINE_RE.match(s):
            continue
        kept_lines.append(line)
    if len(kept_lines) > 0:
        print("\n".join(kept_lines), flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=int(proc.returncode),
            cmd=["bash", str(sh)],
            output=output,
        )


def _wait_outputs_with_rich_subtasks(
    *,
    step_text: str,
    items: List[dict[str, Any]],
    poll_sec: float = 2.0,
    max_visible: int = 5,
) -> None:
    if not (_HAS_RICH_PROGRESS and _HAS_CLI_STATUS and sys.stdout.isatty()):
        while not _all_exists([p for it in items for p in it["outputs"]]):
            time.sleep(max(0.5, poll_sec))
        return

    assert Progress is not None
    assert SpinnerColumn is not None
    assert BarColumn is not None
    assert TextColumn is not None

    total = len(items)
    if total == 0:
        return

    done: set[int] = set()
    for i, item in enumerate(items):
        outputs = list(item.get("outputs", []))
        if _all_exists(outputs):
            done.add(i)

    task_map: dict[int, int] = {}
    progress = Progress(
        SpinnerColumn(
            spinner_name=get_rich_spinner_name(),
            style="cyan",
            finished_text=" ",
        ),
        TextColumn("{task.fields[idx]} {task.fields[name]}"),
        BarColumn(),
        TextColumn("{task.percentage:>6.1f}%"),
        transient=True,
    )

    def _add_visible() -> None:
        for i, item in enumerate(items):
            if len(task_map) >= int(max_visible):
                break
            if i in done or i in task_map:
                continue
            task_id = progress.add_task(
                description="",
                total=1,
                completed=0,
                idx=f"[{i+1}/{total}]",
                name=str(item.get("id", f"item-{i+1}")),
            )
            task_map[i] = task_id

    with progress:
        _add_visible()
        while len(done) < total:
            advanced = False
            for i, item in enumerate(items):
                if i in done:
                    continue
                outputs = list(item.get("outputs", []))
                if not _all_exists(outputs):
                    continue
                done.add(i)
                task_id = task_map.pop(i, None)
                if task_id is not None:
                    progress.update(task_id, completed=1)
                    try:
                        progress.remove_task(task_id)
                    except Exception:
                        pass
                advanced = True
            _add_visible()
            if len(done) >= total:
                break
            if not advanced:
                time.sleep(max(0.5, poll_sec))


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
        return (
            f'csub -J {job} -o ./log/%J.o -e ./log/%J.e -q {queue} '
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
                        max_visible=5,
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
