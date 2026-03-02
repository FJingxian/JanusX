import subprocess
import time
from pathlib import Path
from typing import List, Literal, Optional, Union

PathLike = Union[Path, str]

try:
    from janusx.script._common.status import CliStatus, print_success

    _HAS_CLI_STATUS = True
except Exception:
    CliStatus = None  # type: ignore[assignment]
    _HAS_CLI_STATUS = False


def _step_label(step_idx: int, total_steps: int, step_names: Optional[List[str]]) -> str:
    if step_names is not None and step_idx < len(step_names):
        name = str(step_names[step_idx])
    else:
        name = f"Step-{step_idx}"
    return f"Step {step_idx + 1}/{total_steps}: {name}"


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
            with CliStatus(step_text, enabled=True, show_elapsed=True) as task:
                try:
                    subprocess.run(["bash", str(sh)], check=True)
                    while True:
                        if all(Path(p).exists() for p in ofiles):
                            break
                        time.sleep(5)
                except Exception:
                    task.fail(f"{step_text} ...Failed")
                    raise
                task.complete(f"{step_text} ...Finished")
            continue

        print(f"Run {step_text}: {sh}")
        subprocess.run(["bash", str(sh)], check=True)
        while True:
            if all(Path(p).exists() for p in ofiles):
                print(f"\n{step_text} completed!")
                break
            print(f"\rWaiting outputs... time={time.time() - tstart:.1f}s", end="")
            time.sleep(60)
