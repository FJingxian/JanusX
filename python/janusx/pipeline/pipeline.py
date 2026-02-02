import time
import subprocess
from typing import Any, List, Union, Literal, Dict
from pathlib import Path
PathLike = Union[Path, str]



def wrap_cmd(cmd: str, job: str, threads: int, scheduler: Literal["nohup", "csub"]='nohup', queue: str = "c01",singularity:str="") -> str:
    
    if scheduler == "nohup":
        safe_cmd = cmd.replace('"', '\\"')
        return (
            f'nohup bash -c "{singularity} {safe_cmd}" '
            f'> ./log/{job}.o 2> ./log/{job}.e'
        )
    elif scheduler == "csub":
        return f'csub -J {job} -o ./log/%J.o -e ./log/%J.e -q {queue} -n {threads} "{singularity} {cmd}"'

def node(ifiles: List[PathLike], ofiles: List[PathLike]) -> Literal[-1, 0, 1]:
    outexists = all(Path(p).exists() for p in ofiles)
    if outexists:
        return 1  # pass

    inexists = all(Path(p).exists() for p in ifiles)
    if inexists:
        return 0  # run
    return -1     # back

def pipeline(AllCMD: List[str],
             Alli: List[List[PathLike]],
             Allo: List[List[PathLike]],
             scheduler: str = "csub",
             nohup_max_jobs: int = 0) -> None:
    Path("tmp").mkdir(mode=0o755, parents=True, exist_ok=True)

    for num, cmd in enumerate(AllCMD):
        ifiles = Alli[num]
        ofiles = Allo[num]

        status = node(ifiles, ofiles)

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
            print(f"Step-{num} completed! (skip)")
            continue

        if status == -1:
            raise ValueError(f"Step-{num} inputs not exists: {ifiles}")

        # status == 0
        print(f"Run Step-{num}: {sh}")
        # 真实执行 + 检查退出码
        subprocess.run(["bash", str(sh)], check=True)

        # 等待输出出现（如果命令同步产生输出，这里一般很快结束）
        while True:
            if node(ifiles, ofiles) == 1:
                print(f"\nStep-{num} completed!")
                break
            print(f"\rWaiting outputs... time={time.time() - tstart:.1f}s", end="")
            time.sleep(60)