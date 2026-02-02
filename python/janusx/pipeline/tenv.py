import os
import subprocess
from typing import Sequence, Optional, Mapping, Union, List
import shutil

def _build(args: Sequence[str], timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a lightweight command to verify availability."""
    return subprocess.run(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
def _which(cmd: str) -> Optional[str]:
    """Return full path if command exists in PATH."""
    return shutil.which(cmd)

if __name__ == '__main__':
    
    status = _build(['conda','--version',''])
    print(status)