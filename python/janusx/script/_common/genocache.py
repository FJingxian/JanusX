from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from janusx.gfreader import set_genotype_cache_dir
from .pathcheck import safe_expanduser, safe_resolve


def configure_genotype_cache_from_out(out_value: Union[str, None]) -> Union[str, None]:
    """
    Set JANUSX genotype cache directory from CLI output directory.
    Returns normalized absolute cache directory path.
    """
    if out_value is None:
        return None
    raw = str(out_value).strip()
    if raw == "":
        return None
    out_path = safe_expanduser(raw)
    cache_dir = out_path if out_path.suffix == "" else out_path.parent
    cache_dir = safe_resolve(cache_dir)
    os.makedirs(cache_dir, mode=0o755, exist_ok=True)
    return set_genotype_cache_dir(str(cache_dir))
