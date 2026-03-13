from __future__ import annotations

from typing import Optional


_CJK_FONT_READY: Optional[bool] = None


def contains_cjk(text: object) -> bool:
    for ch in str(text):
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3000 <= code <= 0x303F
            or 0xFF00 <= code <= 0xFFEF
        ):
            return True
    return False


def ensure_cjk_font() -> bool:
    """
    Ensure matplotlib has at least one CJK-capable sans font configured.
    Returns True when a candidate font is found, otherwise False.
    """
    global _CJK_FONT_READY
    if _CJK_FONT_READY is not None:
        return _CJK_FONT_READY

    import matplotlib as mpl
    from matplotlib import font_manager as mpl_font_manager

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "PingFang SC",
        "Heiti SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in mpl_font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), None)
    if selected is None:
        _CJK_FONT_READY = False
        return False

    current = mpl.rcParams.get("font.sans-serif", [])
    if not isinstance(current, list):
        current = [str(current)]
    mpl.rcParams["font.sans-serif"] = [selected] + [x for x in current if x != selected]
    mpl.rcParams["axes.unicode_minus"] = False
    _CJK_FONT_READY = True
    return True
