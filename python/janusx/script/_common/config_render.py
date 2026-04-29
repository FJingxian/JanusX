import logging
import os
import sys
from typing import Any, Optional, Sequence

from .pathcheck import format_path_for_display
from .status import stdout_is_tty

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    _HAS_RICH = True
except Exception:
    Console = None  # type: ignore[assignment]
    Group = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    box = None  # type: ignore[assignment]
    _HAS_RICH = False


def _is_light_terminal() -> bool:
    """
    Best-effort light-background terminal detection.

    Priority:
    1) Explicit override via JANUSX_TERM_BG=light|dark
    2) COLORFGBG (use last numeric token as background index)
    """
    forced = str(os.environ.get("JANUSX_TERM_BG", "")).strip().lower()
    if forced in {"light", "white", "1", "true", "yes", "on"}:
        return True
    if forced in {"dark", "black", "0", "false", "no", "off"}:
        return False

    cfgbg = str(os.environ.get("COLORFGBG", "")).strip()
    if cfgbg != "":
        tokens = (
            cfgbg.replace(";", " ")
            .replace(":", " ")
            .replace(",", " ")
            .split()
        )
        ints: list[int] = []
        for tok in tokens:
            try:
                ints.append(int(tok))
            except Exception:
                continue
        if len(ints) > 0:
            bg = int(ints[-1])
            # ANSI palette convention: 0-7 dark, 8-15 bright.
            # Treat bright backgrounds as light terminals.
            return bool(bg >= 7)

    # Conservative default: dark theme keeps existing CLI look.
    return False


def get_config_color_styles() -> dict[str, str]:
    """
    Return high-contrast style tokens for config panels.
    """
    if _is_light_terminal():
        return {
            "title": "bold",
            "key": "bold blue",
            "value": "default",
            "section": "bold blue",
            "border": "blue",
        }
    return {
        "title": "bold",
        "key": "bold cyan",
        "value": "default",
        "section": "bold cyan",
        "border": "green",
    }


def _emit_info_to_file_handlers(logger: logging.Logger, message: str) -> None:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        args=(),
        exc_info=None,
    )
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def _emit_info_to_stream_handlers(logger: logging.Logger, message: str) -> None:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        args=(),
        exc_info=None,
    )
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.handle(record)


def truncate_line(
    text: object,
    *,
    max_chars: int = 60,
    overflow_mark: str = "***",
) -> str:
    s = str(text)
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    mark = str(overflow_mark)
    if mark == "":
        return s[:max_chars]
    if max_chars <= len(mark):
        return mark[:max_chars]
    return s[: (max_chars - len(mark))] + mark


def _is_path_like_key(key: str) -> bool:
    k = str(key).strip().lower()
    markers = (
        "file",
        "path",
        "prefix",
        "dir",
        "folder",
        "genotype",
        "phenotype",
    )
    return any(m in k for m in markers)


def _format_config_value_for_display(key: str, value: object) -> str:
    s = str(value)
    if _is_path_like_key(key):
        return format_path_for_display(s)
    return s


def _render_rich_panel(
    *,
    app_title: str,
    config_title: str,
    host: str,
    sections: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    footer_rows: Sequence[tuple[str, str]],
    key_width: int,
    line_max_chars: int,
    overflow_mark: str,
) -> bool:
    if not (_HAS_RICH and stdout_is_tty()):
        return False

    try:
        assert Console is not None
        assert Group is not None
        assert Panel is not None
        assert Table is not None
        assert Text is not None
        assert box is not None

        styles = get_config_color_styles()

        def _kv_table(rows: Sequence[tuple[str, str]]) -> Any:
            table = Table(
                show_header=False,
                box=None,
                pad_edge=False,
                expand=False,
            )
            key_w = max(8, int(key_width))
            table.add_column(style=styles["key"], no_wrap=True, width=key_w, justify="left")
            table.add_column(style=styles["value"], no_wrap=True, justify="left")
            for key, val in rows:
                key_txt = str(key)
                val_max = max(1, int(line_max_chars) - key_w - 2)
                val_txt = truncate_line(val, max_chars=val_max, overflow_mark=overflow_mark)
                table.add_row(key_txt, val_txt)
            return table

        app_title_txt = truncate_line(app_title, max_chars=line_max_chars, overflow_mark=overflow_mark)
        host_txt = truncate_line(f"Host: {host}", max_chars=line_max_chars, overflow_mark=overflow_mark)
        panel_title_txt = truncate_line(config_title, max_chars=line_max_chars, overflow_mark=overflow_mark)
        parts: list[Any] = [
            Text(app_title_txt, style=styles["title"]),
            Text(host_txt),
            Text(""),
        ]
        for sec_name, sec_rows in sections:
            parts.append(Text(str(sec_name), style=styles["section"]))
            parts.append(_kv_table(sec_rows))
            parts.append(Text(""))

        if len(footer_rows) > 0:
            parts.append(Text("Output", style=styles["section"]))
            parts.append(_kv_table(footer_rows))

        panel = Panel(
            Group(*parts),
            title=panel_title_txt,
            border_style=styles["border"],
            expand=False,
        )
        Console().print(panel)
        return True
    except Exception:
        return False


def emit_cli_configuration(
    logger: logging.Logger,
    *,
    app_title: str,
    config_title: str,
    host: str,
    sections: Sequence[tuple[str, Sequence[tuple[str, object]]]],
    footer_rows: Optional[Sequence[tuple[str, object]]] = None,
    emit_to_stdout: bool = True,
    line_max_chars: int = 60,
    overflow_mark: str = "***",
) -> None:
    def _dedupe_rows(rows: Sequence[tuple[str, object]]) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for key, val in rows:
            pair = (str(key), str(val))
            if pair in seen:
                continue
            seen.add(pair)
            out.append((str(key), val))
        return out

    sec_norm: list[tuple[str, list[tuple[str, str]]]] = []
    for sec_name, sec_rows in sections:
        sec_rows = _dedupe_rows(sec_rows)
        rows_str = [
            (str(k), _format_config_value_for_display(str(k), v))
            for k, v in sec_rows
        ]
        if len(rows_str) > 0:
            sec_norm.append((str(sec_name), rows_str))

    footer_norm = [
        (str(k), _format_config_value_for_display(str(k), v))
        for k, v in (footer_rows or [])
    ]
    key_width = 8
    for _, rows in sec_norm:
        for k, _ in rows:
            key_width = max(key_width, len(str(k)))
    for k, _ in footer_norm:
        key_width = max(key_width, len(str(k)))

    def _fmt_kv(key: str, val: str, *, truncate: bool) -> str:
        pad = max(1, key_width - len(key))
        line = f"  {key}:{' ' * pad}{val}"
        if truncate:
            return truncate_line(line, max_chars=line_max_chars, overflow_mark=overflow_mark)
        return line

    divider_full = "*" * 60
    divider_terminal = "*" * max(1, int(line_max_chars))

    full_lines: list[str] = [divider_full, str(config_title), divider_full]
    terminal_lines: list[str] = [divider_terminal, str(config_title), divider_terminal]
    for sec_name, sec_rows in sec_norm:
        full_lines.append(f"{sec_name}:")
        terminal_lines.append(f"{sec_name}:")
        for key, val in sec_rows:
            full_lines.append(_fmt_kv(key, val, truncate=False))
            terminal_lines.append(_fmt_kv(key, val, truncate=True))
    if len(footer_norm) > 0:
        full_lines.append("Output:")
        terminal_lines.append("Output:")
        for key, val in footer_norm:
            full_lines.append(_fmt_kv(key, val, truncate=False))
            terminal_lines.append(_fmt_kv(key, val, truncate=True))
    full_lines.append(divider_full + "\n")
    terminal_lines.append(divider_terminal + "\n")

    if bool(emit_to_stdout):
        rich_rendered = _render_rich_panel(
            app_title=str(app_title),
            config_title=str(config_title),
            host=str(host),
            sections=sec_norm,
            footer_rows=footer_norm,
            key_width=key_width,
            line_max_chars=line_max_chars,
            overflow_mark=overflow_mark,
        )

        if not rich_rendered:
            _emit_info_to_stream_handlers(logger, str(app_title))
            _emit_info_to_stream_handlers(logger, f"Host: {host}\n")
            for line in terminal_lines:
                _emit_info_to_stream_handlers(logger, line)

    _emit_info_to_file_handlers(logger, str(app_title))
    _emit_info_to_file_handlers(logger, f"Host: {host}\n")
    for line in full_lines:
        _emit_info_to_file_handlers(logger, line)
