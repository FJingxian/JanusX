import os
import sys
import logging
import warnings
import time
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def _console_symbol(preferred: str, fallback: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        str(preferred).encode(enc)
        return str(preferred)
    except Exception:
        return str(fallback)


class _LevelPrefixFormatter(logging.Formatter):
    """Add fixed prefix for warning/error log records."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.ERROR:
            return msg if msg.startswith("Error: ") else f"Error: {msg}"
        if record.levelno == logging.WARNING:
            return msg if msg.startswith("Warning: ") else f"Warning: {msg}"
        return msg


class _ColorLevelPrefixFormatter(_LevelPrefixFormatter):
    """Prefix warning/error and optionally colorize console output."""

    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _RESET = "\033[0m"
    _FAIL = _console_symbol("\u2718\ufe0e", "[X]")
    _WARN = "!"

    def __init__(self, *, enable_color: bool) -> None:
        super().__init__()
        self.enable_color = bool(enable_color)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.ERROR:
            msg = f"{self._FAIL} {msg}"
        elif record.levelno == logging.WARNING:
            msg = f"{self._WARN} {msg}"
        if not self.enable_color:
            return msg
        if record.levelno >= logging.ERROR:
            return f"{self._RED}{msg}{self._RESET}"
        if record.levelno == logging.WARNING:
            return f"{self._YELLOW}{msg}{self._RESET}"
        return msg


def _enable_windows_ansi(stream) -> bool:
    """Enable ANSI escape on Windows consoles; no-op on non-Windows."""
    if os.name != "nt":
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if mode.value & enable_vt:
            return True
        return kernel32.SetConsoleMode(handle, mode.value | enable_vt) != 0
    except Exception:
        return False


def _supports_color(stream) -> bool:
    """Best-effort cross-platform color support detection."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":
        return _enable_windows_ansi(stream)
    return True


def _fallback_log_path(log_file_path: str) -> str:
    base, ext = os.path.splitext(str(log_file_path))
    suffix = ext if ext else ".log"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    cand = f"{base}.{stamp}.{pid}{suffix}"
    idx = 1
    while os.path.exists(cand):
        idx += 1
        cand = f"{base}.{stamp}.{pid}.{idx}{suffix}"
    return cand


def setup_logging(log_file_path):
    """Configure logging to file and stdout."""
    chosen_log_path = str(log_file_path)
    fallback_note = None
    try:
        out_dir = os.path.dirname(chosen_log_path)
        if out_dir:
            os.makedirs(out_dir, mode=0o755, exist_ok=True)
    except Exception:
        pass
    if os.path.exists(chosen_log_path) and chosen_log_path.lower().endswith(".log"):
        try:
            os.remove(chosen_log_path)
        except PermissionError:
            alt = _fallback_log_path(chosen_log_path)
            fallback_note = (
                f"Log file is in use: {chosen_log_path}. "
                f"Using fallback log file: {alt}"
            )
            chosen_log_path = alt
        except Exception:
            alt = _fallback_log_path(chosen_log_path)
            fallback_note = (
                f"Cannot reset log file: {chosen_log_path}. "
                f"Using fallback log file: {alt}"
            )
            chosen_log_path = alt
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    logger.handlers.clear()
    # Set log format
    file_formatter = _LevelPrefixFormatter()
    console_formatter = _ColorLevelPrefixFormatter(
        enable_color=_supports_color(sys.stdout)
    )
    # File handler
    try:
        file_handler = logging.FileHandler(chosen_log_path, mode="w", encoding='utf-8')
    except PermissionError:
        alt = _fallback_log_path(chosen_log_path)
        fallback_note = (
            f"Log file is in use: {chosen_log_path}. "
            f"Using fallback log file: {alt}"
        )
        chosen_log_path = alt
        file_handler = logging.FileHandler(chosen_log_path, mode="w", encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Route Python warnings.warn(...) through logging so WARNING style/color is unified.
    logging.captureWarnings(True)
    pywarn_logger = logging.getLogger("py.warnings")
    pywarn_logger.handlers.clear()
    pywarn_logger.propagate = True
    warnings.simplefilter("default")
    if fallback_note:
        logger.warning(fallback_note)
    return logger
