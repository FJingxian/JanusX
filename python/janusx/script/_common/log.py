import os
import sys
import logging
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


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

    def __init__(self, *, enable_color: bool) -> None:
        super().__init__()
        self.enable_color = bool(enable_color)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
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


def setup_logging(log_file_path):
    """Configure logging to file and stdout."""
    if os.path.exists(log_file_path) and log_file_path[-4:]=='.log':
        os.remove(log_file_path)
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
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
