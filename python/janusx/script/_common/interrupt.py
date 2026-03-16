#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
from typing import Optional

_FORCE_EXITING = False
_HANDLERS_INSTALLED = False


def _terminate_active_children() -> None:
    # Python multiprocessing workers.
    try:
        for child in list(mp.active_children()):
            try:
                child.terminate()
            except Exception:
                pass
        for child in list(mp.active_children()):
            try:
                child.join(timeout=0.25)
            except Exception:
                pass
    except Exception:
        pass

    # Other child processes (subprocess, native workers).
    try:
        import psutil  # type: ignore

        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)
        for proc in children:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            _, alive = psutil.wait_procs(children, timeout=0.5)
        except Exception:
            alive = []
        for proc in alive:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        pass


def force_exit(code: int = 130, msg: Optional[str] = None) -> None:
    global _FORCE_EXITING
    if _FORCE_EXITING:
        os._exit(int(code))
    _FORCE_EXITING = True
    if msg:
        try:
            print(str(msg), file=sys.stderr, flush=True)
        except Exception:
            pass
    try:
        _terminate_active_children()
    finally:
        try:
            logging.shutdown()
        finally:
            os._exit(int(code))


def install_interrupt_handlers() -> None:
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return

    def _handler(signum, _frame) -> None:
        sigterm = int(getattr(signal, "SIGTERM", -1))
        sigbreak = int(getattr(signal, "SIGBREAK", -1))
        s = int(signum)
        if s == sigterm:
            force_exit(143, "Terminated.")
        elif s == sigbreak:
            force_exit(131, "Interrupted by user (Ctrl+Break).")
        else:
            force_exit(130, "Interrupted by user (Ctrl+C).")

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

    _HANDLERS_INSTALLED = True

