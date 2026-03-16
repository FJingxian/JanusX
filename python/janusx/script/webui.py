#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JanusX WebUI CLI entrypoint.

Core WebUI implementation is located in `janusx.ui`.
"""

from __future__ import annotations

from janusx.ui import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()


