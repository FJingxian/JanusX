from __future__ import annotations

import os

from . import adamixture as _impl


def main() -> None:
    os.environ["JANUSX_POPSTRUCT_BRAND"] = "fastpop"
    _impl.main()


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    main()
