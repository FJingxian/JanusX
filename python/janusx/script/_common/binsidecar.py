from __future__ import annotations

# Legacy JanusX binary matrix payload magic.
BIN01_MAGIC = b"JXBIN001"

# Legacy k-mer sidecar for BIN01 caches: .bin.site
BIN_SITE_MAGIC = b"JXBSITE1"
BIN_SITE_HEADER_SIZE = 24

# Legacy Garfield-style sidecar: .bsite
LEGACY_BSITE_MAGIC = b"JXBSIT02"
LEGACY_BSITE_HEADER_SIZE = 36
LEGACY_BSITE_VERSION = 1
