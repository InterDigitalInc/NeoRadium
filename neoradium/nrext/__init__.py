# Copyright (c) 2026, InterDigital AI Lab
# C-extension acceleration for NeoRadium.
# Imports the compiled _ext module when available; callers should check HAS_C_EXT
# before using any symbol from this package.

try:
    from ._ext import getCrc, decodeLBP
    HAS_C_EXT = True
except ImportError:
    getCrc    = None
    decodeLBP = None
    HAS_C_EXT = False
    
    from ..utils import warnOnce, DOCS_LOC
    warnOnce("C extension not available; using slower pure-Python fallback.\n"
             "For more information please visit: " + DOCS_LOC +
             "source/installation.html#troubleshooting",
             category=RuntimeWarning)
