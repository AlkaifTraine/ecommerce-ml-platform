"""Uniform logging for scripts and services.

A library must not seize the root logger, and this module previously did.

The failure it caused is worth recording. `get_logger` attached a
StreamHandler bound to `sys.stdout`. Under Airflow, `sys.stdout` is not a
stream - it is a `StreamLogWriter` that forwards whatever it receives back
into the logging system. So one `log.info(...)` became:

    log.info -> our handler -> writes to sys.stdout -> which is a logger
             -> logs      -> our handler -> writes to sys.stdout -> ...

an unbounded cycle that ended in `RecursionError` with no traceback, because
the recursion consumed the stack before the error could be written. The same
code ran perfectly from a terminal, where stdout is a real stream, which made
it look like an Airflow problem for some time.

Two defences now:

1. If the root logger already has handlers, someone else owns logging
   configuration (Airflow, pytest, uvicorn) and we attach nothing.
2. When we do configure, we bind to `sys.__stdout__` - the ORIGINAL stdout,
   which no framework can replace - so the cycle cannot form even if stdout
   is redirected later.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED

    if not _CONFIGURED:
        root = logging.getLogger()
        if root.handlers:
            # Airflow, pytest or a server already configured logging. Adding a
            # handler here would duplicate every line at best, and at worst
            # (Airflow) create the stdout/logger cycle described above.
            _CONFIGURED = True
        else:
            # sys.__stdout__, never sys.stdout: the former is the real stream
            # captured at interpreter start and cannot be swapped for a logger.
            handler = logging.StreamHandler(sys.__stdout__)
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            root.addHandler(handler)
            root.setLevel(level)
            _CONFIGURED = True

    return logging.getLogger(name)
