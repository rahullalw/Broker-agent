"""Environment, the D24 timezone contract and the D9 process contract.

Every datetime in AllSet is naive local time, and every comparison assumes that
local time is IST. That is true on a developer laptop in Ahmedabad and quietly
false on a UTC server, so `assert_ist()` turns the quiet failure into a loud one.

`assert_single_worker()` is the same idea applied to the process model: the
debounce buffers, the per-conversation locks and the shared coalescing events all
live in plain dicts in this process (D9). A second worker does not corrupt them,
which is worse — it gets its own copy, so two buyers' turns run concurrently
against one SQLite file and coalescing silently stops working.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

from dotenv import load_dotenv

IST_OFFSET = timedelta(hours=5, minutes=30)

# The POSIX `std offset` form. glibc reads IANA names like "Asia/Kolkata"; the
# MSVC runtime does not — it reads "Asia" as a zone at UTC+0 and "/Kolkata" as a
# daylight name, landing on UTC+1 and shifting datetime.now() by 4.5 hours.
WINDOWS_IST_TZ = "IST-05:30"
_IANA_IST_NAMES = ("Asia/Kolkata", "Asia/Calcutta")

load_dotenv()


def _normalise_tz() -> None:
    """Make TZ mean IST on this platform, for this process where possible and
    for child processes always."""
    if os.name == "nt":
        # This process latched TZ when the interpreter started, so rewriting it
        # now cannot fix the current clock — assert_ist() is what catches that.
        # It does keep subprocesses (uvicorn reloads, the seed CLI, pytest
        # children) from inheriting a value their runtime would misread.
        if os.environ.get("TZ", "") in _IANA_IST_NAMES:
            os.environ["TZ"] = WINDOWS_IST_TZ
        os.environ.setdefault("TZ", WINDOWS_IST_TZ)
    else:
        os.environ.setdefault("TZ", "Asia/Kolkata")
        time.tzset()


_normalise_tz()


def local_offset() -> timedelta:
    """The UTC offset `datetime.now()` is actually producing right now.

    Measured rather than derived: `time.timezone`/`time.daylight` describe the
    zone's rules, not which of them is in force, and both go wrong on Windows
    when TZ is unparsable.
    """
    return datetime.now().astimezone().utcoffset()


def assert_ist() -> None:
    """Raise unless local time is IST, naming the fix for this platform."""
    offset = local_offset()
    if offset == IST_OFFSET:
        return

    hours = offset.total_seconds() / 3600
    remedy = "Set TZ=Asia/Kolkata before starting."
    if os.name == "nt":
        remedy += (
            f" On Windows the C runtime cannot read IANA zone names — "
            f"TZ=Asia/Kolkata resolves to UTC+1, not IST. "
            f"Use TZ={WINDOWS_IST_TZ}, or unset TZ and set the system zone to "
            f"India Standard Time."
        )
    raise RuntimeError(
        f"AllSet assumes naive IST datetimes (D24). "
        f"Process offset is {hours:+.2f}h, expected +5.50h. {remedy}"
    )


# uvicorn resolves the worker count in exactly this order: an explicit
# `--workers` flag wins, otherwise `WEB_CONCURRENCY` from the environment
# (uvicorn/config.py). Reading it the same way is what makes the assertion true
# rather than merely reassuring. Note gunicorn spells the flag `-w`.
_WORKER_FLAGS = ("--workers", "-w")


def _argv_workers() -> int | None:
    """The worker count named on the command line, or None if it was not."""
    argv = sys.argv[1:]
    for index, token in enumerate(argv):
        for flag in _WORKER_FLAGS:
            if token == flag and index + 1 < len(argv):
                candidate = argv[index + 1]
            elif token.startswith(f"{flag}="):
                candidate = token[len(flag) + 1:]
            else:
                continue
            try:
                return int(candidate)
            except ValueError:
                return None
    return None


def worker_count() -> int:
    """How many workers this process was launched with, resolved uvicorn's way."""
    named = _argv_workers()
    if named is not None:
        return named
    raw = os.environ.get("WEB_CONCURRENCY", "").strip()
    try:
        return int(raw) if raw else 1
    except ValueError:
        return 1          # uvicorn will reject it far more loudly than we could


def assert_single_worker() -> None:
    """Raise unless this process is the only one serving the app (D9)."""
    workers = worker_count()
    if workers <= 1:
        return
    raise RuntimeError(
        f"AllSet runs single-worker (D9): the debounce buffers, the "
        f"per-conversation locks and the coalescing events are in-process state, "
        f"and a second worker gets its own copy of all three. Launched with "
        f"{workers} workers. Start it as "
        f"`uvicorn app.main:app --workers 1` and leave WEB_CONCURRENCY unset."
    )


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else value


def _env_int(name: str, default: int) -> int:
    """Empty means "unset"; "0" is a real value (D10 relies on DEBOUNCE_MS=0)."""
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on")


def reload() -> ModuleType:
    """Re-read the environment into module globals. Returns this module so tests
    can write `config.reload().DEBOUNCE_MS`."""
    module = sys.modules[__name__]
    module.OPENROUTER_API_KEY = _env_str("OPENROUTER_API_KEY", "")
    module.MODEL_PRIMARY = _env_str("MODEL_PRIMARY", "google/gemini-3.1-flash-lite")
    module.MODEL_FALLBACK = _env_str("MODEL_FALLBACK", "google/gemini-3.5-flash-lite")
    module.MODEL_LAST_RESORT = _env_str("MODEL_LAST_RESORT", "google/gemma-4-31b-it:free")
    module.DATABASE_URL = _env_str("DATABASE_URL", "sqlite:///allset.db")
    module.DEBOUNCE_MS = _env_int("DEBOUNCE_MS", 3000)
    module.DEBOUNCE_MAX_MS = _env_int("DEBOUNCE_MAX_MS", 10000)
    module.LLM_TIMEOUT_S = _env_int("LLM_TIMEOUT_S", 20)
    module.FORCE_GUARD_FAIL = _env_bool("FORCE_GUARD_FAIL", False)
    return module


def db_path() -> Path:
    """Filesystem path behind DATABASE_URL. `--reset` deletes this file (D21)."""
    prefix = "sqlite:///"
    if not DATABASE_URL.startswith(prefix):
        raise RuntimeError(
            f"AllSet only supports sqlite:/// URLs (D23), got {DATABASE_URL!r}"
        )
    return Path(DATABASE_URL[len(prefix):])


OPENROUTER_API_KEY: str
MODEL_PRIMARY: str
MODEL_FALLBACK: str
MODEL_LAST_RESORT: str
DATABASE_URL: str
DEBOUNCE_MS: int
DEBOUNCE_MAX_MS: int
LLM_TIMEOUT_S: int
FORCE_GUARD_FAIL: bool

reload()
