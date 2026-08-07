"""D24 — naive IST everywhere. The assertion is the only thing standing between
this codebase and silently-wrong datetimes on a UTC server."""
import os
import subprocess
import sys
import textwrap

import pytest

from app import config

WINDOWS = os.name == "nt"

# The value that actually yields IST for the platform's C runtime. glibc reads
# IANA names; the MSVC runtime only understands the POSIX `std offset` form and
# misparses "Asia/Kolkata" as UTC+0 with a DST rule.
IST_TZ = "IST-05:30" if WINDOWS else "Asia/Kolkata"


def _run_with_tz(tz: str | None, body: str = "assert_ist()") -> subprocess.CompletedProcess:
    """assert_ist() judges the offset the process was born with, so only a fresh
    interpreter can tell us what a differently-configured server would do."""
    env = {k: v for k, v in os.environ.items() if k != "TZ"}
    if tz is not None:
        env["TZ"] = tz
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            from app.config import assert_ist, local_offset
            {body}
            print("IST-OK")
        """)],
        capture_output=True, text=True, env=env, cwd=os.getcwd(),
    )


class TestAssertIst:
    def test_passes_under_ist(self):
        result = _run_with_tz(IST_TZ)
        assert result.returncode == 0, result.stderr
        assert "IST-OK" in result.stdout

    def test_passes_when_tz_is_unset_and_the_system_clock_is_ist(self):
        # The ordinary case on a developer machine in Ahmedabad.
        result = _run_with_tz(None)
        assert result.returncode == 0, result.stderr

    def test_raises_under_utc(self):
        result = _run_with_tz("UTC")
        assert result.returncode != 0
        assert "RuntimeError" in result.stderr

    def test_error_names_the_decision_and_the_fix(self):
        result = _run_with_tz("UTC")
        assert "D24" in result.stderr
        assert "TZ=Asia/Kolkata" in result.stderr
        assert "+5.50h" in result.stderr
        assert "+0.00h" in result.stderr

    def test_passes_in_this_process(self):
        config.assert_ist()  # must not raise on a correctly configured machine


@pytest.mark.skipif(not WINDOWS, reason="MSVC-runtime-specific TZ parsing")
class TestWindowsIanaFootgun:
    """On Windows, TZ=Asia/Kolkata does not mean IST — the C runtime reads it as
    UTC+1 and datetime.now() lands 4.5 hours early. config rewrites TZ to the
    POSIX form on import, which repairs the process as long as nothing has read
    the clock yet; assert_ist() is the backstop for when something has."""

    def test_iana_tz_is_repaired_when_config_is_imported_first(self):
        result = _run_with_tz("Asia/Kolkata")
        assert result.returncode == 0, result.stderr
        assert "IST-OK" in result.stdout

    def test_repaired_clock_reports_ist_not_utc_plus_one(self):
        result = _run_with_tz("Asia/Kolkata", body=(
            "from datetime import timedelta; "
            "assert local_offset() == timedelta(hours=5, minutes=30), local_offset()"
        ))
        assert result.returncode == 0, result.stderr

    def test_tz_is_rewritten_so_child_processes_inherit_a_parsable_value(self):
        result = _run_with_tz("Asia/Kolkata", body=(
            "import os; assert os.environ['TZ'] == 'IST-05:30', os.environ['TZ']"
        ))
        assert result.returncode == 0, result.stderr

    def test_clock_latched_before_config_import_is_caught_not_silently_wrong(self):
        # The CRT latches its zone on the first localtime() call. If that happens
        # before config can rewrite TZ, the process is stuck at UTC+1 — and must say so.
        result = _run_with_tz("Asia/Kolkata", body="", )
        assert result.returncode == 0  # sanity: the good ordering still works
        latched = subprocess.run(
            [sys.executable, "-c", textwrap.dedent("""
                from datetime import datetime
                datetime.now()
                from app.config import assert_ist
                assert_ist()
            """)],
            capture_output=True, text=True, cwd=os.getcwd(),
            env={**{k: v for k, v in os.environ.items() if k != "TZ"}, "TZ": "Asia/Kolkata"},
        )
        assert latched.returncode != 0, "a mis-latched clock must not pass silently"
        assert "+1.00h" in latched.stderr
        assert "IST-05:30" in latched.stderr


class TestLocalOffset:
    def test_reports_the_offset_datetime_now_is_actually_producing(self):
        from datetime import datetime, timedelta
        assert config.local_offset() == timedelta(hours=5, minutes=30)
        # and it agrees with the wall clock the rest of the app will read
        assert datetime.now().astimezone().utcoffset() == config.local_offset()


class TestSettings:
    def test_defaults_match_the_env_contract(self):
        assert config.MODEL_PRIMARY == "google/gemini-3.1-flash-lite"
        assert config.DATABASE_URL == "sqlite:///allset.db"

    def test_the_model_chain_runs_gemini_then_gemini_then_free(self):
        # 15 req/min and 500 req/day on the Gemini pair via BYOK; the pinned
        # :free model is what still answers when both are exhausted mid-demo.
        assert config.MODEL_PRIMARY == "google/gemini-3.1-flash-lite"
        assert config.MODEL_FALLBACK == "google/gemini-3.5-flash-lite"
        assert config.MODEL_LAST_RESORT.endswith(":free")
        assert config.DEBOUNCE_MS == 3000
        assert config.DEBOUNCE_MAX_MS == 10000
        assert config.LLM_TIMEOUT_S == 20
        assert config.FORCE_GUARD_FAIL is False

    def test_numeric_settings_are_ints_not_strings(self):
        for name in ("DEBOUNCE_MS", "DEBOUNCE_MAX_MS", "LLM_TIMEOUT_S"):
            assert isinstance(getattr(config, name), int), name

    def test_debounce_zero_is_honoured_for_the_scripted_run(self, monkeypatch):
        # D10 — DEBOUNCE_MS=0 must survive; a falsy-check bug would reset it to 3000.
        monkeypatch.setenv("DEBOUNCE_MS", "0")
        assert config.reload().DEBOUNCE_MS == 0

    def test_db_path_derived_from_database_url(self):
        assert config.db_path().name == "allset.db"

    def test_non_sqlite_url_is_refused(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/allset")
        config.reload()
        with pytest.raises(RuntimeError, match="D23"):
            config.db_path()

    @pytest.mark.parametrize("raw,expected", [("1", True), ("0", False), ("true", True), ("", False)])
    def test_force_guard_fail_parses_truthily(self, monkeypatch, raw, expected):
        monkeypatch.setenv("FORCE_GUARD_FAIL", raw)
        assert config.reload().FORCE_GUARD_FAIL is expected


class TestAssertSingleWorker:
    """D9 — the debounce buffers, the per-conversation locks and the coalescing
    events are plain dicts in this process. A second worker does not corrupt
    them, which is the dangerous part: it gets its own copy, so two turns for one
    buyer run concurrently and coalescing silently stops working.

    The resolution order mirrors uvicorn's own (uvicorn/config.py): an explicit
    `--workers` wins, otherwise `WEB_CONCURRENCY`.
    """

    def test_a_plain_launch_is_one_worker(self, monkeypatch):
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app"])
        assert config.worker_count() == 1
        config.assert_single_worker()

    def test_workers_one_is_explicitly_fine(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app", "--workers", "1"])
        config.assert_single_worker()

    @pytest.mark.parametrize("argv", [
        ["uvicorn", "app.main:app", "--workers", "4"],
        ["uvicorn", "app.main:app", "--workers=4"],
        ["gunicorn", "app.main:app", "-w", "4"],
        ["gunicorn", "app.main:app", "-w=4"],
    ])
    def test_more_than_one_worker_on_the_command_line_is_refused(self, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(RuntimeError, match="single-worker"):
            config.assert_single_worker()

    def test_web_concurrency_is_read_the_way_uvicorn_reads_it(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app"])
        monkeypatch.setenv("WEB_CONCURRENCY", "3")
        assert config.worker_count() == 3
        with pytest.raises(RuntimeError, match="single-worker"):
            config.assert_single_worker()

    def test_an_explicit_flag_beats_the_environment(self, monkeypatch):
        # uvicorn only consults WEB_CONCURRENCY when `workers` was not passed.
        monkeypatch.setenv("WEB_CONCURRENCY", "8")
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app", "--workers", "1"])
        assert config.worker_count() == 1

    def test_the_message_names_the_fix(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app", "--workers", "2"])
        with pytest.raises(RuntimeError) as raised:
            config.assert_single_worker()
        assert "--workers 1" in str(raised.value)
        assert "WEB_CONCURRENCY" in str(raised.value)

    @pytest.mark.parametrize("raw", ["", "not-a-number"])
    def test_an_unreadable_worker_count_is_left_to_uvicorn_to_reject(
        self, monkeypatch, raw
    ):
        # Guessing 4 here would refuse to start a server that is perfectly fine;
        # uvicorn's own int() will produce a far better error than we could.
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app"])
        monkeypatch.setenv("WEB_CONCURRENCY", raw)
        assert config.worker_count() == 1


@pytest.fixture(autouse=True)
def _restore_settings():
    yield
    config.reload()
