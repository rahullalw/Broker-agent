"""Shared fixtures.

Every test that touches the database gets its own file-backed SQLite engine and
monkeypatches `db.engine` onto it. Tools and the agent loop reach the database
through `db.get_session()`, which reads that global at call time, so patching
the attribute is enough — no dependency injection, no import-order games.

**Live tests are opt-in.** `pytest` runs the offline suite and spends nothing;
`pytest --live` adds the files that call real Gemini through OpenRouter. That
default is not timidity, it is the quota (Phase 5 §3): BYOK gives 500 requests a
day across the top two rungs, one turn costs two or three of them, and a suite
that runs live on every save burns the demo's budget before the demo.
"""
import pytest
from sqlmodel import Session, create_engine

from app import config as app_config, db, seed


# --------------------------------------------------------------------------
# The live opt-in
# --------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run the tests that call OpenRouter for real (costs quota).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: calls a real model through OpenRouter; needs --live."
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        if not app_config.OPENROUTER_API_KEY:
            pytest.exit(
                "--live needs OPENROUTER_API_KEY set (put it in .env).",
                returncode=1,
            )
        return
    skip = pytest.mark.skip(reason="live test — re-run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path, monkeypatch):
    target = create_engine(
        "sqlite:///" + (tmp_path / "allset.db").as_posix(),
        connect_args={"check_same_thread": False},
    )
    db.init_db(target)
    monkeypatch.setattr(db, "engine", target)
    yield target
    target.dispose()


@pytest.fixture
def seeded(engine):
    """The committed 25 properties / 3 buyers / 3 conversations (D21)."""
    with Session(engine) as session:
        seed.seed_all(session)
    return engine


@pytest.fixture
def session(seeded):
    with Session(seeded) as s:
        yield s
