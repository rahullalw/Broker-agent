"""Phase 2 §7 — the dispatcher.

Two jobs: run the tool and write exactly one `tool_calls` row with a real
latency. Malformed JSON, an unknown tool name or a schema-invalid payload all
come back as steering errors — the loop must never crash on model output.
"""
import json
from types import SimpleNamespace

import pytest
from sqlmodel import select

from app.db import Message, ToolCall
from app.dispatch import dispatch

pytestmark = pytest.mark.usefixtures("seeded")


def make_call(name, arguments, call_id="call_1"):
    """The shape the OpenAI SDK hands back on a tool_calls finish."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture
def message_id(session):
    """Tool calls hang off the user message that provoked them (D11 keeps the
    assistant-with-tool_calls row out of the transcript entirely)."""
    message = Message(conversation_id=1, role="user", content="3BHK in Bopal")
    session.add(message)
    session.commit()
    session.refresh(message)
    return message.id


def rows(session):
    return session.exec(select(ToolCall)).all()


class TestHappyPath:
    def test_the_tool_result_comes_straight_back(self, message_id):
        result = dispatch(
            make_call("search_properties", {"bhk": 3, "localities": ["Bopal"]}),
            message_id,
        )
        assert result["exact_matches"]

    def test_exactly_one_row_is_written(self, session, message_id):
        dispatch(make_call("search_properties", {"bhk": 3}), message_id)
        assert len(rows(session)) == 1

    def test_the_row_names_the_tool_and_keeps_both_sides(self, session, message_id):
        dispatch(make_call("get_property_details",
                           {"property_id": 5, "sections": ["legal"]}), message_id)
        row = rows(session)[0]
        assert row.message_id == message_id
        assert row.tool_name == "get_property_details"
        assert row.args == {"property_id": 5, "sections": ["legal"]}
        assert row.result["legal"]["rera_id"]

    def test_latency_is_recorded_and_never_null(self, session, message_id):
        dispatch(make_call("search_properties", {}), message_id)
        row = rows(session)[0]
        assert row.latency_ms is not None
        assert isinstance(row.latency_ms, int)
        assert row.latency_ms >= 0

    def test_each_call_gets_its_own_row(self, session, message_id):
        dispatch(make_call("search_properties", {}), message_id)
        dispatch(make_call("update_buyer_profile", {"buyer_id": 1, "updates": {}}),
                 message_id)
        assert [r.tool_name for r in rows(session)] == [
            "search_properties", "update_buyer_profile",
        ]

    def test_an_empty_argument_string_is_treated_as_no_arguments(self, message_id):
        # Gemini sends "" rather than "{}" when a tool takes no arguments.
        result = dispatch(make_call("search_properties", ""), message_id)
        assert "error" not in result

    def test_writes_are_visible_to_the_next_tool(self, message_id):
        dispatch(make_call("update_buyer_profile",
                           {"buyer_id": 1, "updates": {"bhk_need": 3}}), message_id)
        result = dispatch(make_call("update_buyer_profile",
                                    {"buyer_id": 1, "updates": {}}), message_id)
        assert result["profile"]["bhk_need"] == 3


class TestModelOutputNeverCrashesTheLoop:
    def test_unknown_tool_name_steers(self, message_id):
        result = dispatch(make_call("negotiate_price", {"amount": 100}), message_id)
        assert "error" in result
        assert "negotiate_price" in result["error"]

    def test_unknown_tool_name_names_the_real_tools(self, message_id):
        result = dispatch(make_call("negotiate_price", {}), message_id)
        for name in ("search_properties", "book_site_visit", "escalate_to_broker"):
            assert name in result["error"]

    def test_unknown_tool_name_still_leaves_a_trace(self, session, message_id):
        dispatch(make_call("negotiate_price", {}), message_id)
        row = rows(session)[0]
        assert row.tool_name == "negotiate_price"
        assert "error" in row.result
        assert row.latency_ms is not None

    def test_malformed_json_arguments_steer(self, message_id):
        result = dispatch(make_call("search_properties", "{bhk: 3,,"), message_id)
        assert "error" in result
        assert "JSON" in result["error"]

    def test_malformed_json_arguments_still_leave_a_trace(self, session, message_id):
        dispatch(make_call("search_properties", "{bhk: 3,,"), message_id)
        assert len(rows(session)) == 1
        assert "error" in rows(session)[0].result

    def test_arguments_that_are_not_an_object_steer(self, message_id):
        result = dispatch(make_call("search_properties", "[3]"), message_id)
        assert "error" in result

    def test_an_unexpected_argument_steers(self, message_id):
        result = dispatch(
            make_call("search_properties", {"bhk": 3, "swimming_pool": True}),
            message_id,
        )
        assert "error" in result
        assert "swimming_pool" in result["error"]

    def test_a_missing_required_argument_steers(self, message_id):
        result = dispatch(make_call("book_site_visit", {"buyer_id": 1}), message_id)
        assert "error" in result

    def test_a_tool_that_raises_becomes_a_steering_error(self, monkeypatch, message_id):
        def explode(**kwargs):
            raise RuntimeError("sqlite went for a walk")

        monkeypatch.setitem(
            __import__("app.dispatch", fromlist=["TOOL_MAPPING"]).TOOL_MAPPING,
            "search_properties", explode,
        )
        result = dispatch(make_call("search_properties", {}), message_id)
        assert "error" in result

    def test_a_steering_error_from_the_tool_is_logged_as_the_result(self, session, message_id):
        dispatch(make_call("update_buyer_profile",
                           {"buyer_id": 2, "updates": {"preferred_localities": ["Bopal West"]}}),
                 message_id)
        assert "Bopal West" in rows(session)[0].result["error"]
        assert len(rows(session)) == 1
