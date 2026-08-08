"""Phase 3 §3 — inbound, debounce, lock and the turn.

No network. `llm.llm_call` is scripted so the loop's own decisions are what is
under test: coalescing, the status guard, dispatching a whole parallel batch
before acting on escalation, the step cap, and the text-only transcript (D11).
"""
import asyncio
import json
from types import SimpleNamespace

import openai
import pytest
from sqlmodel import select

from app import agent, config, llm
from app.db import Conversation, Escalation, Message, ToolCall

pytestmark = pytest.mark.usefixtures("seeded")


# --------------------------------------------------------------------------
# A scripted model
# --------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"

    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in (self.tool_calls or [])
            ] or None,
        }


def says(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=FakeMessage(content=text))]
    )


def calls(*specs):
    """specs: (name, args_dict) pairs the model emits in one batch."""
    tool_calls = [
        SimpleNamespace(
            id=f"call_{i}", type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        for i, (name, args) in enumerate(specs)
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="tool_calls", message=FakeMessage(tool_calls=tool_calls)
        )]
    )


class FakeLLM:
    """Replays a script and records every messages[] it was handed."""

    def __init__(self, script, model="google/gemini-2.5-flash", provider="Google"):
        self.script = list(script)
        self.model = model
        self.provider = provider
        self.seen = []

    async def __call__(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        if isinstance(self.script[0], Exception):
            raise self.script[0]
        response = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return response, self.model, self.provider

    @property
    def call_count(self):
        return len(self.seen)


@pytest.fixture(autouse=True)
def clean_state():
    agent.reset_state()
    yield
    agent.reset_state()


@pytest.fixture
def fast(monkeypatch):
    """Debounce off — the scripted-demo setting (D10)."""
    monkeypatch.setattr(config, "DEBOUNCE_MS", 0)
    monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 10_000)


@pytest.fixture
def model(monkeypatch):
    def install(*script, **kwargs):
        fake = FakeLLM(script, **kwargs)
        monkeypatch.setattr(llm, "llm_call", fake)
        return fake
    return install


def messages_of(session, conversation_id=1):
    return session.exec(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    ).all()


def tool_rows(session):
    return session.exec(select(ToolCall).order_by(ToolCall.id)).all()


# --------------------------------------------------------------------------
# Inbound dedup (D8)
# --------------------------------------------------------------------------

class TestDuplicateInbound:
    async def test_a_repeat_wa_message_id_is_dropped_silently(self, fast, model, session):
        model(says("Sure."))
        first = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        second = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert first["status"] != "duplicate"
        assert second == {"reply": None, "status": "duplicate"}

    async def test_a_duplicate_runs_no_turn(self, fast, model, session):
        fake = model(says("Sure."))
        await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert fake.call_count == 1

    async def test_a_duplicate_stores_no_second_message(self, fast, model, session):
        model(says("Sure."))
        await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert [m.content for m in messages_of(session) if m.role == "user"] == [
            "3BHK in Bopal",
        ]

    async def test_a_message_already_in_the_database_is_a_duplicate(self, fast, model, session):
        session.add(Message(conversation_id=1, role="user", content="old",
                            wa_message_id="wamid.9"))
        session.commit()
        model(says("Sure."))
        assert await agent.on_inbound(1, "wamid.9", "old") == {
            "reply": None, "status": "duplicate",
        }


# --------------------------------------------------------------------------
# Debounce and the shared future (D8, D10)
# --------------------------------------------------------------------------

class TestCoalescing:
    @pytest.fixture
    def slow(self, monkeypatch):
        # The exit criteria in miniature: 150ms sliding, 400ms cap.
        monkeypatch.setattr(config, "DEBOUNCE_MS", 150)
        monkeypatch.setattr(config, "DEBOUNCE_MAX_MS", 400)

    async def test_three_posts_produce_one_turn_and_one_reply(self, slow, model, session):
        fake = model(says("Got it."))

        async def post(index, delay):
            await asyncio.sleep(delay)
            return await agent.on_inbound(1, f"wamid.{index}", f"msg {index}")

        results = await asyncio.gather(post(1, 0), post(2, 0.05), post(3, 0.10))

        assert fake.call_count == 1
        assert results[0] == {"reply": None, "status": "coalesced"}
        assert results[1] == {"reply": None, "status": "coalesced"}
        assert results[2]["reply"] == "Got it."

    async def test_the_last_request_is_the_one_that_answers(self, slow, model):
        model(says("Got it."))

        async def post(index, delay):
            await asyncio.sleep(delay)
            return await agent.on_inbound(1, f"wamid.{index}", f"msg {index}")

        results = await asyncio.gather(post(1, 0), post(2, 0.05))
        assert results[-1]["status"] == "active"

    async def test_the_fragments_arrive_as_one_user_message(self, slow, model, session):
        model(says("Got it."))

        async def post(index, delay, text):
            await asyncio.sleep(delay)
            return await agent.on_inbound(1, f"wamid.{index}", text)

        await asyncio.gather(
            post(1, 0, "Hi"), post(2, 0.05, "3bhk chahiye"), post(3, 0.10, "bopal me"),
        )
        stored = [m for m in messages_of(session) if m.role == "user"][-1]
        assert stored.content == "Hi\n3bhk chahiye\nbopal me"

    async def test_a_long_drip_fires_at_the_hard_cap(self, slow, model):
        """A fast typer must not starve the turn (D10)."""
        fake = model(says("Got it."))
        loop = asyncio.get_running_loop()
        started = loop.time()

        async def drip():
            for index in range(12):
                await agent.on_inbound(1, f"wamid.{index}", f"msg {index}")

        posts = [asyncio.create_task(drip())]
        for index in range(12, 24):
            await asyncio.sleep(0.06)
            posts.append(asyncio.create_task(
                agent.on_inbound(1, f"wamid.{index}", f"msg {index}")
            ))
            if fake.call_count:
                break
        fired = loop.time() - started

        assert fake.call_count == 1
        assert 0.4 <= fired < 0.72, f"fired at {fired:.3f}s, cap is 0.4s"
        for task in posts:
            task.cancel()

    async def test_messages_arriving_during_a_turn_form_the_next_batch(self, slow, model, session):
        model(says("One."), says("Two."))
        first = await agent.on_inbound(1, "wamid.1", "first")
        second = await agent.on_inbound(1, "wamid.2", "second")
        assert first["reply"] == "One."
        assert second["reply"] == "Two."
        assert [m.content for m in messages_of(session) if m.role == "user"][-2:] == [
            "first", "second",
        ]


# --------------------------------------------------------------------------
# The status guard (§7)
# --------------------------------------------------------------------------

class TestPostEscalationSilence:
    @pytest.fixture
    def escalated(self, session):
        convo = session.get(Conversation, 1)
        convo.status = "escalated"
        session.add(convo)
        session.commit()

    async def test_the_agent_says_nothing(self, escalated, fast, model):
        model(says("I should not speak."))
        result = await agent.on_inbound(1, "wamid.1", "kya hua bhai")
        assert result["reply"] is None
        assert result["status"] == "escalated"

    async def test_the_model_is_never_called(self, escalated, fast, model):
        fake = model(says("I should not speak."))
        await agent.on_inbound(1, "wamid.1", "kya hua bhai")
        assert fake.call_count == 0

    async def test_the_message_is_stored_for_the_broker(self, escalated, fast, model, session):
        model(says("nope"))
        await agent.on_inbound(1, "wamid.1", "kya hua bhai")
        stored = [m for m in messages_of(session) if m.role == "user"][-1]
        assert stored.content == "kya hua bhai"
        assert stored.wa_message_id == "wamid.1"

    async def test_no_assistant_row_is_written(self, escalated, fast, model, session):
        model(says("nope"))
        await agent.on_inbound(1, "wamid.1", "kya hua bhai")
        assert [m for m in messages_of(session) if m.role == "assistant"] == []


# --------------------------------------------------------------------------
# The turn
# --------------------------------------------------------------------------

class TestPlainTurn:
    async def test_the_reply_comes_back(self, fast, model):
        model(says("Bopal me 3BHK dekhte hain. Budget kitna hai?"))
        result = await agent.on_inbound(1, "wamid.1", "3BHK chahiye")
        assert result["reply"] == "Bopal me 3BHK dekhte hain. Budget kitna hai?"
        assert result["status"] == "active"

    async def test_the_assistant_row_records_the_serving_model(self, fast, model, session):
        model(says("Sure."), model="google/gemma-4-31b-it:free", provider="Together")
        await agent.on_inbound(1, "wamid.1", "hi")
        assistant = [m for m in messages_of(session) if m.role == "assistant"][-1]
        assert assistant.model_used == "google/gemma-4-31b-it:free"
        assert assistant.provider == "Together"

    async def test_both_latencies_are_recorded(self, fast, model, session):
        model(says("Sure."))
        await agent.on_inbound(1, "wamid.1", "hi")
        assistant = [m for m in messages_of(session) if m.role == "assistant"][-1]
        assert assistant.latency_ms is not None          # drain -> reply (D20)
        assert assistant.inbound_latency_ms is not None  # last inbound -> reply

    async def test_the_drain_to_reply_clock_stays_under_eight_seconds(self, fast, model, session):
        model(says("Sure."))
        await agent.on_inbound(1, "wamid.1", "hi")
        assistant = [m for m in messages_of(session) if m.role == "assistant"][-1]
        assert assistant.latency_ms < 8_000

    async def test_the_system_prompt_is_first_and_the_buyer_message_last(self, fast, model):
        fake = model(says("Sure."))
        await agent.on_inbound(1, "wamid.1", "3BHK chahiye")
        sent = fake.seen[0]
        assert sent[0]["role"] == "system"
        assert "AllSet" in sent[0]["content"]
        assert sent[-1] == {"role": "user", "content": "3BHK chahiye"}

    async def test_the_transcript_replays_at_most_twenty_text_turns(self, fast, model, session):
        for index in range(30):
            session.add(Message(conversation_id=1, role="assistant", content=f"a{index}"))
        session.commit()
        fake = model(says("Sure."))
        await agent.on_inbound(1, "wamid.1", "hi")
        assert len(fake.seen[0]) == 21, "system + 20 turns"


class TestToolTurn:
    async def test_the_tool_result_is_fed_back_before_the_reply(self, fast, model):
        fake = model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Bopal me teen options hain."),
        )
        result = await agent.on_inbound(1, "wamid.1", "3BHK in Bopal")
        assert result["reply"] == "Bopal me teen options hain."

        second_call = fake.seen[1]
        assert second_call[-1]["role"] == "tool"
        assert second_call[-1]["tool_call_id"] == "call_0"
        assert "Shrishti Elara" in second_call[-1]["content"]

    async def test_tool_calls_rows_are_written(self, fast, model, session):
        model(
            calls(("search_properties", {"bhk": 3})),
            says("Here you go."),
        )
        await agent.on_inbound(1, "wamid.1", "3BHK")
        rows = tool_rows(session)
        assert [r.tool_name for r in rows] == ["search_properties"]
        assert rows[0].latency_ms is not None

    async def test_tool_calls_hang_off_the_buyer_message(self, fast, model, session):
        model(calls(("search_properties", {"bhk": 3})), says("Here you go."))
        await agent.on_inbound(1, "wamid.1", "3BHK")
        user_message = [m for m in messages_of(session) if m.role == "user"][-1]
        assert tool_rows(session)[0].message_id == user_message.id

    async def test_the_turn_result_carries_the_trace(self, fast, model):
        model(calls(("search_properties", {"bhk": 3})), says("Here you go."))
        result = await agent.on_inbound(1, "wamid.1", "3BHK")
        assert [c["tool_name"] for c in result["tool_calls"]] == ["search_properties"]
        assert result["tool_calls"][0]["result"]["exact_matches"]

    async def test_the_transcript_stays_text_only(self, fast, model, session):
        """D11 — assistant-with-tool_calls and role:tool live and die in the turn."""
        model(
            calls(("search_properties", {"bhk": 3}),
                  ("update_buyer_profile", {"buyer_id": 1, "updates": {"bhk_need": 3}})),
            says("Done."),
        )
        await agent.on_inbound(1, "wamid.1", "3BHK")
        assert {m.role for m in messages_of(session)} == {"user", "assistant"}
        for message in messages_of(session):
            assert "tool_call" not in (message.content or "")

    async def test_a_steering_error_is_handed_back_rather_than_crashing(self, fast, model):
        model(
            calls(("update_buyer_profile",
                   {"buyer_id": 1, "updates": {"preferred_localities": ["Bopal West"]}})),
            says("Kaunsi area?"),
        )
        result = await agent.on_inbound(1, "wamid.1", "Bopal West")
        assert result["reply"] == "Kaunsi area?"
        assert "error" in result["tool_calls"][0]["result"]


class TestParallelBatch:
    """Gemini can emit [search_properties, escalate_to_broker] together."""

    @pytest.fixture
    def batch(self, model):
        return model(
            calls(("search_properties", {"bhk": 3}),
                  ("escalate_to_broker",
                   {"conversation_id": 1, "reason": "buyer wants a human", "urgency": "high"})),
            says("never reached"),
        )

    async def test_every_call_in_the_batch_is_dispatched(self, batch, fast, session):
        await agent.on_inbound(1, "wamid.1", "can you sort this out for me")
        assert [r.tool_name for r in tool_rows(session)] == [
            "search_properties", "escalate_to_broker",
        ]

    async def test_escalation_is_acted_on_after_the_whole_batch(self, batch, fast, session):
        result = await agent.on_inbound(1, "wamid.1", "can you sort this out for me")
        assert result["status"] == "escalated"
        assert session.get(Conversation, 1).status == "escalated"

    async def test_the_loop_does_not_go_round_again(self, batch, fast):
        await agent.on_inbound(1, "wamid.1", "can you sort this out for me")
        # The brief-writer runs after the handoff; the loop itself stops dead.
        loop_calls = [sent for sent in batch.seen if "AllSet" in sent[0]["content"]]
        assert len(loop_calls) == 1

    async def test_the_buyer_gets_one_handoff_line(self, batch, fast, session):
        result = await agent.on_inbound(1, "wamid.1", "can you sort this out for me")
        assert result["reply"]
        assistant = [m for m in messages_of(session) if m.role == "assistant"][-1]
        assert assistant.content == result["reply"]

    async def test_the_next_message_gets_silence(self, batch, fast, model, session):
        await agent.on_inbound(1, "wamid.1", "can you sort this out for me")
        again = await agent.on_inbound(1, "wamid.2", "kab tak?")
        assert again["reply"] is None
        assert again["status"] == "escalated"


# --------------------------------------------------------------------------
# Failure policy (D20)
# --------------------------------------------------------------------------

class TestStepCap:
    async def test_eight_steps_escalate(self, fast, model, session):
        model(calls(("search_properties", {"bhk": 3})))   # never finishes
        result = await agent.on_inbound(1, "wamid.1", "3BHK")
        assert result["status"] == "escalated"
        escalation = session.exec(select(Escalation)).all()[0]
        assert escalation.reason == "step_cap_breached"
        assert escalation.urgency == "medium"

    async def test_the_cap_is_eight_llm_calls(self, fast, model):
        fake = model(calls(("search_properties", {"bhk": 3})))
        await agent.on_inbound(1, "wamid.1", "3BHK")
        # The brief-writer runs on the escalation that follows, so count only
        # the calls carrying the agent's own system prompt.
        loop_calls = [sent for sent in fake.seen if "AllSet" in sent[0]["content"]]
        assert len(loop_calls) == 8

    async def test_the_buyer_still_gets_the_handoff_line(self, fast, model):
        model(calls(("search_properties", {"bhk": 3})))
        result = await agent.on_inbound(1, "wamid.1", "3BHK")
        assert result["reply"]


class TestInfrastructureFailure:
    """A transient hiccup must not permanently silence the agent."""

    async def test_a_timeout_returns_a_hold_message(self, fast, model):
        model(llm.LLMUnavailable("both models failed"))
        result = await agent.on_inbound(1, "wamid.1", "3BHK")
        assert result["reply"] == agent.HOLD_MESSAGE

    async def test_a_timeout_does_not_escalate(self, fast, model, session):
        model(llm.LLMUnavailable("both models failed"))
        result = await agent.on_inbound(1, "wamid.1", "3BHK")
        assert result["status"] == "active"
        assert session.get(Conversation, 1).status == "active"
        assert session.exec(select(Escalation)).all() == []

    async def test_an_openrouter_error_is_handled_the_same_way(self, fast, model):
        import httpx
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        model(openai.BadRequestError(
            "bad", response=httpx.Response(400, request=request), body=None
        ))
        result = await agent.on_inbound(1, "wamid.1", "3BHK")
        assert result["reply"] == agent.HOLD_MESSAGE
        assert result["status"] == "active"

    async def test_the_next_inbound_is_answered_normally(self, fast, model, monkeypatch):
        model(llm.LLMUnavailable("boom"))
        await agent.on_inbound(1, "wamid.1", "3BHK")
        model(says("Back now."))
        result = await agent.on_inbound(1, "wamid.2", "hello?")
        assert result["reply"] == "Back now."


# --------------------------------------------------------------------------
# Profile injection (D12)
# --------------------------------------------------------------------------

class TestProfileInjection:
    async def test_a_captured_budget_shows_up_in_the_next_turn(self, fast, model):
        model(
            calls(("update_buyer_profile",
                   {"buyer_id": 1, "updates": {"budget_min": 5_500_000,
                                               "budget_max": 6_500_000}})),
            says("Kaunsi area?"),
        )
        await agent.on_inbound(1, "wamid.1", "budget 55 se 65 lakh")

        fake = model(says("Bopal ya Shela?"))
        await agent.on_inbound(1, "wamid.2", "aur batao")
        system = fake.seen[0][0]["content"]
        assert "Known — budget ₹55–65 lakh" in system

    async def test_the_unknown_list_drives_the_next_question(self, fast, model):
        fake = model(says("Budget kitna hai?"))
        await agent.on_inbound(1, "wamid.1", "3BHK chahiye")
        system = fake.seen[0][0]["content"]
        assert "Unknown — budget" in system
        assert "Ask about the first unknown." in system

    async def test_the_block_is_rebuilt_every_turn(self, fast, model):
        first = model(says("one"))
        await agent.on_inbound(1, "wamid.1", "hi")
        from app.tools import update_buyer_profile
        update_buyer_profile(1, {"bhk_need": 3})
        second = model(says("two"))
        await agent.on_inbound(1, "wamid.2", "hi again")
        assert first.seen[0][0]["content"] != second.seen[0][0]["content"]


# --------------------------------------------------------------------------
# The per-conversation lock (D9)
# --------------------------------------------------------------------------

class TestLocking:
    async def test_two_conversations_run_independently(self, fast, model, session):
        model(says("Reply."))
        first, second = await asyncio.gather(
            agent.on_inbound(1, "wamid.1", "hi"),
            agent.on_inbound(2, "wamid.2", "hello"),
        )
        assert first["reply"] == "Reply."
        assert second["reply"] == "Reply."

    async def test_one_turn_at_a_time_per_conversation(self, fast, monkeypatch):
        depth = {"now": 0, "max": 0}

        async def slow_llm(messages, tools=None):
            depth["now"] += 1
            depth["max"] = max(depth["max"], depth["now"])
            await asyncio.sleep(0.02)
            depth["now"] -= 1
            return says("ok"), "m", "p"

        monkeypatch.setattr(llm, "llm_call", slow_llm)
        await asyncio.gather(
            agent.on_inbound(1, "wamid.1", "a"),
            agent.on_inbound(1, "wamid.2", "b"),
            agent.on_inbound(1, "wamid.3", "c"),
        )
        assert depth["max"] == 1


class TestIdentityInjection:
    async def test_the_turn_tells_the_agent_which_buyer_it_is_talking_to(self, fast, model):
        # Conversation 3 belongs to buyer 3; a live model guessed buyer 1 here.
        fake = model(says("Budget kitna hai?"))
        await agent.on_inbound(3, "wamid.1", "3BHK in Satellite under 80 lakh")
        system = fake.seen[0][0]["content"]
        assert "buyer_id 3" in system
        assert "conversation_id 3" in system


class TestShownProperties:
    """The prompt's SHOWN SO FAR block is rebuilt from `tool_calls` every turn,
    for the same reason the profile is (D12): the transcript is text-only, so
    nothing a tool returned survives into the next turn."""

    async def test_a_search_this_turn_is_visible_to_the_next_one(
        self, model, fast, session
    ):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Here are a few."),
        )
        await agent.on_inbound(1, "sp-1", "3BHK in Bopal")

        shown = agent.shown_properties(1)
        assert shown
        assert all({"id", "title", "locality"} == set(row) for row in shown)
        assert all(isinstance(row["id"], int) for row in shown)

    async def test_nothing_searched_means_nothing_shown(self, model, fast):
        model(says("Hi there!"))
        await agent.on_inbound(1, "sp-2", "hello")
        assert agent.shown_properties(1) == []

    async def test_it_reaches_the_next_turn_s_system_prompt(self, model, fast):
        fake = model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Here are a few."),
        )
        await agent.on_inbound(1, "sp-3", "3BHK in Bopal")

        agent.reset_state()
        fake = model(says("Sure."))
        await agent.on_inbound(1, "sp-4", "tell me about the second one")

        system = fake.seen[0][0]["content"]
        assert "SHOWN SO FAR" in system
        assert "#" in system.split("SHOWN SO FAR")[1]

    async def test_one_conversation_cannot_see_another_s_properties(
        self, model, fast
    ):
        model(
            calls(("search_properties", {"bhk": 3, "localities": ["Bopal"]})),
            says("Here are a few."),
        )
        await agent.on_inbound(1, "sp-5", "3BHK in Bopal")
        assert agent.shown_properties(2) == []

    async def test_ids_are_deduplicated_and_capped(self, model, fast):
        for index in range(3):
            agent.reset_state()
            model(
                calls(("search_properties", {"bhk": 3})),
                says("Here are a few."),
            )
            await agent.on_inbound(1, f"sp-cap-{index}", "show me more")

        shown = agent.shown_properties(1)
        ids = [row["id"] for row in shown]
        assert len(ids) == len(set(ids))
        assert len(ids) <= agent.SHOWN_LIMIT
