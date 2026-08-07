# AllSet — autonomous nurture agent (backend v0)

A WhatsApp-shaped first-touch agent for an Ahmedabad property brokerage. It
qualifies a buyer through conversation, surfaces properties that actually fit,
answers factual questions **only from verified data**, and hands the lead to a
human broker the moment it turns hot — carrying enough context that nothing is
re-explained.

What v0 proves: the loop runs, every fact traces back to a tool call, the handoff
is clean, and the agent goes silent once a human owns the lead. Nothing else.

```bash
uv sync
cp .env.example .env        # paste your OpenRouter BYOK key
./demo.sh                   # reseed, serve, run the whole demo, clean up
```

## The three endpoints

```
POST /conversations/{id}/messages   -> {reply | null, tool_calls[], status}
GET  /conversations/{id}            -> transcript + full trace
GET  /broker/inbox                  -> escalated leads + briefs + post-escalation messages
```

Three, and no more. A frontend that renders a chat window, a trace panel and a
broker inbox needs nothing else.

## Running it yourself

```bash
uv run python -m app.seed --reset
DEBOUNCE_MS=0 uv run uvicorn app.main:app --port 8000 --workers 1
```

`--workers 1` is enforced at startup (D9): the debounce buffers, the
per-conversation locks and the coalescing events are in-process state, and a
second worker gets its own copy of all three. Local time must be IST (D24); both
assertions fail loudly at startup rather than quietly at runtime.

## Tests

```bash
uv run pytest              # the offline suite — free, seconds
uv run pytest --live       # ...plus the tests that call a real model
```

Live tests are opt-in because the quota is the binding constraint: BYOK gives 500
requests a day, one turn costs two or three, and failover is silent — an
exhausted quota does not fail, it quietly finishes on a different rung and starts
formatting differently. Run the live files **serially**, never with `pytest -n`.

## Layout

| File | What it owns |
|---|---|
| `app/main.py` | the three endpoints, and the startup assertions |
| `app/agent.py` | inbound, debounce, the per-conversation lock, the turn |
| `app/llm.py` | OpenRouter, the three-rung failover, the circuit breaker |
| `app/tools.py` | the five tools, as plain Python functions |
| `app/guard.py` | the byte-exact grounding post-check and its one repair turn |
| `app/triggers.py` | the deterministic escalation tier (regex, no model) |
| `app/escalation.py` | the post-turn evaluator and the handoff line |
| `app/brief.py` | the broker brief — written half plus deterministic facts |
| `app/prompt.py` | the system prompt: static blocks plus per-turn state |
| `app/seed.py` | 25 committed properties, 3 buyers, fixed ids |

## Docs

- [`docs/manual-test-plan.md`](docs/manual-test-plan.md) — **start here to test or
  demo it by hand.** Setup, the pre-demo checklist, the demo script turn by turn,
  every §13 line as a manual check, and the known boundaries.
- `nurture-agent-v0.md` — the product spec.
- `docs/phase-1..5-*.md` — the build phases and the 25 numbered design decisions
  (D1–D25) that the code comments reference throughout.

## Known v0 boundaries

- **Proactive re-engagement is v1.** It is called a *nurture* agent; v0 is
  reactive only. v1 adds `scheduled_followups` and a cron touch for quiet leads.
- **Seeded inventory.** Every RERA id carries a visible `DEMO` segment; every
  developer and tower name is invented (D22).
- **Single process, single city.** In-memory buffers die on restart; naive IST
  datetimes assume Ahmedabad. Both are explicit v1 seams.
- **The guard checks grounding, not attribution.** It proves every figure in a
  reply came from a tool this turn. It cannot tell that the tool was called about
  the wrong property.
