"""Tool dispatch and the `tool_calls` trace (Phase 2 §7).

`tool_calls` is not logging — it is the demo's trace panel and the grounding
audit trail, so every call writes exactly one row, successes and steering errors
alike. A call that vanished because the arguments were malformed is precisely
the one you want to see when a turn goes wrong.

Nothing in here raises. Malformed JSON, an unknown tool name and a
schema-invalid payload are all things a model does on an ordinary afternoon; the
loop must never crash on model output.
"""
from __future__ import annotations

import inspect
import json
import time

from app import db
from app.db import ToolCall
from app.tools import TOOL_MAPPING


def _error(message: str) -> dict:
    return {"error": message}


def log_tool_call(
    message_id: int, tool_name: str, args: dict, result: dict, latency_ms: int
) -> None:
    with db.get_session() as session:
        session.add(ToolCall(
            message_id=message_id,
            tool_name=tool_name,
            args=args,
            result=result,
            latency_ms=latency_ms,
        ))
        session.commit()


def _parse_arguments(raw) -> tuple[dict, dict | None]:
    """Returns (args, steering error)."""
    if raw is None or raw == "":
        # Gemini sends "" rather than "{}" for a call with no arguments.
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}, _error(
            "Arguments were not valid JSON. Re-send the call with a JSON object, "
            "e.g. {\"bhk\": 3}."
        )
    if not isinstance(parsed, dict):
        return {}, _error(
            f"Arguments must be a JSON object, got {type(parsed).__name__}. "
            f"Re-send the call with named arguments."
        )
    return parsed, None


def _check_signature(tool, name: str, args: dict) -> dict | None:
    """Steer on arguments the tool does not take, before Python turns it into a
    TypeError the model cannot read."""
    try:
        inspect.signature(tool).bind(**args)
    except TypeError as exc:
        return _error(
            f"{name} cannot be called with those arguments: {exc}. "
            f"Check the tool's parameters and re-send."
        )
    return None


def dispatch(call, message_id: int) -> dict:
    """Run one model-emitted tool call and trace it."""
    name = call.function.name
    t0 = time.perf_counter()

    args, error = _parse_arguments(call.function.arguments)
    if error is None:
        tool = TOOL_MAPPING.get(name)
        if tool is None:
            error = _error(
                f"No tool named {name!r}. Available: {', '.join(TOOL_MAPPING)}."
            )
        else:
            error = _check_signature(tool, name, args)

    if error is not None:
        result = error
    else:
        try:
            result = TOOL_MAPPING[name](**args)
        except Exception as exc:  # noqa: BLE001 - a crashed tool is still a turn
            result = _error(f"{name} failed: {exc}. Try different arguments.")

    log_tool_call(
        message_id, name, args, result,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return result
