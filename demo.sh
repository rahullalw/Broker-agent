#!/usr/bin/env bash
#
# demo.sh — the scripted conversation, end to end (Phase 5 §2).
#
# One command. It reseeds, starts the server single-worker with DEBOUNCE_MS=0,
# runs the scripted conversation and the three demo beats, and finishes on the
# broker inbox.
#
#     ./demo.sh                 # the whole thing
#     PAUSE=2 ./demo.sh         # ...with a two-second beat between turns
#     ./demo.sh --raw           # print raw JSON instead of the digest
#
# Why it owns the server rather than assuming one:
#
#   * `seed --reset` deletes the database file, which cannot be done underneath a
#     running server — SQLite hands out an open handle and Windows will not
#     unlink it.
#   * `FORCE_GUARD_FAIL` is read out of the process environment at import, so the
#     guard beat genuinely needs its own server. It gets one, on its own port and
#     its own database, so the leads in the main inbox stay untouched.
#
# Set DEMO_MANAGE_SERVER=0 to run against a server you started yourself. The
# guard beat still starts its own — it has no other way to flip the flag.

set -euo pipefail

PORT="${PORT:-8000}"
GUARD_PORT="${GUARD_PORT:-8001}"
BASE="${BASE:-http://127.0.0.1:$PORT}"
GUARD_BASE="http://127.0.0.1:$GUARD_PORT"
PAUSE="${PAUSE:-0}"
MANAGE="${DEMO_MANAGE_SERVER:-1}"
GUARD_DB="demo-guard.db"
RAW=""
[ "${1:-}" = "--raw" ] && RAW="1"

cd "$(dirname "$0")"

# The venv interpreter directly, not `uv run`: demo.sh has to kill the server it
# started, and a wrapper process in between means killing the wrapper and leaving
# uvicorn holding the port.
if [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ];       then PY=".venv/bin/python"
else PY="python"; fi

LOG_DIR=".demo-logs"
mkdir -p "$LOG_DIR"
SERVER_PIDFILE="$LOG_DIR/server.pid"
GUARD_PIDFILE="$LOG_DIR/guard.pid"

# Git Bash hands out its own PID namespace, so `$!` is an MSYS pid that neither
# taskkill nor the Windows process table has ever heard of — and `.venv/Scripts/
# python.exe` is a shim that re-execs the real interpreter, so even the right pid
# would only kill the wrapper. Letting the server write its own `os.getpid()`
# sidesteps both: the pid in the file belongs to the process actually listening.
LAUNCHER='
import os, sys, uvicorn
open(sys.argv[1], "w").write(str(os.getpid()))
uvicorn.run("app.main:app", host="127.0.0.1", port=int(sys.argv[2]),
            workers=1, log_level="info")
'

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

RULE="---------------------------------------------------------------------------"

beat() {
  printf '\n%s\n  %s\n%s\n' "$RULE" "$1" "$RULE"
}

note() { printf '  %s\n' "$1"; }

FORMATTER='
import json, sys, textwrap

# stdin matters as much as stdout here: the API returns raw UTF-8 (Starlette
# dumps with ensure_ascii=False), and Windows would otherwise decode the rupee
# sign as cp1252 and re-encode the mojibake on the way out.
for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

mode = sys.argv[1]
raw = len(sys.argv) > 2 and sys.argv[2] == "raw"
body = sys.stdin.read()
try:
    data = json.loads(body)
except ValueError:
    print("  !! not JSON:", body[:400]); raise SystemExit(1)

if raw:
    print(json.dumps(data, indent=2, ensure_ascii=False)); raise SystemExit(0)

def wrap(text, indent="           "):
    if text is None: return indent + "(none)"
    lines = []
    for para in str(text).splitlines() or [""]:
        lines += textwrap.wrap(para, 88, initial_indent=indent,
                               subsequent_indent=indent) or [indent]
    return "\n".join(lines)

def digest(name, result):
    if not isinstance(result, dict): return str(result)[:120]
    if "error" in result: return "ERROR " + result["error"][:100]
    if name == "search_properties":
        head = "%d exact, %d near" % (len(result.get("exact_matches") or []),
                                      len(result.get("near_matches") or []))
        cards = (result.get("exact_matches") or []) + (result.get("near_matches") or [])
        rows = ["  #%s %s — %s · %s · %s" % (c["id"], c["title"], c["price_display"],
                                             c["possession_display"], c["fit_reason"])
                for c in cards]
        if result.get("relaxed"): rows.append("  relaxed: " + result["relaxed"])
        if result.get("note"):    rows.append("  note: " + result["note"])
        return head + ("\n" + "\n".join("           " + r for r in rows) if rows else "")
    if name == "get_property_details":
        parts = []
        for key in ("pricing", "possession", "legal", "amenities"):
            if key in result:
                parts.append("%s=%s" % (key, json.dumps(result[key], ensure_ascii=False)))
        return ("#%s %s " % (result.get("id"), result.get("title"))) + " ".join(parts)
    if name == "update_buyer_profile":
        return "profile=%s unknown=%s next=%s" % (
            json.dumps(result.get("profile"), ensure_ascii=False),
            result.get("unknown"), result.get("next_question_hint"))
    if name == "book_site_visit":
        return "confirmed=%s %s @ %s" % (result.get("confirmed"),
                                         result.get("property"), result.get("slot_display"))
    if name == "escalate_to_broker":
        return "escalated reason=%s urgency=%s" % (result.get("reason"), result.get("urgency"))
    return json.dumps(result, ensure_ascii=False)[:200]

def show_calls(calls, indent="  "):
    for call in calls:
        print("%stool     : %s(%s)  %sms" % (
            indent, call["tool_name"],
            json.dumps(call["args"], ensure_ascii=False), call.get("latency_ms")))
        for line in digest(call["tool_name"], call["result"]).splitlines():
            print("%s           %s" % (indent, line.strip() if line.startswith("  ") else line))

if mode == "turn":
    print("  status   : %s" % data.get("status"))
    print("  reply    :"); print(wrap(data.get("reply")))
    show_calls(data.get("tool_calls") or [])

elif mode == "trace":
    print("  conversation %s — status %s" % (data["conversation_id"], data["status"]))
    buyer = data["buyer"]
    print("  buyer    : %s · %s · intent %s" % (buyer["name"], buyer["phone"], buyer["intent_tier"]))
    print("  profile  : %s" % json.dumps(buyer.get("profile"), ensure_ascii=False))
    print("  unknown  : %s" % buyer.get("unknown"))
    for m in data["messages"]:
        tag = "buyer" if m["role"] == "user" else "agent"
        meta = ""
        if m["role"] == "assistant":
            meta = "   [%sms drain->reply · %s · %s]" % (
                m.get("latency_ms"), m.get("model_used"), m.get("provider"))
        print("\n  %-6s #%s%s" % (tag, m["id"], meta))
        print(wrap(m["content"], "           "))
        show_calls(m.get("tool_calls") or [])
    for row in data.get("guard_rejections") or []:
        print("\n  GUARD REJECTION #%s  spans=%s" % (row["id"], row["offending_spans"]))
        print(wrap(row["rejected_text"], "           "))
    if data.get("escalation"):
        print("\n  ESCALATION reason=%s urgency=%s" % (
            data["escalation"]["reason"], data["escalation"]["urgency"]))
        print(wrap(data["escalation"]["brief"], "           "))

elif mode == "inbox":
    print("  %d escalated lead(s)" % data["count"])
    for lead in data["leads"]:
        print("\n  " + "=" * 71)
        print("  conversation %s · %s · %s · %s (urgency %s)" % (
            lead["conversation_id"], lead["buyer"]["name"], lead["buyer"]["phone"],
            lead["reason"], lead["urgency"]))
        print("  " + "=" * 71)
        print(wrap(lead["brief"], "    "))
        print("\n    unanswered since handoff: %d" % lead["unread_count"])
        for m in lead["unread_messages"]:
            print(wrap("#%s  %s" % (m["id"], m["content"]), "      "))
'

fmt() { "$PY" -c "$FORMATTER" "$1" ${RAW:+raw}; }

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

# A turn is bounded by DEBOUNCE_MAX_MS plus a full step-capped loop, so the
# client timeout has to sit well above both.
say() {  # say <base> <conversation> <wa_message_id> <text>
  curl -sS --max-time 120 -X POST "$1/conversations/$2/messages" \
    -H 'Content-Type: application/json' \
    --data "$("$PY" -c '
import json, sys
sys.stdout.write(json.dumps({"text": sys.argv[1], "wa_message_id": sys.argv[2]}))
' "$4" "$3")"
}

turn() {  # turn <conversation> <wa_message_id> <text>
  printf '\n  buyer    : %s\n' "$3"
  say "$BASE" "$1" "$2" "$3" | fmt turn
  sleep "$PAUSE"
}

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

wait_for() {  # wait_for <base> <name>
  for _ in $(seq 1 60); do
    if curl -sf --max-time 2 "$1/broker/inbox" > /dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "!! $2 did not come up — see $LOG_DIR/" >&2
  tail -20 "$LOG_DIR/$2.log" >&2 || true
  return 1
}

# A server left running keeps the port and, worse, keeps an open handle on the
# database file, so the next `seed --reset` cannot unlink it.
stop() {  # stop <pidfile>
  [ -f "${1:-}" ] || return 0
  pid="$(cat "$1")"
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) taskkill //F //T //PID "$pid" > /dev/null 2>&1 || true ;;
    *)                    kill "$pid" 2>/dev/null || true ;;
  esac
  rm -f "$1"
  sleep 1          # let the socket and the sqlite handle actually go
}

cleanup() {
  stop "$GUARD_PIDFILE"
  stop "$SERVER_PIDFILE"
  rm -f "$GUARD_DB" "$GUARD_DB-wal" "$GUARD_DB-shm"
}
trap cleanup EXIT

# ===========================================================================
beat "0 · Fresh world — 25 properties, 3 buyers, fixed ids (D21)"
# ===========================================================================

if [ "$MANAGE" = "1" ]; then
  "$PY" -m app.seed --reset
else
  note "DEMO_MANAGE_SERVER=0 — using the server already on $BASE, not reseeding."
fi

if [ "$MANAGE" = "1" ]; then
  # DEBOUNCE_MS=0 so the scripted run moves at full speed (D10). --workers 1 is
  # not a suggestion: the buffers, locks and coalescing events are in-process
  # state, and app startup refuses to come up any other way (D9).
  rm -f "$SERVER_PIDFILE"
  DEBOUNCE_MS=0 "$PY" -c "$LAUNCHER" "$SERVER_PIDFILE" "$PORT" \
    > "$LOG_DIR/server.log" 2>&1 &
  wait_for "$BASE" "server"
  note "server up on $BASE (pid $(cat "$SERVER_PIDFILE"), DEBOUNCE_MS=0, 1 worker)"
fi

# ===========================================================================
beat "1 · Priya — the scripted conversation (conversation 1)"
# ===========================================================================

# Five turns, not the six in Phase 5 §2, and the budget turn is missing on
# purpose. D16's evaluator hands a lead over as soon as it has budget AND
# locality AND (BHK OR possession). Priya's opening states two of those, so the
# moment she names a budget the conversation escalates on
# `qualification_complete` — correctly, per Phase 4 — and turns 3 to 6 can never
# run. Withholding the budget keeps her qualification incomplete, which is what
# lets the arc reach the booking and hand off on `site_visit_booked` instead:
# the strongest trigger, first in PRECEDENCE, and the one Phase 5 §2 wanted
# turn 6 to fire. Every capability in the §2 table still gets demonstrated.

note "1 · qualification starts        -> update_buyer_profile, asks the first unknown"
turn 1 "demo-p1-t1" "Hi, looking for a 3BHK in Bopal"

note "2 · search on partial qualification -> search_properties, cards with display strings"
turn 1 "demo-p1-t2" "I'll share my budget in a bit — can you show me what you have first?"

note "3 · a named property             -> get_property_details([possession])"
turn 1 "demo-p1-t3" "Tell me about Vrund Meadows — when is possession?"

note "4 · a factual legal question     -> get_property_details([legal]), RERA id verbatim"
turn 1 "demo-p1-t4" "Is it RERA registered?"

note "5 · the booking                  -> book_site_visit, then the evaluator hands off"
turn 1 "demo-p1-t5" "Can I visit this Saturday at 5pm?"

beat "1b · The trace panel — every fact back to the tool call that produced it"
curl -sS --max-time 30 "$BASE/conversations/1" | fmt trace

# ===========================================================================
beat "2 · Rakesh — deterministic escalation, no model in the decision (D18)"
# ===========================================================================

note "The regex tier fires on the coalesced buffer before the loop begins."
note "No search, no qualification — one templated handoff line, then silence."
turn 2 "demo-p2-t1" "price kam karo bhai"

beat "2b · ...and the agent stays silent (D19)"
note "This message is stored for the broker and never answered."
turn 2 "demo-p2-t2" "acha theek hai bhai, kab call karoge?"

beat "2c · A resent WhatsApp id is dropped silently"
note "Same wa_message_id as the message above — no second row, no second turn."
turn 2 "demo-p2-t2" "acha theek hai bhai, kab call karoge?"

# ===========================================================================
beat "3 · Anjali — zero exact matches, labelled near-misses (D14)"
# ===========================================================================

note "Satellite 3BHKs start at ₹1.2 crore in the seeded inventory."
note "search_properties never returns empty: it relaxes, and names what it relaxed."
turn 3 "demo-p3-t1" "3BHK in Satellite under 80 lakh"

# ===========================================================================
beat "4 · The grounding guard, triggered on cue (FORCE_GUARD_FAIL=1)"
# ===========================================================================

note "You cannot make a model hallucinate to order, so the harness appends a"
note "fake price and possession date after generation. Own server, own database:"
note "the leads in the main inbox are untouched."

rm -f "$GUARD_DB" "$GUARD_DB-wal" "$GUARD_DB-shm"
DATABASE_URL="sqlite:///$GUARD_DB" "$PY" -m app.seed --reset > /dev/null
rm -f "$GUARD_PIDFILE"
DATABASE_URL="sqlite:///$GUARD_DB" FORCE_GUARD_FAIL=1 DEBOUNCE_MS=0 \
  "$PY" -c "$LAUNCHER" "$GUARD_PIDFILE" "$GUARD_PORT" \
  > "$LOG_DIR/guard.log" 2>&1 &
wait_for "$GUARD_BASE" "guard"

printf '\n  buyer    : %s\n' "Which 3BHKs do you have in Bopal?"
say "$GUARD_BASE" 1 "demo-guard-t1" "Which 3BHKs do you have in Bopal?" | fmt turn

beat "4b · The rejected output, logged in full"
note "The buyer never saw it. ₹72 lakh and March 2027 appear in no tool result"
note "from this turn, so the guard refused the reply and asked for a repair."
curl -sS --max-time 30 "$GUARD_BASE/conversations/1" | fmt trace
stop "$GUARD_PIDFILE"

# ===========================================================================
beat "5 · The broker inbox — what a human inherits"
# ===========================================================================

curl -sS --max-time 30 "$BASE/broker/inbox" | fmt inbox

printf '\n%s\n  Done. Server log: %s\n%s\n' \
  "$RULE" "$LOG_DIR/server.log" "$RULE"
