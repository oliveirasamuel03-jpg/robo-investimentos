"""Conservative operator STOP protocol stored in the existing bot_state document.

A received request is not a persisted stop. A persisted stop already prevents
new PAPER commits; only a worker observation sets ACKNOWLEDGED. No automatic
retry or process cancellation is implied by either result.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

from core import persistence as p

_TRANSITION_ALLOWED = ContextVar("atlas_explicit_stop_transition", default=False)
_ACTIVE_STOP_STATUSES = {"PERSISTED", "ACKNOWLEDGED"}


@dataclass(frozen=True)
class StopReceipt:
    request_id: str
    status: str
    persisted: bool
    acknowledged: bool
    error_code: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stop_control_status(state: dict) -> dict:
    control = state.get("stop_control", {}) or {}
    return {
        **deepcopy(control),
        "status": str(control.get("status") or "LEGACY"),
        "stopped": state.get("bot_status") == "STOPPED",
        "persisted": bool(control.get("request_id") and control.get("status") in _ACTIVE_STOP_STATUSES),
        "acknowledged": bool(control.get("status") == "ACKNOWLEDGED" and control.get("acknowledged_at")),
        "cycle_activity": deepcopy(state.get("cycle_activity", {}) or {}),
    }


def _is_ack_transition(previous: dict, candidate: dict) -> bool:
    before = previous.get("stop_control", {}) or {}
    after = candidate.get("stop_control", {}) or {}
    if (previous.get("bot_status") != "STOPPED" or candidate.get("bot_status") != "STOPPED"
            or before.get("status") != "PERSISTED" or after.get("status") != "ACKNOWLEDGED"
            or not before.get("request_id") or not after.get("acknowledged_at")
            or not after.get("worker_id")):
        return False
    changing = {"status", "acknowledged_at", "worker_id", "events"}
    if {k: v for k, v in before.items() if k not in changing} != {
            k: v for k, v in after.items() if k not in changing}:
        return False
    before_events = before.get("events", [])
    after_events = after.get("events", [])
    return (isinstance(before_events, list) and isinstance(after_events, list)
            and len(after_events) == len(before_events) + 1
            and after_events[:-1] == before_events
            and after_events[-1].get("event") == "ACKNOWLEDGED"
            and after_events[-1].get("request_id") == before["request_id"])


def guard_stop_transition(previous: dict, candidate: dict) -> None:
    """Generic saves cannot clear a stop or rewrite its audit identity."""
    if _TRANSITION_ALLOWED.get():
        return
    if previous.get("bot_status") == "STOPPED" and candidate.get("bot_status") != "STOPPED":
        raise p.PersistenceConflictError("STOPPED requires an explicit reviewed start; reload and review.")
    if previous.get("stop_control") != candidate.get("stop_control") and previous.get("stop_control"):
        if not _is_ack_transition(previous, candidate):
            raise p.PersistenceConflictError("Stop control changed; reload and review before saving.")


@contextmanager
def _explicit_transition():
    token = _TRANSITION_ALLOWED.set(True)
    try:
        yield
    finally:
        _TRANSITION_ALLOWED.reset(token)


@contextmanager
def bot_state_transaction():
    """Guard and state save share the PostgreSQL transaction or local path lock."""
    from core.config import BOT_STATE_FILE
    with p.persistence_transaction("bot_state"):
        with (nullcontext() if p.database_enabled() else p._local_lock(BOT_STATE_FILE)):
            yield


def _receipt(state: dict, expected_id: str) -> StopReceipt:
    status = stop_control_status(state)
    if not status["stopped"] or status.get("request_id") != expected_id or not status["persisted"]:
        return StopReceipt(expected_id, "UNCONFIRMED", False, False, "VerificationFailed")
    return StopReceipt(expected_id, status["status"], True, status["acknowledged"])


def _require_standalone_request() -> None:
    if p._ACTIVE_CONNECTION.get() is not None:
        raise RuntimeError("Operator STOP receipts require an outer commit; do not request inside a transaction.")


def request_stop(requested_by: str = "operator") -> StopReceipt:
    """Persist STOP once and verify it after commit. Failure never means stopped."""
    from core.state_store import load_bot_state, update_bot_state
    request_id = str(uuid.uuid4())
    requested_at = _now()
    actor = str(requested_by or "operator")
    try:
        _require_standalone_request()
        def stop(state):
            nonlocal request_id
            current = state.get("stop_control", {}) or {}
            if (state.get("bot_status") == "STOPPED" and current.get("request_id")
                    and current.get("status") in _ACTIVE_STOP_STATUSES):
                request_id = str(current["request_id"])
                return state
            event = {"event": "PERSISTED", "request_id": request_id,
                     "requested_at": requested_at, "at": _now(), "actor": actor}
            state["bot_status"] = "STOPPED"
            state["stop_control"] = {
                "request_id": request_id, "requested_at": requested_at,
                "persisted_at": event["at"], "acknowledged_at": "",
                "requested_by": actor, "worker_id": "", "status": "PERSISTED",
                "events": [*deepcopy(current.get("events", [])), event],
            }
            return state
        with _explicit_transition():
            update_bot_state(stop)
        return _receipt(load_bot_state(), request_id)
    except Exception as exc:
        # Include only the exception class, never credentials or connection text.
        return StopReceipt(request_id, "UNCONFIRMED", False, False, type(exc).__name__)


def mutate_acknowledgement(state: dict, worker_id: str = "worker") -> dict:
    """Pure candidate mutation; the caller must commit under its PAPER guard lock.

    Calling this function is not evidence of acknowledgement until its enclosing
    transaction commits. Legacy STOPPED without a request is preserved unchanged.
    """
    result = deepcopy(state)
    current = result.get("stop_control", {}) or {}
    if result.get("bot_status") != "STOPPED" or not current.get("request_id"):
        return result
    if current.get("status") == "ACKNOWLEDGED":
        return result
    if current.get("status") != "PERSISTED":
        return result
    at = _now()
    current["status"] = "ACKNOWLEDGED"
    current["acknowledged_at"] = at
    current["worker_id"] = str(worker_id or "worker")
    current["events"] = [*current.get("events", []), {
        "event": "ACKNOWLEDGED", "request_id": current["request_id"],
        "at": at, "actor": current["worker_id"],
    }]
    result["stop_control"] = current
    return result


def acknowledge_stop(worker_id: str = "worker") -> StopReceipt:
    from core.state_store import load_bot_state, update_bot_state
    request_id = ""
    try:
        _require_standalone_request()
        with bot_state_transaction():
            current = load_bot_state()
            request_id = str((current.get("stop_control", {}) or {}).get("request_id") or "")
            if current.get("bot_status") != "STOPPED" or not request_id:
                return StopReceipt(request_id, "UNCONFIRMED", False, False, "NoPersistedRequest")
            update_bot_state(lambda state: mutate_acknowledgement(state, worker_id))
        return _receipt(load_bot_state(), request_id)
    except Exception as exc:
        return StopReceipt(request_id, "UNCONFIRMED", False, False, type(exc).__name__)


def _control_revision(state: dict):
    control = state.get("stop_control", {}) or {}
    return (state.get("bot_status"), control.get("request_id"), control.get("status"),
            control.get("resumed_at"), control.get("resume_id"))


def start_robot(expected_state: dict, requested_by: str = "operator", bot_mode: str | None = None) -> dict:
    """Explicit START using the control revision displayed to the operator.

    Heartbeats and chart updates do not alter that revision. A STOP/ACK/START
    after the displayed snapshot requires reload; no automatic retry is made.
    Pending unacknowledged requests cannot be resumed.
    """
    from core.state_store import load_bot_state, update_bot_state
    from core.commit_protocol import assert_execution_allowed
    assert_execution_allowed()
    if bot_mode is not None and bot_mode not in {"Automatico", "Semi-automatico"}:
        raise ValueError("Invalid bot mode")
    if not isinstance(expected_state, p.StateSnapshot) or expected_state.expected is None:
        raise p.PersistenceConflictError("A persisted state snapshot is required; reload and review.")
    original = p._decode_json(expected_state.expected, dict, "operator control snapshot")
    with bot_state_transaction():
        current = load_bot_state()
        if expected_state.source != current.source or _control_revision(original) != _control_revision(current):
            raise p.PersistenceConflictError("Control state changed; reload and review before starting.")
        if bot_mode is not None and original.get("bot_mode") != current.get("bot_mode"):
            raise p.PersistenceConflictError("Bot mode changed; reload and review before saving the selected mode.")
        control = current.get("stop_control", {}) or {}
        if current.get("bot_status") == "STOPPED" and control.get("status") == "PERSISTED":
            raise p.PersistenceConflictError("Stop acknowledgement is pending; wait, reload and review.")
        def start(state):
            state["bot_status"] = "RUNNING"
            if bot_mode is not None:
                state["bot_mode"] = bot_mode
            control = deepcopy(state.get("stop_control", {}) or {})
            at, resume_id = _now(), str(uuid.uuid4())
            control.update(status="RESUMED", resumed_at=at, resume_id=resume_id,
                           resumed_by=str(requested_by or "operator"))
            control["events"] = [*control.get("events", []), {
                "event": "RESUMED", "request_id": control.get("request_id", ""),
                "resume_id": resume_id, "at": at, "actor": str(requested_by or "operator"),
            }]
            state["stop_control"] = control
            return state
        with _explicit_transition():
            return update_bot_state(start)
