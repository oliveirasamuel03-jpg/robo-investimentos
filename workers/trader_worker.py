from __future__ import annotations

import time
from datetime import datetime, timedelta

from core.state_store import load_bot_state, save_bot_state, log_event, update_worker_heartbeat
from engines.trader_engine import run_trader_cycle

SLEEP_SECONDS = 60
PAUSED_SLEEP_SECONDS = 5


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def update_runtime_state(last_action: str, next_run_delta_seconds: int) -> None:
    state = load_bot_state()
    state["last_action"] = last_action
    state["last_run_at"] = utc_now_iso()
    state["next_run_at"] = (datetime.utcnow() + timedelta(seconds=next_run_delta_seconds)).isoformat()
    state["worker_status"] = "online"
    state["worker_heartbeat"] = utc_now_iso()
    save_bot_state(state)


def mark_error(message: str) -> None:
    state = load_bot_state()
    state["last_action"] = f"Erro: {message}"
    state["last_run_at"] = utc_now_iso()
    state["next_run_at"] = (datetime.utcnow() + timedelta(seconds=SLEEP_SECONDS)).isoformat()
    state["worker_status"] = "error"
    state["worker_heartbeat"] = utc_now_iso()
    save_bot_state(state)


def build_action_text(result: dict) -> str:
    cycle_result = result.get("cycle_result", {}) if isinstance(result, dict) else {}
    trades_executed = int(cycle_result.get("trades_executed", 0))

    if trades_executed > 0:
        return f"{trades_executed} trade(s) executado(s) nesta rodada"

    return "Rodada concluída sem novas operações"


def worker_loop() -> None:
    log_event("INFO", "Worker 24h iniciado")

    while True:
        try:
            update_worker_heartbeat("online")
            state = load_bot_state()

            if state.get("bot_status") != "RUNNING":
                update_runtime_state(
                    last_action="Robô pausado. Aguardando ativação.",
                    next_run_delta_seconds=PAUSED_SLEEP_SECONDS,
                )
                time.sleep(PAUSED_SLEEP_SECONDS)
                continue

            result = run_trader_cycle()
            action_text = build_action_text(result)

            update_runtime_state(
                last_action=action_text,
                next_run_delta_seconds=SLEEP_SECONDS,
            )
            log_event("INFO", action_text)

        except Exception as e:
            error_msg = str(e)
            log_event("ERROR", f"Erro no worker: {error_msg}")
            mark_error(error_msg)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    worker_loop()
