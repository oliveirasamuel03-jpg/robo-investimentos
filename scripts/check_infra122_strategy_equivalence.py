"""Compare synthetic strategy outcomes against the validated INFRA-1.2.1 source.

Only reads this repository's Git history. Each version runs in its own temporary
storage/process, with no inherited credentials, PAPER required and network denied.
This is an additional audit artifact, not a pytest case or production simulator.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "fc5931254c85d1b36591818fcafa0e204e37c5b3"


def worker(checkout: Path, storage: Path, output: Path) -> None:
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    os.environ.update(ROBO_STORAGE_DIR=str(storage), APP_ENV="development",
                      AUTH_REQUIRED="true", BROKER_MODE="paper", BROKER_PROVIDER="paper")
    blocked = []
    def deny_external(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto",
                     "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
            blocked.append(event)
            raise RuntimeError("Synthetic comparison: network/process denied")
    sys.addaudithook(deny_external)
    from datetime import datetime, timezone
    import paper_trading_engine as paper
    from core.trader_reports import read_trade_reports
    from tests.conftest import fake_market_result
    from tests.test_trader_smoke import _fake_prices

    now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    paper._utc_now = lambda: now
    paper._new_trade_id = lambda asset: f"{asset}-synthetic-fixed-id"
    paper.load_market_data_result = lambda symbols, config, requested_by="worker_cycle": fake_market_result(
        {symbol: _fake_prices() for symbol in symbols}, requested_by)
    cfg = paper.PaperTradingConfig(custom_tickers=["AAPL"], allow_new_entries=True)
    original_evaluate = paper._evaluate_buy_signal
    original_sell = paper._should_sell

    def snapshot(label, result):
        state = result["state"]
        return {
            "label": label,
            "financial_state": {key: state[key] for key in (
                "initial_capital", "cash", "equity", "realized_pnl", "positions",
                "last_prices", "asset_cooldowns", "run_count", "history")},
            "signals": result["signals"],
            "trades_executed": result["trades_executed"],
            "trades": [{key: value for key, value in row.items() if key != "cycle_id"}
                       for row in paper.read_paper_trades()],  # receipt identity is nonfinancial metadata
            "report_count": len(read_trade_reports()),
        }

    observations = []
    observations.append(snapshot("actual_strategy_synthetic_feed", paper.run_paper_cycle(cfg)))
    # Reset only this version's temporary synthetic ledger before controlled BUY/SELL.
    paper.reset_paper_state(cfg)
    paper._evaluate_buy_signal = lambda *_args: {
        "buy": True, "score": 0.95, "rejection_reasons": [],
        "trend_ok": True, "score_ok": True, "rsi_ok": True, "atr_ok": True,
        "structure_ok": True, "momentum_ok": True, "trend_alignment": "aligned",
    }
    paper._should_sell = lambda *_args: (False, "")
    observations.append(snapshot("controlled_buy_allocation", paper.run_paper_cycle(cfg)))
    assert len(observations[-1]["trades"]) == 1
    observations.append(snapshot("same_position_repeated", paper.run_paper_cycle(cfg)))
    paper._should_sell = lambda *_args: (True, "synthetic-close")
    observations.append(snapshot("controlled_sell_accounting", paper.run_paper_cycle(cfg)))
    assert [row["side"] for row in observations[-1]["trades"]] == ["BUY", "SELL"]
    assert observations[-1]["report_count"] == 1
    observations.append(snapshot("closed_position_repeated", paper.run_paper_cycle(cfg)))
    paper._evaluate_buy_signal, paper._should_sell = original_evaluate, original_sell
    output.write_text(json.dumps({"observations": observations,
                                  "network_process_attempts_blocked": len(blocked)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--storage", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/retomada/evidencias/infra122-strategy-equivalence.json")
    args = parser.parse_args()
    if args.worker:
        worker(args.checkout, args.storage, args.output)
        return
    if (ROOT / ".env").exists():
        raise SystemExit("Move .env before isolated synthetic comparison.")
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    env.update(BROKER_MODE="paper", BROKER_PROVIDER="paper", APP_ENV="development")
    archive = subprocess.run(["git", "archive", BASE], cwd=ROOT, env=env,
                             capture_output=True, check=True, timeout=30).stdout
    with tempfile.TemporaryDirectory(prefix="atlas-infra121-equivalence-") as directory:
        temp = Path(directory)
        legacy = temp / "legacy"
        legacy.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(legacy, filter="data")
        results = {}
        warning_output = {}
        for label, checkout in (("legacy", legacy), ("current", ROOT)):
            output = temp / f"{label}.json"
            completed = subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--worker", "--checkout", str(checkout),
                "--storage", str(temp / f"storage-{label}"), "--output", str(output),
            ], cwd=checkout, env=env, capture_output=True, text=True, timeout=60)
            if completed.returncode:
                print(completed.stdout, file=sys.stderr)
                print(completed.stderr, file=sys.stderr)
                raise SystemExit(f"Synthetic {label} comparison failed")
            if completed.stderr:
                print(f"Synthetic {label} warnings/output:", file=sys.stderr)
                print(completed.stderr, file=sys.stderr)
            warning_output[label] = completed.stderr.splitlines()
            results[label] = json.loads(output.read_text())
        assert results["legacy"]["observations"] == results["current"]["observations"], "Strategy outcome changed"
        artifact = {
            "base_commit": BASE,
            "scope": "LOCAL_SYNTHETIC_ONLY",
            "broker_mode": "paper", "broker_provider": "paper",
            "comparison_equal": True,
            "comparisons": [row["label"] for row in results["current"]["observations"]],
            "compared_fields": ["signals", "trades_executed", "trades", "financial_state", "report_count"],
            "results": results,
            "warning_output": warning_output,
            "limitation": "Synthetic outcomes only; controlled BUY/SELL fixes signal outcome to isolate allocation/accounting. No proof across all market conditions.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"comparison_equal": True, "comparisons": artifact["comparisons"],
                          "artifact": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
