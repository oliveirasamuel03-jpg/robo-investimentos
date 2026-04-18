from __future__ import annotations

import json
from pathlib import Path

from paper_trading_engine import PaperTradingConfig, run_paper_cycle


CONFIG_PATH = Path(__file__).resolve().parent / "paper_runner_config.json"


def load_runner_config() -> PaperTradingConfig:
    if not CONFIG_PATH.exists():
        default_cfg = PaperTradingConfig()
        CONFIG_PATH.write_text(
            json.dumps(default_cfg.__dict__, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return default_cfg

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return PaperTradingConfig(**payload)


def main() -> None:
    config = load_runner_config()
    result = run_paper_cycle(config)

    print("\n=== PAPER TRADING CYCLE ===")
    print(f"Signal date:      {result['signal_date']}")
    print(f"Equity before:    {result['equity_before']:.2f}")
    print(f"Equity after:     {result['equity_after']:.2f}")
    print(f"Cash:             {result['cash']:.2f}")
    print(f"Trades executed:  {result['trades_executed']}")
    print(f"Run count:        {result['run_count']}")
    print("Positions:")
    for pos in result["positions"]:
        print(
            f"  - {pos['asset']}: qty={pos['quantity']:.6f} "
            f"px={pos['last_price']:.2f} mv={pos['market_value']:.2f}"
        )


if __name__ == "__main__":
    main()