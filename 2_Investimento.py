from __future__ import annotations

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parent / "pages" / "2_Investimento.py"

runpy.run_path(str(TARGET), run_name="__main__")
