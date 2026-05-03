from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


MODE = "SHADOW_ONLY"
ALIGNMENT_SOURCE = "video_pdf_inspired_checklist_v1"
MAX_RULE_ROWS = 12


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    return bool(value)


def _rule_status(passed: bool, *, partial: bool = False, insufficient: bool = False) -> str:
    if insufficient:
        return "insufficient"
    if passed:
        return "ok"
    if partial:
        return "partial"
    return "divergent"


def _rule_score(status: str) -> float:
    if status == "ok":
        return 1.0
    if status == "partial":
        return 0.5
    if status == "insufficient":
        return 0.2
    return 0.0


def _add_rule(
    rows: list[dict[str, Any]],
    *,
    item: str,
    expected: str,
    detected: str,
    status: str,
    motivo: str,
    weight: float,
) -> None:
    rows.append(
        {
            "item": item,
            "esperado_pelo_video_pdf": expected,
            "detectado_pelo_app": detected,
            "status": status,
            "motivo": motivo,
            "weight": weight,
        }
    )


def default_fib_alignment_audit_state(reason: str = "No Fibonacci video/PDF alignment audit data yet.") -> dict[str, Any]:
    return {
        "fib_alignment_enabled": True,
        "fib_alignment_mode": MODE,
        "fib_alignment_source": ALIGNMENT_SOURCE,
        "fib_alignment_score": None,
        "fib_alignment_status": "insufficient_data",
        "fib_alignment_top_symbol": "",
        "fib_alignment_anchor_low_status": "insufficient",
        "fib_alignment_anchor_high_status": "insufficient",
        "fib_alignment_zone_status": "insufficient",
        "fib_alignment_pivot_status": "insufficient",
        "fib_alignment_bos_status": "insufficient",
        "fib_alignment_entry_confirmation_status": "insufficient",
        "fib_alignment_confluence_status": "insufficient",
        "fib_alignment_missing_evidence": ["no_data"],
        "fib_alignment_why_differs": reason,
        "fib_alignment_recommendation": "insufficient_data",
        "fib_alignment_checklist": [],
        "fib_alignment_last_run_at": "",
    }


def _resolve_top_row(market_structure_audit: dict[str, Any]) -> dict[str, Any]:
    best_candidates = list(market_structure_audit.get("market_structure_best_candidates", []) or [])
    if best_candidates:
        return dict(best_candidates[0] or {})
    return {}


def _build_checklist(top_row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    swing_low = _as_float(top_row.get("relevant_swing_low"))
    swing_high = _as_float(top_row.get("relevant_swing_high"))
    fib_zone = str(top_row.get("current_fib_zone") or "INCONCLUSIVE")
    pivot = _as_bool(top_row.get("pivot_detected"))
    bos = _as_bool(top_row.get("bos_detected"))
    false_breakout = _as_bool(top_row.get("false_breakout_risk"))
    pullback_zone = _as_bool(top_row.get("is_in_pullback_zone"))
    reaction_zone = _as_bool(top_row.get("is_in_reaction_zone"))
    trend_pullback = _as_bool(top_row.get("structure_confirms_trend_pullback"))
    breakout_confirms = _as_bool(top_row.get("structure_confirms_breakout"))
    data_ok = bool(top_row.get("market_structure_minimum_sample_met", False))
    direction = str(top_row.get("structure_direction") or "INCONCLUSIVE")
    why_no_candidate = str(top_row.get("market_structure_why_no_candidate") or "")

    low_ok = swing_low is not None
    high_ok = swing_high is not None
    coherent_anchors = low_ok and high_ok and swing_high != swing_low
    zone_ok = fib_zone in {"MEDIUM_ZONE", "DEEP_ZONE", "SHALLOW_ZONE"}
    reaction_ok = pullback_zone or reaction_zone
    confluence_ok = trend_pullback or breakout_confirms
    entry_ok = pivot and bos and not false_breakout and confluence_ok

    _add_rule(
        rows,
        item="Ancora do fundo",
        expected="fundo relevante identificado",
        detected="swing_low valido" if low_ok else "sem swing_low valido",
        status=_rule_status(low_ok, insufficient=not data_ok),
        motivo="App encontrou um swing low objetivo." if low_ok else "Nao houve fundo relevante objetivo.",
        weight=0.10,
    )
    _add_rule(
        rows,
        item="Ancora do topo",
        expected="topo relevante identificado",
        detected="swing_high valido" if high_ok else "sem swing_high valido",
        status=_rule_status(high_ok, insufficient=not data_ok),
        motivo="App encontrou um swing high objetivo." if high_ok else "Nao houve topo relevante objetivo.",
        weight=0.10,
    )
    _add_rule(
        rows,
        item="Tracado Fibonacci",
        expected="Fibonacci entre fundo/topo coerentes",
        detected=f"direction={direction}" if coherent_anchors else "ancoras incoerentes",
        status=_rule_status(coherent_anchors, insufficient=not data_ok),
        motivo="As ancoras permitem tracado coerente." if coherent_anchors else "Sem ancoras objetivas coerentes para o tracado.",
        weight=0.10,
    )
    _add_rule(
        rows,
        item="Zona Fibonacci",
        expected="preco em zona relevante de reacao",
        detected=fib_zone,
        status=_rule_status(zone_ok, partial=fib_zone == "BREAKOUT_ZONE", insufficient=not data_ok),
        motivo="Zona atual e util para leitura de pullback/reacao." if zone_ok else "Preco nao esta em zona clara de reacao.",
        weight=0.12,
    )
    _add_rule(
        rows,
        item="Classificacao da retracao",
        expected="retracao rasa/media/profunda identificada",
        detected=fib_zone,
        status=_rule_status(fib_zone in {"SHALLOW_ZONE", "MEDIUM_ZONE", "DEEP_ZONE"}, insufficient=not data_ok),
        motivo="O app classificou a retracao em zona objetiva." if fib_zone in {"SHALLOW_ZONE", "MEDIUM_ZONE", "DEEP_ZONE"} else "A retracao ficou neutra/inconclusiva.",
        weight=0.08,
    )
    _add_rule(
        rows,
        item="Pivo confirmado",
        expected="pivot de reacao presente",
        detected="sim" if pivot else "nao",
        status=_rule_status(pivot, insufficient=not data_ok),
        motivo="Ha candle/pivo objetivo de reacao." if pivot else "Faltou pivo objetivo de confirmacao.",
        weight=0.10,
    )
    _add_rule(
        rows,
        item="BOS confirmado",
        expected="rompimento de estrutura confirmado",
        detected="sim" if bos else "nao",
        status=_rule_status(bos, insufficient=not data_ok),
        motivo="O app detectou BOS objetivo." if bos else "Nao houve BOS objetivo ainda.",
        weight=0.10,
    )
    _add_rule(
        rows,
        item="Falso rompimento controlado",
        expected="falso rompimento ausente ou controlado",
        detected="sim" if not false_breakout else "risco_detectado",
        status=_rule_status(not false_breakout, insufficient=not data_ok),
        motivo="Nao ha risco dominante de falso rompimento." if not false_breakout else "Existe risco objetivo de falso rompimento.",
        weight=0.08,
    )
    _add_rule(
        rows,
        item="Pullback com reacao",
        expected="pullback reage em zona tecnica",
        detected="sim" if reaction_ok else "nao",
        status=_rule_status(reaction_ok, partial=pullback_zone and not reaction_zone, insufficient=not data_ok),
        motivo="Preco reagiu em zona tecnica." if reaction_ok else "Ha zona, mas sem reacao suficiente." if pullback_zone else "Nem zona de pullback nem reacao objetiva apareceram.",
        weight=0.07,
    )
    _add_rule(
        rows,
        item="Confluencia com tendencia",
        expected="estrutura alinhada com a tendencia",
        detected="sim" if confluence_ok else "nao",
        status=_rule_status(confluence_ok, partial=trend_pullback or breakout_confirms, insufficient=not data_ok),
        motivo="Estrutura e tendencia/setup estao alinhados." if confluence_ok else "A estrutura nao confirma a tendencia do setup.",
        weight=0.05,
    )
    _add_rule(
        rows,
        item="Confluencia trend_pullback_breakout",
        expected="setup dominante confirmado pela estrutura",
        detected="sim" if trend_pullback else "nao",
        status=_rule_status(trend_pullback, partial=breakout_confirms, insufficient=not data_ok),
        motivo="A estrutura confirma o setup dominante." if trend_pullback else "A estrutura ainda nao confirma claramente o trend_pullback_breakout.",
        weight=0.05,
    )
    _add_rule(
        rows,
        item="Confirmacao de entrada",
        expected="entrada so com confirmacao objetiva",
        detected="confirmada" if entry_ok else "incompleta",
        status=_rule_status(entry_ok, partial=(pivot or bos) and not false_breakout, insufficient=not data_ok),
        motivo="Confirmacao completa segundo checklist inspirado no video/PDF." if entry_ok else "Ainda faltam confirmacoes objetivas para equivaler ao checklist.",
        weight=0.05,
    )

    for row in rows:
        if row["status"] != "ok":
            missing.append(str(row["item"]))
    if why_no_candidate:
        missing.append(why_no_candidate)
    return rows[:MAX_RULE_ROWS], missing


def build_fibonacci_alignment_audit(market_structure_audit: dict[str, Any] | None) -> dict[str, Any]:
    audit = dict(market_structure_audit or {})
    if not audit or not bool(audit.get("market_structure_audit_enabled", False)):
        state = default_fib_alignment_audit_state("Market structure audit unavailable.")
        state["fib_alignment_last_run_at"] = _utc_now_iso()
        return state

    top_row = _resolve_top_row(audit)
    if not top_row:
        state = default_fib_alignment_audit_state("No structural candidate available for Fibonacci alignment.")
        state["fib_alignment_last_run_at"] = _utc_now_iso()
        return state

    checklist, missing = _build_checklist(top_row)
    weighted_total = sum(float(row.get("weight", 0.0) or 0.0) for row in checklist)
    weighted_score = sum(float(row.get("weight", 0.0) or 0.0) * _rule_score(str(row.get("status") or "")) for row in checklist)
    score = round(weighted_score / weighted_total, 4) if weighted_total else 0.0

    if not bool(top_row.get("market_structure_minimum_sample_met", False)):
        status = "insufficient_data"
    elif score >= 0.80:
        status = "strong_alignment"
    elif score >= 0.55:
        status = "partial_alignment"
    elif score >= 0.30:
        status = "weak_alignment"
    else:
        status = "no_sufficient_alignment"

    rule_map = {str(row["item"]): str(row["status"]) for row in checklist}
    if status == "insufficient_data":
        recommendation = "insufficient_data"
    elif rule_map.get("Ancora do fundo") == "divergent" or rule_map.get("Ancora do topo") == "divergent":
        recommendation = "anchors_need_review"
    elif rule_map.get("Zona Fibonacci") in {"ok", "partial"} and (
        rule_map.get("Pivo confirmado") != "ok" or rule_map.get("BOS confirmado") != "ok"
    ):
        recommendation = "fib_zone_matches_but_confirmation_missing"
    elif rule_map.get("Pivo confirmado") != "ok" and rule_map.get("BOS confirmado") != "ok":
        recommendation = "pivot_bos_missing"
    elif status == "strong_alignment":
        recommendation = "video_pdf_alignment_strong"
    elif status == "partial_alignment":
        recommendation = "video_pdf_alignment_partial"
    elif status == "weak_alignment":
        recommendation = "video_pdf_alignment_weak"
    else:
        recommendation = "structure_not_equivalent_to_video"

    divergent_rules = [row for row in checklist if str(row.get("status")) in {"divergent", "insufficient"}]
    why_differs = "; ".join(str(row.get("motivo") or "") for row in divergent_rules[:3]) or "Checklist sem divergencia relevante."

    return {
        "fib_alignment_enabled": True,
        "fib_alignment_mode": MODE,
        "fib_alignment_source": ALIGNMENT_SOURCE,
        "fib_alignment_score": score,
        "fib_alignment_status": status,
        "fib_alignment_top_symbol": str(top_row.get("symbol") or audit.get("market_structure_top_symbol") or ""),
        "fib_alignment_anchor_low_status": rule_map.get("Ancora do fundo", "insufficient"),
        "fib_alignment_anchor_high_status": rule_map.get("Ancora do topo", "insufficient"),
        "fib_alignment_zone_status": rule_map.get("Zona Fibonacci", "insufficient"),
        "fib_alignment_pivot_status": rule_map.get("Pivo confirmado", "insufficient"),
        "fib_alignment_bos_status": rule_map.get("BOS confirmado", "insufficient"),
        "fib_alignment_entry_confirmation_status": rule_map.get("Confirmacao de entrada", "insufficient"),
        "fib_alignment_confluence_status": rule_map.get("Confluencia trend_pullback_breakout", "insufficient"),
        "fib_alignment_missing_evidence": missing[:8],
        "fib_alignment_why_differs": why_differs,
        "fib_alignment_recommendation": recommendation,
        "fib_alignment_checklist": checklist,
        "fib_alignment_last_run_at": _utc_now_iso(),
    }
