from __future__ import annotations

import streamlit as st
from core.config import APP_TITLE
from core.state_store import ensure_storage, load_bot_state
from portfolio.wallet import compute_wallet_snapshot

st.set_page_config(page_title=APP_TITLE, layout="wide")
ensure_storage()
state = load_bot_state()
snapshot = compute_wallet_snapshot(state)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg:#050816;
            --bg2:#0a1022;
            --card:rgba(10,18,36,.92);
            --line:rgba(0,245,255,.15);
            --cyan:#00f5ff;
            --purple:#8b5cf6;
            --pink:#ff2bd6;
            --green:#19f5c1;
            --red:#ff5f7e;
            --text:#eaf6ff;
            --muted:#8aa0b8;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(0,245,255,.12), transparent 28%),
                radial-gradient(circle at top right, rgba(139,92,246,.12), transparent 26%),
                radial-gradient(circle at bottom center, rgba(255,43,214,.08), transparent 24%),
                linear-gradient(180deg, #02040b 0%, #050816 50%, #081224 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(4,10,18,.98), rgba(6,12,24,.99));
            border-right: 1px solid rgba(0,245,255,.12);
        }
        .block-container { max-width: 1500px; padding-top: .9rem; padding-bottom: 2rem; }
        .hero {
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(8,16,30,.96), rgba(10,22,42,.88));
            border: 1px solid rgba(0,245,255,.14);
            box-shadow: 0 0 25px rgba(0,245,255,.08), 0 0 50px rgba(139,92,246,.05);
        }
        .hero-title { font-size: 2rem; font-weight: 800; color: #fff; }
        .hero-sub { color: var(--muted); }
        .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap:.75rem; margin: 1rem 0; }
        .card {
            border-radius: 18px; padding: .95rem 1rem;
            background: linear-gradient(180deg, rgba(10,18,34,.96), rgba(7,14,26,.96));
            border: 1px solid rgba(0,245,255,.12);
            box-shadow: 0 0 16px rgba(0,245,255,.05);
        }
        .kpi-label { color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; }
        .kpi-value { color:#fff; font-size:1.55rem; font-weight:800; }
        .good { color: var(--green); }
        .bad { color: var(--red); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, cls in items:
        cards.append(
            f"<div class='card'><div class='kpi-label'>{label}</div><div class='kpi-value {cls}'>{value}</div></div>"
        )
    st.markdown(f"<div class='grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


inject_css()

st.markdown(
    f"""
    <div class='hero'>
        <div class='hero-title'>{APP_TITLE}</div>
        <div class='hero-sub'>Plataforma trader + investimento · multi-módulo · pausa e retomada do bot</div>
    </div>
    st.write(f"• Investimento habilitado: {state['investment']['enabled']}")
