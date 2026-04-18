from __future__ import annotations

import streamlit as st


def inject_trading_desk_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #050816;
            --bg2: #0b1020;
            --panel: rgba(10,18,36,0.92);
            --panel2: rgba(7,14,28,0.96);
            --line: rgba(0,245,255,0.14);
            --cyan: #00f5ff;
            --green: #19f5c1;
            --red: #ff5f7e;
            --yellow: #ffcc66;
            --purple: #8b5cf6;
            --pink: #ff2bd6;
            --text: #eaf6ff;
            --muted: #8aa0b8;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(0,245,255,0.10), transparent 25%),
                radial-gradient(circle at top right, rgba(139,92,246,0.10), transparent 22%),
                radial-gradient(circle at bottom center, rgba(255,43,214,0.07), transparent 24%),
                linear-gradient(180deg, #03050c 0%, #050816 50%, #081224 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(4,10,18,0.98), rgba(6,12,24,0.99));
            border-right: 1px solid rgba(0,245,255,0.10);
        }

        .block-container {
            max-width: 1550px;
            padding-top: 0.85rem;
            padding-bottom: 1.5rem;
        }

        .desk-hero {
            border-radius: 22px;
            padding: 1.15rem 1.35rem;
            background:
                linear-gradient(135deg, rgba(7,16,30,0.96), rgba(10,22,42,0.88));
            border: 1px solid rgba(0,245,255,0.14);
            box-shadow:
                0 0 18px rgba(0,245,255,0.05),
                0 0 42px rgba(139,92,246,0.04);
            margin-bottom: 0.9rem;
        }

        .desk-title {
            font-size: 2rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0.2rem;
        }

        .desk-sub {
            color: var(--muted);
            font-size: 0.96rem;
        }

        .desk-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.75rem;
            margin: 0.9rem 0 1rem 0;
        }

        .desk-kpi {
            background: linear-gradient(180deg, rgba(10,18,34,0.96), rgba(7,14,26,0.96));
            border: 1px solid rgba(0,245,255,0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 0 14px rgba(0,245,255,0.04);
        }

        .desk-kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .desk-kpi-value {
            color: #ffffff;
            font-size: 1.45rem;
            font-weight: 800;
        }

        .good { color: var(--green) !important; }
        .bad { color: var(--red) !important; }
        .warn { color: var(--yellow) !important; }

        .desk-panel {
            background: linear-gradient(180deg, rgba(8,16,30,0.95), rgba(7,14,24,0.95));
            border: 1px solid rgba(0,245,255,0.10);
            border-radius: 18px;
            padding: 0.95rem 1rem 0.8rem 1rem;
            box-shadow: 0 0 14px rgba(0,245,255,0.04);
            margin-bottom: 0.9rem;
        }

        .desk-section-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
            color: white;
        }

        .desk-tag {
            display: inline-block;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(0,245,255,0.16);
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            color: var(--text);
            background: rgba(0,245,255,0.04);
            font-size: 0.8rem;
        }

        .desk-log {
            border-left: 3px solid rgba(0,245,255,0.28);
            padding-left: 0.6rem;
            margin-bottom: 0.65rem;
            color: var(--muted);
            font-size: 0.9rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(0,245,255,0.18);
            background: linear-gradient(135deg, rgba(12,22,40,0.98), rgba(9,16,30,0.98));
            color: #f2fbff;
            font-weight: 700;
            min-height: 2.8rem;
            box-shadow: 0 0 12px rgba(0,245,255,0.05);
        }

        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(0,245,255,0.08);
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(10,18,34,0.96), rgba(7,14,26,0.96));
            border: 1px solid rgba(0,245,255,0.10);
            border-radius: 16px;
            padding: 0.5rem 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="desk-hero">
            <div class="desk-title">{title}</div>
            <div class="desk-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_grid(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, css_class in items:
        cards.append(
            f"""
            <div class="desk-kpi">
                <div class="desk-kpi-label">{label}</div>
                <div class="desk-kpi-value {css_class}">{value}</div>
            </div>
            """
        )
    st.markdown(f'<div class="desk-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def panel_open(title: str) -> None:
    st.markdown(
        f"""
        <div class="desk-panel">
            <div class="desk-section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_tags(tags: list[str]) -> None:
    html = "".join([f'<span class="desk-tag">{tag}</span>' for tag in tags])
    st.markdown(html, unsafe_allow_html=True)
