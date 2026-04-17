from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bot_engine import (
    APP_DIR,
    StrategyConfig,
    ensure_runtime_files,
    load_bot_state,
    load_runtime_config,
    read_logs,
    reset_bot_state,
    run_backtest_for_ticker,
    save_runtime_config,
    scan_market,
)


st.set_page_config(
    page_title="Invest Pro Bot",
    page_icon="IP",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #07111f;
                --bg-soft: #0f1b2d;
                --card: rgba(15, 27, 45, 0.88);
                --line: rgba(148, 163, 184, 0.22);
                --text: #ebf2ff;
                --muted: #97a6ba;
                --green: #3ddc97;
                --red: #ff6b6b;
                --gold: #ffca6d;
                --blue: #5ab2ff;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(90, 178, 255, 0.18), transparent 30%),
                    radial-gradient(circle at top right, rgba(61, 220, 151, 0.12), transparent 24%),
                    linear-gradient(180deg, #040b14 0%, #07111f 45%, #08131d 100%);
                color: var(--text);
            }

            .block-container {
                padding-top: 1.1rem;
                padding-bottom: 2rem;
            }

            .hero, .panel {
                border: 1px solid var(--line);
                background: var(--card);
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.22);
            }

            .hero {
                padding: 1.5rem 1.7rem;
                margin-bottom: 1rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 2.2rem;
                letter-spacing: -0.04em;
            }

            .hero p {
                color: var(--muted);
                margin: 0.45rem 0 0 0;
            }

            .panel {
                padding: 1rem 1rem 0.5rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float) -> str:
    return f"{value:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Invest Pro Bot</h1>
            <p>Painel pessoal para acompanhar, configurar e operar seu robo em modo automatico com trilha de auditoria.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_form(runtime: dict) -> tuple[list[str], StrategyConfig, date]:
    st.sidebar.title("Controle do robo")
    tickers_text = st.sidebar.text_area("Ativos monitorados", value=", ".join(runtime["tickers"]))
    tickers = [item.strip().upper() for item in tickers_text.split(",") if item.strip()]

    start_date = st.sidebar.date_input("Data inicial do backtest", value=date.fromisoformat(runtime["backtest_start"]))
    capital_total = st.sidebar.number_input(
        "Capital inicial da conta",
        min_value=100.0,
        value=float(runtime["capital_total"]),
        step=100.0,
    )
    capital_operacao = st.sidebar.number_input(
        "Capital maximo por operacao",
        min_value=50.0,
        value=float(runtime["capital_por_operacao"]),
        step=50.0,
    )
    intervalo = st.sidebar.slider(
        "Intervalo de verificacao (min)",
        min_value=1,
        max_value=60,
        value=int(runtime["interval_minutes"]),
    )
    media_curta = st.sidebar.slider("Media curta", 5, 30, int(runtime["strategy"]["media_curta"]))
    media_longa = st.sidebar.slider("Media longa", 10, 60, int(runtime["strategy"]["media_longa"]))
    media_tendencia = st.sidebar.slider("Media tendencia", 20, 200, int(runtime["strategy"]["media_tendencia"]))
    periodo_rsi = st.sidebar.slider("Periodo RSI", 7, 30, int(runtime["strategy"]["periodo_rsi"]))
    rsi_compra = st.sidebar.slider("RSI max compra", 40, 80, int(runtime["strategy"]["rsi_compra_max"]))
    rsi_venda = st.sidebar.slider("RSI min venda", 50, 90, int(runtime["strategy"]["rsi_venda_min"]))
    stop_loss = st.sidebar.slider("Stop loss (%)", 1.0, 15.0, float(runtime["strategy"]["stop_loss_pct"]), 0.5)
    taxa = st.sidebar.slider("Taxa operacao (%)", 0.0, 2.0, float(runtime["strategy"]["taxa_operacao_pct"]), 0.05)

    strategy = StrategyConfig(
        capital_inicial=float(capital_total),
        capital_por_operacao=float(capital_operacao),
        media_curta=media_curta,
        media_longa=media_longa,
        media_tendencia=media_tendencia,
        periodo_rsi=periodo_rsi,
        rsi_compra_max=float(rsi_compra),
        rsi_venda_min=float(rsi_venda),
        stop_loss_pct=float(stop_loss),
        taxa_operacao_pct=float(taxa),
    )

    runtime["tickers"] = tickers
    runtime["backtest_start"] = start_date.isoformat()
    runtime["capital_total"] = float(capital_total)
    runtime["capital_por_operacao"] = float(capital_operacao)
    runtime["interval_minutes"] = int(intervalo)
    runtime["strategy"] = strategy.to_dict()
    return tickers, strategy, start_date


def render_controls(runtime: dict) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Centro de operacao")
    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.button("Salvar configuracao", use_container_width=True):
        save_runtime_config(runtime)
        st.success("Configuracao salva.")
    if col2.button("Ativar robo", use_container_width=True):
        runtime["enabled"] = True
        save_runtime_config(runtime)
        st.success("Robo marcado como ativo.")
    if col3.button("Pausar robo", use_container_width=True):
        runtime["enabled"] = False
        save_runtime_config(runtime)
        st.warning("Robo pausado.")
    if col4.button("Executar agora", type="primary", use_container_width=True):
        report = scan_market()
        if report["errors"]:
            st.error("Execucao concluida com alertas. Veja os logs abaixo.")
        else:
            st.success("Ciclo do robo executado com sucesso.")
        st.json(report)
    if col5.button("Resetar paper", use_container_width=True):
        reset_bot_state(float(runtime["capital_total"]))
        st.success("Conta paper reiniciada com o capital configurado.")

    modo = runtime.get("mode", "paper")
    status = "ATIVO" if runtime.get("enabled") else "PAUSADO"
    st.caption(f"Modo atual: `{modo}` | Status: `{status}` | Runner: [bot_runner.py]({Path(APP_DIR, 'bot_runner.py')})")
    if modo == "real":
        st.error("Modo real exige integracao da corretora. Atualmente o projeto executa ordens apenas em simulacao.")
    else:
        st.info("Modo paper ativo. O robo atualiza saldo, posicoes e historico sem enviar ordens reais.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_status_cards(runtime: dict, state: dict) -> None:
    portfolio_value = float(state["cash"])
    for position in state["positions"].values():
        portfolio_value += float(position["quantity"]) * float(position["last_price"])
    pnl = portfolio_value - float(runtime["capital_total"])

    cols = st.columns(4)
    cols[0].metric("Caixa", format_currency(float(state["cash"])))
    cols[1].metric("Patrimonio", format_currency(portfolio_value), format_currency(pnl))
    cols[2].metric("Posicoes abertas", str(len(state["positions"])))
    cols[3].metric("Ciclos executados", str(int(state.get("run_count", 0))))


def build_price_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Preco",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["media_curta"], name="Media curta", line=dict(color="#5ab2ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["media_longa"], name="Media longa", line=dict(color="#ffca6d")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["media_tendencia"], name="Tendencia", line=dict(color="#3ddc97")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#ff6b6b")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#8d99ae", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#8d99ae", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=720, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=55, b=20))
    fig.update_layout(title=f"{ticker} | Leitura tecnica")
    return fig


def render_backtest(runtime: dict, tickers: list[str], strategy: StrategyConfig, start_date: date) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Painel de validacao da estrategia")

    if not tickers:
        st.info("Adicione pelo menos um ativo para gerar o backtest.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if st.button("Atualizar backtest", use_container_width=True):
        st.session_state["refresh_backtest"] = True

    if st.session_state.get("refresh_backtest", True):
        reports = []
        data_map = {}
        for ticker in tickers:
            try:
                report = run_backtest_for_ticker(ticker, strategy, start_date)
            except Exception:
                continue
            reports.append(report["summary"])
            data_map[ticker] = report["data"]
        st.session_state["backtest_reports"] = reports
        st.session_state["backtest_data_map"] = data_map
        st.session_state["refresh_backtest"] = False

    reports = st.session_state.get("backtest_reports", [])
    data_map = st.session_state.get("backtest_data_map", {})

    if not reports:
        st.warning("Nao foi possivel montar o backtest com os filtros atuais.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ranking_df = pd.DataFrame(reports).sort_values("Retorno %", ascending=False).reset_index(drop=True)
    ranking_view = ranking_df.copy()
    ranking_view["Lucro"] = ranking_view["Lucro"].map(format_currency)
    ranking_view["Valor final"] = ranking_view["Valor final"].map(format_currency)
    ranking_view["Retorno %"] = ranking_view["Retorno %"].map(format_pct)
    ranking_view["Drawdown %"] = ranking_view["Drawdown %"].map(format_pct)
    st.dataframe(ranking_view, use_container_width=True, hide_index=True)

    ticker = st.selectbox("Ativo para detalhe tecnico", ranking_df["Ativo"].tolist())
    df = data_map[ticker]
    st.plotly_chart(build_price_figure(df, ticker), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_positions_and_logs(state: dict) -> None:
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Posicoes abertas")
        if state["positions"]:
            positions_df = pd.DataFrame(
                [
                    {
                        "Ativo": ticker,
                        "Quantidade": pos["quantity"],
                        "Preco medio": format_currency(pos["avg_price"]),
                        "Ultimo preco": format_currency(pos["last_price"]),
                        "Stop": format_currency(pos["stop_price"]),
                        "RSI": round(pos["last_rsi"], 2),
                        "Atualizado em": pos["updated_at"],
                    }
                    for ticker, pos in state["positions"].items()
                ]
            )
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhuma posicao aberta no momento.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Ultimos eventos")
        logs = read_logs(limit=12)
        if logs:
            log_df = pd.DataFrame(logs)
            st.dataframe(log_df[["timestamp", "level", "message"]], use_container_width=True, hide_index=True)
        else:
            st.info("Sem logs ainda.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_files() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Arquivos do robo")
    files = [
        ("Configuracao", Path(APP_DIR, "runtime_config.json")),
        ("Estado", Path(APP_DIR, "bot_state.json")),
        ("Logs", Path(APP_DIR, "bot_logs.jsonl")),
        ("Runner", Path(APP_DIR, "bot_runner.py")),
    ]
    for label, path in files:
        st.write(f"{label}: [{path.name}]({path})")
    st.caption("O `bot_runner.py` e o processo que voce deixa rodando para executar o robo continuamente.")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    ensure_runtime_files()
    inject_css()
    render_header()

    runtime = load_runtime_config()
    tickers, strategy, start_date = sidebar_form(runtime)
    state = load_bot_state()

    render_controls(runtime)
    render_status_cards(runtime, state)
    st.divider()
    render_backtest(runtime, tickers, strategy, start_date)
    st.divider()
    render_positions_and_logs(state)
    st.divider()
    render_files()


if __name__ == "__main__":
    main()
