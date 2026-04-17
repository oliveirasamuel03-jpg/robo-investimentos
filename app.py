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
                --bg: #06111a;
                --bg-soft: #0f1d29;
                --card: rgba(11, 24, 36, 0.82);
                --line: rgba(133, 255, 210, 0.18);
                --text: #f4fbff;
                --muted: #90a8b7;
                --green: #7dffcf;
                --red: #ff7a90;
                --gold: #ffd36e;
                --blue: #61d9ff;
                --cyan: #5afff2;
            }

            .stApp {
                background:
                    radial-gradient(circle at 10% 10%, rgba(97, 217, 255, 0.22), transparent 20%),
                    radial-gradient(circle at 88% 12%, rgba(125, 255, 207, 0.18), transparent 18%),
                    radial-gradient(circle at 50% 90%, rgba(255, 211, 110, 0.10), transparent 24%),
                    linear-gradient(180deg, #02070d 0%, #06111a 42%, #041019 100%);
                color: var(--text);
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 2.5rem;
                max-width: 1380px;
            }

            .hero, .panel {
                border: 1px solid var(--line);
                background: var(--card);
                border-radius: 24px;
                box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
                backdrop-filter: blur(18px);
            }

            .hero {
                padding: 1.8rem 1.9rem;
                margin-bottom: 1.2rem;
                overflow: hidden;
                position: relative;
                background:
                    linear-gradient(135deg, rgba(11, 24, 36, 0.92), rgba(5, 18, 30, 0.82)),
                    linear-gradient(135deg, rgba(97, 217, 255, 0.08), rgba(125, 255, 207, 0.05));
            }

            .hero h1 {
                margin: 0;
                font-size: 2.9rem;
                letter-spacing: -0.04em;
                line-height: 0.95;
            }

            .hero p {
                color: var(--muted);
                margin: 0.55rem 0 0 0;
                font-size: 1.05rem;
                max-width: 760px;
            }

            .panel {
                padding: 1.05rem 1.05rem 0.6rem 1.05rem;
                background:
                    linear-gradient(180deg, rgba(12, 25, 37, 0.88), rgba(7, 18, 28, 0.9));
            }

            .hero::before {
                content: "";
                position: absolute;
                width: 300px;
                height: 300px;
                right: -90px;
                top: -120px;
                background: radial-gradient(circle, rgba(97, 217, 255, 0.24), transparent 62%);
                pointer-events: none;
            }

            .hero::after {
                content: "";
                position: absolute;
                width: 260px;
                height: 260px;
                right: 80px;
                bottom: -160px;
                background: radial-gradient(circle, rgba(125, 255, 207, 0.18), transparent 60%);
                pointer-events: none;
            }

            .hero-badges {
                display: flex;
                gap: 0.65rem;
                flex-wrap: wrap;
                margin-bottom: 1rem;
            }

            .hero-badge {
                padding: 0.38rem 0.8rem;
                border-radius: 999px;
                border: 1px solid rgba(97, 217, 255, 0.22);
                background: rgba(255, 255, 255, 0.04);
                color: var(--text);
                font-size: 0.82rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            .kpi-card {
                padding: 1rem 1rem 1.05rem 1rem;
                border-radius: 22px;
                border: 1px solid rgba(97, 217, 255, 0.16);
                background: linear-gradient(180deg, rgba(10, 24, 35, 0.94), rgba(7, 17, 28, 0.92));
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 14px 40px rgba(0,0,0,0.18);
                min-height: 132px;
            }

            .kpi-label {
                color: var(--muted);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .kpi-value {
                color: var(--text);
                font-size: 2rem;
                font-weight: 700;
                margin-top: 0.55rem;
                letter-spacing: -0.04em;
            }

            .kpi-delta {
                margin-top: 0.4rem;
                color: var(--cyan);
                font-size: 0.95rem;
            }

            .steps-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.9rem;
                margin-top: 0.35rem;
                margin-bottom: 0.6rem;
            }

            .step-card {
                border-radius: 20px;
                padding: 1rem;
                border: 1px solid rgba(125, 255, 207, 0.15);
                background: linear-gradient(180deg, rgba(13, 28, 42, 0.86), rgba(8, 18, 29, 0.9));
                min-height: 138px;
            }

            .step-number {
                color: var(--cyan);
                font-size: 0.8rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .step-title {
                color: var(--text);
                font-size: 1.08rem;
                font-weight: 700;
                margin-top: 0.55rem;
                margin-bottom: 0.4rem;
            }

            .step-copy {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.45;
            }

            .highlight-box {
                border-radius: 18px;
                padding: 0.95rem 1rem;
                border: 1px solid rgba(255, 211, 110, 0.16);
                background: linear-gradient(90deg, rgba(255, 211, 110, 0.08), rgba(97, 217, 255, 0.06));
                color: var(--text);
                margin-bottom: 0.8rem;
            }

            .stButton > button {
                border-radius: 14px;
                border: 1px solid rgba(97, 217, 255, 0.16);
                background: linear-gradient(135deg, rgba(13, 31, 44, 0.98), rgba(8, 20, 30, 0.96));
                color: var(--text);
                font-weight: 600;
                min-height: 3rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.18);
                transition: all 0.18s ease;
            }

            .stButton > button:hover {
                border-color: rgba(97, 217, 255, 0.36);
                transform: translateY(-1px);
                box-shadow: 0 14px 32px rgba(0,0,0,0.22);
            }

            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, rgba(91, 255, 242, 0.22), rgba(97, 217, 255, 0.18));
                border: 1px solid rgba(91, 255, 242, 0.34);
                box-shadow: 0 0 0 1px rgba(91, 255, 242, 0.08), 0 0 28px rgba(91, 255, 242, 0.16);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(5, 14, 22, 0.96), rgba(7, 18, 28, 0.98));
                border-right: 1px solid rgba(97, 217, 255, 0.10);
            }

            [data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(97, 217, 255, 0.12);
            }

            h3 {
                letter-spacing: -0.03em;
            }

            @media (max-width: 900px) {
                .hero h1 {
                    font-size: 2.2rem;
                }
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
            <div class="hero-badges">
                <span class="hero-badge">Painel Pessoal</span>
                <span class="hero-badge">Visual Futurista</span>
                <span class="hero-badge">Robo em Simulacao</span>
            </div>
            <h1>Invest Pro Bot</h1>
            <p>Um painel simples para testar a estrategia, acompanhar o saldo e ligar ou pausar o robo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_form(runtime: dict) -> tuple[list[str], StrategyConfig, date]:
    st.sidebar.title("Configuracao rapida")
    st.sidebar.caption("Preencha apenas o basico. O restante ja vem ajustado para voce.")

    tickers_text = st.sidebar.text_area(
        "Acoes para acompanhar",
        value=", ".join(runtime["tickers"]),
        help="Digite os codigos separados por virgula. Ex.: VALE3.SA, PETR4.SA, ITUB4.SA",
    )
    tickers = [item.strip().upper() for item in tickers_text.split(",") if item.strip()]

    start_date = st.sidebar.date_input("Analisar dados desde", value=date.fromisoformat(runtime["backtest_start"]))
    capital_total = st.sidebar.number_input(
        "Quanto dinheiro voce quer simular",
        min_value=100.0,
        value=float(runtime["capital_total"]),
        step=100.0,
    )
    capital_operacao = st.sidebar.number_input(
        "Valor maximo por compra",
        min_value=50.0,
        value=float(runtime["capital_por_operacao"]),
        step=50.0,
    )
    intervalo = st.sidebar.slider(
        "A cada quantos minutos verificar",
        min_value=1,
        max_value=60,
        value=int(runtime["interval_minutes"]),
    )

    perfil = st.sidebar.selectbox(
        "Perfil da estrategia",
        ["Equilibrado", "Conservador", "Agressivo"],
        index=0,
        help="Escolha um perfil pronto em vez de mexer em varios numeros tecnicos.",
    )

    if perfil == "Conservador":
        strategy = StrategyConfig(
            capital_inicial=float(capital_total),
            capital_por_operacao=float(capital_operacao),
            media_curta=9,
            media_longa=26,
            media_tendencia=72,
            periodo_rsi=14,
            rsi_compra_max=62,
            rsi_venda_min=70,
            stop_loss_pct=2.0,
            taxa_operacao_pct=0.1,
        )
    elif perfil == "Agressivo":
        strategy = StrategyConfig(
            capital_inicial=float(capital_total),
            capital_por_operacao=float(capital_operacao),
            media_curta=7,
            media_longa=18,
            media_tendencia=40,
            periodo_rsi=10,
            rsi_compra_max=72,
            rsi_venda_min=78,
            stop_loss_pct=4.5,
            taxa_operacao_pct=0.1,
        )
    else:
        strategy = StrategyConfig(
            capital_inicial=float(capital_total),
            capital_por_operacao=float(capital_operacao),
            media_curta=9,
            media_longa=21,
            media_tendencia=50,
            periodo_rsi=14,
            rsi_compra_max=68,
            rsi_venda_min=75,
            stop_loss_pct=3.0,
            taxa_operacao_pct=0.1,
        )

    with st.sidebar.expander("Ver detalhes da estrategia"):
        st.write(f"Perfil escolhido: `{perfil}`")
        st.write(f"Media curta: `{strategy.media_curta}`")
        st.write(f"Media longa: `{strategy.media_longa}`")
        st.write(f"Stop loss: `{strategy.stop_loss_pct}%`")

    runtime["tickers"] = tickers
    runtime["backtest_start"] = start_date.isoformat()
    runtime["capital_total"] = float(capital_total)
    runtime["capital_por_operacao"] = float(capital_operacao)
    runtime["interval_minutes"] = int(intervalo)
    runtime["profile"] = perfil
    runtime["strategy"] = strategy.to_dict()
    return tickers, strategy, start_date


def render_controls(runtime: dict) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("O que voce quer fazer agora?")
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
    st.caption(f"Modo: `{modo}` | Status do robo: `{status}` | Arquivo do runner: [bot_runner.py]({Path(APP_DIR, 'bot_runner.py')})")
    if modo == "real":
        st.error("Modo real exige integracao da corretora. Atualmente o projeto executa ordens apenas em simulacao.")
    else:
        st.info("Voce esta em simulacao. O robo treina e registra tudo sem mexer em dinheiro real.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_status_cards(runtime: dict, state: dict) -> None:
    portfolio_value = float(state["cash"])
    for position in state["positions"].values():
        portfolio_value += float(position["quantity"]) * float(position["last_price"])
    pnl = portfolio_value - float(runtime["capital_total"])

    cols = st.columns(4)
    cards = [
        ("Dinheiro parado", format_currency(float(state["cash"])), "Saldo livre para novas entradas"),
        ("Valor total", format_currency(portfolio_value), format_currency(pnl)),
        ("Compras abertas", str(len(state["positions"])), "Posicoes em acompanhamento"),
        ("Verificacoes feitas", str(int(state.get("run_count", 0))), "Ciclos executados pelo robo"),
    ]
    for col, (label, value, delta) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta">{delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_steps() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Como usar")
    st.markdown(
        """
        <div class="steps-grid">
            <div class="step-card">
                <div class="step-number">Passo 1</div>
                <div class="step-title">Escolha as acoes</div>
                <div class="step-copy">Digite os codigos na barra lateral e informe o valor que deseja simular.</div>
            </div>
            <div class="step-card">
                <div class="step-number">Passo 2</div>
                <div class="step-title">Selecione o perfil</div>
                <div class="step-copy">Use um perfil pronto para nao precisar mexer em configuracoes tecnicas.</div>
            </div>
            <div class="step-card">
                <div class="step-number">Passo 3</div>
                <div class="step-title">Teste a estrategia</div>
                <div class="step-copy">Clique em atualizar backtest e veja se o resultado visual faz sentido para voce.</div>
            </div>
            <div class="step-card">
                <div class="step-number">Passo 4</div>
                <div class="step-title">Ligue o robo</div>
                <div class="step-copy">Ative o robo e deixe o runner funcionando para que ele acompanhe tudo sozinho.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


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
    st.subheader("Resultado da estrategia")

    if not tickers:
        st.info("Adicione pelo menos uma acao para gerar a simulacao.")
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
        st.warning("Nao foi possivel montar a simulacao com os dados atuais.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ranking_df = pd.DataFrame(reports).sort_values("Retorno %", ascending=False).reset_index(drop=True)
    ranking_view = ranking_df.copy()
    ranking_view["Lucro"] = ranking_view["Lucro"].map(format_currency)
    ranking_view["Valor final"] = ranking_view["Valor final"].map(format_currency)
    ranking_view["Retorno %"] = ranking_view["Retorno %"].map(format_pct)
    ranking_view["Drawdown %"] = ranking_view["Drawdown %"].map(format_pct)

    melhor = ranking_df.iloc[0]
    st.markdown(
        f"""
        <div class="highlight-box">
            Melhor ativo da simulacao: <strong>{melhor['Ativo']}</strong> com retorno de <strong>{format_pct(float(melhor['Retorno %']))}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(ranking_view, use_container_width=True, hide_index=True)

    ticker = st.selectbox("Escolha uma acao para ver o grafico", ranking_df["Ativo"].tolist())
    df = data_map[ticker]
    st.plotly_chart(build_price_figure(df, ticker), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_positions_and_logs(state: dict) -> None:
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Compras em andamento")
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
            st.info("Nenhuma compra aberta no momento.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Ultimas movimentacoes")
        logs = read_logs(limit=12)
        if logs:
            log_df = pd.DataFrame(logs)
            st.dataframe(log_df[["timestamp", "level", "message"]], use_container_width=True, hide_index=True)
        else:
            st.info("Sem logs ainda.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_files() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Arquivos importantes")
    files = [
        ("Configuracao", Path(APP_DIR, "runtime_config.json")),
        ("Estado", Path(APP_DIR, "bot_state.json")),
        ("Logs", Path(APP_DIR, "bot_logs.jsonl")),
        ("Runner", Path(APP_DIR, "bot_runner.py")),
    ]
    for label, path in files:
        st.write(f"{label}: [{path.name}]({path})")
    st.caption("O `bot_runner.py` e o arquivo que fica rodando para o robo continuar trabalhando sozinho.")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    ensure_runtime_files()
    inject_css()
    render_header()

    runtime = load_runtime_config()
    tickers, strategy, start_date = sidebar_form(runtime)
    state = load_bot_state()

    render_steps()
    st.divider()
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
