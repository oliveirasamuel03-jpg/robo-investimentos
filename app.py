import os
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Invest Pro",
    page_icon="IP",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_TICKERS = [
    "VALE3.SA",
    "PETR4.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "BBAS3.SA",
    "WEGE3.SA",
    "MGLU3.SA",
    "RENT3.SA",
    "SUZB3.SA",
    "PRIO3.SA",
]


@dataclass
class StrategyConfig:
    capital_inicial: float
    media_curta: int
    media_longa: int
    media_tendencia: int
    periodo_rsi: int
    rsi_compra_max: float
    rsi_venda_min: float
    stop_loss_pct: float
    taxa_operacao_pct: float


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
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }

            .hero {
                padding: 1.6rem 1.8rem;
                border: 1px solid var(--line);
                background: linear-gradient(135deg, rgba(11, 22, 40, 0.96), rgba(7, 17, 31, 0.88));
                border-radius: 22px;
                margin-bottom: 1.1rem;
                box-shadow: 0 22px 60px rgba(0, 0, 0, 0.24);
            }

            .hero h1 {
                margin: 0;
                font-size: 2.25rem;
                letter-spacing: -0.04em;
                color: var(--text);
            }

            .hero p {
                margin: 0.45rem 0 0 0;
                color: var(--muted);
                font-size: 1rem;
            }

            .panel {
                border: 1px solid var(--line);
                background: var(--card);
                border-radius: 18px;
                padding: 1rem 1rem 0.3rem 1rem;
                box-shadow: 0 16px 45px rgba(0, 0, 0, 0.18);
            }

            .login-card {
                max-width: 440px;
                margin: 7vh auto 0 auto;
                padding: 2rem;
                border-radius: 24px;
                border: 1px solid var(--line);
                background: linear-gradient(180deg, rgba(9, 18, 32, 0.96), rgba(7, 17, 31, 0.9));
                box-shadow: 0 22px 70px rgba(0, 0, 0, 0.28);
            }

            .login-card h2 {
                color: var(--text);
                margin-bottom: 0.35rem;
            }

            .login-card p {
                color: var(--muted);
                margin-bottom: 1.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_credentials() -> tuple[str, str]:
    secrets = getattr(st, "secrets", {})
    username = secrets.get("INVEST_PRO_USERNAME", os.getenv("INVEST_PRO_USERNAME", "admin"))
    password = secrets.get("INVEST_PRO_PASSWORD", os.getenv("INVEST_PRO_PASSWORD", "1234"))
    return username, password


def ensure_session_state() -> None:
    if "logado" not in st.session_state:
        st.session_state.logado = False
    if "analise_pronta" not in st.session_state:
        st.session_state.analise_pronta = False
    if "resultados" not in st.session_state:
        st.session_state.resultados = []
    if "dados_map" not in st.session_state:
        st.session_state.dados_map = {}
    if "trades_map" not in st.session_state:
        st.session_state.trades_map = {}
    if "equity_map" not in st.session_state:
        st.session_state.equity_map = {}
    if "resumo_carteira" not in st.session_state:
        st.session_state.resumo_carteira = {}


def render_login() -> None:
    user_ok, pass_ok = get_credentials()

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown("<h2>Acesso ao Invest Pro</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p>Entre para liberar o dashboard profissional com ranking, backtest e comparativo de ativos.</p>",
        unsafe_allow_html=True,
    )

    usuario = st.text_input("Usuario")
    senha = st.text_input("Senha", type="password")

    if st.button("Entrar", use_container_width=True):
        if usuario == user_ok and senha == pass_ok:
            st.session_state.logado = True
            st.success("Login realizado com sucesso.")
            st.rerun()
        else:
            st.error("Usuario ou senha incorretos.")

    st.caption("Dica: altere as credenciais com variaveis de ambiente ou `st.secrets` antes de publicar.")
    st.markdown("</div>", unsafe_allow_html=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).title() for col in data.columns]
    return data


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_market_data(tickers: tuple[str, ...], start_date: date, end_date: date) -> dict[str, pd.DataFrame]:
    data_map: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
        )
        df = normalize_columns(df)
        if not df.empty:
            data_map[ticker] = df
    return data_map


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def enrich_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]
    data["media_curta"] = close.rolling(config.media_curta).mean()
    data["media_longa"] = close.rolling(config.media_longa).mean()
    data["media_tendencia"] = close.rolling(config.media_tendencia).mean()
    data["rsi"] = calc_rsi(close, config.periodo_rsi)
    data["sinal"] = 0
    data.loc[data["media_curta"] > data["media_longa"], "sinal"] = 1
    data.loc[data["media_curta"] < data["media_longa"], "sinal"] = -1
    return data.dropna().copy()


def backtest_strategy(
    ticker: str, df: pd.DataFrame, config: StrategyConfig
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = enrich_indicators(df, config)
    if data.empty:
        raise ValueError(f"{ticker} sem dados suficientes para analise.")

    capital = config.capital_inicial
    qtd_acoes = 0.0
    preco_compra = 0.0
    taxa = config.taxa_operacao_pct / 100
    stop_loss = 1 - (config.stop_loss_pct / 100)
    operations: list[dict] = []
    equity_points: list[dict] = []

    for current_date, row in data.iterrows():
        preco = float(row["Close"])
        sinal = int(row["sinal"])
        rsi = float(row["rsi"])
        media_tendencia = float(row["media_tendencia"])

        compra_liberada = (
            sinal == 1
            and capital > 0
            and rsi <= config.rsi_compra_max
            and preco >= media_tendencia
        )
        venda_por_tendencia = sinal == -1 or rsi >= config.rsi_venda_min
        venda_por_stop = qtd_acoes > 0 and preco <= preco_compra * stop_loss

        if compra_liberada:
            qtd_acoes = (capital * (1 - taxa)) / preco
            operations.append(
                {
                    "Data": current_date.date(),
                    "Tipo": "Compra",
                    "Preco": round(preco, 2),
                    "Quantidade": round(qtd_acoes, 4),
                    "RSI": round(rsi, 2),
                    "Motivo": "Cruzamento + filtro de tendencia",
                }
            )
            capital = 0.0
            preco_compra = preco

        elif qtd_acoes > 0 and (venda_por_tendencia or venda_por_stop):
            valor_saida = qtd_acoes * preco * (1 - taxa)
            operations.append(
                {
                    "Data": current_date.date(),
                    "Tipo": "Venda",
                    "Preco": round(preco, 2),
                    "Quantidade": round(qtd_acoes, 4),
                    "RSI": round(rsi, 2),
                    "Motivo": "Stop loss" if venda_por_stop else "Saida tecnica",
                }
            )
            capital = valor_saida
            qtd_acoes = 0.0
            preco_compra = 0.0

        patrimonio = capital + (qtd_acoes * preco)
        equity_points.append(
            {
                "Data": current_date,
                "Patrimonio": patrimonio,
                "Close": preco,
                "Sinal": sinal,
                "RSI": rsi,
            }
        )

    equity_df = pd.DataFrame(equity_points).set_index("Data")
    buy_hold = config.capital_inicial * (data["Close"] / float(data["Close"].iloc[0]))
    valor_final = float(equity_df["Patrimonio"].iloc[-1])
    retorno_pct = ((valor_final / config.capital_inicial) - 1) * 100
    retorno_buy_hold_pct = ((float(buy_hold.iloc[-1]) / config.capital_inicial) - 1) * 100

    trades_df = pd.DataFrame(operations)
    total_vendas = 0
    if not trades_df.empty:
        total_vendas = int((trades_df["Tipo"] == "Venda").sum())

    metricas = {
        "Ativo": ticker,
        "Valor final": valor_final,
        "Lucro": valor_final - config.capital_inicial,
        "Retorno %": retorno_pct,
        "Buy and hold %": retorno_buy_hold_pct,
        "Drawdown %": max_drawdown(equity_df["Patrimonio"]) * 100,
        "Operacoes": int(len(trades_df)),
        "Vendas": total_vendas,
        "Ultimo fechamento": float(data["Close"].iloc[-1]),
        "Ultimo RSI": float(data["rsi"].iloc[-1]),
    }

    equity_df["BuyHold"] = buy_hold.reindex(equity_df.index)
    return metricas, data, trades_df, equity_df


def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float) -> str:
    return f"{value:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def sidebar_controls() -> tuple[list[str], date, date, StrategyConfig]:
    st.sidebar.title("Centro de controle")
    st.sidebar.caption("Ajuste ativos, periodo e regras da estrategia antes de rodar o backtest.")

    ticker_text = st.sidebar.text_area(
        "Ativos",
        value=", ".join(DEFAULT_TICKERS),
        help="Use codigos separados por virgula. Ex.: VALE3.SA, PETR4.SA, ITUB4.SA",
    )
    tickers = [item.strip().upper() for item in ticker_text.split(",") if item.strip()]

    col_a, col_b = st.sidebar.columns(2)
    start_date = col_a.date_input("Inicio", value=date(2020, 1, 1))
    end_date = col_b.date_input("Fim", value=date.today())

    st.sidebar.markdown("### Estrategia")
    capital = st.sidebar.number_input("Capital inicial", min_value=100.0, value=1000.0, step=100.0)
    media_curta = st.sidebar.slider("Media curta", min_value=5, max_value=30, value=9)
    media_longa = st.sidebar.slider("Media longa", min_value=10, max_value=60, value=21)
    media_tendencia = st.sidebar.slider("Media de tendencia", min_value=30, max_value=200, value=50)
    periodo_rsi = st.sidebar.slider("Periodo RSI", min_value=7, max_value=30, value=14)
    rsi_compra_max = st.sidebar.slider("RSI maximo para compra", min_value=40, max_value=80, value=68)
    rsi_venda_min = st.sidebar.slider("RSI minimo para venda", min_value=50, max_value=90, value=75)
    stop_loss_pct = st.sidebar.slider("Stop loss (%)", min_value=1.0, max_value=15.0, value=3.0, step=0.5)
    taxa_operacao_pct = st.sidebar.slider("Taxa por operacao (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)

    config = StrategyConfig(
        capital_inicial=float(capital),
        media_curta=media_curta,
        media_longa=media_longa,
        media_tendencia=media_tendencia,
        periodo_rsi=periodo_rsi,
        rsi_compra_max=float(rsi_compra_max),
        rsi_venda_min=float(rsi_venda_min),
        stop_loss_pct=float(stop_loss_pct),
        taxa_operacao_pct=float(taxa_operacao_pct),
    )
    return tickers, start_date, end_date, config


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Invest Pro</h1>
            <p>Dashboard de analise tecnica com backtest, ranking dinamico, curva de patrimonio e leitura rapida de oportunidades.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(tickers: list[str], start_date: date, end_date: date, config: StrategyConfig) -> None:
    if not tickers:
        st.warning("Informe pelo menos um ativo para iniciar a analise.")
        return

    if start_date >= end_date:
        st.warning("A data final precisa ser maior que a data inicial.")
        return

    market_data = load_market_data(tuple(tickers), start_date, end_date)
    if not market_data:
        st.error("Nenhum dado foi carregado. Confira os codigos digitados e tente novamente.")
        return

    resultados = []
    dados_map: dict[str, pd.DataFrame] = {}
    trades_map: dict[str, pd.DataFrame] = {}
    equity_map: dict[str, pd.DataFrame] = {}

    progress = st.progress(0, text="Rodando analise dos ativos...")
    total = len(market_data)

    for idx, (ticker, df) in enumerate(market_data.items(), start=1):
        try:
            metricas, data, trades, equity = backtest_strategy(ticker, df, config)
        except Exception:
            continue

        resultados.append(metricas)
        dados_map[ticker] = data
        trades_map[ticker] = trades
        equity_map[ticker] = equity
        progress.progress(idx / total, text=f"Processando {ticker} ({idx}/{total})")

    progress.empty()

    if not resultados:
        st.error("Nao foi possivel concluir o backtest para os ativos selecionados.")
        return

    ranking_df = pd.DataFrame(resultados).sort_values("Retorno %", ascending=False).reset_index(drop=True)
    carteira = ranking_df.head(3).copy()

    st.session_state.resultados = ranking_df
    st.session_state.dados_map = dados_map
    st.session_state.trades_map = trades_map
    st.session_state.equity_map = equity_map
    st.session_state.resumo_carteira = {
        "melhor_ativo": ranking_df.iloc[0]["Ativo"],
        "retorno_medio_top3": carteira["Retorno %"].mean(),
        "lucro_total_top3": carteira["Lucro"].sum(),
        "capital_total_top3": config.capital_inicial * len(carteira),
    }
    st.session_state.analise_pronta = True


def render_overview(ranking_df: pd.DataFrame) -> None:
    resumo = st.session_state.resumo_carteira
    top = ranking_df.iloc[0]
    positivos = int((ranking_df["Lucro"] > 0).sum())
    cols = st.columns(4)
    cols[0].metric("Melhor ativo", top["Ativo"], format_pct(top["Retorno %"]))
    cols[1].metric("Lucro top 3", format_currency(resumo["lucro_total_top3"]))
    cols[2].metric("Retorno medio top 3", format_pct(resumo["retorno_medio_top3"]))
    cols[3].metric("Ativos positivos", f"{positivos}/{len(ranking_df)}")


def render_ranking(ranking_df: pd.DataFrame) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Ranking de performance")

    ranking_view = ranking_df.copy()
    ranking_view["Valor final"] = ranking_view["Valor final"].map(format_currency)
    ranking_view["Lucro"] = ranking_view["Lucro"].map(format_currency)
    ranking_view["Retorno %"] = ranking_view["Retorno %"].map(format_pct)
    ranking_view["Buy and hold %"] = ranking_view["Buy and hold %"].map(format_pct)
    ranking_view["Drawdown %"] = ranking_view["Drawdown %"].map(format_pct)
    ranking_view["Ultimo fechamento"] = ranking_view["Ultimo fechamento"].map(format_currency)
    ranking_view["Ultimo RSI"] = ranking_view["Ultimo RSI"].map(lambda x: f"{x:.2f}")

    st.dataframe(ranking_view, use_container_width=True, hide_index=True)
    csv_bytes = ranking_df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar ranking em CSV", csv_bytes, "ranking_invest_pro.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


def build_price_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
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
    fig.add_trace(
        go.Scatter(x=df.index, y=df["media_curta"], name="Media curta", line=dict(color="#5ab2ff", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["media_longa"], name="Media longa", line=dict(color="#ffca6d", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["media_tendencia"], name="Tendencia", line=dict(color="#3ddc97", width=1.5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#ff6b6b", width=2)),
        row=2,
        col=1,
    )

    fig.add_hline(y=70, line_dash="dash", line_color="#8d99ae", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#8d99ae", row=2, col=1)
    fig.update_layout(
        title=f"{ticker} | Candlestick, medias e RSI",
        template="plotly_dark",
        height=760,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def build_equity_figure(ticker: str) -> go.Figure:
    equity = st.session_state.equity_map[ticker]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["Patrimonio"],
            name="Estrategia",
            line=dict(color="#3ddc97", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["BuyHold"],
            name="Buy and hold",
            line=dict(color="#5ab2ff", width=2),
        )
    )
    fig.update_layout(
        title=f"{ticker} | Curva de patrimonio",
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="Patrimonio (R$)",
    )
    return fig


def render_asset_detail(ranking_df: pd.DataFrame) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Detalhe do ativo")
    ticker = st.selectbox("Escolha um ativo", ranking_df["Ativo"].tolist())
    detail = ranking_df.loc[ranking_df["Ativo"] == ticker].iloc[0]
    df = st.session_state.dados_map[ticker]
    trades = st.session_state.trades_map[ticker]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lucro", format_currency(detail["Lucro"]), format_pct(detail["Retorno %"]))
    col2.metric("Buy and hold", format_pct(detail["Buy and hold %"]))
    col3.metric("Drawdown", format_pct(detail["Drawdown %"]))
    col4.metric("Operacoes", int(detail["Operacoes"]))

    st.plotly_chart(build_price_figure(df, ticker), use_container_width=True)
    st.plotly_chart(build_equity_figure(ticker), use_container_width=True)

    if trades.empty:
        st.info("Nenhuma operacao foi executada para este ativo com os filtros atuais.")
    else:
        trades_view = trades.copy()
        trades_view["Preco"] = trades_view["Preco"].map(format_currency)
        st.dataframe(trades_view, use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar operacoes em CSV",
            trades.to_csv(index=False).encode("utf-8"),
            f"operacoes_{ticker}.csv",
            "text/csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_publish_notes() -> None:
    with st.expander("Como publicar"):
        st.markdown(
            """
            1. Instale as dependencias com `pip install -r requirements.txt`
            2. Execute localmente com `streamlit run app.py`
            3. No Streamlit Community Cloud, envie `app.py`, `requirements.txt` e configure `INVEST_PRO_USERNAME` e `INVEST_PRO_PASSWORD` em Secrets
            """
        )


def main() -> None:
    inject_css()
    ensure_session_state()

    if not st.session_state.logado:
        render_login()
        return

    render_header()
    tickers, start_date, end_date, config = sidebar_controls()

    top_controls = st.columns([1.2, 1, 1.2])
    rodar = top_controls[0].button("Rodar analise", type="primary", use_container_width=True)
    limpar = top_controls[1].button("Limpar resultados", use_container_width=True)
    top_controls[2].caption(
        "Os dados ficam em cache por 30 minutos para deixar o dashboard mais rapido."
    )

    if limpar:
        st.session_state.analise_pronta = False
        st.session_state.resultados = []
        st.session_state.dados_map = {}
        st.session_state.trades_map = {}
        st.session_state.equity_map = {}
        st.session_state.resumo_carteira = {}
        st.rerun()

    if rodar:
        with st.spinner("Baixando dados e calculando o backtest..."):
            run_analysis(tickers, start_date, end_date, config)

    if not st.session_state.analise_pronta:
        st.info("Ajuste os parametros na sidebar e clique em `Rodar analise` para gerar o dashboard.")
        render_publish_notes()
        return

    ranking_df = st.session_state.resultados
    render_overview(ranking_df)
    st.divider()
    render_ranking(ranking_df)
    st.divider()
    render_asset_detail(ranking_df)
    render_publish_notes()


if __name__ == "__main__":
    main()
