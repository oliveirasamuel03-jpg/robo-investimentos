# Invest Pro

Dashboard de analise tecnica em Streamlit com:

- login configuravel por `st.secrets` ou variaveis de ambiente
- backtest de estrategia com medias moveis, RSI, stop loss e taxa
- ranking de ativos
- grafico candlestick com medias e RSI
- curva de patrimonio versus buy and hold
- exportacao de ranking e operacoes em CSV

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Credenciais

Por padrao:

- usuario: `admin`
- senha: `1234`

Para publicar com seguranca, defina:

- `INVEST_PRO_USERNAME`
- `INVEST_PRO_PASSWORD`

Ou use `st.secrets`.
