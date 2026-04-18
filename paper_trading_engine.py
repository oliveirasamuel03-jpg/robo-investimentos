import streamlit as st
import plotly.graph_objects as go

from data_engine import load_data
from ml_engine import create_features, create_labels, EnsembleModel
from backtest_engine import backtest

st.set_page_config(layout="wide")
st.title("📊 Invest Pro Bot - Institutional Quant System")

prices = load_data()

features = create_features(prices)
labels = create_labels(prices).dropna()

model = EnsembleModel()
model.fit(features.fillna(0), labels.iloc[:len(features)])

probs = model.predict_proba(features.fillna(0))

signals = (probs > 0.55).astype(int)

equity = backtest(prices.iloc[-len(signals):], signals)

fig = go.Figure()
fig.add_trace(go.Scatter(y=equity, name="Equity Curve"))

st.plotly_chart(fig)

st.metric("Final Equity", float(equity.iloc[-1]))
