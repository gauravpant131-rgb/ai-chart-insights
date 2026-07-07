"""
Simple Chart Insights App -- Free, Rule-Based, No API Key Needed
==================================================================
Single-file Streamlit app: pick a stock/gold/silver, see the chart, and get
a plain technical recommendation (Bullish/Bearish/Neutral/Watch) computed
entirely from indicator rules -- no Anthropic API, no cost, no key needed.

Run with: streamlit run simple_chart_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Chart Insights (Free)", page_icon="📈", layout="wide")

ASSET_PRESETS = {
    "Indian Stock": None,
    "Gold (COMEX futures proxy)": "GC=F",
    "Silver (COMEX futures proxy)": "SI=F",
}

REC_COLORS = {"Bullish": "#0B6623", "Bearish": "#B22222", "Neutral": "#8a8a3c", "Watch": "#B8860B"}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def compute_indicators(df):
    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    last_close = float(close.iloc[-1])
    swing_high = float(df["High"].tail(60).max())
    swing_low = float(df["Low"].tail(60).min())

    return {
        "last_close": round(last_close, 2),
        "sma20": round(float(sma20.iloc[-1]), 2) if pd.notna(sma20.iloc[-1]) else None,
        "sma50": round(float(sma50.iloc[-1]), 2) if pd.notna(sma50.iloc[-1]) else None,
        "rsi": round(float(rsi.iloc[-1]), 1) if pd.notna(rsi.iloc[-1]) else None,
        "macd": round(float(macd.iloc[-1]), 3) if pd.notna(macd.iloc[-1]) else None,
        "macd_signal": round(float(signal.iloc[-1]), 3) if pd.notna(signal.iloc[-1]) else None,
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
    }, sma20, sma50


def rule_based_recommendation(ind):
    signals = []
    if ind["sma20"] and ind["sma50"]:
        if ind["last_close"] > ind["sma20"] > ind["sma50"]:
            trend, signals = "Uptrend", signals + [("Price above rising SMA20/50", 1)]
        elif ind["last_close"] < ind["sma20"] < ind["sma50"]:
            trend, signals = "Downtrend", signals + [("Price below falling SMA20/50", -1)]
        else:
            trend, signals = "Sideways/Range-bound", signals + [("Moving averages mixed", 0)]
    else:
        trend = "Sideways/Range-bound"

    if ind["rsi"] is not None:
        if ind["rsi"] >= 70:
            signals.append((f"RSI {ind['rsi']} overbought", -1))
        elif ind["rsi"] <= 30:
            signals.append((f"RSI {ind['rsi']} oversold", 1))
        else:
            signals.append((f"RSI {ind['rsi']} neutral", 0))

    if ind["macd"] is not None and ind["macd_signal"] is not None:
        if ind["macd"] > ind["macd_signal"]:
            signals.append(("MACD bullish crossover", 1))
        else:
            signals.append(("MACD bearish crossover", -1))

    directions = [d for _, d in signals if d != 0]
    score = sum(directions)
    if score >= 2:
        rec = "Bullish"
    elif score <= -2:
        rec = "Bearish"
    elif score == 0 and directions:
        rec = "Neutral"
    else:
        rec = "Watch"

    confidence = "High" if directions and (all(d > 0 for d in directions) or all(d < 0 for d in directions)) else "Medium" if directions else "Low"

    return {
        "trend": trend,
        "recommendation": rec,
        "confidence": confidence,
        "pattern": "; ".join(d for d, _ in signals),
        "support": ind["swing_low"],
        "resistance": ind["swing_high"],
    }


st.title("📈 Chart Insights (Free -- No API Key Needed)")
st.caption("Rule-based technical reads. Informational only, not investment advice.")

col1, col2 = st.columns([2, 1])
with col1:
    asset_type = st.selectbox("Asset", list(ASSET_PRESETS.keys()))
    if ASSET_PRESETS[asset_type] is None:
        symbol = st.text_input("NSE symbol", value="GRSE").strip().upper()
        ticker = f"{symbol}.NS" if symbol else ""
    else:
        ticker = ASSET_PRESETS[asset_type]
        st.text_input("Ticker", value=ticker, disabled=True)
with col2:
    period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)

if st.button("Analyze", type="primary") and ticker:
    with st.spinner("Fetching and analyzing..."):
        df = fetch_data(ticker, period=period)
    if df.empty:
        st.error("No data found. Check the symbol.")
    else:
        ind, sma20, sma50 = compute_indicators(df)
        result = rule_based_recommendation(ind)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                      low=df["Low"], close=df["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50", line=dict(width=1)))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        color = REC_COLORS.get(result["recommendation"], "#555")
        st.markdown(
            f"""<div style="border-left:6px solid {color};padding:14px 18px;background:rgba(0,0,0,0.03);border-radius:6px;">
            <span style="font-size:1.2em;font-weight:700;color:{color};">{result['recommendation']}</span>
            <span style="margin-left:10px;color:#666;">Confidence: {result['confidence']}</span>
            <div style="margin-top:8px;">Trend: {result['trend']} | Support: {result['support']} | Resistance: {result['resistance']}</div>
            <div style="margin-top:8px;">{result['pattern']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Rule-based technical read -- not investment advice.")
else:
    st.info("Pick an asset and click Analyze.")