
import json
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import google.generativeai as genai

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

MODEL_NAME = "gemini-1.5-flash" 
AI_CACHE_TTL_SECONDS = 3600 
RATE_LIMIT_DELAY_SECONDS = 1.5 
RATE_LIMIT_BATCH_SIZE = 10       
RATE_LIMIT_BATCH_PAUSE_SECONDS = 6
MAX_RETRIES = 3

ASSET_PRESETS = {
    "Indian Stock": None,
    "Gold (Intl Spot proxy - COMEX futures)": "GC=F",
    "Silver (Intl Spot proxy - COMEX futures)": "SI=F",
}

WATCHLIST_ALIASES = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
}

PERIOD_OPTIONS = {
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
}

INTERVAL_OPTIONS = {
    "Daily": "1d",
    "Weekly": "1wk",
}

ANALYSIS_MODES = {
    "Free (Rule-based, no API needed)": "rule",
    "AI Narrative (Gemini API - uses free credits)": "ai",
}

# ----------------------------------------------------------------------------
# DATA FETCH & INDICATORS (Same as original)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df.dropna()

def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi14 = 100 - (100 / (1 + (close.diff().clip(lower=0).rolling(14).mean() / (-close.diff().clip(upper=0)).rolling(14).mean())))
    
    return {
        "last_close": round(float(close.iloc[-1]), 2),
        "sma20": round(float(sma20.iloc[-1]), 2),
        "sma50": round(float(sma50.iloc[-1]), 2),
        "rsi14": round(float(rsi14.iloc[-1]), 1),
    }, {"sma20": sma20, "sma50": sma50}

# ----------------------------------------------------------------------------
# GEMINI CALL
# ----------------------------------------------------------------------------

def call_gemini_for_insight(symbol: str, asset_label: str, indicators: dict, interval: str) -> dict:
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in st.secrets or environment.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    user_prompt = f"""Asset: {asset_label} ({symbol})
Timeframe: {interval} candles
Latest technical snapshot:
{json.dumps(indicators, indent=2)}

You are a technical analysis assistant. Analyze the data and return ONLY valid JSON matching this schema:
{{
  "pattern_identified": "string",
  "trend": "Uptrend | Downtrend | Sideways/Range-bound",
  "key_support": number or null,
  "key_resistance": number or null,
  "recommendation": "Bullish | Bearish | Neutral | Watch",
  "confidence": "High | Medium | Low",
  "rationale": "2-4 sentences explaining the read",
  "risk_note": "1-2 sentences on key risk"
}}
"""

    response = model.generate_content(
        user_prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

# (Keep existing render_tab, render_single_asset, render_watchlist, etc. functions here, 
# just ensure they call call_gemini_for_insight instead of call_claude_for_insight)
# ... [Rest of the file logic]
