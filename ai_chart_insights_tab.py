"""
AI Chart Insights Tab
=====================
A drop-in Streamlit tab that gives AI-generated chart-pattern narratives and
a classified BUY / SELL / HOLD recommendation for:
  - Indian equities (NSE, via yfinance ".NS" suffix)
  - Gold (international spot proxy: COMEX Gold futures, ticker "GC=F")
  - Silver (international spot proxy: COMEX Silver futures, ticker "SI=F")

HOW TO INTEGRATE INTO YOUR EXISTING DASHBOARD
----------------------------------------------
1. Copy this file into your project folder (same folder as app.py).
2. In your main app.py, where you define your tabs, e.g.:

       tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
           "Ratings", "Backtest", "Portfolio", ..., "AI Chart Insights"
       ])

   add one more tab and call:

       from ai_chart_insights_tab import render_tab
       with tab7:
           render_tab()

3. Add your Gemini API key to Streamlit secrets (Settings -> Secrets on
   Streamlit Community Cloud, or .streamlit/secrets.toml locally):

       GEMINI_API_KEY = "AIza..."

   (Get a free key at https://aistudio.google.com/apikey)

4. Add to requirements.txt: google-genai, yfinance, plotly (pandas/numpy you
   already have).

DESIGN NOTES
------------
- Gemini is asked to return strict JSON (pattern, trend, levels, recommendation,
  confidence, rationale, risk_note) so the UI can render a clean recommendation
  badge instead of parsing free text.
- A hard disclaimer is baked into both the system prompt and the UI -- this
  tool is for informational/educational chart reading, not investment advice.
- Model is configurable at the top (MODEL_NAME).
- The free Rule-Based mode still works with zero API calls/cost, and now
  drives its verdict off a wider indicator set (Stochastic, ADX, OBV,
  Fibonacci retracement, volume) to produce a BUY / SELL / HOLD call.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash"  # swap to "gemini-2.5-pro" for deeper reasoning
AI_CACHE_TTL_SECONDS = 3600  # don't re-spend API credits re-analyzing the same snapshot within this window
RATE_LIMIT_DELAY_SECONDS = 1.5  # pause between AI calls in a batch scan
RATE_LIMIT_BATCH_SIZE = 10       # extra pause after every N calls, to stay well under RPM limits
RATE_LIMIT_BATCH_PAUSE_SECONDS = 6
MAX_RETRIES = 3

ASSET_PRESETS = {
    "Indian Stock": None,  # user types the symbol
    "Gold (Intl Spot proxy - COMEX futures)": "GC=F",
    "Silver (Intl Spot proxy - COMEX futures)": "SI=F",
}

# Shorthand tickers accepted in the watchlist box, in addition to plain NSE symbols
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
    "AI Narrative (Gemini API - uses credits)": "ai",
}

# BUY / SELL / HOLD palette used everywhere (cards, badges, tables)
REC_COLORS = {
    "BUY": "#0F9D58",
    "SELL": "#D93025",
    "HOLD": "#F4A100",
}
REC_BG = {
    "BUY": "rgba(15,157,88,0.10)",
    "SELL": "rgba(217,48,37,0.10)",
    "HOLD": "rgba(244,161,0,0.12)",
}
REC_EMOJI = {"BUY": "\U0001F4C8", "SELL": "\U0001F4C9", "HOLD": "\u23F8\uFE0F"}


# ----------------------------------------------------------------------------
# STYLE -- inject once per tab render for a colorful, professional GUI
# ----------------------------------------------------------------------------

def inject_custom_css():
    st.markdown(
        """
        <style>
        .aci-hero {
            background: linear-gradient(120deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
            padding: 22px 28px;
            border-radius: 16px;
            color: #ffffff;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .aci-hero h1 { margin: 0; font-size: 1.6em; }
        .aci-hero p { margin: 6px 0 0 0; color: #cfe8f2; font-size: 0.95em; }

        .aci-price-card {
            background: linear-gradient(135deg, #ffffff 0%, #f4f7fb 100%);
            border: 1px solid #e6e9f0;
            border-radius: 14px;
            padding: 18px 22px;
            box-shadow: 0 4px 14px rgba(20,30,60,0.06);
            margin-bottom: 14px;
        }
        .aci-price-label {
            font-size: 0.85em;
            color: #6b7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .aci-price-value {
            font-size: 2.6em;
            font-weight: 800;
            line-height: 1.1;
            margin: 2px 0;
            color: #111827;
        }
        .aci-price-change {
            font-size: 1.05em;
            font-weight: 700;
            padding: 2px 10px;
            border-radius: 8px;
            display: inline-block;
        }
        .aci-up { color: #0F9D58; background: rgba(15,157,88,0.10); }
        .aci-down { color: #D93025; background: rgba(217,48,37,0.10); }
        .aci-flat { color: #6b7280; background: rgba(107,114,128,0.10); }

        .aci-rec-badge {
            display: inline-block;
            font-size: 1.5em;
            font-weight: 800;
            padding: 6px 22px;
            border-radius: 999px;
            letter-spacing: 0.03em;
        }

        .aci-card {
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 4px 14px rgba(20,30,60,0.06);
        }
        .aci-metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .aci-metric {
            background: #f8fafc;
            border: 1px solid #edf0f5;
            border-radius: 10px;
            padding: 10px 12px;
        }
        .aci-metric .lbl { font-size: 0.75em; color: #6b7280; font-weight: 600; text-transform: uppercase; }
        .aci-metric .val { font-size: 1.15em; font-weight: 700; color: #111827; }

        .stButton>button {
            border-radius: 10px;
            font-weight: 700;
            border: none;
            background: linear-gradient(120deg, #2c5364, #203a43);
            color: white;
        }
        .stButton>button:hover { background: linear-gradient(120deg, #2c5364, #0f2027); color: #ffe8b8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# DATA FETCH
# ----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_quote(ticker: str) -> dict:
    """Best-effort near-real-time quote (fast_info), independent of the
    historical candle cache, so the headline price refreshes more often
    than the chart data."""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        last = fi.get("last_price") or fi.get("lastPrice")
        prev = fi.get("previous_close") or fi.get("previousClose")
        day_high = fi.get("day_high") or fi.get("dayHigh")
        day_low = fi.get("day_low") or fi.get("dayLow")
        currency = fi.get("currency", "")
        return {
            "last_price": float(last) if last is not None else None,
            "previous_close": float(prev) if prev is not None else None,
            "day_high": float(day_high) if day_high is not None else None,
            "day_low": float(day_low) if day_low is not None else None,
            "currency": currency,
        }
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# TECHNICAL INDICATORS
# ----------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"] if "Volume" in df.columns else pd.Series(dtype=float)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_upper = mid + 2 * std
    bb_lower = mid - 2 * std

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    # --- Stochastic Oscillator (%K, %D) ---
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    # --- ADX (trend strength) ---
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = tr.rolling(14).sum().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx14 = dx.rolling(14).mean()

    # --- On-Balance Volume (trend of volume-weighted flow) ---
    if not vol.empty:
        obv = (np.sign(close.diff().fillna(0)) * vol).fillna(0).cumsum()
        obv_sma20 = obv.rolling(20).mean()
        obv_rising = bool(obv.iloc[-1] > obv_sma20.iloc[-1]) if pd.notna(obv_sma20.iloc[-1]) else None
    else:
        obv_rising = None

    # --- Fibonacci retracement over the recent swing ---
    lookback = min(60, len(df))
    recent = df.tail(lookback)
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())
    span = swing_high - swing_low
    fib_levels = {
        "0.0%": round(swing_high, 2),
        "23.6%": round(swing_high - 0.236 * span, 2),
        "38.2%": round(swing_high - 0.382 * span, 2),
        "50.0%": round(swing_high - 0.5 * span, 2),
        "61.8%": round(swing_high - 0.618 * span, 2),
        "100%": round(swing_low, 2),
    } if span > 0 else {}

    last_close = float(close.iloc[-1])
    vol_avg20 = float(vol.rolling(20).mean().iloc[-1]) if not vol.empty else None
    vol_last = float(vol.iloc[-1]) if not vol.empty else None
    vol_ratio = round(vol_last / vol_avg20, 2) if (vol_last and vol_avg20) else None

    # --- Supertrend (10, 3) -- the standard desk default ---
    supertrend_series, supertrend_dir = compute_supertrend(df, period=10, multiplier=3.0)
    supertrend_direction = "Bullish" if int(supertrend_dir.iloc[-1]) == 1 else "Bearish"
    supertrend_up = supertrend_series.where(supertrend_dir == 1)
    supertrend_down = supertrend_series.where(supertrend_dir == -1)

    # --- Anchored VWAP -- anchored at the most recent Supertrend flip, so it
    # reflects the volume-weighted average price of the *current* trend leg
    # rather than an arbitrary fixed window. Falls back to the start of the
    # fetched period if no flip occurred (e.g. short lookback). ---
    flip_positions = np.flatnonzero(supertrend_dir.diff().fillna(0).to_numpy() != 0)
    vwap_anchor_pos = int(flip_positions[-1]) if len(flip_positions) else 0
    vwap_series = compute_anchored_vwap(df, vwap_anchor_pos)
    vwap_anchor_date = str(df.index[vwap_anchor_pos].date())

    # --- RSI & MACD-histogram divergence over the last 60 candles ---
    rsi_div = detect_divergence(df, rsi14, "RSI", order=3, lookback=60)
    macd_div = detect_divergence(df, macd_hist, "MACD histogram", order=3, lookback=60)

    # --- ATR-based stop/target brackets -- reuses the atr14 already computed
    # above (no new series). 1.5x ATR stop / 2.5x ATR target is a standard
    # desk default giving roughly a 1:1.7 reward-to-risk skeleton. ---
    atr_last = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else None
    if atr_last:
        stop_long = round(last_close - 1.5 * atr_last, 2)
        target_long = round(last_close + 2.5 * atr_last, 2)
        stop_short = round(last_close + 1.5 * atr_last, 2)
        target_short = round(last_close - 2.5 * atr_last, 2)
    else:
        stop_long = target_long = stop_short = target_short = None

    def _safe(series):
        try:
            v = series.iloc[-1]
            return round(float(v), 2) if pd.notna(v) else None
        except Exception:
            return None

    return {
        "last_close": round(last_close, 2),
        "sma20": _safe(sma20),
        "sma50": _safe(sma50),
        "sma200": _safe(sma200) if len(df) >= 200 else None,
        "rsi14": _safe(rsi14),
        "macd": round(float(macd.iloc[-1]), 3) if pd.notna(macd.iloc[-1]) else None,
        "macd_signal": round(float(signal.iloc[-1]), 3) if pd.notna(signal.iloc[-1]) else None,
        "macd_hist": round(float(macd_hist.iloc[-1]), 3) if pd.notna(macd_hist.iloc[-1]) else None,
        "bb_upper": _safe(bb_upper),
        "bb_lower": _safe(bb_lower),
        "atr14": _safe(atr14),
        "stoch_k": _safe(stoch_k),
        "stoch_d": _safe(stoch_d),
        "adx14": _safe(adx14),
        "obv_rising": obv_rising,
        "vol_ratio_vs_avg20": vol_ratio,
        "fib_levels": fib_levels,
        "swing_high_60d": round(swing_high, 2),
        "swing_low_60d": round(swing_low, 2),
        "vol_last": vol_last,
        "vol_avg20": vol_avg20,
        "supertrend": _safe(supertrend_series),
        "supertrend_direction": supertrend_direction,
        "rsi_divergence": rsi_div["type"],
        "rsi_divergence_detail": rsi_div["detail"],
        "macd_divergence": macd_div["type"],
        "macd_divergence_detail": macd_div["detail"],
        "vwap_anchored": _safe(vwap_series),
        "vwap_anchor_date": vwap_anchor_date,
        "stop_long": stop_long,
        "target_long": target_long,
        "stop_short": stop_short,
        "target_short": target_short,
    }, {
        "sma20": sma20, "sma50": sma50, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "supertrend_up": supertrend_up, "supertrend_down": supertrend_down,
        "rsi14": rsi14, "macd_hist": macd_hist,
        "rsi_divergence": rsi_div, "macd_divergence": macd_div,
        "vwap": vwap_series,
    }


# ----------------------------------------------------------------------------
# PRO-TRADER ADD-ONS: SUPERTREND + RSI/MACD DIVERGENCE
# ----------------------------------------------------------------------------
# These plug straight into the existing indicator engine: compute_indicators()
# calls both helpers below and folds their output into the same `indicators`
# dict and `overlays` dict that already flow through to the chart, the rule
# engine, the indicator grid, and the AI prompt (which just json.dumps the
# indicators dict) -- so nothing downstream had to change shape.

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Classic ATR-based Supertrend. Returns (supertrend_series, direction_series)
    where direction is +1 (uptrend / price above the line) or -1 (downtrend)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()  # Wilder-style smoothing

    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(df)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(n):
        if i == 0:
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            direction.iloc[i] = 1
            supertrend.iloc[i] = final_lower.iloc[i]
            continue

        final_upper.iloc[i] = (
            basic_upper.iloc[i]
            if (basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1])
            else final_upper.iloc[i - 1]
        )
        final_lower.iloc[i] = (
            basic_lower.iloc[i]
            if (basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1])
            else final_lower.iloc[i - 1]
        )

        if direction.iloc[i - 1] == 1:
            direction.iloc[i] = -1 if close.iloc[i] < final_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return supertrend, direction


def compute_anchored_vwap(df: pd.DataFrame, anchor_pos: int) -> pd.Series:
    """Volume-weighted average price computed cumulatively from `anchor_pos`
    to the end of the dataframe. Values before the anchor are NaN (undefined)."""
    vwap = pd.Series(index=df.index, dtype=float)
    if "Volume" not in df.columns or df["Volume"].fillna(0).sum() == 0:
        return vwap
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol = typical * df["Volume"]
    cum_tp_vol = tp_vol.iloc[anchor_pos:].cumsum()
    cum_vol = df["Volume"].iloc[anchor_pos:].cumsum().replace(0, np.nan)
    vwap.iloc[anchor_pos:] = cum_tp_vol / cum_vol
    return vwap


def _find_swing_points(vals: np.ndarray, order: int = 3):
    """Lightweight local-extrema finder (no scipy dependency). Returns positional
    indices of swing highs and swing lows within `vals`."""
    n = len(vals)
    highs_idx, lows_idx = [], []
    for i in range(order, n - order):
        window = vals[i - order:i + order + 1]
        if np.isnan(window).any():
            continue
        center = vals[i]
        if center == window.max() and (window == center).sum() == 1:
            highs_idx.append(i)
        if center == window.min() and (window == center).sum() == 1:
            lows_idx.append(i)
    return highs_idx, lows_idx


def detect_divergence(df: pd.DataFrame, indicator_series: pd.Series, label: str,
                       order: int = 3, lookback: int = 60) -> dict:
    """Compares the last two price swing highs/lows against the same indicator
    (RSI or MACD histogram) at those same points -- the standard definition of
    regular bullish/bearish divergence. Returns plot-ready timestamp/value pairs
    so the chart can draw the trendlines, not just report them as text."""
    lb = min(lookback, len(df))
    price = df["Close"].tail(lb)
    ind = indicator_series.reindex(df.index).tail(lb)
    idx = price.index
    p_vals = price.values
    i_vals = ind.values

    result = {"type": None, "detail": None, "bear_points": None, "bull_points": None}
    if lb < (2 * order + 5):
        return result

    highs_idx, lows_idx = _find_swing_points(p_vals, order=order)

    if len(highs_idx) >= 2:
        a, b = highs_idx[-2], highs_idx[-1]
        if p_vals[b] > p_vals[a] and pd.notna(i_vals[a]) and pd.notna(i_vals[b]) and i_vals[b] < i_vals[a]:
            result["type"] = "Bearish"
            result["detail"] = (
                f"Price made a higher high ({p_vals[a]:.2f} -> {p_vals[b]:.2f}) while {label} made a "
                f"lower high ({i_vals[a]:.2f} -> {i_vals[b]:.2f}) -- classic bearish divergence."
            )
            result["bear_points"] = [(idx[a], p_vals[a], i_vals[a]), (idx[b], p_vals[b], i_vals[b])]

    if len(lows_idx) >= 2:
        a, b = lows_idx[-2], lows_idx[-1]
        if p_vals[b] < p_vals[a] and pd.notna(i_vals[a]) and pd.notna(i_vals[b]) and i_vals[b] > i_vals[a]:
            bull_detail = (
                f"Price made a lower low ({p_vals[a]:.2f} -> {p_vals[b]:.2f}) while {label} made a "
                f"higher low ({i_vals[a]:.2f} -> {i_vals[b]:.2f}) -- classic bullish divergence."
            )
            if result["type"] is None:
                result["type"] = "Bullish"
                result["detail"] = bull_detail
            else:
                result["detail"] = result["detail"] + " " + bull_detail
            result["bull_points"] = [(idx[a], p_vals[a], i_vals[a]), (idx[b], p_vals[b], i_vals[b])]

    return result


# ----------------------------------------------------------------------------
# FREE RULE-BASED ANALYSIS -- no API key, no cost, works forever
# ----------------------------------------------------------------------------

def rule_based_insight(indicators: dict) -> dict:
    """Produces the exact same schema as the AI insight, but derived entirely
    from coded technical rules across a wider indicator set -- zero API
    calls, zero cost. Outputs an explicit BUY / SELL / HOLD call."""
    last_close = indicators.get("last_close")
    sma20 = indicators.get("sma20")
    sma50 = indicators.get("sma50")
    sma200 = indicators.get("sma200")
    rsi = indicators.get("rsi14")
    macd = indicators.get("macd")
    macd_signal = indicators.get("macd_signal")
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    stoch_k = indicators.get("stoch_k")
    stoch_d = indicators.get("stoch_d")
    adx = indicators.get("adx14")
    obv_rising = indicators.get("obv_rising")
    vol_ratio = indicators.get("vol_ratio_vs_avg20")
    swing_high = indicators.get("swing_high_60d")
    swing_low = indicators.get("swing_low_60d")
    atr = indicators.get("atr14")

    signals = []  # (description, direction) where direction: +1 bullish, -1 bearish, 0 neutral

    if sma20 is not None and sma50 is not None and last_close is not None:
        if last_close > sma20 > sma50:
            trend = "Uptrend"
            signals.append(("Price above rising SMA20, which is above SMA50", 1))
        elif last_close < sma20 < sma50:
            trend = "Downtrend"
            signals.append(("Price below falling SMA20, which is below SMA50", -1))
        else:
            trend = "Sideways/Range-bound"
            signals.append(("Moving averages mixed or overlapping", 0))
    else:
        trend = "Sideways/Range-bound"
        signals.append(("Not enough history yet for a full moving-average read", 0))

    if sma200 is not None and last_close is not None:
        if last_close > sma200:
            signals.append(("Price above long-term SMA200 (bullish backdrop)", 1))
        else:
            signals.append(("Price below long-term SMA200 (bearish backdrop)", -1))

    if rsi is not None:
        if rsi >= 70:
            signals.append((f"RSI at {rsi} is in overbought territory", -1))
        elif rsi <= 30:
            signals.append((f"RSI at {rsi} is in oversold territory", 1))
        else:
            signals.append((f"RSI at {rsi} is in neutral territory", 0))

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            signals.append(("MACD is above its signal line (bullish crossover)", 1))
        else:
            signals.append(("MACD is below its signal line (bearish crossover)", -1))

    if bb_upper is not None and bb_lower is not None and last_close is not None and bb_upper > bb_lower:
        band_pos = (last_close - bb_lower) / (bb_upper - bb_lower)
        if band_pos >= 0.95:
            signals.append(("Price is testing the upper Bollinger Band", -1))
        elif band_pos <= 0.05:
            signals.append(("Price is testing the lower Bollinger Band", 1))

    if stoch_k is not None and stoch_d is not None:
        if stoch_k >= 80:
            signals.append((f"Stochastic %K at {stoch_k} signals overbought", -1))
        elif stoch_k <= 20:
            signals.append((f"Stochastic %K at {stoch_k} signals oversold", 1))
        elif stoch_k > stoch_d:
            signals.append(("Stochastic %K crossing above %D (early bullish)", 1))
        else:
            signals.append(("Stochastic %K crossing below %D (early bearish)", -1))

    if adx is not None:
        if adx >= 25:
            # ADX confirms trend strength -- amplify the prevailing trend direction
            if trend == "Uptrend":
                signals.append((f"ADX at {adx} confirms a strong trend (supports the uptrend)", 1))
            elif trend == "Downtrend":
                signals.append((f"ADX at {adx} confirms a strong trend (supports the downtrend)", -1))
            else:
                signals.append((f"ADX at {adx} shows a strong trend despite mixed averages", 0))
        else:
            signals.append((f"ADX at {adx} shows a weak/choppy trend", 0))

    if obv_rising is not None:
        if obv_rising:
            signals.append(("On-Balance Volume is rising (buying pressure)", 1))
        else:
            signals.append(("On-Balance Volume is falling (selling pressure)", -1))

    if vol_ratio is not None and vol_ratio >= 1.5:
        # High relative volume amplifies whatever the prevailing short-term move is
        if macd is not None and macd_signal is not None and macd > macd_signal:
            signals.append((f"Volume is {vol_ratio}x the 20-day average, confirming upside move", 1))
        elif macd is not None and macd_signal is not None:
            signals.append((f"Volume is {vol_ratio}x the 20-day average, confirming downside move", -1))

    supertrend_direction = indicators.get("supertrend_direction")
    if supertrend_direction == "Bullish":
        signals.append((f"Supertrend (10,3) is bullish -- price holding above {indicators.get('supertrend')}", 1))
    elif supertrend_direction == "Bearish":
        signals.append((f"Supertrend (10,3) is bearish -- price under {indicators.get('supertrend')}", -1))

    # Divergence is a higher-conviction reversal signal, so it's weighted 2x a normal signal.
    rsi_divergence = indicators.get("rsi_divergence")
    if rsi_divergence == "Bullish":
        signals.append((f"Bullish RSI divergence: {indicators.get('rsi_divergence_detail')}", 2))
    elif rsi_divergence == "Bearish":
        signals.append((f"Bearish RSI divergence: {indicators.get('rsi_divergence_detail')}", -2))

    macd_divergence = indicators.get("macd_divergence")
    if macd_divergence == "Bullish":
        signals.append((f"Bullish MACD-histogram divergence: {indicators.get('macd_divergence_detail')}", 2))
    elif macd_divergence == "Bearish":
        signals.append((f"Bearish MACD-histogram divergence: {indicators.get('macd_divergence_detail')}", -2))

    directions = [d for _, d in signals if d != 0]
    score = sum(directions)

    if score >= 3:
        recommendation = "BUY"
    elif score <= -3:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    if directions:
        pos_count = sum(1 for d in directions if d > 0)
        neg_count = sum(1 for d in directions if d < 0)
        total = len(directions)
        if pos_count == total or neg_count == total:
            confidence = "High"
        elif abs(pos_count - neg_count) / total >= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        confidence = "Low"

    pattern_bits = [desc for desc, _ in signals]
    pattern_identified = "; ".join(pattern_bits) if pattern_bits else "No clear pattern - insufficient data"
    rationale = "Rule-based technical read across trend, momentum, volatility and volume: " + "; ".join(pattern_bits) + "."

    risk_note = "Automated rule-based technical read (no AI narrative) -- not investment advice."
    if atr is not None and last_close:
        vol_pct = (atr / last_close) * 100
        risk_note += f" Recent volatility (ATR) is about {vol_pct:.1f}% of price; size positions accordingly."

    return {
        "pattern_identified": pattern_identified,
        "trend": trend,
        "key_support": swing_low,
        "key_resistance": swing_high,
        "recommendation": recommendation,
        "confidence": confidence,
        "rationale": rationale,
        "risk_note": risk_note,
        "signal_score": score,
    }


# ----------------------------------------------------------------------------
# CHART
# ----------------------------------------------------------------------------

def build_candlestick_chart(df: pd.DataFrame, overlays: dict, title: str, fib_levels: dict = None) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.58, 0.20, 0.22],
        vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#0F9D58", decreasing_line_color="#D93025",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=overlays["sma20"], name="SMA 20",
                              line=dict(color="#2c5364", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=overlays["sma50"], name="SMA 50",
                              line=dict(color="#F4A100", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=overlays["bb_upper"], name="BB Upper",
                              line=dict(color="rgba(120,120,140,0.55)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=overlays["bb_lower"], name="BB Lower",
                              line=dict(color="rgba(120,120,140,0.55)", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(44,83,100,0.06)"), row=1, col=1)

    # --- Supertrend overlay -- split into up/down segments so the line color
    # flips green/red exactly where the trend flips, like a desk terminal. ---
    if "supertrend_up" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=overlays["supertrend_up"], name="Supertrend (Bullish)",
                                  line=dict(color="#0F9D58", width=2), connectgaps=False), row=1, col=1)
    if "supertrend_down" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=overlays["supertrend_down"], name="Supertrend (Bearish)",
                                  line=dict(color="#D93025", width=2), connectgaps=False), row=1, col=1)

    if "vwap" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=overlays["vwap"], name="Anchored VWAP",
                                  line=dict(color="#8E44AD", width=1.6, dash="dash")), row=1, col=1)

    if fib_levels:
        fib_colors = ["#9AA5B1", "#7C8CA6", "#5E7CE2", "#F4A100", "#D93025", "#9AA5B1"]
        for (label, level), color in zip(fib_levels.items(), fib_colors):
            fig.add_hline(y=level, line=dict(color=color, width=0.8, dash="dot"),
                          annotation_text=f"Fib {label}", annotation_position="right",
                          annotation_font_size=9, row=1, col=1)

    # --- RSI panel with overbought/oversold guides + divergence trendlines ---
    if "rsi14" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=overlays["rsi14"], name="RSI (14)",
                                  line=dict(color="#5E7CE2", width=1.4)), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="rgba(217,48,37,0.5)", width=1, dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="rgba(15,157,88,0.5)", width=1, dash="dot"), row=2, col=1)

        rsi_div = overlays.get("rsi_divergence") or {}
        for key, color, label in (("bear_points", "#D93025", "Bearish Divergence"),
                                   ("bull_points", "#0F9D58", "Bullish Divergence")):
            pts = rsi_div.get(key)
            if pts:
                (t1, p1, r1), (t2, p2, r2) = pts
                fig.add_trace(go.Scatter(x=[t1, t2], y=[p1, p2], mode="lines+markers", name=f"{label} (Price)",
                                          line=dict(color=color, width=2, dash="dash"),
                                          marker=dict(size=7, color=color)), row=1, col=1)
                fig.add_trace(go.Scatter(x=[t1, t2], y=[r1, r2], mode="lines+markers", name=f"{label} (RSI)",
                                          line=dict(color=color, width=2, dash="dash"),
                                          marker=dict(size=7, color=color), showlegend=False), row=2, col=1)

    if "Volume" in df.columns:
        vol_colors = np.where(df["Close"] >= df["Open"], "#0F9D58", "#D93025")
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                              marker_color=vol_colors, opacity=0.6), row=3, col=1)

    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False, height=760,
        margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(250,251,253,1)",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    return fig


# ----------------------------------------------------------------------------
# LIVE PRICE HEADER -- big, bold, unmissable
# ----------------------------------------------------------------------------

def render_price_header(ticker: str, asset_label: str, fallback_close: float, fallback_prev: float = None):
    quote = fetch_live_quote(ticker)
    last_price = quote.get("last_price") or fallback_close
    prev_close = quote.get("previous_close") or fallback_prev
    currency = quote.get("currency") or ("INR" if ticker.endswith(".NS") else "USD")
    symbol_map = {"INR": "\u20B9", "USD": "$"}
    ccy_symbol = symbol_map.get(currency, currency + " ")

    change, pct = None, None
    if last_price is not None and prev_close:
        change = last_price - prev_close
        pct = (change / prev_close) * 100

    if change is None:
        change_html = ""
    elif change > 0:
        change_html = f'<span class="aci-price-change aci-up">\u25B2 {ccy_symbol}{change:,.2f} ({pct:+.2f}%)</span>'
    elif change < 0:
        change_html = f'<span class="aci-price-change aci-down">\u25BC {ccy_symbol}{abs(change):,.2f} ({pct:+.2f}%)</span>'
    else:
        change_html = '<span class="aci-price-change aci-flat">\u2014 No change</span>'

    day_range = ""
    if quote.get("day_low") and quote.get("day_high"):
        day_range = (f'<div class="aci-metric"><div class="lbl">Day Range</div>'
                     f'<div class="val">{ccy_symbol}{quote["day_low"]:,.2f} \u2013 {ccy_symbol}{quote["day_high"]:,.2f}</div></div>')

    prev_html = ""
    if prev_close:
        prev_html = (f'<div class="aci-metric"><div class="lbl">Prev Close</div>'
                     f'<div class="val">{ccy_symbol}{prev_close:,.2f}</div></div>')

    price_str = f"{ccy_symbol}{last_price:,.2f}" if last_price is not None else "N/A"

    st.markdown(
        f"""
        <div class="aci-price-card">
            <div class="aci-price-label">{asset_label} &middot; {ticker}</div>
            <div class="aci-price-value">{price_str}</div>
            {change_html}
            <div class="aci-metric-grid">
                {prev_html}
                {day_range}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# GEMINI CALL
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a technical analysis assistant embedded in a personal investing \
dashboard for an experienced Indian retail investor. You read price/indicator data and \
describe the chart pattern and technical setup in plain, precise language.

Rules:
- Base your read strictly on the numeric data given. Do not invent price history you weren't given.
- Identify classical chart/technical patterns only if the data genuinely supports them \
(e.g. uptrend/downtrend, consolidation, breakout above resistance, oversold/overbought, \
moving average crossover, Bollinger squeeze, Fibonacci confluence). Do not force a pattern \
name if none fits -- say the structure is unclear or range-bound instead.
- The data includes a Supertrend(10,3) direction and any detected RSI/MACD-histogram \
divergence. Treat a live divergence (bullish or bearish) as a higher-conviction reversal \
signal that can override a purely trend-following read, and call this out explicitly in \
the rationale when present. Treat the Supertrend direction as a trend-confirmation filter. \
The data also includes an anchored VWAP (anchored at the most recent Supertrend flip) -- price \
holding above it within an uptrend, or below it within a downtrend, adds confluence to your read.
- Your recommendation field is a technical-stance classification (BUY / SELL / HOLD), not \
investment advice. Always include a risk_note.
- Return ONLY valid JSON, no markdown fences, no preamble, matching exactly this schema:

{
  "pattern_identified": "string, e.g. 'Ascending triangle near resistance' or 'No clear pattern - range-bound'",
  "trend": "Uptrend | Downtrend | Sideways/Range-bound",
  "key_support": number or null,
  "key_resistance": number or null,
  "recommendation": "BUY | SELL | HOLD",
  "confidence": "High | Medium | Low",
  "rationale": "2-4 sentences explaining the read, referencing the specific indicator values given",
  "risk_note": "1-2 sentences on what would invalidate this read or key risk"
}
"""


def call_gemini_for_insight(symbol: str, asset_label: str, indicators: dict, interval: str) -> dict:
    if genai is None:
        raise RuntimeError("The 'google-genai' package is not installed. Add it to requirements.txt.")

    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in st.secrets or environment.")

    client = genai.Client(api_key=api_key)

    user_prompt = f"""Asset: {asset_label} ({symbol})
Timeframe: {interval} candles

Latest technical snapshot:
{json.dumps(indicators, indent=2)}

Give your read as JSON per the schema."""

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    max_output_tokens=700,
                ),
            )
            text = (resp.text or "").strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            last_error = e
            time.sleep((2 ** attempt) * RATE_LIMIT_DELAY_SECONDS)

    raise RuntimeError(f"AI call failed after {MAX_RETRIES} attempts: {last_error}")


@st.cache_data(ttl=AI_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_ai_insight(symbol: str, asset_label: str, indicators_json: str, interval: str, model_name: str) -> dict:
    """Cache wrapper keyed on the actual indicator values (as a JSON string, so it's
    hashable). Identical indicator snapshots within AI_CACHE_TTL_SECONDS reuse the
    previous Gemini response instead of paying for a fresh API call. A genuinely new
    price snapshot (different indicators) produces a different cache key automatically."""
    indicators = json.loads(indicators_json)
    return call_gemini_for_insight(symbol, asset_label, indicators, interval)


def get_ai_insight(symbol: str, asset_label: str, indicators: dict, interval: str, force_refresh: bool = False) -> dict:
    indicators_json = json.dumps(indicators, sort_keys=True)
    if force_refresh:
        _cached_ai_insight.clear()
    return _cached_ai_insight(symbol, asset_label, indicators_json, interval, MODEL_NAME)


def get_insight(mode: str, symbol: str, asset_label: str, indicators: dict, interval: str, force_refresh: bool = False) -> dict:
    """Single entry point used by the UI. mode is 'rule' or 'ai' (see ANALYSIS_MODES)."""
    insight = rule_based_insight(indicators) if mode == "rule" else get_ai_insight(symbol, asset_label, indicators, interval, force_refresh)
    return _attach_risk_bracket(insight, indicators)


def _attach_risk_bracket(insight: dict, indicators: dict) -> dict:
    """Adds an ATR-based stop/target bracket to whichever insight came back
    (AI or rule-based), keyed off that insight's own BUY/SELL/HOLD call so
    both analysis modes get the same risk-sizing treatment for free."""
    rec = str(insight.get("recommendation", "HOLD")).upper()
    if rec == "BUY":
        insight["suggested_stop"] = indicators.get("stop_long")
        insight["suggested_target"] = indicators.get("target_long")
    elif rec == "SELL":
        insight["suggested_stop"] = indicators.get("stop_short")
        insight["suggested_target"] = indicators.get("target_short")
    else:
        insight["suggested_stop"] = None
        insight["suggested_target"] = None
    return insight


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------

def render_recommendation_card(insight: dict):
    rec = str(insight.get("recommendation", "HOLD")).upper()
    if rec not in REC_COLORS:
        rec = "HOLD"
    color = REC_COLORS[rec]
    bg = REC_BG[rec]
    emoji = REC_EMOJI[rec]

    st.markdown(
        f"""
        <div class="aci-card" style="background:{bg}; border-left: 8px solid {color};">
            <span class="aci-rec-badge" style="background:{color}; color:white;">{emoji} {rec}</span>
            <span style="margin-left: 14px; color: #444; font-weight: 600;">
                Confidence: {insight.get('confidence', 'n/a')}
            </span>
            <div style="margin-top: 12px; font-weight: 700; color:#1f2937;">
                {insight.get('pattern_identified', '')}
            </div>
            <div style="margin-top: 6px; color: #374151;">
                Trend: <b>{insight.get('trend', 'n/a')}</b> &nbsp;|&nbsp;
                Support: <b>{insight.get('key_support', 'n/a')}</b> &nbsp;|&nbsp;
                Resistance: <b>{insight.get('key_resistance', 'n/a')}</b>
            </div>
            {f'''<div style="margin-top: 6px; color: #374151;">
                Suggested Stop: <b style="color:#D93025;">{insight.get('suggested_stop')}</b> &nbsp;|&nbsp;
                Suggested Target: <b style="color:#0F9D58;">{insight.get('suggested_target')}</b>
                <span style="color:#9AA5B1;">(1.5x / 2.5x ATR14 bracket)</span>
            </div>''' if insight.get('suggested_stop') is not None else ''}
            <div style="margin-top: 10px; color:#1f2937;">{insight.get('rationale', '')}</div>
            <div style="margin-top: 8px; font-style: italic; color: #6b7280; font-size: 0.9em;">
                \u26A0\uFE0F Risk: {insight.get('risk_note', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_indicator_grid(ind: dict):
    def fmt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "\u2014"

    items = [
        ("SMA 20", fmt(ind.get("sma20"))),
        ("SMA 50", fmt(ind.get("sma50"))),
        ("SMA 200", fmt(ind.get("sma200"))),
        ("RSI (14)", fmt(ind.get("rsi14"))),
        ("MACD", fmt(ind.get("macd"))),
        ("MACD Signal", fmt(ind.get("macd_signal"))),
        ("Stoch %K", fmt(ind.get("stoch_k"))),
        ("Stoch %D", fmt(ind.get("stoch_d"))),
        ("ADX (14)", fmt(ind.get("adx14"))),
        ("ATR (14)", fmt(ind.get("atr14"))),
        ("OBV Trend", "Rising \U0001F4C8" if ind.get("obv_rising") else ("Falling \U0001F4C9" if ind.get("obv_rising") is False else "\u2014")),
        ("Volume vs 20d Avg", fmt(ind.get("vol_ratio_vs_avg20"), "x")),
        ("60D Support", fmt(ind.get("swing_low_60d"))),
        ("60D Resistance", fmt(ind.get("swing_high_60d"))),
        ("Supertrend (10,3)", fmt(ind.get("supertrend"))),
        ("Supertrend Direction", "\U0001F7E2 Bullish" if ind.get("supertrend_direction") == "Bullish"
         else ("\U0001F534 Bearish" if ind.get("supertrend_direction") == "Bearish" else "\u2014")),
        ("RSI Divergence", f"\u26A0\uFE0F {ind['rsi_divergence']}" if ind.get("rsi_divergence") else "None"),
        ("MACD Divergence", f"\u26A0\uFE0F {ind['macd_divergence']}" if ind.get("macd_divergence") else "None"),
        ("Anchored VWAP", f"{fmt(ind.get('vwap_anchored'))} (since {ind.get('vwap_anchor_date', 'n/a')})"),
    ]
    html = '<div class="aci-metric-grid">'
    for label, val in items:
        html += f'<div class="aci-metric"><div class="lbl">{label}</div><div class="val">{val}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    if ind.get("rsi_divergence_detail"):
        st.caption(f"\U0001F4C9 RSI: {ind['rsi_divergence_detail']}")
    if ind.get("macd_divergence_detail"):
        st.caption(f"\U0001F4C9 MACD: {ind['macd_divergence_detail']}")

    fib = ind.get("fib_levels")
    if fib:
        st.markdown("**Fibonacci retracement (last 60 candles):**")
        fib_html = '<div class="aci-metric-grid">'
        for label, level in fib.items():
            fib_html += f'<div class="aci-metric"><div class="lbl">{label}</div><div class="val">{level}</div></div>'
        fib_html += "</div>"
        st.markdown(fib_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# MAIN TAB RENDER FUNCTION -- call this from your app.py
# ----------------------------------------------------------------------------

def render_single_asset():
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        asset_type = st.selectbox("Asset type", list(ASSET_PRESETS.keys()))
        if ASSET_PRESETS[asset_type] is None:
            symbol_input = st.text_input(
                "NSE symbol (e.g. RELIANCE, TCS, GRSE)", value="GRSE"
            ).strip().upper()
            ticker = f"{symbol_input}.NS" if symbol_input else ""
            asset_label = symbol_input
        else:
            ticker = ASSET_PRESETS[asset_type]
            asset_label = asset_type
            st.text_input("Ticker (auto)", value=ticker, disabled=True)

    with col2:
        period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()), index=1)
    with col3:
        interval_label = st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()))

    fetch_clicked = st.button("\U0001F50D Fetch Chart", type="primary")

    state_key = "ai_chart_df"
    if fetch_clicked and ticker:
        with st.spinner(f"Fetching {ticker}..."):
            df = fetch_price_data(ticker, PERIOD_OPTIONS[period_label], INTERVAL_OPTIONS[interval_label])
        if df.empty:
            st.error("No data returned. Check the symbol and try again.")
            st.session_state.pop(state_key, None)
        else:
            st.session_state[state_key] = {
                "df": df, "ticker": ticker, "asset_label": asset_label,
                "interval_label": interval_label,
            }

    if state_key in st.session_state:
        data = st.session_state[state_key]
        df, ticker, asset_label, interval_label = (
            data["df"], data["ticker"], data["asset_label"], data["interval_label"]
        )

        indicators, overlays = compute_indicators(df)

        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else None
        render_price_header(ticker, asset_label, indicators["last_close"], prev_close)

        fig = build_candlestick_chart(df, overlays, f"{asset_label} ({ticker})", indicators.get("fib_levels"))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("\U0001F4CA Full technical snapshot (all indicators)", expanded=True):
            render_indicator_grid(indicators)

        mode_label = st.radio(
            "Analysis mode", list(ANALYSIS_MODES.keys()), horizontal=True, key="single_mode"
        )
        mode = ANALYSIS_MODES[mode_label]

        btn_col, refresh_col = st.columns([2, 1])
        with btn_col:
            btn_label = "\u26A1 Get Insight" if mode == "rule" else "\U0001F916 Get AI Insight"
            get_insight_clicked = st.button(btn_label, type="primary")
        with refresh_col:
            force_refresh = st.checkbox(
                "Force refresh",
                help="Bypass the 1-hour cache and call the AI again even if this exact snapshot was analyzed recently.",
                disabled=(mode == "rule"),
            )

        if get_insight_clicked:
            if mode == "ai" and genai is None:
                st.error("Install the `google-genai` package and add it to requirements.txt, or switch to Free (Rule-based) mode.")
            else:
                spinner_msg = "Running rule-based analysis..." if mode == "rule" else "Analyzing chart with Gemini..."
                with st.spinner(spinner_msg):
                    try:
                        insight = get_insight(mode, ticker, asset_label, indicators, interval_label, force_refresh)
                        st.session_state["ai_chart_insight"] = insight
                        st.session_state["ai_chart_insight_mode"] = mode
                    except Exception as e:
                        st.error(f"Insight generation failed: {e}")

        if "ai_chart_insight" in st.session_state:
            render_recommendation_card(st.session_state["ai_chart_insight"])
            used_mode = st.session_state.get("ai_chart_insight_mode", mode)
            if used_mode == "rule":
                st.caption(
                    f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} \u00b7 Free rule-based engine, no API used \u00b7 "
                    "For informational/educational purposes only, not investment advice."
                )
            else:
                st.caption(
                    f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} \u00b7 Model: {MODEL_NAME} \u00b7 "
                    f"Cached for {AI_CACHE_TTL_SECONDS // 60} min per unique snapshot \u00b7 "
                    "For informational/educational purposes only, not investment advice."
                )
    else:
        st.info("Select an asset and click 'Fetch Chart' to begin.")


# ----------------------------------------------------------------------------
# WATCHLIST SCAN -- run insight across many symbols at once
# ----------------------------------------------------------------------------

def _resolve_watchlist_ticker(raw: str) -> tuple:
    """Returns (yfinance_ticker, display_label) for a user-entered watchlist token."""
    token = raw.strip().upper()
    if not token:
        return None, None
    if token in WATCHLIST_ALIASES:
        return WATCHLIST_ALIASES[token], token.title()
    return f"{token}.NS", token


def _guess_symbol_column(columns) -> str:
    keywords = ["symbol", "ticker", "scrip", "stock", "nse", "name"]
    lower_cols = {c: str(c).strip().lower() for c in columns}
    for kw in keywords:
        for col, low in lower_cols.items():
            if kw in low:
                return col
    return list(columns)[0]


def render_watchlist():
    st.caption(
        "Scan several symbols in one pass. Enter NSE stock symbols, and/or `GOLD` / `SILVER` "
        "for the metals, separated by commas. Each symbol's AI insight is cached for "
        f"{AI_CACHE_TTL_SECONDS // 60} minutes, so re-running the scan on unchanged charts costs no extra API calls."
    )

    with st.expander("\U0001F4C2 Load symbols from your portfolio Excel", expanded=False):
        uploaded = st.file_uploader(
            "Upload your portfolio workbook (.xlsx)", type=["xlsx", "xls", "csv"], key="wl_portfolio_upload"
        )
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    port_df = pd.read_csv(uploaded)
                    sheet_used = None
                else:
                    xls = pd.ExcelFile(uploaded)
                    sheet_names = xls.sheet_names
                    sheet_used = st.selectbox("Sheet", sheet_names, key="wl_portfolio_sheet")
                    port_df = pd.read_excel(xls, sheet_name=sheet_used)

                if port_df.empty:
                    st.warning("That sheet looks empty.")
                else:
                    default_col = _guess_symbol_column(port_df.columns)
                    symbol_col = st.selectbox(
                        "Which column has the stock symbol/name?",
                        list(port_df.columns),
                        index=list(port_df.columns).index(default_col),
                        key="wl_portfolio_symbol_col",
                    )
                    st.dataframe(port_df[[symbol_col]].head(10), use_container_width=True, hide_index=True)

                    include_metals = st.checkbox("Also include GOLD, SILVER", value=True, key="wl_portfolio_include_metals")

                    if st.button("Load symbols into watchlist", key="wl_portfolio_load_btn"):
                        raw_symbols = (
                            port_df[symbol_col]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .str.upper()
                            .str.replace(r"\.NS$", "", regex=True)  # in case symbols already have .NS
                        )
                        symbols = [s for s in dict.fromkeys(raw_symbols) if s]  # de-dupe, preserve order
                        if include_metals:
                            for m in ("GOLD", "SILVER"):
                                if m not in symbols:
                                    symbols.append(m)
                        st.session_state["wl_symbols_text"] = ", ".join(symbols)
                        st.success(f"Loaded {len(symbols)} symbols from '{uploaded.name}'"
                                   + (f" (sheet: {sheet_used})" if sheet_used else "")
                                   + " below.")
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")

    symbols_raw = st.text_area(
        "Watchlist symbols (comma-separated)",
        value=st.session_state.get("wl_symbols_text", "RELIANCE, TCS, INFY, GRSE, GOLD, SILVER"),
        height=70,
        key="wl_symbols_text",
    )
    wl_col1, wl_col2, wl_col3 = st.columns([1, 1, 1])
    with wl_col1:
        period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()), index=1, key="wl_period")
    with wl_col2:
        interval_label = st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()), key="wl_interval")
    with wl_col3:
        force_refresh = st.checkbox("Force refresh all", key="wl_force_refresh")

    mode_label = st.radio(
        "Analysis mode", list(ANALYSIS_MODES.keys()), horizontal=True, key="wl_mode"
    )
    mode = ANALYSIS_MODES[mode_label]
    if mode == "rule":
        st.caption("Free mode: no API calls, no rate limiting needed -- scans run at full speed.")

    run_clicked = st.button("\u25B6\uFE0F Run Watchlist Scan", type="primary")

    if run_clicked:
        tokens = [s for s in symbols_raw.split(",") if s.strip()]
        if not tokens:
            st.warning("Enter at least one symbol.")
        else:
            results = []
            progress = st.progress(0.0, text="Starting scan...")
            for i, raw in enumerate(tokens):
                ticker, label = _resolve_watchlist_ticker(raw)
                progress.progress((i) / len(tokens), text=f"Scanning {label}...")
                row = {"Symbol": label, "Ticker": ticker}
                try:
                    df = fetch_price_data(ticker, PERIOD_OPTIONS[period_label], INTERVAL_OPTIONS[interval_label])
                    if df.empty:
                        row.update({"Error": "No data returned"})
                        results.append(row)
                        continue
                    indicators, _ = compute_indicators(df)
                    insight = get_insight(mode, ticker, label, indicators, interval_label, force_refresh)
                    row.update({
                        "Last Close": indicators.get("last_close"),
                        "Trend": insight.get("trend"),
                        "Recommendation": insight.get("recommendation"),
                        "Confidence": insight.get("confidence"),
                        "Pattern": insight.get("pattern_identified"),
                        "Support": insight.get("key_support"),
                        "Resistance": insight.get("key_resistance"),
                        "Rationale": insight.get("rationale"),
                        "Risk": insight.get("risk_note"),
                    })
                except Exception as e:
                    row.update({"Error": str(e)})
                results.append(row)

                # Pace requests so a long watchlist doesn't burst past API rate limits.
                # No API calls in free/rule mode, so no need to pace at all.
                is_last = (i == len(tokens) - 1)
                if mode == "ai" and not is_last:
                    if (i + 1) % RATE_LIMIT_BATCH_SIZE == 0:
                        progress.progress((i + 1) / len(tokens), text=f"Pausing briefly ({RATE_LIMIT_BATCH_PAUSE_SECONDS}s) to respect API rate limits...")
                        time.sleep(RATE_LIMIT_BATCH_PAUSE_SECONDS)
                    else:
                        time.sleep(RATE_LIMIT_DELAY_SECONDS)
            progress.progress(1.0, text="Scan complete.")
            st.session_state["watchlist_results"] = results
            st.session_state["watchlist_mode"] = mode

    if "watchlist_results" in st.session_state:
        results = st.session_state["watchlist_results"]
        summary_rows = [
            {
                "Symbol": r["Symbol"],
                "Last Close": r.get("Last Close", "-"),
                "Trend": r.get("Trend", r.get("Error", "-")),
                "Recommendation": r.get("Recommendation", "-"),
                "Confidence": r.get("Confidence", "-"),
                "Support": r.get("Support", "-"),
                "Resistance": r.get("Resistance", "-"),
            }
            for r in results
        ]
        summary_df = pd.DataFrame(summary_rows)

        def _color_rec(val):
            color = REC_COLORS.get(str(val).upper())
            return f"color: {color}; font-weight: 700;" if color else ""

        styler = summary_df.style
        # pandas >=2.1 renamed Styler.applymap -> Styler.map (applymap was removed
        # entirely in pandas 3.0). Use whichever this environment has.
        style_fn = getattr(styler, "map", None) or getattr(styler, "applymap")
        st.dataframe(
            style_fn(_color_rec, subset=["Recommendation"]),
            use_container_width=True,
            hide_index=True,
        )

        for r in results:
            if "Error" in r:
                st.warning(f"{r['Symbol']}: {r['Error']}")
                continue
            rec = str(r.get("Recommendation", "HOLD")).upper()
            emoji = REC_EMOJI.get(rec, "")
            with st.expander(f"{emoji} {r['Symbol']} \u2014 {rec} ({r['Confidence']} confidence)"):
                st.write(f"**Pattern:** {r['Pattern']}")
                st.write(f"**Trend:** {r['Trend']} | **Support:** {r['Support']} | **Resistance:** {r['Resistance']}")
                st.write(r["Rationale"])
                st.caption(f"Risk: {r['Risk']}")

        used_mode = st.session_state.get("watchlist_mode", mode)
        if used_mode == "rule":
            st.caption(
                f"Scan run: {datetime.now().strftime('%d %b %Y, %H:%M')} \u00b7 Free rule-based engine, no API used \u00b7 "
                "For informational purposes only, not investment advice."
            )
        else:
            st.caption(
                f"Scan run: {datetime.now().strftime('%d %b %Y, %H:%M')} \u00b7 Model: {MODEL_NAME} \u00b7 "
                "For informational purposes only, not investment advice."
            )
    else:
        st.info("Enter symbols above and click 'Run Watchlist Scan'.")


# ----------------------------------------------------------------------------
# TOP-LEVEL TAB ENTRY POINT -- call this from your app.py
# ----------------------------------------------------------------------------

def render_tab():
    inject_custom_css()
    st.markdown(
        """
        <div class="aci-hero">
            <h1>\U0001F916\U0001F4C8 AI Chart Insights</h1>
            <p>Live price tracking + AI &amp; rule-based technical reads for NSE/BSE stocks, Gold and Silver.
            Educational and research use only \u2014 not investment advice, always do your own due diligence or
            consult a SEBI-registered advisor before acting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sub_tab1, sub_tab2 = st.tabs(["\U0001F4CD Single Asset", "\U0001F4CB Watchlist Scan"])
    with sub_tab1:
        render_single_asset()
    with sub_tab2:
        render_watchlist()
