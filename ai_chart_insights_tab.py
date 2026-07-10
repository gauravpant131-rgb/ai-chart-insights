"""
AI Chart Insights Tab
=====================
A drop-in Streamlit tab that gives AI-generated chart-pattern narratives and
a classified (Bullish / Bearish / Neutral / Watch) recommendation for:
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

3. Add your Anthropic API key to Streamlit secrets (Settings -> Secrets on
   Streamlit Community Cloud, or .streamlit/secrets.toml locally):

       ANTHROPIC_API_KEY = "sk-ant-..."

4. Add to requirements.txt:  anthropic, yfinance, plotly  (pandas/numpy you
   already have).

DESIGN NOTES
------------
- Claude is asked to return strict JSON (pattern, trend, levels, recommendation,
  confidence, rationale, risk_note) so the UI can render a clean recommendation
  badge instead of parsing free text.
- A hard disclaimer is baked into both the system prompt and the UI -- this
  tool is for informational/educational chart reading, not investment advice.
- Model is configurable at the top (MODEL_NAME). Sonnet gives richer
  narratives; swap to Haiku for a faster/cheaper tab if you're calling it a
  lot.
"""

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
except ImportError:
    google_genai = None
    google_genai_types = None


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

MODEL_NAME = "claude-sonnet-5"  # swap to "claude-haiku-4-5-20251001" for speed/cost
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"  # highest free-tier RPM/RPD; swap to "gemini-2.5-flash" for richer narratives
AI_CACHE_TTL_SECONDS = 3600  # don't re-spend API credits re-analyzing the same snapshot within this window
RATE_LIMIT_DELAY_SECONDS = 1.5  # pause between Claude calls in a batch scan
RATE_LIMIT_BATCH_SIZE = 10       # extra pause after every N calls, to stay well under RPM limits
RATE_LIMIT_BATCH_PAUSE_SECONDS = 6
GEMINI_RATE_LIMIT_DELAY_SECONDS = 4.5  # Gemini free tier is ~15 RPM -- stay comfortably under that
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

# Official NSE equity master list -- symbol + full company name, used for name-based search
NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equity/EQUITY_L.csv"
NSE_LIST_CACHE_TTL_SECONDS = 86400  # new listings are infrequent; refresh once a day

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
    "AI Narrative (Gemini - free tier)": "gemini",
    "AI Narrative (Claude API - uses credits)": "ai",
}


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


# ----------------------------------------------------------------------------
# NSE COMPANY NAME LOOKUP -- enables name-based search, e.g. "Garden Reach" -> GRSE
# ----------------------------------------------------------------------------

def _nse_session() -> requests.Session:
    """NSE's archive endpoint rejects bare requests without a browser-like
    session; hitting the homepage first sets the cookies it expects."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass  # best-effort; the CSV request below will fail cleanly if this didn't help
    return session


@st.cache_data(ttl=NSE_LIST_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_nse_company_list() -> pd.DataFrame:
    """Fetches the official NSE-listed equities master list (SYMBOL + NAME).
    Cached for 24h. Returns an empty DataFrame on any failure (network block,
    NSE site change, etc.) so the rest of the app keeps working with plain
    symbol entry -- this feature degrades gracefully rather than breaking anything."""
    try:
        session = _nse_session()
        resp = session.get(NSE_EQUITY_LIST_URL, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        symbol_col = next((c for c in df.columns if c.strip().upper() == "SYMBOL"), None)
        name_col = next((c for c in df.columns if "NAME" in c.strip().upper()), None)
        if not symbol_col or not name_col:
            return pd.DataFrame(columns=["SYMBOL", "NAME"])
        out = df[[symbol_col, name_col]].rename(columns={symbol_col: "SYMBOL", name_col: "NAME"})
        out["SYMBOL"] = out["SYMBOL"].astype(str).str.strip().str.upper()
        out["NAME"] = out["NAME"].astype(str).str.strip()
        out = out[(out["SYMBOL"] != "") & (out["NAME"] != "")]
        return out.drop_duplicates(subset="SYMBOL").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["SYMBOL", "NAME"])


def search_nse_companies(query: str, nse_df: pd.DataFrame, limit: int = 15):
    """Returns up to `limit` (SYMBOL, NAME) matches for a free-text query,
    matching against both symbol and company name, best matches first."""
    if nse_df.empty or not query or not query.strip():
        return []
    q = query.strip().upper()
    mask = nse_df["SYMBOL"].str.contains(q, na=False, regex=False) | \
           nse_df["NAME"].str.upper().str.contains(q, na=False, regex=False)
    matches = nse_df[mask].copy()
    if matches.empty:
        return []

    def _rank(row):
        if row["SYMBOL"] == q:
            return 0
        if row["SYMBOL"].startswith(q):
            return 1
        if row["NAME"].upper().startswith(q):
            return 2
        return 3

    matches["_rank"] = matches.apply(_rank, axis=1)
    matches = matches.sort_values(["_rank", "NAME"]).head(limit)
    return list(zip(matches["SYMBOL"], matches["NAME"]))


def resolve_symbol_or_name(token: str, nse_df: pd.DataFrame) -> str:
    """Given a raw watchlist token that might be a symbol OR a company name
    (e.g. from a broker's portfolio export), resolves it to the best-guess
    NSE symbol. Falls back to returning the token unchanged if no match is
    found or the NSE list isn't available -- fully backward compatible with
    plain-symbol input."""
    token_u = token.strip().upper()
    if nse_df.empty or not token_u:
        return token_u

    if (nse_df["SYMBOL"] == token_u).any():
        return token_u  # already a valid symbol

    exact_name = nse_df[nse_df["NAME"].str.upper() == token_u]
    if not exact_name.empty:
        return exact_name.iloc[0]["SYMBOL"]

    contains_name = nse_df[nse_df["NAME"].str.upper().str.contains(token_u, na=False, regex=False)]
    if not contains_name.empty:
        return contains_name.iloc[0]["SYMBOL"]

    return token_u  # unrecognized -- leave as-is, same as prior behavior


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

    lookback = min(60, len(df))
    recent = df.tail(lookback)
    swing_high = recent["High"].max()
    swing_low = recent["Low"].min()

    last_close = float(close.iloc[-1])
    vol_avg20 = float(vol.rolling(20).mean().iloc[-1]) if not vol.empty else None
    vol_last = float(vol.iloc[-1]) if not vol.empty else None

    return {
        "last_close": round(last_close, 2),
        "sma20": round(float(sma20.iloc[-1]), 2) if not np.isnan(sma20.iloc[-1]) else None,
        "sma50": round(float(sma50.iloc[-1]), 2) if not np.isnan(sma50.iloc[-1]) else None,
        "sma200": round(float(sma200.iloc[-1]), 2) if len(df) >= 200 and not np.isnan(sma200.iloc[-1]) else None,
        "rsi14": round(float(rsi14.iloc[-1]), 1) if not np.isnan(rsi14.iloc[-1]) else None,
        "macd": round(float(macd.iloc[-1]), 3) if not np.isnan(macd.iloc[-1]) else None,
        "macd_signal": round(float(signal.iloc[-1]), 3) if not np.isnan(signal.iloc[-1]) else None,
        "bb_upper": round(float(bb_upper.iloc[-1]), 2) if not np.isnan(bb_upper.iloc[-1]) else None,
        "bb_lower": round(float(bb_lower.iloc[-1]), 2) if not np.isnan(bb_lower.iloc[-1]) else None,
        "atr14": round(float(atr14.iloc[-1]), 2) if not np.isnan(atr14.iloc[-1]) else None,
        "swing_high_60d": round(float(swing_high), 2),
        "swing_low_60d": round(float(swing_low), 2),
        "vol_last": vol_last,
        "vol_avg20": vol_avg20,
    }, {"sma20": sma20, "sma50": sma50, "bb_upper": bb_upper, "bb_lower": bb_lower}


# ----------------------------------------------------------------------------
# FREE RULE-BASED ANALYSIS -- no API key, no cost, works forever
# ----------------------------------------------------------------------------

def rule_based_insight(indicators: dict) -> dict:
    """Produces the exact same schema as the AI insight, but derived entirely
    from coded technical rules -- zero API calls, zero cost."""
    last_close = indicators.get("last_close")
    sma20 = indicators.get("sma20")
    sma50 = indicators.get("sma50")
    rsi = indicators.get("rsi14")
    macd = indicators.get("macd")
    macd_signal = indicators.get("macd_signal")
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
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

    directions = [d for _, d in signals if d != 0]
    score = sum(directions)

    if score >= 2:
        recommendation = "Bullish"
    elif score <= -2:
        recommendation = "Bearish"
    elif score == 0 and directions:
        recommendation = "Neutral"
    else:
        recommendation = "Watch"

    if directions:
        pos_count = sum(1 for d in directions if d > 0)
        neg_count = sum(1 for d in directions if d < 0)
        total = len(directions)
        if pos_count == total or neg_count == total:
            confidence = "High"
        elif abs(pos_count - neg_count) >= 1:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        confidence = "Low"

    pattern_bits = [desc for desc, _ in signals]
    pattern_identified = "; ".join(pattern_bits) if pattern_bits else "No clear pattern - insufficient data"
    rationale = "Rule-based technical read: " + "; ".join(pattern_bits) + "."

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
    }


# ----------------------------------------------------------------------------
# CHART
# ----------------------------------------------------------------------------

def build_candlestick_chart(df: pd.DataFrame, overlays: dict, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#0B6623", decreasing_line_color="#B22222",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=overlays["sma20"], name="SMA 20",
                              line=dict(color="#1f77b4", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=overlays["sma50"], name="SMA 50",
                              line=dict(color="#ff7f0e", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=overlays["bb_upper"], name="BB Upper",
                              line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=overlays["bb_lower"], name="BB Lower",
                              line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(150,150,150,0.07)"))
    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False, height=520,
        margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ----------------------------------------------------------------------------
# CLAUDE CALL
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a technical analysis assistant embedded in a personal investing \
dashboard for an experienced Indian retail investor. You read price/indicator data and \
describe the chart pattern and technical setup in plain, precise language.

Rules:
- Base your read strictly on the numeric data given. Do not invent price history you weren't given.
- Identify classical chart/technical patterns only if the data genuinely supports them \
(e.g. uptrend/downtrend, consolidation, breakout above resistance, oversold/overbought, \
moving average crossover, Bollinger squeeze). Do not force a pattern name if none fits -- \
say the structure is unclear or range-bound instead.
- Your recommendation field is a technical-stance classification, not investment advice. \
Always include a risk_note.
- Return ONLY valid JSON, no markdown fences, no preamble, matching exactly this schema:

{
  "pattern_identified": "string, e.g. 'Ascending triangle near resistance' or 'No clear pattern - range-bound'",
  "trend": "Uptrend | Downtrend | Sideways/Range-bound",
  "key_support": number or null,
  "key_resistance": number or null,
  "recommendation": "Bullish | Bearish | Neutral | Watch",
  "confidence": "High | Medium | Low",
  "rationale": "2-4 sentences explaining the read, referencing the specific indicator values given",
  "risk_note": "1-2 sentences on what would invalidate this read or key risk"
}
"""


def call_claude_for_insight(symbol: str, asset_label: str, indicators: dict, interval: str) -> dict:
    if anthropic is None:
        raise RuntimeError("The 'anthropic' package is not installed. Add it to requirements.txt.")

    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in st.secrets or environment.")

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = f"""Asset: {asset_label} ({symbol})
Timeframe: {interval} candles

Latest technical snapshot:
{json.dumps(indicators, indent=2)}

Give your read as JSON per the schema."""

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL_NAME,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(block.text for block in resp.content if block.type == "text").strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except anthropic.RateLimitError as e:
            last_error = e
            wait = (2 ** attempt) * RATE_LIMIT_DELAY_SECONDS
            time.sleep(wait)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_error = e
            time.sleep(RATE_LIMIT_DELAY_SECONDS)

    raise RuntimeError(f"AI call failed after {MAX_RETRIES} attempts: {last_error}")


@st.cache_data(ttl=AI_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_ai_insight(symbol: str, asset_label: str, indicators_json: str, interval: str, model_name: str) -> dict:
    """Cache wrapper keyed on the actual indicator values (as a JSON string, so it's
    hashable). Identical indicator snapshots within AI_CACHE_TTL_SECONDS reuse the
    previous Claude response instead of paying for a fresh API call. A genuinely new
    price snapshot (different indicators) produces a different cache key automatically."""
    indicators = json.loads(indicators_json)
    return call_claude_for_insight(symbol, asset_label, indicators, interval)


def get_ai_insight(symbol: str, asset_label: str, indicators: dict, interval: str, force_refresh: bool = False) -> dict:
    indicators_json = json.dumps(indicators, sort_keys=True)
    if force_refresh:
        _cached_ai_insight.clear()
    return _cached_ai_insight(symbol, asset_label, indicators_json, interval, MODEL_NAME)


# ----------------------------------------------------------------------------
# GEMINI CALL (free tier alternative to Claude)
# ----------------------------------------------------------------------------

def call_gemini_for_insight(symbol: str, asset_label: str, indicators: dict, interval: str) -> dict:
    if google_genai is None:
        raise RuntimeError("The 'google-genai' package is not installed. Add it to requirements.txt.")

    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in st.secrets or environment.")

    client = google_genai.Client(api_key=api_key)

    user_prompt = f"""Asset: {asset_label} ({symbol})
Timeframe: {interval} candles

Latest technical snapshot:
{json.dumps(indicators, indent=2)}

Give your read as JSON per the schema."""

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=user_prompt,
                config=google_genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    max_output_tokens=600,
                ),
            )
            text = (resp.text or "").strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "429" in msg or "resource_exhausted" in msg or "rate" in msg:
                wait = (2 ** attempt) * RATE_LIMIT_DELAY_SECONDS
                time.sleep(wait)
            else:
                time.sleep(RATE_LIMIT_DELAY_SECONDS)

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")


@st.cache_data(ttl=AI_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_gemini_insight(symbol: str, asset_label: str, indicators_json: str, interval: str, model_name: str) -> dict:
    indicators = json.loads(indicators_json)
    return call_gemini_for_insight(symbol, asset_label, indicators, interval)


def get_gemini_insight(symbol: str, asset_label: str, indicators: dict, interval: str, force_refresh: bool = False) -> dict:
    indicators_json = json.dumps(indicators, sort_keys=True)
    if force_refresh:
        _cached_gemini_insight.clear()
    return _cached_gemini_insight(symbol, asset_label, indicators_json, interval, GEMINI_MODEL_NAME)


def get_insight(mode: str, symbol: str, asset_label: str, indicators: dict, interval: str, force_refresh: bool = False) -> dict:
    """Single entry point used by the UI. mode is 'rule', 'gemini', or 'ai' (see ANALYSIS_MODES)."""
    if mode == "rule":
        return rule_based_insight(indicators)
    if mode == "gemini":
        return get_gemini_insight(symbol, asset_label, indicators, interval, force_refresh)
    return get_ai_insight(symbol, asset_label, indicators, interval, force_refresh)


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------

REC_COLORS = {
    "Bullish": "#0B6623",
    "Bearish": "#B22222",
    "Neutral": "#8a8a3c",
    "Watch": "#B8860B",
}


def render_recommendation_card(insight: dict):
    rec = insight.get("recommendation", "Watch")
    color = REC_COLORS.get(rec, "#555555")
    st.markdown(
        f"""
        <div style="border-left: 6px solid {color}; padding: 12px 16px;
                    background: rgba(0,0,0,0.03); border-radius: 6px; margin-bottom: 10px;">
            <span style="font-size: 1.1em; font-weight: 700; color: {color};">
                {rec}
            </span>
            <span style="margin-left: 10px; color: #666;">
                Confidence: {insight.get('confidence', 'n/a')}
            </span>
            <div style="margin-top: 8px; font-weight: 600;">
                {insight.get('pattern_identified', '')}
            </div>
            <div style="margin-top: 4px; color: #333;">
                Trend: {insight.get('trend', 'n/a')} &nbsp;|&nbsp;
                Support: {insight.get('key_support', 'n/a')} &nbsp;|&nbsp;
                Resistance: {insight.get('key_resistance', 'n/a')}
            </div>
            <div style="margin-top: 8px;">{insight.get('rationale', '')}</div>
            <div style="margin-top: 8px; font-style: italic; color: #777; font-size: 0.9em;">
                Risk: {insight.get('risk_note', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# MAIN TAB RENDER FUNCTION -- call this from your app.py
# ----------------------------------------------------------------------------

def render_single_asset():
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        asset_type = st.selectbox("Asset type", list(ASSET_PRESETS.keys()))
        if ASSET_PRESETS[asset_type] is None:
            entry_mode = st.radio(
                "Enter by", ["Symbol", "Company Name"], horizontal=True, key="single_entry_mode"
            )
            if entry_mode == "Symbol":
                symbol_input = st.text_input(
                    "NSE symbol (e.g. RELIANCE, TCS, GRSE)", value="GRSE"
                ).strip().upper()
                ticker = f"{symbol_input}.NS" if symbol_input else ""
                asset_label = symbol_input
            else:
                nse_df = fetch_nse_company_list()
                if nse_df.empty:
                    st.warning("Couldn't load the NSE company list right now -- switch to 'Symbol' entry instead.")
                    ticker, asset_label = "", ""
                else:
                    query = st.text_input(
                        "Search company name (e.g. 'Garden Reach', 'Tata Consultancy')",
                        key="single_name_query",
                    )
                    matches = search_nse_companies(query, nse_df)
                    if matches:
                        options = [f"{sym} — {name}" for sym, name in matches]
                        choice = st.selectbox("Matching companies", options, key="single_name_choice")
                        chosen_symbol = choice.split(" — ")[0]
                        ticker = f"{chosen_symbol}.NS"
                        asset_label = chosen_symbol
                    elif query:
                        st.info("No matches found. Try a different search term or switch to 'Symbol' entry.")
                        ticker, asset_label = "", ""
                    else:
                        ticker, asset_label = "", ""
        else:
            ticker = ASSET_PRESETS[asset_type]
            asset_label = asset_type
            st.text_input("Ticker (auto)", value=ticker, disabled=True)

    with col2:
        period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()), index=1)
    with col3:
        interval_label = st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()))

    fetch_clicked = st.button("Fetch Chart", type="primary")

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
        fig = build_candlestick_chart(df, overlays, f"{asset_label} ({ticker})")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Latest technical snapshot"):
            st.json(indicators)

        mode_label = st.radio(
            "Analysis mode", list(ANALYSIS_MODES.keys()), horizontal=True, key="single_mode"
        )
        mode = ANALYSIS_MODES[mode_label]

        btn_col, refresh_col = st.columns([2, 1])
        with btn_col:
            btn_label = "Get Insight" if mode == "rule" else "Get AI Insight"
            get_insight_clicked = st.button(btn_label, type="primary")
        with refresh_col:
            force_refresh = st.checkbox(
                "Force refresh",
                help="Bypass the 1-hour cache and call the AI again even if this exact snapshot was analyzed recently.",
                disabled=(mode == "rule"),
            )

        if get_insight_clicked:
            if mode == "ai" and anthropic is None:
                st.error("Install the `anthropic` package and add it to requirements.txt, or switch modes.")
            elif mode == "gemini" and google_genai is None:
                st.error("Install the `google-genai` package and add it to requirements.txt, or switch modes.")
            else:
                spinner_msg = {
                    "rule": "Running rule-based analysis...",
                    "gemini": "Analyzing chart with Gemini...",
                    "ai": "Analyzing chart with Claude...",
                }[mode]
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
                    f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} · Free rule-based engine, no API used · "
                    "For informational purposes only, not investment advice."
                )
            elif used_mode == "gemini":
                st.caption(
                    f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} · Model: {GEMINI_MODEL_NAME} (free tier) · "
                    f"Cached for {AI_CACHE_TTL_SECONDS // 60} min per unique snapshot · "
                    "For informational purposes only, not investment advice."
                )
            else:
                st.caption(
                    f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} · Model: {MODEL_NAME} · "
                    f"Cached for {AI_CACHE_TTL_SECONDS // 60} min per unique snapshot · "
                    "For informational purposes only, not investment advice."
                )
    else:
        st.info("Select an asset and click 'Fetch Chart' to begin.")


# ----------------------------------------------------------------------------
# WATCHLIST SCAN -- run AI insight across many symbols at once
# ----------------------------------------------------------------------------

def _resolve_watchlist_ticker(raw: str, nse_df: pd.DataFrame = None) -> tuple:
    """Returns (yfinance_ticker, display_label) for a user-entered watchlist token.
    Accepts a plain NSE symbol (unchanged, original behavior), GOLD/SILVER aliases,
    or -- if nse_df is supplied -- a company name (e.g. 'Garden Reach Shipbuilders'),
    which gets resolved to its NSE symbol via resolve_symbol_or_name()."""
    token = raw.strip().upper()
    if not token:
        return None, None
    if token in WATCHLIST_ALIASES:
        return WATCHLIST_ALIASES[token], token.title()
    if nse_df is not None and not nse_df.empty:
        resolved = resolve_symbol_or_name(token, nse_df)
        return f"{resolved}.NS", resolved
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

    with st.expander("📂 Load symbols from your portfolio Excel", expanded=False):
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

    run_clicked = st.button("Run Watchlist Scan", type="primary")

    if run_clicked:
        tokens = [s for s in symbols_raw.split(",") if s.strip()]
        if not tokens:
            st.warning("Enter at least one symbol.")
        else:
            nse_df = fetch_nse_company_list()
            if nse_df.empty:
                st.caption("⚠️ Couldn't load the NSE company-name list right now -- symbols/names will be used as typed.")
            results = []
            progress = st.progress(0.0, text="Starting scan...")
            for i, raw in enumerate(tokens):
                ticker, label = _resolve_watchlist_ticker(raw, nse_df)
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
                if mode == "gemini" and not is_last:
                    time.sleep(GEMINI_RATE_LIMIT_DELAY_SECONDS)
                elif mode == "ai" and not is_last:
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
            color = REC_COLORS.get(val)
            return f"color: {color}; font-weight: 700;" if color else ""

        st.dataframe(
            summary_df.style.applymap(_color_rec, subset=["Recommendation"]),
            use_container_width=True,
            hide_index=True,
        )

        for r in results:
            if "Error" in r:
                st.warning(f"{r['Symbol']}: {r['Error']}")
                continue
            with st.expander(f"{r['Symbol']} — {r['Recommendation']} ({r['Confidence']} confidence)"):
                st.write(f"**Pattern:** {r['Pattern']}")
                st.write(f"**Trend:** {r['Trend']} | **Support:** {r['Support']} | **Resistance:** {r['Resistance']}")
                st.write(r["Rationale"])
                st.caption(f"Risk: {r['Risk']}")

        used_mode = st.session_state.get("watchlist_mode", mode)
        if used_mode == "rule":
            st.caption(
                f"Scan run: {datetime.now().strftime('%d %b %Y, %H:%M')} · Free rule-based engine, no API used · "
                "For informational purposes only, not investment advice."
            )
        else:
            st.caption(
                f"Scan run: {datetime.now().strftime('%d %b %Y, %H:%M')} · Model: {MODEL_NAME} · "
                "For informational purposes only, not investment advice."
            )
    else:
        st.info("Enter symbols above and click 'Run Watchlist Scan'.")


# ----------------------------------------------------------------------------
# TOP-LEVEL TAB ENTRY POINT -- call this from your app.py
# ----------------------------------------------------------------------------

def render_tab():
    st.subheader("🤖 AI Chart Insights")
    st.caption(
        "AI-generated technical reads for stocks, gold and silver charts. "
        "This is informational chart analysis, not investment advice -- always do your own "
        "due diligence and consider consulting a SEBI-registered advisor before acting."
    )

    sub_tab1, sub_tab2 = st.tabs(["Single Asset", "Watchlist Scan"])
    with sub_tab1:
        render_single_asset()
    with sub_tab2:
        render_watchlist()
