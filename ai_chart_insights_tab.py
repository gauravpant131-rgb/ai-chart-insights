import json
import os
import streamlit as st
import yfinance as yf
import google.generativeai as genai

# --- CONFIG & STYLE ---
MODEL_NAME = "gemini-1.5-flash"

# CSS for a modern, elegant look
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .stButton>button { border-radius: 20px; font-weight: bold; background-color: #4a90e2; color: white; }
    .insight-card { 
        padding: 20px; border-radius: 15px; border-left: 8px solid #4a90e2; 
        background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- GEMINI INTEGRATION ---
def call_gemini(indicators):
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key: return {"error": "API Key missing"}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"Analyze this market data: {json.dumps(indicators)}. Return JSON with fields: pattern_identified, trend, recommendation (Bullish/Bearish/Neutral), confidence, rationale, risk_note."
    
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

# --- CORE LOGIC ---
@st.cache_data(ttl=900)
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
    return df.dropna() if not df.empty else df

def compute_indicators(df):
    close = df["Close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    return {"last_close": float(close.iloc[-1]), "sma20": float(sma20), "sma50": float(sma50)}

# --- UI ---
def render_tab():
    st.title("📈 AI Market Insight Pro")
    st.markdown("---")
    
    ticker = st.text_input("Enter NSE Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
    btn = st.button("🚀 Analyze Now")

    if btn:
        with st.spinner("Gemini is analyzing..."):
            df = fetch_data(ticker)
            if not df.empty:
                ind = compute_indicators(df)
                res = call_gemini(ind)
                
                st.markdown(f'<div class="insight-card">', unsafe_allow_html=True)
                st.subheader(f"Analysis for {ticker}")
                st.metric("Recommendation", res.get("recommendation", "N/A"))
                st.write(f"**Pattern:** {res.get('pattern_identified')}")
                st.write(f"**Rationale:** {res.get('rationale')}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Invalid Ticker or No Data.")