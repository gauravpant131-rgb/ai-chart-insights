"""
AI Chart Insights -- Standalone App
====================================
Entry point for running this as its own Streamlit app (separate from your
main ratings dashboard). Deploy this file as the "Main file path" on
Streamlit Community Cloud.
"""

import streamlit as st
from ai_chart_insights_tab import render_tab

st.set_page_config(
    page_title="AI Chart Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_tab()
