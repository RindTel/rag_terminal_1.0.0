"""
RAG - Local RAG application using Qwen 2.5 7B 
Main Streamlit application — retro CRT terminal aesthetic.
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.ui import render_app

st.set_page_config(
    page_title="TERMINAL",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CRT 
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&display=swap');

/* ── Root variables ── */
:root {
    --phosphor:      #33ff33;
    --phosphor-dim:  #1a7a1a;
    --phosphor-glow: #00ff00;
    --phosphor-dark: #0d1f0d;
    --bg:            #050a05;
    --bg-panel:      #070e07;
    --bg-inset:      #040804;
    --amber:         #ffb000;
    --amber-dim:     #7a5500;
    --red-crt:       #ff3333;
    --border-crt:    #1a4d1a;
    --scanline:      rgba(0,0,0,0.35);
    --font-mono:     'Share Tech Mono', 'Courier New', monospace;
    --font-display:  'VT323', monospace;
}

/* ── Global reset to terminal feel ── */
html, body, [class*="css"] {
    font-family: var(--font-mono) !important;
    background-color: var(--bg) !important;
    color: var(--phosphor) !important;
}

.stApp {
    background-color: var(--bg) !important;
    background-image: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        var(--scanline) 2px,
        var(--scanline) 4px
    ) !important;
}

/* ── CRT animations ── */
@keyframes flicker {
    0%   { opacity: 1; }
    92%  { opacity: 1; }
    93%  { opacity: 0.94; }
    94%  { opacity: 1; }
    96%  { opacity: 0.97; }
    100% { opacity: 1; }
}
@keyframes cursor-blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
@keyframes text-flicker {
    0%,100% { text-shadow: 0 0 8px var(--phosphor-glow), 0 0 2px var(--phosphor-glow); }
    50%      { text-shadow: 0 0 4px var(--phosphor-glow); }
}
@keyframes boot-in {
    from { opacity: 0; transform: scaleY(0.02); filter: brightness(3); }
    to   { opacity: 1; transform: scaleY(1);    filter: brightness(1); }
}

.main .block-container { animation: flicker 8s infinite; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-panel) !important;
    border-right: 1px solid var(--border-crt) !important;
    background-image: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.4) 2px, rgba(0,0,0,0.4) 4px
    ) !important;
}
section[data-testid="stSidebar"] * { color: var(--phosphor) !important; }

/* ── Header ── */
.crt-header {
    padding: 24px 0 18px 0;
    border-bottom: 1px solid var(--border-crt);
    margin-bottom: 24px;
    animation: boot-in 0.6s ease-out;
}
.crt-title {
    font-family: var(--font-display) !important;
    font-size: 2.8rem !important;
    color: var(--phosphor) !important;
    letter-spacing: 0.08em;
    text-shadow: 0 0 10px var(--phosphor-glow), 0 0 30px rgba(0,255,0,0.3);
    margin: 0 !important; padding: 0 !important;
    animation: text-flicker 4s infinite;
}
.crt-subtitle {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--phosphor-dim);
    letter-spacing: 0.15em;
    margin-top: 4px;
}
.crt-subtitle .cursor {
    display: inline-block;
    animation: cursor-blink 1s step-end infinite;
    color: var(--phosphor);
}

/* ── Section labels ── */
.crt-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--phosphor-dim);
    margin: 18px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.crt-label::before { content: '>'; color: var(--phosphor); margin-right: 2px; }
.crt-label::after  { content: ''; flex: 1; height: 1px; background: var(--border-crt); }

/* ── Status pills ── */
.crt-pill {
    display: inline-block;
    padding: 2px 10px;
    border: 1px solid var(--border-crt);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    background: var(--bg-inset);
    margin-right: 6px;
    margin-bottom: 4px;
}
.crt-pill-ok   { color: var(--phosphor); border-color: var(--phosphor-dim); }
.crt-pill-warn { color: var(--amber);    border-color: var(--amber-dim);    }
.crt-pill-err  { color: var(--red-crt);  border-color: #7a0000;             }

/* ── Chat messages ── */
.msg-user, .msg-assistant {
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin-bottom: 18px;
}
.msg-role-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: var(--phosphor-dim);
    text-transform: uppercase;
}
.bubble-user {
    background: var(--bg-inset);
    border: 1px solid var(--amber-dim);
    border-left: 3px solid var(--amber);
    padding: 10px 14px;
    font-family: var(--font-mono);
    font-size: 0.88rem;
    color: var(--amber);
    line-height: 1.6;
    text-shadow: 0 0 4px rgba(255,176,0,0.4);
}
.bubble-user::before {
    content: 'C:\\USER> ';
    color: var(--phosphor-dim);
    font-size: 0.78rem;
    margin-right: 4px;
}
.bubble-assistant {
    background: var(--bg-inset);
    border: 1px solid var(--border-crt);
    border-left: 3px solid var(--phosphor);
    padding: 10px 14px;
    font-family: var(--font-mono);
    font-size: 0.88rem;
    color: var(--phosphor);
    line-height: 1.7;
    text-shadow: 0 0 6px rgba(0,255,0,0.25);
}

/* ── Source blocks ── */
.source-crt {
    background: var(--bg-inset);
    border: 1px dashed var(--border-crt);
    padding: 8px 12px;
    margin-top: 6px;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--phosphor-dim);
    line-height: 1.5;
}
.source-crt-header {
    color: var(--phosphor);
    font-size: 0.70rem;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
    text-transform: uppercase;
}
.source-crt-header::before { content: '// '; color: var(--phosphor-dim); }

/* ── File card ── */
.file-crt {
    background: var(--bg-inset);
    border: 1px solid var(--border-crt);
    padding: 7px 12px;
    margin-bottom: 6px;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--phosphor);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.file-crt:hover { border-color: var(--phosphor); }
.file-crt-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 170px;
}
.file-crt-name::before { content: '> '; color: var(--phosphor-dim); }
.file-crt-size { color: var(--phosphor-dim); font-size: 0.68rem; }

/* ── Empty state ── */
.crt-empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--phosphor-dim);
}
.crt-empty-art {
    font-family: var(--font-display);
    font-size: 5rem;
    color: var(--border-crt);
    margin-bottom: 8px;
    display: block;
}
.crt-empty-text {
    font-size: 0.82rem;
    letter-spacing: 0.08em;
}
.blink { animation: cursor-blink 1.2s step-end infinite; }

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: var(--bg-inset) !important;
    border: 1px solid var(--border-crt) !important;
    border-radius: 0 !important;
    color: var(--phosphor) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
    caret-color: var(--phosphor) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--phosphor) !important;
    box-shadow: 0 0 8px rgba(0,255,0,0.2) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--phosphor-dim) !important; opacity: 0.6 !important; }
.stTextInput label, .stTextArea label {
    color: var(--phosphor-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
}

.stButton > button {
    background: var(--bg-inset) !important;
    border: 1px solid var(--border-crt) !important;
    border-radius: 0 !important;
    color: var(--phosphor) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.1s !important;
}
.stButton > button:hover {
    background: var(--border-crt) !important;
    color: var(--phosphor-glow) !important;
    text-shadow: 0 0 6px var(--phosphor-glow) !important;
}
.stButton > button[kind="primary"] {
    border-color: var(--phosphor) !important;
    color: var(--phosphor) !important;
    text-shadow: 0 0 6px rgba(0,255,0,0.5) !important;
}

.stFileUploader {
    background: var(--bg-inset) !important;
    border: 1px dashed var(--border-crt) !important;
    border-radius: 0 !important;
}
.stFileUploader label, .stFileUploader small, .stFileUploader p {
    color: var(--phosphor-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

.stSelectbox > div > div {
    background-color: var(--bg-inset) !important;
    border: 1px solid var(--border-crt) !important;
    border-radius: 0 !important;
    color: var(--phosphor) !important;
    font-family: var(--font-mono) !important;
}
.stSelectbox label { color: var(--phosphor-dim) !important; font-family: var(--font-mono) !important; font-size: 0.72rem !important; }
.stSlider label    { color: var(--phosphor-dim) !important; font-family: var(--font-mono) !important; font-size: 0.72rem !important; }

.stAlert, div[data-testid="stNotification"] {
    background: var(--bg-inset) !important;
    border: 1px solid var(--border-crt) !important;
    border-radius: 0 !important;
    color: var(--phosphor) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.80rem !important;
}

.streamlit-expanderHeader {
    background: var(--bg-inset) !important;
    border: 1px solid var(--border-crt) !important;
    border-radius: 0 !important;
    color: var(--phosphor-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.76rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-inset) !important;
    border: 1px solid var(--border-crt) !important;
    border-top: none !important;
}

.stProgress > div > div > div { background: var(--phosphor) !important; }
.stProgress > div > div       { background: var(--border-crt) !important; }
.stSpinner > div { border-top-color: var(--phosphor) !important; }
.stCaption, small { color: var(--phosphor-dim) !important; font-family: var(--font-mono) !important; font-size: 0.70rem !important; }
.stToast { background: var(--bg-panel) !important; border: 1px solid var(--border-crt) !important; border-radius: 0 !important; font-family: var(--font-mono) !important; color: var(--phosphor) !important; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
h1, h2, h3, h4 { font-family: var(--font-display) !important; color: var(--phosphor) !important; text-shadow: 0 0 8px rgba(0,255,0,0.3) !important; letter-spacing: 0.05em !important; }
code, pre { font-family: var(--font-mono) !important; background: var(--bg-inset) !important; color: var(--phosphor) !important; border: 1px solid var(--border-crt) !important; border-radius: 0 !important; }

* { scrollbar-width: thin; scrollbar-color: var(--border-crt) var(--bg); }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-crt); }
::-webkit-scrollbar-thumb:hover { background: var(--phosphor-dim); }
</style>
""", unsafe_allow_html=True)

render_app()
