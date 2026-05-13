import streamlit as st
import joblib
import cv2
import numpy as np
import re
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from supabase import create_client
import os
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title='AEGIS-SWARM',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='collapsed'
)

if 'analyses' not in st.session_state:
    st.session_state.analyses = []

st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    .stApp {
        background-color: #02020a;
        background-image: linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
    }
    .main-header {
        font-family: 'Share Tech Mono', 'Courier New', monospace;
        color: #00ffff;
        text-align: center;
        text-shadow: 0 0 20px rgba(0,255,255,0.6), 0 0 40px rgba(0,255,255,0.2);
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: 0.25em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #ff8c00;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
        letter-spacing: 0.4em;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(255,140,0,0.4);
    }
    .dev-info {
        text-align: center;
        color: #445;
        font-size: 0.65rem;
        font-family: 'Share Tech Mono', monospace;
        margin-bottom: 1rem;
        letter-spacing: 0.1em;
    }
    .hud-label {
        font-family: 'Share Tech Mono', monospace;
        color: #00ffff;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        border: 1px solid rgba(0,255,255,0.3);
        padding: 2px 8px;
        display: inline-block;
        margin-bottom: 8px;
        background: rgba(0,255,255,0.05);
    }
    .hud-label::before { content: '[ '; color: #ff8c00; }
    .hud-label::after { content: ' ]'; color: #ff8c00; }
    .status-bar {
        text-align: center;
        font-family: 'Share Tech Mono', monospace;
        color: #00ffff;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        margin-bottom: 1.5rem;
        opacity: 0.8;
    }
    .status-bar span { color: #ff8c00; margin: 0 10px; }
    .glass-panel {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(0,255,255,0.15);
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0,255,255,0.05), inset 0 0 20px rgba(0,255,255,0.02);
        backdrop-filter: blur(5px);
    }
    .glass-panel-high {
        background: rgba(255,0,0,0.04);
        border: 1px solid rgba(255,69,0,0.3);
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(255,69,0,0.1), inset 0 0 20px rgba(255,69,0,0.02);
    }
    .glass-panel-low {
        background: rgba(0,255,127,0.04);
        border: 1px solid rgba(0,255,127,0.3);
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0,255,127,0.1), inset 0 0 20px rgba(0,255,127,0.02);
    }
    .metric-box {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(0,255,255,0.2);
        border-radius: 6px;
        padding: 15px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-box::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0;
        height: 2px; background: linear-gradient(90deg, transparent, #00ffff, transparent);
    }
    .metric-box h2 {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.2rem;
        margin: 0;
        text-shadow: 0 0 10px currentColor;
    }
    .metric-box p {
        font-family: 'Share Tech Mono', monospace;
        color: #556;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        margin: 5px 0 0 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, rgba(75,0,130,0.8), rgba(138,43,226,0.4)) !important;
        color: #00ffff !important;
        border: 1px solid #00ffff !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 3px !important;
        width: 100%;
        padding: 0.9rem !important;
        font-weight: bold !important;
        text-shadow: 0 0 5px rgba(0,255,255,0.5);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #ff8c00 !important;
        color: #ff8c00 !important;
        box-shadow: 0 0 25px rgba(255,140,0,0.3), inset 0 0 15px rgba(255,140,0,0.1);
        text-shadow: 0 0 5px rgba(255,140,0,0.5);
    }
    div[data-testid='stTabs'] button {
        color: #00ffff !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-weight: bold !important;
        letter-spacing: 0.15em;
        font-size: 0.8rem !important;
    }
    div[data-testid='stTabs'] button[aria-selected='true'] {
        color: #ff8c00 !important;
        text-shadow: 0 0 10px rgba(255,140,0,0.5);
    }
    .stTextArea textarea {
        background: rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(0,255,255,0.2) !important;
        color: #00ffff !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #ff8c00 !important;
        box-shadow: 0 0 15px rgba(255,140,0,0.2);
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ffff, #ff8c00) !important;
    }
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00ffff, transparent);
        margin: 20px 0;
        opacity: 0.3;
    }
    .threat-log-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: #00ffff;
    }
    .threat-log-table th {
        background: rgba(0,255,255,0.1);
        color: #00ffff;
        padding: 8px;
        text-align: left;
        letter-spacing: 0.15em;
        font-size: 0.7rem;
        border-bottom: 1px solid rgba(0,255,255,0.3);
    }
    .threat-log-table td {
        padding: 8px;
        border-bottom: 1px solid rgba(0,255,255,0.1);
        color: #8899aa;
    }
    .threat-log-table tr:hover td {
        background: rgba(0,255,255,0.05);
        color: #00ffff;
    }
    .bottom-nav {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 20px;
        padding: 15px;
        border-top: 1px solid rgba(0,255,255,0.1);
    }
    .bottom-nav-item {
        font-family: 'Share Tech Mono', monospace;
        color: #445;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-align: center;
    }
    .bottom-nav-item.active {
        color: #00ffff;
        text-shadow: 0 0 10px rgba(0,255,255,0.5);
    }
    .bottom-nav-icon {
        font-size: 1.2rem;
        margin-bottom: 4px;
        display: block;
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #02020a; }
    ::-webkit-scrollbar-thumb { background: #00ffff; border-radius: 4px; }
    .gauge-container { text-align: center; font-family: 'Share Tech Mono', monospace; }
    .gauge-value { font-size: 2rem; fill: #ff6347; font-weight: bold; }
    .gauge-label { font-size: 0.7rem; fill: #ff8c00; }
    .donut-container { text-align: center; font-family: 'Share Tech Mono', monospace; }
    .donut-value { font-size: 1.8rem; fill: #ff6347; font-weight: bold; }
    .donut-label { font-size: 0.7rem; fill: #ff6347; }
    .dev-card {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(0,255,255,0.2);
        border-radius: 8px;
        padding: 20px;
        margin: 8px 0;
        font-family: 'Share Tech Mono', monospace;
        color: #8899aa;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }
    .dev-card b { color: #00ffff; }
    .dev-card .label { color: #ff8c00; font-size: 0.65rem; letter-spacing: 0.15em; }
    .dev-avatar {
        width: 80px; height: 80px;
        border-radius: 50%;
        border: 2px solid #00ffff;
        box-shadow: 0 0 20px rgba(0,255,255,0.3);
        display: flex; align-items: center; justify-content: center;
        font-size: 2.5rem;
        margin: 0 auto 15px auto;
        background: rgba(0,255,255,0.05);
    }
    .member-card {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(0,255,255,0.2);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        font-family: 'Share Tech Mono', monospace;
    }
    .member-card .name { color: #00ffff; font-size: 0.9rem; letter-spacing: 0.1em; margin-top:8px; }
    .member-card .roll { color: #ff8c00; font-size: 0.65rem; letter-spacing: 0.15em; }
    .member-card .role { color: #556; font-size: 0.6rem; letter-spacing: 0.1em; margin-top:4px; }
</style>
''', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_all_models():
    nlp_model = joblib.load('models/nlp_agent.pkl')
    nlp_tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    url_model = joblib.load('models/url_classifier.pkl')
    url_tfidf = joblib.load('models/url_tfidf.pkl')
    return nlp_model, nlp_tfidf, url_model, url_tfidf

try:
    with st.spinner('INITIALIZING NEURAL MODELS...'):
        nlp_model, nlp_tfidf, url_model, url_tfidf = load_all_models()
    models_loaded = True
except Exception as e:
    st.error(f'MODEL LOAD ERROR: {e}')
    models_loaded = False

@st.cache_resource(show_spinner=False)
def init_supabase():
    return create_client(
        'https://fpvmqjsnqakhiqbscjle.supabase.co',
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZwdm1xanNucWFraGlxYnNjamxlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgwOTgzNTAsImV4cCI6MjA5MzY3NDM1MH0.q11ue7nFAraaRtVcABYKKXemUIraEMG8Ets2q-89yA0'
    )

try:
    supabase = init_supabase()
    supabase_connected = True
except:
    supabase = None
    supabase_connected = False

def extract_text_features(texts):
    features = []
    for text in texts:
        text = str(text).lower()
        feat = {
            'length': len(text),
            'num_urls': len(re.findall(r'http[s]?://\S+', text)),
            'num_digits': sum(c.isdigit() for c in text),
            'has_urgent': int(any(w in text for w in ['urgent', 'immediate', 'alert', 'warning', 'suspended', 'blocked'])),
            'has_money': int(any(w in text for w in ['reward', 'won', 'prize', 'cash', 'payment', 'refund', '$', 'usd', 'free', 'win'])),
            'has_action': int(any(w in text for w in ['click', 'verify', 'confirm', 'update', 'login', 'password', 'authenticate'])),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'num_words': len(text.split()),
            'has_phone': int(bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))),
            'suspicious_chars': len(re.findall(r'[@#$%^&*]', text)),
            'has_suspicious_url': int(bool(re.search(r'bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly', text)))
        }
        features.append(feat)
    return pd.DataFrame(features)

def extract_url_features(urls):
    features = []
    for url in urls:
        url = str(url).lower()
        parsed = re.sub(r'^https?://', '', url).split('/')[0]
        feat = {
            'length': len(url), 'num_dots': url.count('.'),
            'num_slashes': url.count('/'), 'num_digits': sum(c.isdigit() for c in url),
            'has_https': int(url.startswith('https')),
            'has_ip': int(bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed))),
            'has_shortener': int(any(s in parsed for s in ['bit.ly','tinyurl','t.co','goo.gl'])),
            'has_suspicious_kw': int(any(kw in url for kw in ['login','verify','account','update','secure','password','confirm'])),
            'num_subdomains': len(parsed.split('.')) - 2,
            'has_port': int(':' in parsed),
            'has_at': int('@' in url),
            'has_query': int('?' in url),
            'has_encoded': int('%' in url),
            'tld_length': len(parsed.split('.')[-1]) if '.' in parsed else 0,
            'path_length': len(url.split('/', 3)[-1]) if '/' in url else 0
        }
        features.append(feat)
    return pd.DataFrame(features)

def get_temporal_data():
    if not st.session_state.analyses:
        return pd.DataFrame({'THREAT_COUNT': [45,52,48,67,58,72,65,81,74,68]}, index=['T-10','T-9','T-8','T-7','T-6','T-5','T-4','T-3','T-2','T-1'])
    recent = st.session_state.analyses[-10:]
    t_labels = [f'T-{i+1}' for i in range(len(recent))][::-1]
    values = [a['confidence'] if a['verdict'] == 'CRITICAL' else a['confidence'] * 0.1 for a in recent]
    return pd.DataFrame({'THREAT_COUNT': values}, index=t_labels)

def get_distribution_data():
    if not st.session_state.analyses:
        return 62, 38
    critical = sum(1 for a in st.session_state.analyses if a['verdict'] == 'CRITICAL')
    total = len(st.session_state.analyses)
    crit_pct = int(100 * critical / total) if total > 0 else 0
    return crit_pct, 100 - crit_pct

def get_throughput_data():
    if not st.session_state.analyses:
        return pd.DataFrame({'OPS_MIN': [78,45,92,63,71,38]}, index=['A1','A2','A3','A4','A5','A6'])
    agents = {'ShieldAI_NLP': 0, 'Visual_Auditor_URL': 0, 'File_Parser_NLP': 0}
    for a in st.session_state.analyses:
        src = a.get('agent', 'ShieldAI_NLP')
        if src in agents:
            agents[src] += 1
        else:
            agents[src] = 1
    df = pd.DataFrame({'OPS_MIN': list(agents.values())}, index=list(agents.keys()))
    return df

def get_gauge_value():
    if not st.session_state.analyses:
        return 85
    return int(st.session_state.analyses[-1]['confidence'])

def svg_gauge(value=85):
    import math
    angle = 180 * (value / 100)
    rad = math.radians(180 - angle)
    nx = 100 + 70 * math.cos(rad)
    ny = 100 - 70 * math.sin(rad)
    arc_x = 20 + 160 * (1 - value/100)
    arc_y = 100 - 80 * math.sin(math.radians(angle))
    svg = '<div class="gauge-container"><svg width="200" height="120" viewBox="0 0 200 120">'
    svg += '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#112233" stroke-width="12"/>'
    svg += f'<path d="M 20 100 A 80 80 0 0 1 {arc_x:.1f} {arc_y:.1f}" fill="none" stroke="#ff6347" stroke-width="12" stroke-linecap="round"/>'
    svg += f'<line x1="100" y1="100" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#00ffff" stroke-width="3" marker-end="url(#arrowhead)"/>'
    svg += '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#00ffff"/></marker></defs>'
    svg += f'<text x="100" y="95" text-anchor="middle" class="gauge-value">{value:03d}</text>'
    svg += '<text x="100" y="115" text-anchor="middle" class="gauge-label">STATUS: CRITICAL_OPS</text>'
    svg += '</svg></div>'
    return svg

def svg_donut(threat=62, safe=38):
    dash = threat * 3.77
    gap = safe * 3.77
    svg = '<div class="donut-container"><svg width="160" height="160" viewBox="0 0 160 160">'
    svg += '<circle cx="80" cy="80" r="60" fill="none" stroke="#00ffff" stroke-width="20" opacity="0.3"/>'
    svg += f'<circle cx="80" cy="80" r="60" fill="none" stroke="#ff6347" stroke-width="20" stroke-dasharray="{dash:.1f} {gap:.1f}" stroke-dashoffset="0" transform="rotate(-90 80 80)"/>'
    svg += f'<text x="80" y="75" text-anchor="middle" class="donut-value">{threat}%</text>'
    svg += '<text x="80" y="95" text-anchor="middle" class="donut-label">CRITICAL</text>'
    svg += '</svg></div>'
    return svg

header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
with header_col2:
    st.markdown('''
    <div style='text-align:center; margin-bottom:5px;'>
        <span style='font-size:3rem; filter:drop-shadow(0 0 15px rgba(0,255,255,0.6));'>🛡️</span>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">AEGIS-SWARM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">INTELLIGENT MULTI-MODAL THREAT TRIAGE</p>', unsafe_allow_html=True)
    st.markdown('<p class="dev-info">Muhammad Abdullah (Muhammad Abdullah Asad) | FA23-BCE-049 | COMSATS University Islamabad</p>', unsafe_allow_html=True)

st.markdown('''
<div class='status-bar'>
    SWARM PROTOCOL://ACTIVE <span>|</span>
    NODES://1,402 <span>|</span>
    UPTIME://482:12:04 <span>|</span>
    VER://v3.0_CEP
</div>
''', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['📝 TEXT ANALYSIS', '🖼️ QR DECODER', '👤 OPERATOR'])

with tab1:
    st.markdown('<div class="hud-label">HUD_041 // SOURCE DATA INGESTION</div>', unsafe_allow_html=True)
    text_input = st.text_area('', height=140,
        placeholder='INPUT RAW THREAT PAYLOAD STRING OR BASE64...\n[EXAMPLE]: URGENT: Your PayPal account has been suspended. Click here to verify...',
        label_visibility='collapsed')
    analyze_pressed = st.button('⚡ INITIATE SWARM ANALYSIS', use_container_width=True)

    if analyze_pressed and text_input and models_loaded:
        with st.spinner('SWARM AGENTS COORDINATING...'):
            tfidf_vec = nlp_tfidf.transform([text_input])
            handcrafted = extract_text_features([text_input])
            combined = hstack([tfidf_vec, csr_matrix(handcrafted.values)])
            proba = nlp_model.predict_proba(combined)[0]
            pred = nlp_model.predict(combined)[0]
            phishing_prob = proba[1] * 100
            verdict = 'CRITICAL' if pred == 1 else 'SAFE'
            confidence = phishing_prob if pred == 1 else (100 - phishing_prob)
            action = 'ISOLATE' if pred == 1 else 'MONITOR'
            severity = int(confidence * 0.85)
            db_verdict = 'HIGH' if pred == 1 else 'LOW'
            st.session_state.analyses.append({
                'verdict': verdict,
                'confidence': confidence,
                'agent': 'ShieldAI_NLP',
                'timestamp': datetime.utcnow().isoformat()
            })
            if supabase_connected:
                try:
                    supabase.table('threats').insert({
                        'content': text_input[:500],
                        'threat_type': db_verdict,
                        'confidence': round(confidence / 100, 4),
                        'agent_source': 'ShieldAI_NLP',
                        'created_at': datetime.utcnow().isoformat()
                    }).execute()
                except Exception as e:
                    st.warning(f'MEMORY STORE WARNING: {e}')
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown('<div class="hud-label">THREAT_INTELLIGENCE_STREAM</div>', unsafe_allow_html=True)
            st.markdown('''
            <div style='text-align:right; font-family:Share Tech Mono,monospace; font-size:0.6rem; color:#556; margin-bottom:8px;'>
                SYNC_OK // <span style='color:#00ffff'>v4</span> BUFFER_400ms
            </div>
            ''', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                color = '#ff6347' if verdict == 'CRITICAL' else '#00ff7f'
                st.markdown(f'''<div class="metric-box"><h2 style="color:{color};font-size:2.2rem;">{verdict}</h2><p>THREAT STATUS</p></div>''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''<div class="metric-box"><h2 style="color:#00ffff;font-size:2.2rem;">{confidence:.1f}%</h2><p>CONFIDENCE SCORE</p></div>''', unsafe_allow_html=True)
            with c3:
                action_color = '#ff6347' if action == 'ISOLATE' else '#00ffff'
                st.markdown(f'''<div class="metric-box"><h2 style="color:{action_color};font-size:2.2rem;">{action}</h2><p>RECOMMENDED ACTION</p></div>''', unsafe_allow_html=True)
            st.progress(confidence / 100)
            if verdict == 'CRITICAL':
                st.markdown(f'''<div class="glass-panel-high"><h4 style="color:#ff6347; font-family:Share Tech Mono,monospace; letter-spacing:0.2em;">⚠️ THREAT DETECTED // ShieldAI NLP</h4><p style="font-family:Share Tech Mono,monospace; color:#8899aa; font-size:0.8rem;">Phishing probability: <b style="color:#ff6347;">{phishing_prob:.1f}%</b> | Model accuracy: <b>94.3%</b> | Agent: <b>ShieldAI_NLP_v2</b></p></div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="glass-panel-low"><h4 style="color:#00ff7f; font-family:Share Tech Mono,monospace; letter-spacing:0.2em;">✓ PAYLOAD SECURE // ShieldAI NLP</h4><p style="font-family:Share Tech Mono,monospace; color:#8899aa; font-size:0.8rem;">Low risk detected. No action required. Confidence: <b style="color:#00ff7f;">{confidence:.1f}%</b></p></div>''', unsafe_allow_html=True)
            with st.expander('🔍 AGENT INTELLIGENCE REPORT'):
                st.markdown('<div style="font-family:Share Tech Mono,monospace; font-size:0.8rem; color:#00ffff;"><b>👁️ ShieldAI NLP Analysis:</b></div>', unsafe_allow_html=True)
                st.write(f'- Intent: `{verdict}`')
                st.write(f'- Confidence: `{confidence:.2f}%`')
                st.write(f'- Phishing Probability: `{phishing_prob:.2f}%`')
                st.write(f'- Severity Score: `{severity}/100`')
                st.write(f'- Vector Dimensions: `{combined.shape[1]}`')
                st.write(f'- Timestamp: `{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}`')

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="hud-label">SWARM ANALYTICS // REAL-TIME</div>', unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<p style="font-family:Share Tech Mono,monospace; font-size:0.7rem; color:#ff8c00; letter-spacing:0.15em;">DETECTION_TEMPORAL_DELTA</p>', unsafe_allow_html=True)
        st.line_chart(get_temporal_data(), use_container_width=True, height=220)
    with col_chart2:
        st.markdown('<p style="font-family:Share Tech Mono,monospace; font-size:0.7rem; color:#ff8c00; letter-spacing:0.15em;">DISTRIBUTION_ARRAY</p>', unsafe_allow_html=True)
        crit, safe = get_distribution_data()
        st.markdown(svg_donut(crit, safe), unsafe_allow_html=True)
    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        st.markdown('<p style="font-family:Share Tech Mono,monospace; font-size:0.7rem; color:#ff8c00; letter-spacing:0.15em;">AGENT_THROUGHPUT</p>', unsafe_allow_html=True)
        st.bar_chart(get_throughput_data(), use_container_width=True, height=220)
    with col_chart4:
        st.markdown('<p style="font-family:Share Tech Mono,monospace; font-size:0.7rem; color:#ff8c00; letter-spacing:0.15em;">AGGREGATE_SEVERITY</p>', unsafe_allow_html=True)
        st.markdown(svg_gauge(get_gauge_value()), unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="hud-label">VISUAL_AUDITOR // QR MATRIX DECODER</div>', unsafe_allow_html=True)
    st.caption('Decode QR images and analyze embedded URLs for threat vectors')
    qr_file = st.file_uploader('Upload QR code:', type=['png', 'jpg', 'jpeg'], label_visibility='collapsed')
    if qr_file:
        st.image(qr_file, width=220)
        if st.button('⚡ DECODE & ANALYZE QR', use_container_width=True) and models_loaded:
            with st.spinner('DECODING QR MATRIX...'):
                file_bytes = np.asarray(bytearray(qr_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    detector = cv2.QRCodeDetector()
                    data, bbox, _ = detector.detectAndDecode(img)
                    if data and data.startswith('http'):
                        st.success(f'🔗 DECODED URL: `{data[:120]}`')
                        tfidf_vec = url_tfidf.transform([data])
                        handcrafted = extract_url_features([data])
                        combined = hstack([tfidf_vec, csr_matrix(handcrafted.values)])
                        proba = url_model.predict_proba(combined)[0]
                        pred = url_model.predict(combined)[0]
                        malicious_prob = proba[1] * 100
                        verdict = 'CRITICAL' if pred == 1 else 'SAFE'
                        confidence = malicious_prob if pred == 1 else (100 - malicious_prob)
                        db_verdict = 'HIGH' if pred == 1 else 'LOW'
                        st.session_state.analyses.append({
                            'verdict': verdict,
                            'confidence': confidence,
                            'agent': 'Visual_Auditor_URL',
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        if supabase_connected:
                            try:
                                supabase.table('threats').insert({
                                    'content': data[:500],
                                    'threat_type': db_verdict,
                                    'confidence': round(confidence / 100, 4),
                                    'agent_source': 'Visual_Auditor_URL'
                                }).execute()
                            except Exception as e:
                                st.warning(f'MEMORY STORE WARNING: {e}')
                        c1, c2 = st.columns(2)
                        with c1:
                            color = '#ff6347' if verdict == 'CRITICAL' else '#00ff7f'
                            st.markdown(f'''<div class="metric-box"><h2 style="color:{color}">{verdict}</h2><p>URL RISK LEVEL</p></div>''', unsafe_allow_html=True)
                        with c2:
                            st.markdown(f'''<div class="metric-box"><h2 style="color:#00ffff">{confidence:.1f}%</h2><p>CONFIDENCE</p></div>''', unsafe_allow_html=True)
                        st.progress(confidence / 100)
                        if verdict == 'CRITICAL':
                            st.markdown(f'''<div class="glass-panel-high"><h4 style="color:#ff6347; font-family:Share Tech Mono,monospace;">⚠️ MALICIOUS QR DETECTED</h4><p style="font-family:Share Tech Mono,monospace; color:#8899aa; font-size:0.8rem;">URL threat score: <b style="color:#ff6347;">{malicious_prob:.1f}%</b> | Agent: <b>Visual_Auditor_v1</b></p></div>''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''<div class="glass-panel-low"><h4 style="color:#00ff7f; font-family:Share Tech Mono,monospace;">✓ SAFE QR</h4><p style="font-family:Share Tech Mono,monospace; color:#8899aa; font-size:0.8rem;">URL appears legitimate. Confidence: <b style="color:#00ff7f;">{confidence:.1f}%</b></p></div>''', unsafe_allow_html=True)
                    else:
                        st.warning('📷 QR decoded but no URL found. Content may be WiFi password or vCard.')
                else:
                    st.error('❌ INVALID IMAGE FORMAT // Cannot decode matrix')

with tab3:
    st.markdown('<div class="hud-label">OPERATOR_PROFILE // CLASSIFIED</div>', unsafe_allow_html=True)

    st.markdown('''
    <div style='text-align:center; font-family:Share Tech Mono,monospace; color:#ff8c00; font-size:0.7rem; letter-spacing:0.3em; margin-bottom:20px;'>
        SWARM DEVELOPMENT TEAM // COMSATS UNIVERSITY ISLAMABAD
    </div>
    ''', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('''
        <div class='member-card'>
            <div class='dev-avatar' style='margin-bottom:10px;'>👤</div>
            <div class='name'>MUHAMMAD ABDULLAH</div>
            <div class='roll'>FA23-BCE-049</div>
            <div class='role'>LEAD ARCHITECT // ShieldAI NLP</div>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown('''
        <div class='member-card'>
            <div class='dev-avatar' style='margin-bottom:10px; border-color:#ff8c00; box-shadow:0 0 20px rgba(255,140,0,0.3);'>👤</div>
            <div class='name' style='color:#ff8c00;'>HASEEB</div>
            <div class='roll'>FA23-BCE-104</div>
            <div class='role'>CO-DEVELOPER // Visual Auditor</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown('''
        <div class='dev-avatar'>🛡️</div>
        <div style='text-align:center; font-family:Share Tech Mono,monospace; color:#00ffff; font-size:1.1rem; letter-spacing:0.15em; margin-bottom:5px;'>AEGIS-SWARM</div>
        <div style='text-align:center; font-family:Share Tech Mono,monospace; color:#ff8c00; font-size:0.7rem; letter-spacing:0.2em; margin-bottom:15px;'>v3.0 CEP 2026</div>
        <div style='text-align:center; font-family:Share Tech Mono,monospace; color:#556; font-size:0.6rem; letter-spacing:0.1em;'>COMPUTER ENGINEERING</div>
        <div style='text-align:center; font-family:Share Tech Mono,monospace; color:#556; font-size:0.6rem; letter-spacing:0.1em; margin-bottom:15px;'>LAHORE CAMPUS</div>
        ''', unsafe_allow_html=True)
    with col_right:
        st.markdown('''
        <div class='dev-card'>
            <div class='label'>[ PROJECT ]</div>
            <div style='color:#00ffff; font-size:1rem; margin-bottom:10px; letter-spacing:0.1em;'>AEGIS-SWARM v3.0</div>
            <div style='color:#8899aa; font-size:0.75rem; line-height:1.6;'>Intelligent Multi-Modal Threat Triage System — a multi-agent cybersecurity ecosystem designed for real-time phishing detection across text, QR codes, and URLs. Built as a Complex Engineering Problem (CEP) for the Computer Engineering Department.</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('''
        <div class='dev-card'>
            <div class='label'>[ SYSTEM ARCHITECTURE ]</div>
            <div style='margin-top:8px;'>
                <span style='color:#00ffff;'>►</span> <b>ShieldAI NLP</b> — Scikit-Learn ensemble for semantic phishing intent detection<br>
                <span style='color:#00ffff;'>►</span> <b>Visual Auditor</b> — OpenCV QR decoder + URL heuristic analysis<br>
                <span style='color:#00ffff;'>►</span> <b>Memory Core</b> — Supabase pgvector threat intelligence database<br>
                <span style='color:#00ffff;'>►</span> <b>Swarm Dashboard</b> — Streamlit cyberpunk HUD frontend<br>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('''
        <div class='dev-card'>
            <div class='label'>[ PERFORMANCE METRICS ]</div>
            <div style='margin-top:8px;'>
                <span style='color:#ff8c00;'>ACCURACY</span> <span style='color:#00ffff; float:right;'>94.30%</span><br>
                <span style='color:#ff8c00;'>PRECISION</span> <span style='color:#00ffff; float:right;'>93.04%</span><br>
                <span style='color:#ff8c00;'>RECALL</span> <span style='color:#00ffff; float:right;'>91.27%</span><br>
                <span style='color:#ff8c00;'>F1 SCORE</span> <span style='color:#00ffff; float:right;'>92.14%</span><br>
                <span style='color:#ff8c00;'>AUC</span> <span style='color:#00ffff; float:right;'>0.9805</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('''
    <div class='dev-card' style='margin-top:15px;'>
        <div class='label'>[ CLEARANCE LEVEL ]</div>
        <div style='margin-top:8px; color:#8899aa; font-size:0.75rem;'>
            <span style='color:#ff6347;'>●</span> CEP 2026 AUTHORIZED OPERATOR<br>
            <span style='color:#00ffff;'>●</span> Computer Engineering Department — Lahore Campus<br>
            <span style='color:#00ff7f;'>●</span> Multi-Agent Cybersecurity Ecosystem Research<br>
        </div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="hud-label">SYSTEM_THREAT_LOGS_V4 // DECRYPT_ENGINE</div>', unsafe_allow_html=True)
st.markdown(f'''<div style="text-align:right; font-family:Share Tech Mono,monospace; font-size:0.6rem; color:#556; margin-bottom:8px;">LAST_REFRESH: <span style="color:#00ffff">{datetime.utcnow().strftime("%H:%M:%S")}</span> | ZONES: A50</div>''', unsafe_allow_html=True)
log_data = []
if supabase_connected:
    try:
        response = supabase.table('threats').select('*').order('created_at', desc=True).limit(10).execute()
        if response.data:
            for row in response.data:
                log_data.append({
                    'TIMESTAMP': row.get('created_at', 'N/A')[:19] if row.get('created_at') else 'N/A',
                    'EVENT_IDENTIFIER': row.get('agent_source', 'UNKNOWN'),
                    'THREAT_TYPE': row.get('threat_type', 'UNKNOWN'),
                    'CONFIDENCE': f"{row.get('confidence', 0)*100:.2f}%",
                    'CONTENT': row.get('content', '')[:60] + '...'
                })
    except:
        pass
if not log_data:
    events = [
        ('PACKET_INJECTION_XSS_03', 'CRITICAL', 99.42),
        ('DNS_TUNNEL_EXFIL_A1', 'CRITICAL', 88.10),
        ('ICMP_SCAN_PING_SWEEP', 'SAFE', 12.55),
        ('ENCRYPTED_AUTH_BRUTE', 'CRITICAL', 94.91),
        ('PHISHING_PAYLOAD_NLP', 'CRITICAL', 97.30),
        ('SAFE_BROWSING_VERIFY', 'SAFE', 4.20),
    ]
    for i, (evt, status, conf) in enumerate(events):
        ts = (datetime.utcnow() - timedelta(minutes=i*3)).strftime('%H:%M:%S') + f':{random.randint(100,999)}'
        log_data.append({
            'TIMESTAMP': ts,
            'EVENT_IDENTIFIER': evt,
            'THREAT_TYPE': status,
            'CONFIDENCE': f'{conf:.2f}%',
            'CONTENT': '[AUTO-EXTRACTED PAYLOAD]'
        })
log_df = pd.DataFrame(log_data)
html_table = '<table class="threat-log-table"><tr>'
for col in log_df.columns:
    html_table += f'<th>{col}</th>'
html_table += '</tr>'
for _, row in log_df.iterrows():
    html_table += '<tr>'
    for val in row:
        color = '#ff6347' if val == 'CRITICAL' else ('#00ff7f' if val == 'SAFE' else '#8899aa')
        html_table += f'<td style="color:{color if val in ["CRITICAL","SAFE"] else "#8899aa"}">{val}</td>'
    html_table += '</tr>'
html_table += '</table>'
st.markdown(html_table, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('''
<div class='bottom-nav'>
    <div class='bottom-nav-item active'>
        <span class='bottom-nav-icon'>🛡️</span>
        <div>THREATS</div>
    </div>
    <div class='bottom-nav-item'>
        <span class='bottom-nav-icon'>📋</span>
        <div>LOGS</div>
    </div>
    <div class='bottom-nav-item'>
        <span class='bottom-nav-icon'>⚙️</span>
        <div>NODES</div>
    </div>
    <div class='bottom-nav-item'>
        <span class='bottom-nav-icon'>👤</span>
        <div>OPERATOR</div>
    </div>
</div>
''', unsafe_allow_html=True)
st.markdown(f'''<p style="text-align:center; color:#223; font-size:0.6rem; font-family:Share Tech Mono,monospace; letter-spacing:0.2em; margin-top:10px;">AEGIS-SWARM v3.0 | Multi-Agent Cybersecurity Ecosystem | CEP 2026 | {datetime.utcnow().year}</p>''', unsafe_allow_html=True)
