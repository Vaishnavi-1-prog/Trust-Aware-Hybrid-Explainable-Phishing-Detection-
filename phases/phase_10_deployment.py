# -*- coding: utf-8 -*-
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Path to your dataset folder
import os

folder_path = "/content/drive/MyDrive/phishing_project_datasets"

# List all files in the folder
files = os.listdir(folder_path)

print("Files inside phishing_project_datasets:\n")

for file in files:
    print(file)

"""Input (URL / Header / Message)
        ↓
Feature Extraction
        ↓
RF Model (URL)
BiLSTM Model (Message)
Header Rule Engine
        ↓
Weighted Consensus Engine
        ↓
Trust Score (0–10)
        ↓
Verdict + Explainable Charts
"""

# ==========================================================
# TRUST-AWARE HYBRID EXPLAINABLE AI PHISHING DETECTION SYSTEM
# ==========================================================

!pip install gradio plotly tensorflow scikit-learn pandas numpy --quiet

import gradio as gr
import numpy as np
import pickle
import joblib
import re
import os
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
from urllib.parse import urlparse
from google.colab import drive

# ----------------------------------------------------------
# MOUNT DRIVE
# ----------------------------------------------------------

drive.mount('/content/drive')
BASE = "/content/drive/MyDrive/phishing_project_datasets/"

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------

def load_models():
    rf, tok, lstm = None, None, None
    try:
        rf = joblib.load(BASE + "rf_phishing_model.pkl")
        print("✅ RF model loaded.")
    except Exception as e:
        print(f"⚠️ RF model: {e}")
    try:
        with open(BASE + "tokenizer.pkl", "rb") as f:
            tok = pickle.load(f)
        print("✅ Tokenizer loaded.")
    except Exception as e:
        print(f"⚠️ Tokenizer: {e}")
    try:
        p = BASE + "bilstm_email_model"
        lstm = tf.keras.models.load_model(p if os.path.isdir(p) else p + ".h5")
        print("✅ BiLSTM loaded.")
    except Exception as e:
        print(f"⚠️ BiLSTM: {e}")
    return rf, tok, lstm

rf_model, tokenizer, bilstm = load_models()
MAX_LEN = 200

# ----------------------------------------------------------
# TRUSTED DOMAIN CHECK — uses REGISTERED DOMAIN only
#
# KEY FIX: We extract only the last two labels of the hostname.
#   http://paypal-secure-login.verify-account.ru
#     → registered domain = "verify-account.ru"  ← NOT trusted
#   https://www.amazon.in
#     → registered domain = "amazon.in"           ← trusted
#
# This prevents brand-name-in-subdomain spoofing attacks.
# ----------------------------------------------------------

TRUSTED_REGISTERED_DOMAINS = {
    "amazon.in", "amazon.com", "amazon.co.uk", "amazon.de",
    "google.com", "google.co.in", "google.co.uk", "googleapis.com", "gmail.com",
    "microsoft.com", "live.com", "outlook.com", "office.com", "azure.com",
    "apple.com", "icloud.com",
    "paypal.com", "paypal.in",         # ONLY real PayPal domains
    "hdfcbank.com", "sbi.co.in", "icicibank.com", "axisbank.com",
    "irctc.co.in", "flipkart.com", "zomato.com", "swiggy.in",
    "linkedin.com", "twitter.com", "x.com", "instagram.com",
    "facebook.com", "youtube.com", "github.com", "netflix.com",
}

def get_registered_domain(url):
    """Extract last two hostname labels = registered domain."""
    try:
        host = urlparse(url).netloc.lower()
        host = re.sub(r':\d+$', '', host)        # strip port
        parts = [p for p in host.split('.') if p]
        return '.'.join(parts[-2:]) if len(parts) >= 2 else host
    except:
        return ""

def is_trusted_url(url):
    return get_registered_domain(url) in TRUSTED_REGISTERED_DOMAINS

# ----------------------------------------------------------
# URL FEATURE EXTRACTOR
# ----------------------------------------------------------

PHISHING_KEYWORDS = [
    "login", "secure", "account", "update", "verify", "bank",
    "confirm", "paypal", "ebay", "signin", "password",
    "credential", "free", "winner", "click", "urgent", "suspend"
]

def extract_url_features(url):
    try:
        parsed = urlparse(url)
        host  = parsed.netloc or ""
        path  = parsed.path   or ""
        query = parsed.query  or ""

        features = [
            len(url), url.count('.'), url.count('-'), url.count('@'),
            url.count('?'), url.count('%'), url.count('='), url.count('&'),
            url.count('_'), url.count('/'), url.count('#'), url.count('~'),
            len(host), host.count('.'), host.count('-'),
            len(path), len(query), len(parsed.scheme),
            1 if parsed.scheme == "https" else 0,
            1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        ]
        for kw in PHISHING_KEYWORDS:
            features.append(1 if kw in url.lower() else 0)

        def entropy(s):
            if not s: return 0
            probs = [s.count(c)/len(s) for c in set(s)]
            return -sum(p * np.log2(p) for p in probs if p > 0)

        features.append(entropy(host))
        features.append(entropy(path))
        features.append(sum(c.isdigit() for c in url) / max(len(url), 1))

        features = np.array(features, dtype=float)
        if rf_model is not None:
            exp = rf_model.n_features_in_
            if len(features) < exp:
                features = np.concatenate([features, np.zeros(exp - len(features))])
            elif len(features) > exp:
                features = features[:exp]
        return features.reshape(1, -1)

    except Exception as e:
        print("URL Feature Error:", e)
        return np.zeros((1, rf_model.n_features_in_ if rf_model else 41))

# ----------------------------------------------------------
# MESSAGE ANALYSIS (BiLSTM)
# ----------------------------------------------------------

def analyze_message(text):
    if bilstm is None or tokenizer is None:
        return 0.5
    try:
        seq = tokenizer.texts_to_sequences([text])
        seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
        return float(bilstm.predict(seq, verbose=0)[0][0])
    except Exception as e:
        print("BiLSTM Error:", e)
        return 0.5

# ----------------------------------------------------------
# HEADER RULE ENGINE  (v3 — strengthened)
# ----------------------------------------------------------

HEADER_KW_SIGNALS = {
    "spf fail":          3,
    "dkim fail":         3,
    "dmarc fail":        3,
    "spoof":             3,
    "relay":             1,
    "unauthorized":      3,
    "fake":              3,
    "suspicious":        2,
    "blacklisted":       4,
    "forged":            4,
    "x-spam-flag: yes":  4,
    "x-spam-status: yes":4,
}

def header_analysis(header):
    score = 0
    hits  = []
    hl    = header.lower()

    # 1. Keyword signals
    for sig, w in HEADER_KW_SIGNALS.items():
        if sig in hl:
            score += w
            hits.append(f"{sig} (+{w})")

    # 2. Reply-To domain mismatch
    #    From: someone@paypal.com  +  Reply-To: attacker@evil.ru  → phishing
    from_m    = re.search(r'from:\s*[^@\s]*@([\w.\-]+)', hl)
    replyto_m = re.search(r'reply-to:\s*[^@\s]*@([\w.\-]+)', hl)
    if from_m and replyto_m:
        fd = from_m.group(1)
        rd = replyto_m.group(1)
        if fd != rd:
            score += 4
            hits.append(f"Reply-To mismatch: from={fd} reply-to={rd} (+4)")

    # 3. Private / loopback IP in Received header
    #    Real mail servers never originate from 192.168.x.x / 10.x.x.x / 127.x.x.x
    priv = re.search(
        r'received:.*?'
        r'(192\.168\.\d{1,3}\.\d{1,3}'
        r'|10\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        r'|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}'
        r'|127\.\d{1,3}\.\d{1,3}\.\d{1,3})',
        hl
    )
    if priv:
        score += 4
        hits.append(f"Private IP in Received: {priv.group(1)} (+4)")

    # 4. "unknown" specifically inside a Received: line
    if re.search(r'received:.*?unknown', hl):
        score += 2
        hits.append("Unknown sender host in Received header (+2)")

    # 5. HTTP (non-HTTPS) in header
    if 'http://' in hl:
        score += 1
        hits.append("Non-HTTPS link in header (+1)")

    return score, hits

# ----------------------------------------------------------
# CHARTS
# ----------------------------------------------------------

def url_importance_chart(features):
    if rf_model is None:
        fig = go.Figure()
        fig.update_layout(title="RF model not loaded",
                          paper_bgcolor="#0d1117", font_color="white")
        return fig
    try:
        imps = rf_model.feature_importances_
        n    = min(len(imps), features.shape[1])
        df   = pd.DataFrame({"feature": [f"F{i}" for i in range(n)],
                              "importance": imps[:n]
                             }).sort_values("importance", ascending=False).head(10)
        fig = go.Figure(go.Bar(x=df.importance, y=df.feature,
                               orientation='h', marker_color='crimson'))
        fig.update_layout(title="Top URL Feature Importances (RF)",
                          plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                          font_color="white")
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error: {e}",
                          paper_bgcolor="#0d1117", font_color="white")
        return fig

def risk_gauge(score):
    color = "red" if score <= 4 else ("orange" if score <= 6 else "green")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Trust Score", "font": {"color": "white"}},
        gauge={
            "axis":    {"range": [0, 10], "tickcolor": "white"},
            "bar":     {"color": color},
            "bgcolor": "#0d1117",
            "steps": [
                {"range": [0,  4], "color": "#ff4c4c"},
                {"range": [4,  7], "color": "#ffa500"},
                {"range": [7, 10], "color": "#2ecc71"},
            ]
        }
    ))
    fig.update_layout(paper_bgcolor="#0d1117", font_color="white")
    return fig

def word_importance_chart(text):
    try:
        SW = {"the","a","an","is","in","it","of","to","and","or","for","on","with",
              "was","are","this","that","i","you","we","he","she","they","at","be",
              "by","as","so","if","but","not","from","your","our","will","have",
              "has","had","do","did","my","me","us","his","her","its","their",
              "been","all","can","just","about","up","out","no","re","dear","thank"}
        words    = re.findall(r'\b[a-z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in SW]
        freq     = pd.Series(filtered or words).value_counts().head(10)
        fig = go.Figure(go.Bar(x=freq.values, y=freq.index,
                               orientation='h', marker_color='royalblue'))
        fig.update_layout(title="Top Message Keywords",
                          plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                          font_color="white")
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error: {e}",
                          paper_bgcolor="#0d1117", font_color="white")
        return fig

# ----------------------------------------------------------
# MAIN ANALYSIS — CONSENSUS VOTING
# ----------------------------------------------------------

def analyze(url, header, body):
    try:
        explanations = []
        signals      = {}
        url_features = None
        trusted_url  = False

        # ── SIGNAL 1: URL (Random Forest) ────────────────────
        if url and url.strip():
            trusted_url = is_trusted_url(url)
            reg_domain  = get_registered_domain(url)

            if rf_model is not None:
                url_features = extract_url_features(url)
                try:
                    prob       = rf_model.predict_proba(url_features)[0]
                    phish_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
                    signals["url"] = phish_prob

                    if phish_prob > 0.7:
                        explanations.append(
                            f"URL flagged as phishing by RF ({phish_prob:.1%})")
                    elif phish_prob > 0.4:
                        explanations.append(
                            f"URL looks suspicious (RF: {phish_prob:.1%})")
                    else:
                        explanations.append(
                            f"URL appears clean (RF: {phish_prob:.1%})")

                    if trusted_url:
                        explanations.append(
                            f"URL registered domain '{reg_domain}' is a verified trusted domain.")
                    else:
                        # Check for brand-name-in-subdomain spoofing
                        brand_spoof = any(
                            brand in url.lower()
                            for brand in ["paypal","amazon","google","apple",
                                          "microsoft","netflix","ebay","bank",
                                          "hdfc","sbi","icici"]
                        )
                        if brand_spoof:
                            explanations.append(
                                f"Brand name found in URL but registered domain "
                                f"'{reg_domain}' is NOT a trusted domain — "
                                f"possible spoofing attack!")
                            # Boost URL risk score when brand spoofing detected
                            signals["url"] = max(signals["url"], 0.80)

                except Exception as e:
                    explanations.append(f"URL model error: {e}")
            else:
                explanations.append("RF model not loaded — URL skipped.")
        else:
            explanations.append("No URL provided.")

        # ── SIGNAL 2: EMAIL BODY (BiLSTM) ────────────────────
        if body and body.strip():
            raw_score = analyze_message(body)

            if raw_score == 0.5 and bilstm is None:
                explanations.append("BiLSTM not loaded — body skipped.")
            else:
                signals["bilstm"] = raw_score

                # Trusted domain + clean header → discount BiLSTM
                # (handles legitimate transactional email false positives)
                # This is done AFTER header check below, so we defer it.

                if raw_score > 0.85:
                    explanations.append(
                        f"BiLSTM detected phishing language (score: {raw_score:.2f})")
                elif raw_score > 0.55:
                    explanations.append(
                        f"Spam indicators in message (BiLSTM: {raw_score:.2f})")
                else:
                    explanations.append(
                        f"Message body appears legitimate (BiLSTM: {raw_score:.2f})")
        else:
            explanations.append("No message body provided.")

        # ── SIGNAL 3: HEADER RULE ENGINE ─────────────────────
        if header and header.strip():
            h_score, hits = header_analysis(header)
            header_risk   = min(h_score / 10.0, 1.0)
            signals["header"] = header_risk

            if h_score > 0:
                explanations.append(
                    f"Header issues found: {', '.join(hits)}")
            else:
                explanations.append("No suspicious header signals found.")
        else:
            explanations.append("No email header provided.")

        # ── PRE-VOTE: BiLSTM DISCOUNT FOR TRUSTED + CLEAN ────
        # Only discount BiLSTM when BOTH conditions are true:
        # 1. URL's registered domain is actually trusted
        # 2. Header shows no suspicious signals
        header_clean = signals.get("header", 0) < 0.1
        if trusted_url and header_clean and "bilstm" in signals:
            orig = signals["bilstm"]
            signals["bilstm"] = 0.10
            explanations.append(
                f"BiLSTM score ({orig:.2f}) discounted — "
                f"trusted domain + clean headers override body flag.")

        # ── WEIGHTED CONSENSUS ───────────────────────────────
        # URL=40%, BiLSTM=40%, Header=20%
        WEIGHTS      = {"url": 0.55, "bilstm": 0.25, "header": 0.20}
        total_weight = sum(WEIGHTS[k] for k in signals)

        if total_weight == 0:
            weighted_risk = 0.5
        else:
            weighted_risk = (
                sum(signals[k] * WEIGHTS[k] for k in signals) / total_weight
            )

        # ── TRUST SCORE & VERDICT ────────────────────────────
        trust_score = max(0, min(10, int((1.0 - weighted_risk) * 10)))

        if trust_score <= 4:
            result = "🚨 PHISHING — High Risk"
        elif trust_score <= 6:
            result = "⚠️ SUSPICIOUS / SPAM"
        else:
            result = "✅ LIKELY LEGITIMATE"

        sig_summary = " | ".join(
            f"{k.upper()}: {v:.2f}" for k, v in signals.items()
        )
        explanations.append(f"\nConsensus: {sig_summary}")
        explanations.append(
            f"Weighted risk = {weighted_risk:.2f} → Trust Score = {trust_score}/10")

        explanation_text = "\n".join(explanations)

        # ── CHARTS ───────────────────────────────────────────
        gauge_fig = risk_gauge(trust_score)

        if url_features is not None:
            url_fig = url_importance_chart(url_features)
        else:
            url_fig = go.Figure()
            url_fig.update_layout(title="No URL data",
                                   paper_bgcolor="#0d1117", font_color="white")

        if body and body.strip():
            word_fig = word_importance_chart(body)
        else:
            word_fig = go.Figure()
            word_fig.update_layout(title="No body data",
                                    paper_bgcolor="#0d1117", font_color="white")

        return result, str(trust_score), explanation_text, \
               gauge_fig, url_fig, word_fig

    except Exception as e:
        import traceback
        err   = traceback.format_exc()
        empty = go.Figure()
        return f"Error: {e}", "0", err, empty, empty, empty

# ----------------------------------------------------------
# GRADIO UI
# ----------------------------------------------------------

css = """
body, .gradio-container {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Segoe UI', sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    color: white !important; font-weight: bold; border-radius: 8px;
}
.gr-textbox textarea, .gr-textbox input {
    background: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
label { color: #8b949e !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔐 Hybrid AI Phishing Detection Platform
    ### Random Forest (55%) + BiLSTM (25%) + Header Rules (20%) | Consensus Voting
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            url_input    = gr.Textbox(
                label="🌐 URL to Analyze",
                placeholder="https://suspicious-login.example.com/verify")
            header_input = gr.Textbox(
                label="📧 Email Header", lines=5,
                placeholder="Paste raw email headers here...")
            body_input   = gr.Textbox(
                label="💬 SMS / Email Body", lines=7,
                placeholder="Paste the email or SMS message content here...")
            analyze_btn  = gr.Button("🔍 Analyze Threat", variant="primary")

        with gr.Column(scale=1):
            result_out      = gr.Textbox(label="🎯 Verdict", interactive=False)
            trust_out       = gr.Textbox(label="🛡️ Trust Score (0–10)", interactive=False)
            explanation_out = gr.Textbox(label="📋 Explanation",
                                         lines=12, interactive=False)

    gr.Markdown("### 📊 Visual Analysis")

    with gr.Row():
        gauge_out = gr.Plot(label="Trust Score Gauge")
    with gr.Row():
        url_chart_out  = gr.Plot(label="URL Feature Importance")
        word_chart_out = gr.Plot(label="Message Keywords")

    gr.Markdown("""
    ---
    | Trust Score | Verdict |
    |---|---|
    | 0 – 4.9 | 🚨 PHISHING |
    | 5 – 6.9 | ⚠️ SUSPICIOUS / SPAM |
    | 7 – 10 | ✅ LIKELY LEGITIMATE |

    **Weights:** URL 55% · BiLSTM 25% · Header 20%
    Trusted domain discount applies only when registered domain is verified **and** headers are clean.
    """)

    analyze_btn.click(
        fn=analyze,
        inputs=[url_input, header_input, body_input],
        outputs=[result_out, trust_out, explanation_out,
                 gauge_out, url_chart_out, word_chart_out]
    )

demo.launch(debug=True, share=True)

"""advancement try - fs"""
