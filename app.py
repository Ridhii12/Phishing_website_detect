# ============================================================
# app.py  –  PhishGuard Flask Backend
#
# This is an OPTIONAL Python server that exposes your trained
# scikit-learn / XGBoost model as a REST API.
#
# The Chrome extension can call this instead of (or in addition to)
# the built-in JS scoring engine.
#
# How to use:
#   1. Train and save your model from the Jupyter notebook
#      (it saves phishing_detector_model.pkl and phishing_scaler.pkl)
#   2. Put those .pkl files in the same folder as this file
#   3. Run:  python app.py
#   4. The API will be at http://localhost:5000
#
# The extension sends POST /predict  { "url": "https://example.com" }
# The server responds with          { "score": 42, "label": "suspicious", "flags": [...] }
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
import numpy as np
import pandas as pd
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)   # Allow requests from Chrome extension (cross-origin)

# ── Load the trained model and scaler ────────────────────────────────
MODEL_PATH  = 'phishing_detector_model.pkl'
SCALER_PATH = 'phishing_scaler.pkl'

model  = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ ML model loaded from disk")
else:
    print("⚠️  Model files not found. Run the Jupyter notebook first to generate them.")
    print("    Expected files:", MODEL_PATH, "and", SCALER_PATH)


# ── Feature extraction (mirrors ml_engine.js and the notebook) ───────
SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'secure', 'account', 'update', 'bank',
    'confirm', 'password', 'credential', 'suspend', 'signin',
    'click', 'free', 'prize', 'ebay', 'paypal', 'alert',
    'billing', 'support', 'wallet', 'recover', 'unlock'
]
SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd']


def extract_features(url: str) -> dict:
    """Extract numeric features from a URL string."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ''
        pathname = parsed.path   or ''
        scheme   = parsed.scheme or ''
        port     = parsed.port
    except Exception:
        hostname, pathname, scheme, port = '', '', '', None

    url_lower = url.lower()

    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'

    features = {
        'url_length':               len(url),
        'has_ip':                   1 if re.match(ip_pattern, hostname) else 0,
        'count_at':                 url.count('@'),
        'count_hyphen':             hostname.count('-'),
        'count_dots':               url.count('.'),
        'has_https':                1 if scheme == 'https' else 0,
        'domain_length':            len(hostname),
        'path_depth':               pathname.count('/'),
        'suspicious_keyword_count': sum(w in url_lower for w in SUSPICIOUS_KEYWORDS),
        'digit_count':              sum(c.isdigit() for c in url),
        'special_char_count':       sum(url.count(c) for c in '%=?&#'),
        'has_port':                 1 if port is not None else 0,
        'has_www':                  1 if hostname.startswith('www') else 0,
        'is_shortened':             1 if any(s in url_lower for s in SHORTENERS) else 0,
        'digit_ratio_domain':       sum(c.isdigit() for c in hostname) / max(len(hostname), 1),
        'short_domain':             1 if len(hostname.replace('www.','').split('.')[0]) < 4 else 0,
        'has_redirect':             1 if ('redirect' in url_lower or 'forward' in url_lower) else 0,
    }
    return features


def rule_based_score(features: dict) -> dict:
    """
    Fallback scoring if the ML model isn't loaded.
    Same logic as ml_engine.js.
    """
    score = 0
    flags = []

    if features['has_ip']:
        score += 35; flags.append('Uses IP address instead of domain name')
    if features['count_at'] > 0:
        score += 30; flags.append('Contains @ symbol')
    if features['is_shortened']:
        score += 25; flags.append('URL shortener detected')
    if features['has_redirect']:
        score += 20; flags.append('Contains redirect in URL')
    if features['suspicious_keyword_count'] >= 3:
        score += 20; flags.append(f"{features['suspicious_keyword_count']} suspicious keywords")
    elif features['suspicious_keyword_count'] >= 1:
        score += 10; flags.append(f"{features['suspicious_keyword_count']} suspicious keyword(s)")
    if features['count_hyphen'] >= 3:
        score += 15; flags.append(f"{features['count_hyphen']} hyphens in domain")
    if features['url_length'] > 75:
        score += 12; flags.append(f"Very long URL ({features['url_length']} chars)")
    if features['count_dots'] >= 5:
        score += 12; flags.append(f"Many dots ({features['count_dots']}) in URL")
    if features['has_port']:
        score += 10; flags.append('Non-standard port number')
    if features['digit_ratio_domain'] > 0.35:
        score += 10; flags.append('High digit ratio in domain')

    if features['has_https'] and not features['has_ip']:
        score = max(0, score - 5)
    if features['has_www']:
        score = max(0, score - 3)

    score = min(100, max(0, score))

    if score < 20:     label = 'safe'
    elif score < 50:   label = 'suspicious'
    else:              label = 'phishing'

    return {'score': score, 'label': label, 'flags': flags}


# ── API Endpoints ─────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint — lets the extension verify the server is running."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.

    Request body (JSON):
        { "url": "https://example.com" }

    Response (JSON):
        {
            "url":      "https://example.com",
            "score":    12,
            "label":    "safe",
            "flags":    [],
            "features": { ... },
            "engine":   "ML_ACTIVE" | "rule_based"
        }
    """
    data = request.get_json(silent=True)

    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" field in request body'}), 400

    url = str(data['url']).strip()
    if not url:
        return jsonify({'error': 'URL cannot be empty'}), 400

    features = extract_features(url)

    # ── Use ML model if available ──────────────────────────────────────
    if model is not None and scaler is not None:
        try:
            # Build feature vector in the same order as training
            feature_order = [
                'url_length', 'has_ip', 'count_at', 'count_hyphen',
                'count_dots', 'has_https', 'domain_length', 'path_depth',
                'suspicious_keyword_count', 'digit_count', 'special_char_count',
                'has_port', 'has_www', 'is_shortened', 'digit_ratio_domain',
                'short_domain', 'has_redirect'
            ]
            feat_vector = pd.DataFrame([[features.get(k, 0) for k in feature_order]], columns=feature_order)
            feat_scaled = scaler.transform(feat_vector)

            # Predict probability of phishing
            proba = model.predict_proba(feat_scaled)[0]
            phishing_prob = proba[1]
            score = int(round(phishing_prob * 100))

            if score < 20:     label = 'safe'
            elif score < 50:   label = 'suspicious'
            else:              label = 'phishing'

            # Generate flags (same as rule-based for interpretability)
            flags = rule_based_score(features)['flags']

            return jsonify({
                'url':      url,
                'score':    score,
                'label':    label,
                'flags':    flags,
                'features': features,
                'engine':   'ml_model'
            })

        except Exception as e:
            print(f"ML model error: {e} — falling back to rule-based")

    # ── Fallback: rule-based scoring ──────────────────────────────────
    result = rule_based_score(features)
    return jsonify({
        'url':      url,
        'score':    result['score'],
        'label':    result['label'],
        'flags':    result['flags'],
        'features': features,
        'engine':   'rule_based'
    })


@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint — scan multiple URLs at once.

    Request body:
        { "urls": ["https://a.com", "http://b.com"] }
    """
    data = request.get_json(silent=True)
    if not data or 'urls' not in data:
        return jsonify({'error': 'Missing "urls" field'}), 400

    results = []
    for url in data['urls'][:50]:   # Limit to 50 URLs per request
        features = extract_features(url)
        result   = rule_based_score(features)
        results.append({
            'url':   url,
            'score': result['score'],
            'label': result['label'],
            'flags': result['flags']
        })

    return jsonify({'results': results})


# ── Run the server ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🛡️  PhishGuard Flask Backend")
    print("=" * 40)
    print("API running at: http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health      – health check")
    print("  POST /predict     – analyse one URL")
    print("  POST /batch       – analyse many URLs")
    print("=" * 40 + "\n")

    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False    # Always False in production
    )
