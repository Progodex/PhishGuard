"""
PhishGuard - web demo.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

from features import extract

bundle = joblib.load("model/phishguard.joblib")
MODEL, FEATS, NAME = bundle["model"], bundle["features"], bundle["name"]

app = Flask(__name__)

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>PhishGuard</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#f4f6fa;margin:0;color:#1a202c}
 .wrap{max-width:720px;margin:60px auto;padding:0 20px}
 h1{margin:0;font-size:34px;color:#1a365d} .sub{color:#4a5568;margin:6px 0 28px}
 form{display:flex;gap:10px}
 input{flex:1;padding:14px;font-size:16px;border:1px solid #cbd5e0;border-radius:8px}
 button{padding:14px 26px;font-size:16px;background:#2b6cb0;color:#fff;border:0;border-radius:8px;cursor:pointer}
 .card{margin-top:28px;padding:24px;border-radius:12px;background:#fff;box-shadow:0 2px 10px rgba(0,0,0,.06)}
 .safe{border-left:8px solid #38a169} .bad{border-left:8px solid #e53e3e}
 .verdict{font-size:28px;font-weight:700} .safe .verdict{color:#2f855a} .bad .verdict{color:#c53030}
 .bar{height:12px;background:#e2e8f0;border-radius:6px;margin:12px 0}
 .fill{height:12px;border-radius:6px}
 li{margin:4px 0} .foot{margin-top:30px;color:#718096;font-size:13px}
</style></head><body><div class="wrap">
<h1>PhishGuard</h1>
<div class="sub">Lightweight ML phishing URL detector &middot; model: {{name}}</div>
<form method="post"><input name="url" placeholder="Paste a URL, e.g. paypal-login.secure-verify.xyz/signin" value="{{url}}" required>
<button>Check</button></form>
{% if result %}
<div class="card {{ 'bad' if result.phish else 'safe' }}">
 <div class="verdict">{{ 'PHISHING' if result.phish else 'LEGITIMATE' }}</div>
 <div>Phishing probability: <b>{{ '%.1f'|format(result.prob*100) }}%</b></div>
 <div class="bar"><div class="fill" style="width:{{result.prob*100}}%;background:{{ '#e53e3e' if result.phish else '#38a169' }}"></div></div>
 {% if result.flags %}<b>Warning signs found:</b><ul>{% for f in result.flags %}<li>{{f}}</li>{% endfor %}</ul>
 {% else %}<div>No common warning signs found in the URL structure.</div>{% endif %}
</div>{% endif %}
<div class="foot">Uses only the URL text: no DNS, WHOIS or page download, so it runs on edge/IoT devices.</div>
</div></body></html>
"""


def warning_signs(f):
    flags = []
    if f["length_url"] > 54: flags.append(f"Long URL ({f['length_url']} characters)")
    if f["domain_in_ip"]: flags.append("Uses an IP address instead of a domain name")
    if f["qty_at_url"]: flags.append("Contains '@', which can hide the real destination")
    if f["qty_hyphen_domain"] >= 2: flags.append(f"{f['qty_hyphen_domain']} hyphens in the domain")
    if f["qty_dot_domain"] >= 4: flags.append(f"Many sub-domains ({f['qty_dot_domain']} dots in domain)")
    if f["email_in_url"]: flags.append("An email address is embedded in the URL")
    if f["qty_slash_url"] >= 4: flags.append(f"Deep path ({f['qty_slash_url']} slashes)")
    if f["qty_equal_url"] + f["qty_and_url"] >= 3: flags.append("Many query parameters")
    return flags


def predict(url):
    f = extract(url)
    prob = float(MODEL.predict_proba(pd.DataFrame([f])[FEATS])[0, 1])
    return {"phish": prob >= 0.5, "prob": prob, "flags": warning_signs(f)}


@app.route("/", methods=["GET", "POST"])
def home():
    url = request.form.get("url", "")
    result = predict(url) if url else None
    return render_template_string(PAGE, url=url, result=result, name=NAME)


if __name__ == "__main__":
    app.run(debug=False)
