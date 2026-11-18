"""
PhishGuard - lexical feature extraction.

Computes 39 features directly from the URL string, with no DNS, WHOIS or
page download. That keeps the detector fast and usable on low-power
IoT gateways. Feature names match the Vrbancic et al. (2020) phishing
dataset so the same code is used for training and live prediction.
"""
import re
from urllib.parse import urlsplit

# Special characters counted in the full URL and in the domain
SPECIAL_CHARS = {
    "dot": ".", "hyphen": "-", "underline": "_", "slash": "/",
    "questionmark": "?", "equal": "=", "at": "@", "and": "&",
    "exclamation": "!", "space": " ", "tilde": "~", "comma": ",",
    "plus": "+", "asterisk": "*", "hashtag": "#", "dollar": "$",
    "percent": "%",
}

IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def feature_names():
    names = [f"qty_{k}_url" for k in SPECIAL_CHARS] + ["length_url"]
    names += [f"qty_{k}_domain" for k in SPECIAL_CHARS]
    names += ["qty_vowels_domain", "domain_length", "domain_in_ip", "email_in_url"]
    return names


def normalise(url: str) -> str:
    """The dataset stores URLs without the http(s):// scheme, so strip it."""
    url = url.strip()
    return re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)


def get_domain(url_no_scheme: str) -> str:
    domain = urlsplit("http://" + url_no_scheme).hostname or ""
    return domain.lower()


def extract(url: str) -> dict:
    u = normalise(url)
    domain = get_domain(u)
    f = {}
    for name, ch in SPECIAL_CHARS.items():
        f[f"qty_{name}_url"] = u.count(ch)
    f["length_url"] = len(u)
    for name, ch in SPECIAL_CHARS.items():
        f[f"qty_{name}_domain"] = domain.count(ch)
    f["qty_vowels_domain"] = sum(domain.count(v) for v in "aeiou")
    f["domain_length"] = len(domain)
    f["domain_in_ip"] = int(bool(IP_PATTERN.match(domain)))
    f["email_in_url"] = int(bool(EMAIL_PATTERN.search(u)))
    return f
