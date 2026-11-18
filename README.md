# PhishGuard: Lightweight ML Phishing URL Detection for Web and IoT

PhishGuard decides whether a URL is **phishing** or **legitimate** using only the text of the URL. It makes no DNS, WHOIS or page-download requests, so it is fast and small enough to run on edge and IoT gateways.

## Results (test set: 17,730 URLs)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | 72.90% | 90.72% | 24.09% | 38.08% | 0.924 |
| Logistic Regression | 89.68% | 89.55% | 79.41% | 84.18% | 0.958 |
| Decision Tree | 92.61% | 90.74% | 87.57% | 89.13% | 0.955 |
| Gradient Boosting | 92.88% | 90.18% | 89.10% | 89.64% | 0.978 |
| **Random Forest** | **93.88%** | **92.44%** | **89.62%** | **91.01%** | **0.979** |

**Lightweight trade-off:** a Random Forest that uses all 111 dataset features, including network lookups, reaches 96.98% accuracy. PhishGuard gives up about 3 points to remove every network call. The Gradient Boosting model is 0.37 MB and takes about 2 ms per URL, which makes it the best fit for IoT devices.

## How it works

```
URL ──► features.py (39 lexical features) ──► trained model ──► LEGITIMATE / PHISHING + warning signs
```

The 39 features are counts of 17 special characters in the URL and in the domain, URL and domain length, the number of vowels in the domain, an IP-address-as-domain flag and an e-mail-in-URL flag.

## Run it

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python train.py      # downloads the dataset, trains 5 models, saves the model and charts to results/
python app.py        # open http://127.0.0.1:5000
```

Try these URLs:

| URL | Expected |
|---|---|
| `www.google.com` | LEGITIMATE |
| `https://www.sbi.co.in` | LEGITIMATE |
| `paypal-login.secure-verify.xyz/signin/update.php?user=abc&id=99` | PHISHING |
| `secure-account-update-apple.com.verify-id.info/login/` | PHISHING |
| `http://192.168.4.21/bank/login.php` | PHISHING |

## Project structure

| File | Purpose |
|---|---|
| `features.py` | Lexical feature extraction, shared by training and prediction |
| `train.py` | Trains and compares LR, Naive Bayes, Decision Tree, Gradient Boosting and Random Forest, then saves metrics and plots |
| `app.py` | Flask web app that shows the verdict, the probability and warning signs |
| `results/metrics.json` | All metrics from the reference run |

## Limitations

- Most legitimate URLs in the dataset are bare domains, so long legitimate links (articles, videos) can be over-flagged.
- Shortened links (bit.ly) hide the real destination.
- A phishing page hosted on a clean-looking, compromised domain can't be detected from URL text alone.

## Dataset

G. Vrbančić, I. Fister Jr., and V. Podgorelec, "Datasets for phishing websites detection," *Data in Brief*, vol. 33, 106438, 2020. [GitHub](https://github.com/GregaVrbancic/Phishing-Dataset)
