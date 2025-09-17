# generate_phishing_large.py
"""
Generate a very large balanced phishing/legit dataset (URLs + email-like texts)
Outputs CSVs: Phishing_dataset_full.csv, Phishing_train.csv, Phishing_val.csv, Phishing_test.csv
Adjust NUM_TOTAL to control size.
"""
import random
import os
from urllib.parse import urlparse, urlencode
import pandas as pd
import numpy as np
import html, re

random.seed(42)
np.random.seed(42)

# ---------- Config ----------
OUT_DIR = os.path.expanduser("~/Downloads")
OUT_FULL = os.path.join(OUT_DIR, "Phishing_dataset_full_large.csv")
OUT_TRAIN = os.path.join(OUT_DIR, "Phishing_train_large.csv")
OUT_VAL   = os.path.join(OUT_DIR, "Phishing_val_large.csv")
OUT_TEST  = os.path.join(OUT_DIR, "Phishing_test_large.csv")

# *** غيّر الرقم هنا لحجم البيانات الكلي ***
NUM_TOTAL = 200000   # مثال: 200k صف. يمكن تغييره إلى 100000, 500000, إلخ.
RATIO_TEXTS = 0.5    # نسبة النصوص (رسائل) من كل فئة؛ الباقي روابط
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ---------- Seed lists (يمكن توسيعها) ----------
LEGIT_DOMAINS = [
    "google.com","facebook.com","wikipedia.org","amazon.com","microsoft.com",
    "apple.com","netflix.com","github.com","linkedin.com","paypal.com",
    "cnn.com","bbc.com","nytimes.com","stackoverflow.com","apple.com"
]
PHISH_BASE_DOMAINS = [
    "secure-bank-info.com","paypal-security-update.com","amazon.verify-user.com",
    "netflix.account-update.com","free-giftcard-win.com","gmail-security-check.com",
    "ebay-account-verify.com","appleid-security-reset.com","dropbox.verify-user.com",
    "linkedin.message-alert.com"
]

LEGIT_TEXTS_EN = [
    "Your order has been shipped and will arrive soon.",
    "Your subscription has been renewed successfully.",
    "Meeting scheduled for tomorrow at 10 AM.",
    "Your invoice for order #12345 is attached.",
    "Your account login was successful from a new device."
]
PHISHING_TEXTS_EN = [
    "Your account has been locked. Verify now: <URL>",
    "Update your payment details to avoid suspension: <URL>",
    "Congratulations! You won a prize. Claim now: <URL>",
    "Urgent: confirm your account information here: <URL>",
    "Unusual activity detected. Verify your login at: <URL>"
]

LEGIT_TEXTS_AR = [
    "تم شحن طلبك وسوف يصل قريباً.",
    "تم تجديد اشتراكك بنجاح.",
    "تم جدولة الاجتماع غداً الساعة 10 صباحاً.",
    "الفاتورة مرفقة بالرسالة لمراجعتك.",
    "تم تسجيل الدخول إلى حسابك من جهاز جديد."
]
PHISHING_TEXTS_AR = [
    "تم إيقاف حسابك، قم بالتأكيد هنا: <URL>",
    "تحديث معلومات الدفع لتجنّب الإيقاف: <URL>",
    "تهانينا! فزت بجائزة، استلمها هنا: <URL>",
    "تنبيه أمني، تحقق من حسابك الآن: <URL>",
    "التحقق مطلوب لتجنب الإيقاف: <URL>"
]

# ---------- Helpers ----------
def random_path(phish=False):
    parts = ["", "login", "signin", "auth", "account", "verify", "user", "home", "dashboard", "payment"]
    n = random.randint(0,2)
    segs = random.choices(parts, k=n)
    p = "/".join([s for s in segs if s])
    if p:
        if random.random() < (0.35 if phish else 0.15):
            p += random.choice([".php", ".html", ".asp"])
        return "/" + p
    return ""

def add_query_if_any(url, p_prob=0.35):
    if random.random() < p_prob:
        params = {"id": random.randint(1,999999), "ref": random.choice(["email","sms","ad"]), "utm": random.choice(["a","b","c"])}
        return url + "?" + urlencode(params)
    return url

def phishify_domain(d):
    host = d
    if random.random() < 0.45:
        host = host.replace(".", "-" + str(random.randint(1,999)) + ".", 1)
    if random.random() < 0.35:
        host = random.choice(["secure","login","verify","account"]) + "-" + host
    if random.random() < 0.3:
        host = random.choice(["auth","api","id"]) + "." + host
    scheme = random.choice(["http","http","https"])
    return f"{scheme}://{host}"

def legitify_domain(d):
    host = d
    if not host.startswith("www.") and random.random() < 0.3:
        host = "www." + host
    scheme = "https"
    return f"{scheme}://{host}"

def compute_url_features(url):
    if url is None:
        url = ""
    if '://' not in url:
        full = "http://" + url
    else:
        full = url
    p = urlparse(full)
    scheme = p.scheme
    netloc = p.netloc
    path = p.path or ""
    query = p.query or ""
    url_length = len(full)
    has_ip_address = 1 if netloc.count('.')==3 and all(part.isdigit() for part in netloc.split('.') if part) else 0
    dot_count = netloc.count('.') + path.count('.')
    https_flag = 1 if scheme == 'https' else 0
    uniq_chars = len(set(full))
    url_entropy = round((uniq_chars / (len(full)+1)) * 10, 2)
    token_count = max(1, len([t for t in (path + ("?"+query if query else "")).split('/') if t]))
    subdomain_count = max(0, netloc.count('.') - 1)
    query_param_count = 1 if query else 0
    tld = netloc.split('.')[-1] if '.' in netloc else ''
    tld_length = len(tld)
    path_length = len(path)
    has_hyphen_in_domain = 1 if '-' in netloc else 0
    number_of_digits = sum(c.isdigit() for c in full)
    tld_pop_map = {'com':1000,'org':300,'net':200,'io':100,'co':150,'info':50}
    tld_popularity = tld_pop_map.get(tld, 10)
    suspicious_file_extension = 1 if any(full.endswith(ext) for ext in ['.exe','.zip','.php','.asp']) else 0
    domain_name_length = len(netloc)
    percentage_numeric_chars = round((number_of_digits / (len(full)+1)) * 100, 2)
    return {
        "url_length": url_length,
        "has_ip_address": has_ip_address,
        "dot_count": dot_count,
        "https_flag": https_flag,
        "url_entropy": url_entropy,
        "token_count": token_count,
        "subdomain_count": subdomain_count,
        "query_param_count": query_param_count,
        "tld_length": tld_length,
        "path_length": path_length,
        "has_hyphen_in_domain": has_hyphen_in_domain,
        "number_of_digits": number_of_digits,
        "tld_popularity": tld_popularity,
        "suspicious_file_extension": suspicious_file_extension,
        "domain_name_length": domain_name_length,
        "percentage_numeric_chars": percentage_numeric_chars
    }

def clean_text_for_emails(s):
    if s is None:
        return ""
    s = html.unescape(str(s))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------- Generate ----------
def generate_dataset(num_total=NUM_TOTAL, ratio_texts=RATIO_TEXTS):
    half = num_total // 2
    per_class = half
    num_texts_per_class = int(per_class * ratio_texts)
    num_links_per_class = per_class - num_texts_per_class

    rows = []

    # Legit texts
    for _ in range(num_texts_per_class):
        if random.random() < 0.8:
            txt = random.choice(LEGIT_TEXTS_EN)
        else:
            txt = random.choice(LEGIT_TEXTS_AR)
        rows.append({"text": clean_text_for_emails(txt), "label": 0})

    # Legit links
    for _ in range(num_links_per_class):
        base = random.choice(LEGIT_DOMAINS)
        url = legitify_domain(base)
        if random.random() < 0.6:
            url += random_path(phish=False)
        url = add_query_if_any(url, p_prob=0.25)
        feat = compute_url_features(url)
        row = {"text": url, "label": 0}
        row.update(feat)
        rows.append(row)

    # Phishing texts
    for _ in range(num_texts_per_class):
        if random.random() < 0.8:
            txt = random.choice(PHISHING_TEXTS_EN).replace("<URL>", random.choice(PHISH_BASE_DOMAINS))
        else:
            txt = random.choice(PHISHING_TEXTS_AR).replace("<URL>", random.choice(PHISH_BASE_DOMAINS))
        rows.append({"text": clean_text_for_emails(txt), "label": 1})

    # Phishing links
    for _ in range(num_links_per_class):
        base = random.choice(PHISH_BASE_DOMAINS)
        url = phishify_domain(base)
        if random.random() < 0.9:
            url += random_path(phish=True)
        url = add_query_if_any(url, p_prob=0.6)
        feat = compute_url_features(url)
        row = {"text": url, "label": 1}
        row.update(feat)
        rows.append(row)

    # Normalize: ensure all feature columns exist
    feature_keys = ["text","label",
        "url_length","has_ip_address","dot_count","https_flag","url_entropy","token_count",
        "subdomain_count","query_param_count","tld_length","path_length","has_hyphen_in_domain",
        "number_of_digits","tld_popularity","suspicious_file_extension","domain_name_length","percentage_numeric_chars"
    ]
    normalized = []
    for r in rows:
        nr = {}
        nr["text"] = r.get("text","")
        nr["label"] = int(r.get("label",0))
        # if features present use them, otherwise compute from text if looks like URL else zeros
        present = all(k in r for k in feature_keys[2:])
        if present:
            for k in feature_keys[2:]:
                nr[k] = r.get(k, 0)
        else:
            t = nr["text"]
            if ("http" in t) or ("www." in t) or (t.count(".")>=1 and len(t.split())==1):
                feats = compute_url_features(t)
                for k in feature_keys[2:]:
                    nr[k] = feats.get(k,0)
            else:
                for k in feature_keys[2:]:
                    nr[k] = 0
        normalized.append(nr)

    df_out = pd.DataFrame(normalized, columns=feature_keys)
    df_out = df_out.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_out

def split_and_save(df, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    from sklearn.model_selection import train_test_split
    X = df
    y = df['label']
    test_size = test_ratio
    val_size = val_ratio / (1 - test_size)
    X_temp, X_test = train_test_split(X, test_size=test_size, stratify=y, random_state=42)
    y_temp = X_temp['label']
    X_train, X_val = train_test_split(X_temp, test_size=val_size, stratify=y_temp, random_state=42)
    X.to_csv(OUT_FULL, index=False)
    X_train.to_csv(OUT_TRAIN, index=False)
    X_val.to_csv(OUT_VAL, index=False)
    X_test.to_csv(OUT_TEST, index=False)
    print("Saved files to", OUT_DIR)
    print("Full:", OUT_FULL, "train:", OUT_TRAIN, "val:", OUT_VAL, "test:", OUT_TEST)
    print("Counts -> total:", len(X), "train:", len(X_train), "val:", len(X_val), "test:", len(X_test))
    print("Label counts (full):")
    print(X['label'].value_counts())

if __name__ == "__main__":
    print("Generating dataset size:", NUM_TOTAL)
    df = generate_dataset(NUM_TOTAL, RATIO_TEXTS)
    split_and_save(df)
    print("Done.")