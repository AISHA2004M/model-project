import pandas as pd
import numpy as np
import re

# ===============================
# توليد بيانات dummy متوازنة أكبر
# ===============================
def generate_large_dummy_data(n_per_class=5000):
    # Legit
    legit_texts = [f"Welcome to our website, user {i}" for i in range(n_per_class)]
    legit_urls = [f"https://www.legit{i}.com/login" for i in range(n_per_class)]

    # Phishing
    phishing_texts = [f"Your account has been compromised, click here {i}" for i in range(n_per_class)]
    phishing_urls = [f"http://192.168.{i%255}.{i%255}/login.php" for i in range(n_per_class)]

    # DataFrames
    df_legit = pd.DataFrame({
        'text': legit_texts,
        'URL': legit_urls,
        'label': 0
    })

    df_phish = pd.DataFrame({
        'text': phishing_texts,
        'URL': phishing_urls,
        'label': 1
    })

    df = pd.concat([df_legit, df_phish]).sample(frac=1, random_state=42).reset_index(drop=True)

    # ميزات URL يدوية بسيطة
    df['url_length'] = df['URL'].apply(len)
    df['has_ip_address'] = df['URL'].apply(lambda x: 1 if re.search(r'\d+\.\d+\.\d+\.\d+', x) else 0)
    df['dot_count'] = df['URL'].apply(lambda x: x.count('.'))
    df['https_flag'] = df['URL'].apply(lambda x: 1 if x.startswith('https') else 0)
    df['subdomain_count'] = df['URL'].apply(lambda x: x.count('.') - 1)

    return df

# توليد 10000 صف (5000 Legit + 5000 Phishing)
df_large = generate_large_dummy_data(5000)

# حفظ CSV
df_large.to_csv('Phishing_large_balanced.csv', index=False)
print("تم إنشاء وحفظ ملف CSV بنجاح: Phishing_large_balanced.csv")