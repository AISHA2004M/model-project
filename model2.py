# ===============================
# Phishing Detection - Training + Interactive QA
# ===============================
import pandas as pd
import numpy as np
import re
import html
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer, util
import torch
import joblib
import os

# ===============================
# 1️⃣ قراءة البيانات
# ===============================
DATA_FILE = 'Phishing_large_dummy_data.csv'
MODEL_FILE = 'logistic_model.pkl'
VECT_FILE = 'tfidf_vectorizer.pkl'

df = pd.read_csv(DATA_FILE)

def clean_text(s):
    if pd.isna(s):
        return ''
    s = str(s)
    s = html.unescape(s)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = s.lower().strip()
    return s

df['text'] = df['text'].apply(clean_text)

# ===============================
# 2️⃣ الميزات اليدوية للـ URL
# ===============================
url_feature_cols = [
    'url_length', 'has_ip_address', 'dot_count', 'https_flag', 'url_entropy', 'token_count',
    'subdomain_count', 'query_param_count', 'tld_length', 'path_length', 'has_hyphen_in_domain',
    'number_of_digits', 'tld_popularity', 'suspicious_file_extension', 'domain_name_length',
    'percentage_numeric_chars'
]
existing_url_features = [col for col in url_feature_cols if col in df.columns]
X_url_features = df[existing_url_features].fillna(0)

# ===============================
# 3️⃣ تدريب النموذج أو تحميله
# ===============================
if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
    print("تحميل النموذج والـ TF-IDF الموجودين...")
    clf = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
else:
    print("تدريب نموذج جديد...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_text_vec = vectorizer.fit_transform(df['text'])
    X_final = hstack([X_text_vec, X_url_features.values])
    y_final = df['label'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("=== Logistic Regression Model (Large Data) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # حفظ النموذج والـ TF-IDF
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(vectorizer, VECT_FILE)
    print("تم حفظ النموذج وTF-IDF!")

# ===============================
# 4️⃣ دالة تصنيف أي نص أو رابط
# ===============================
def classify_any_text(text, url_features=None):
    vec = vectorizer.transform([text])
    if url_features is None or len(existing_url_features) == 0:
        url_feat = np.zeros((1, len(existing_url_features)))
    else:
        url_feat = np.array([url_features.get(col,0) for col in existing_url_features]).reshape(1,-1)
    X_input = hstack([vec, url_feat])
    pred = clf.predict(X_input)[0]
    return "Phishing" if pred==1 else "Legit"

# ===============================
# 5️⃣ نموذج QA اختياري وسريع
# ===============================
qa_model = SentenceTransformer('all-MiniLM-L6-v2')

def answer_question(question, df_subset=None, top_k=5):
    if df_subset is None:
        df_subset = df
    embeddings = qa_model.encode(df_subset['text'].tolist(), convert_to_tensor=True)
    q_emb = qa_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0]
    top_results = torch.topk(scores, k=min(top_k, len(df_subset)))
    for score, idx in zip(top_results.values, top_results.indices):
        print(f"Score: {score:.4f}, Text: {df_subset.iloc[int(idx)]['text']}, Label: {df_subset.iloc[int(idx)]['label']}")

# ===============================
# 6️⃣ الوضع التفاعلي
# ===============================
def interactive_mode():
    print("مرحبا! أدخلي نص أو رابط لتصنيفه، أو اكتبي 'QA:سؤالك' لأمثلة، واكتبي 'exit' للخروج.")
    while True:
        user_input = input("\nأدخل النص / الرابط أو سؤال QA: ").strip()
        if user_input.lower() == 'exit':
            print("تم إنهاء الجلسة.")
            break
        elif user_input.lower().startswith('qa:'):
            question = user_input[3:].strip()
            answer_question(question, df_subset=df.head(100))  # subset صغير لتسريع
        else:
            pred = classify_any_text(user_input)
            print("Prediction:", pred)

# ===============================
# 7️⃣ بدء التفاعل
# ===============================
if __name__ == "__main__":
    interactive_mode()