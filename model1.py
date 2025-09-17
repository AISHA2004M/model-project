# ===============================
# Phishing Detection - Improved Model
# ===============================
import pandas as pd
import numpy as np
import re
import html
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer, util
import torch
import joblib

# ===============================
# 1️⃣ قراءة البيانات
# ===============================
DATA_FILE = '/Users/yahyamohnd/Downloads/URL_final.csv'
df = pd.read_csv(DATA_FILE)

# ===============================
# 2️⃣ تنظيف النصوص
# ===============================
def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'http\S+', '', text)  # إزالة الروابط
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # إزالة الرموز
    text = text.lower().strip()
    return text

df['text'] = df['text'].apply(clean_text)

# ===============================
# 3️⃣ ميزات URL
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
# 4️⃣ TF-IDF للنصوص (ngrams + أكثر ميزات)
# ===============================
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_text_vec = vectorizer.fit_transform(df['text'])

# ===============================
# 5️⃣ دمج النصوص + الميزات
# ===============================
X_final = hstack([X_text_vec, X_url_features.values])
y_final = df['label'].astype(int)

# ===============================
# 6️⃣ تقسيم بيانات التدريب والاختبار
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# ===============================
# 7️⃣ نموذج Random Forest
# ===============================
clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ===============================
# 8️⃣ تقييم النموذج
# ===============================
y_pred = clf.predict(X_test)
print("=== Random Forest Model (Large Data) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===============================
# 9️⃣ نموذج QA - اختياري
# ===============================
qa_model = SentenceTransformer('all-MiniLM-L6-v2')

def answer_question(question, df_subset=None, top_k=5):
    if df_subset is None:
        df_subset = df
    texts = df_subset['text'].tolist()
    embeddings = qa_model.encode(texts, convert_to_tensor=True)
    q_emb = qa_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0]
    top_results = torch.topk(scores, k=min(top_k, len(df_subset)))
    for score, idx in zip(top_results.values, top_results.indices):
        idx_int = idx.item()
        print(f"Score: {score:.4f}, Text: {df_subset.iloc[idx_int]['text']}, Label: {df_subset.iloc[idx_int]['label']}")

# ===============================
# 🔟 دالة تصنيف أي نص أو رابط
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
# 1️⃣1️⃣ الوضع التفاعلي
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
            answer_question(question, df_subset=df.head(100))
        else:
            pred = classify_any_text(user_input)
            print("Prediction:", pred)

# ===============================
# 1️⃣2️⃣ حفظ النموذج
# ===============================
joblib.dump(clf, 'rf_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("تم حفظ النموذج وTF-IDF بنجاح!")

# ===============================
# 1️⃣3️⃣ تشغيل الوضع التفاعلي
# ===============================
if __name__ == "__main__":
    interactive_mode()