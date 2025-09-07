import pandas as pd
import re
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib
from utils import extract_text

nltk.download("stopwords", quiet=True)
stopwords = set(nltk.corpus.stopwords.words("english"))

data = pd.read_csv("data/UpdatedResumeDataSet.csv", encoding="utf-8")

def clean_resume(text):
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stopwords])

data["cleaned_resume"] = data["Resume"].apply(clean_resume)

le = LabelEncoder()
data["CategoryEncoded"] = le.fit_transform(data["Category"])

vectorizer = TfidfVectorizer(max_features=1500, stop_words="english")
X = vectorizer.fit_transform(data["cleaned_resume"])
y = data["CategoryEncoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("📊 Model Evaluation:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Save model + vectorizer + label encoder
joblib.dump(clf, "src/model.pkl")
joblib.dump(vectorizer, "src/vectorizer.pkl")
joblib.dump(le, "src/labelencoder.pkl")

def predict_resume(file_path, top_n=3):
    resume_text = extract_text(file_path)
    cleaned = clean_resume(resume_text)
    vec = vectorizer.transform([cleaned])
    proba = clf.predict_proba(vec)[0]
    top_indices = np.argsort(proba)[::-1][:top_n]
    results = [(le.inverse_transform([i])[0], proba[i]*100) for i in top_indices]
    return results
