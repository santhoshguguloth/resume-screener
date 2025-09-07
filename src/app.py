import os
from flask import Flask, request, render_template, redirect
import joblib
from utils import extract_text
import re
import nltk

# Initialize Flask app
app = Flask(__name__, template_folder="templates")  # ensure Flask looks in the right folder

# Load ML models
clf = joblib.load("src/model.pkl")
vectorizer = joblib.load("src/vectorizer.pkl")
le = joblib.load("src/labelencoder.pkl")

# NLTK stopwords
nltk.download("stopwords", quiet=True)
stopwords = set(nltk.corpus.stopwords.words("english"))

# Upload folder
UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Resume cleaning function
def clean_resume(text):
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stopwords])

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "resume" not in request.files:
            return redirect(request.url)

        file = request.files["resume"]
        if file.filename == "":
            return redirect(request.url)

        # Save file safely
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)

        # Extract and preprocess text
        text = extract_text(filename)
        cleaned = clean_resume(text)
        vec = vectorizer.transform([cleaned])
        proba = clf.predict_proba(vec)[0]

        # Top prediction
        top_index = proba.argmax()
        predicted_category = le.inverse_transform([top_index])[0]
        confidence = proba[top_index] * 100

        # Decision threshold
        decision = "✅ Selected" if confidence >= 70 else "❌ Not Selected"

        # Render result
        return render_template("result.html",
                               filename=file.filename,
                               results=[(predicted_category, confidence)],
                               decision=decision)

    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
