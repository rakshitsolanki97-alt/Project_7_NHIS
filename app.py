
from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load("disaster_tweet_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words]
    return " ".join(filtered_tokens)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    tweet = ""

    if request.method == "POST":
        tweet = request.form["tweet"]

        processed_text = preprocess_text(tweet)
        vectorized_text = vectorizer.transform([processed_text])

        predicted_class = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]

        prediction = (
            "Disaster"
            if predicted_class == 1
            else "Non-Disaster")

        confidence = round(
            probabilities[predicted_class] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        tweet=tweet)


if __name__ == "__main__":
    app.run(debug=True)
