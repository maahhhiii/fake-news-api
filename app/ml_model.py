import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))


def predict_news(news_text: str):

    news_vector = vectorizer.transform([news_text])

    prediction = model.predict(news_vector)[0]

    if prediction == 1:
        result = "REAL"
    else:
        result = "FAKE"

    return {
        "prediction": result
    }