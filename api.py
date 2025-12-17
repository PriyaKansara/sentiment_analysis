from fastapi import FastAPI
from pydantic import BaseModel
import pickle

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

app = FastAPI(title="Sentiment Analysis API")

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    text_vec = vectorizer.transform([review.text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec).max()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "sentiment": sentiment,
        "probability": float(probability)
    }
