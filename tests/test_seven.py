import pickle
import os
import numpy as np

MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


def test_model_files_exist():
    assert os.path.exists(MODEL_PATH), "sentiment_model.pkl is missing"
    assert os.path.exists(VECTORIZER_PATH), "tfidf_vectorizer.pkl is missing"


def test_model_prediction_shape():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    sample_text = ["I love this product"]
    vectorized = vectorizer.transform(sample_text)
    prediction = model.predict(vectorized)

    assert prediction.shape == (1,)
