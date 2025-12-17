import streamlit as st
import requests

st.title("📝 Sentiment Analysis")

text = st.text_area("Enter your review")

if st.button("Analyze"):
    if text.strip():
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": text}
        )

        result = response.json()

        if result["sentiment"] == "Positive":
            st.success(f"Positive 😊 (Confidence: {result['probability']:.2f})")
        else:
            st.error(f"Negative 😞 (Confidence: {result['probability']:.2f})")
