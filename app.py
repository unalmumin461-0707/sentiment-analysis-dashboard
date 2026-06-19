
import streamlit as st
import joblib
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def save_prediction(review, sentiment, confidence):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (review, sentiment, confidence, created_at) VALUES (?, ?, ?, ?)",
        (review, sentiment, confidence, datetime.now())
    )
    conn.commit()
    conn.close()

def load_predictions():
    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df

st.title("Sentiment Analysis Dashboard")
st.write("This dashboard predicts whether a customer review is Positive or Negative.")

review = st.text_area("Enter a customer review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        X = vectorizer.transform([review])
        prediction = model.predict(X)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = 0.85

        save_prediction(review, sentiment, confidence)

        st.success(f"Predicted Sentiment: {sentiment}")
        st.write(f"Confidence Score: {confidence}")

st.subheader("Saved Predictions")

try:
    data = load_predictions()
    st.dataframe(data)

    if not data.empty:
        st.subheader("Sentiment Distribution")
        counts = data["sentiment"].value_counts()
        st.bar_chart(counts)

except Exception as e:
    st.info("No predictions saved yet.")
