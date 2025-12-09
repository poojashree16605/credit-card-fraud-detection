import streamlit as st
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt

# Load ML model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection (Offline Mode)")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Remove Class if present
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    st.subheader("Data Preview:")
    st.write(df.head())

    if st.button("Check Fraud"):
        st.info("Processing... Please wait...")

        # Predict
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["prediction"] = predictions
        df["probability"] = probabilities

        st.subheader("Prediction Results:")
        st.write(df.head())

        # Fraud count
        fraud = df[df["prediction"] == 1]
        st.error(f"⚠️ Detected {len(fraud)} Fraudulent Transactions")

        # Download results
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("⬇ Download Results CSV",
                           csv_buffer.getvalue(),
                           "fraud_results.csv",
                           "text/csv")

        # Plot
        fig, ax = plt.subplots()
        df["prediction"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Fraud vs Non-Fraud")
        st.pyplot(fig)
