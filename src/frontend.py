import streamlit as st
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Page settings
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detection – Offline Mode")

# Load trained model
model = joblib.load("src/artifacts/fraud_model.pkl")

uploaded = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Remove Class column if present
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    st.subheader("Data Preview:")
    st.write(df.head())

    if st.button("Check Fraud"):
        st.info("Processing… Please wait...")

        # MODEL PREDICTION (NO API)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["prediction"] = predictions
        df["probability"] = probabilities

        st.subheader("Prediction Results:")
        st.write(df.head())

        fraud_cases = df[df["prediction"] == 1]
        st.error(f"⚠️ Fraud Detected: {len(fraud_cases)} transactions")

        # Download CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇ Download Fraud Report CSV",
            csv_data,
            "fraud_results.csv",
            "text/csv"
        )

        # Visualizations
        st.subheader("📊 Fraud Detection Charts")

        # Bar chart
        fig, ax = plt.subplots()
        sns.countplot(x=df["prediction"], palette="coolwarm", ax=ax)
        ax.set_xticklabels(["Not Fraud (0)", "Fraud (1)"])
        st.pyplot(fig)

        # Probability distribution
        fig2, ax2 = plt.subplots()
        sns.histplot(df["probability"], bins=40, kde=True, color="purple", ax=ax2)
        ax2.set_title("Probability Distribution")
        st.pyplot(fig2)


