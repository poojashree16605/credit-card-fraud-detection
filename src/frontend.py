import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Settings
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

uploaded = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Remove Class column if present
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    st.subheader("Data Preview:")
    st.write(df.head())

    if st.button("Check Fraud"):
        st.info("Processing... Please wait...")

        # Prepare data for API
        payload = {"data": df.to_dict(orient="records")}

        try:
            res = requests.post("http://localhost:8000/predict_batch", json=payload)

            if res.ok:
                result = res.json()

                # Add prediction results
                df["prediction"] = result["predictions"]
                df["probability"] = result["probabilities"]

                st.subheader("Prediction Results:")
                st.write(df.head())

                # Count frauds
                flagged = df[df["prediction"] == 1]
                st.error(f"⚠️ Detected {len(flagged)} Fraudulent Transactions")

                # -------------------------------
                # 📌 DOWNLOAD RESULTS CSV
                # -------------------------------
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="⬇ Download Full Predictions CSV",
                    data=csv_data,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

                # -------------------------------
                # 📊 VISUALIZATION DASHBOARD
                # -------------------------------
                st.subheader("📊 Fraud Detection Visualizations")

                # 1️⃣ Fraud vs Non-Fraud Count
                st.write("### 1️⃣ Fraud vs Non-Fraud Count")
                count_fig, ax = plt.subplots()
                sns.countplot(x=df["prediction"], palette="coolwarm", ax=ax)
                ax.set_xticklabels(["Not Fraud (0)", "Fraud (1)"])
                st.pyplot(count_fig)

                # 2️⃣ Fraud Amount Distribution (if Amount column exists)
                if "Amount" in df.columns:
                    st.write("### 2️⃣ Fraud Amount Distribution")
                    amt_fig, ax = plt.subplots()
                    sns.histplot(
                        data=df,
                        x="Amount",
                        hue="prediction",
                        bins=50,
                        kde=True,
                        palette="coolwarm",
                        ax=ax
                    )
                    st.pyplot(amt_fig)

                # 3️⃣ Fraud Probability Distribution
                st.write("### 3️⃣ Fraud Probability Distribution")
                prob_fig, ax = plt.subplots()
                sns.histplot(df["probability"], bins=50, kde=True, color="purple", ax=ax)
                ax.set_title("Probability Distribution")
                st.pyplot(prob_fig)

                # 4️⃣ Pie Chart
                st.write("### 4️⃣ Fraud vs Non-Fraud Pie Chart")
                pie_fig, ax = plt.subplots()
                sizes = df["prediction"].value_counts()
                labels = ["Not Fraud", "Fraud"]
                ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["skyblue", "red"])
                ax.set_title("Fraud Percentage")
                st.pyplot(pie_fig)

            else:
                st.error("API Error: Could not get predictions. Check backend logs.")

        except Exception as e:
            st.error(f"Error connecting to API: {e}")

