# 📝 **README.md — Credit Card Fraud Detection System**

```md
# 💳 Credit Card Fraud Detection System  
A complete end-to-end Machine Learning project that detects fraudulent credit card transactions using a Random Forest model, deployed with **FastAPI (backend)** and **Streamlit (frontend)**.

---

## 🚀 Project Architecture

```

User → Streamlit App → FastAPI Backend → ML Model → Prediction → Dashboard

```

---

## 🎯 Features

### 🔹 **Frontend (Streamlit)**
- Upload transaction CSV file  
- Automatic preprocessing  
- Fraud probability prediction  
- Interactive dashboards:
  - Fraud vs Non-Fraud count  
  - Probability distribution  
  - Amount distribution  
  - Pie charts  
- Downloadable results CSV  

### 🔹 **Backend (FastAPI)**
- `/` → Health check endpoint  
- `/predict_batch` → Predict fraud for multiple transactions  
- Accepts JSON payload of CSV records  
- Returns predictions + fraud probability  

### 🔹 **Machine Learning**
- Random Forest Classifier trained on **Kaggle Credit Card Fraud Dataset**  
- Handles feature imbalance  
- Model stored as `fraud_model.pkl`

---

## 📦 Tech Stack

| Component     | Technology |
|---------------|------------|
| Frontend      | Streamlit |
| Backend API   | FastAPI + Uvicorn |
| ML Model      | scikit-learn + pandas + numpy |
| Deployment    | Streamlit Cloud + Render |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure

```

credit-card-fraud/
│
├── data/
│   └── creditcard.csv
│
├── src/
│   ├── api.py                 # FastAPI backend
│   ├── frontend.py            # Streamlit UI
│   ├── preprocessing.py       # Data processing functions
│   ├── train_model.py         # ML training script
│   └── artifacts/
│       └── fraud_model.pkl    # Saved ML model
│
├── requirements.txt
└── README.md

```

---

## 🔥 Deployment Links

### 🌐 **Frontend (Streamlit)**  
👉 *Add your deployed Streamlit link here*

### ⚙️ **Backend API (FastAPI on Render)**  
`https://fraud-api-pooja.onrender.com`

### API Endpoint  
```

POST /predict_batch
Content-Type: application/json

````

Sample request:
```json
{
  "data": [
    {
      "Time": 0,
      "V1": -1.3598,
      "V2": -0.0728,
      "V3": 2.5363,
      "V4": 1.3782
    }
  ]
}
````

---

## 🖥️ Run Locally

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/poojashree16605/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Start Backend

```bash
uvicorn src.api:app --reload
```

### 4️⃣ Start Streamlit App

```bash
streamlit run src/frontend.py
```

---

## 💡 Future Enhancements

* Add user authentication
* Build dashboard for live monitoring
* Deploy full-stack version
* Add SMS/email alerts for fraud detection

---

## 🤝 Contributions

Contributions are welcome!
Feel free to fork this repository and submit a pull request.

---

## 📞 Contact

**Pooja Shree**
GitHub: [https://github.com/poojashree16605](https://github.com/poojashree16605)
LinkedIn: https://www.linkedin.com/in/poojashree-s16/

```


