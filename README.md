# 🚢 AI/ML Titanic Survival Prediction

An end-to-end Machine Learning project that predicts Titanic passenger survival using a clean, modular Python pipeline and an interactive Streamlit web app.

The app allows real-time passenger input and predicts survival probability using the trained machine learning model.

---

## 📁 Project Structure

```
AI/ML Titanic-Survival-Prediction/
│
├── data/
│   ├── raw/                   # Auto-downloaded train.csv & test.csv
│   └── processed/             # Cleaned, feature-engineered dataset
│
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── utils.py               # Path constants & logger
│   ├── data_loader.py         # Load / auto-fetch raw data
│   ├── preprocessing.py       # Imputation & encoding
│   ├── feature_engineering.py # Title, FamilySize, IsAlone, FarePerPerson
│   ├── model_training.py      # Train RF / GB / LR / XGBoost, select best
│   └── evaluation.py          # Metrics JSON + feature importance plot
│
├── models/
│   ├── best_model.pkl         # Best classifier (joblib)
│   └── scaler.pkl             # StandardScaler (joblib)
│
├── reports/
│   ├── model_metrics.json     # Accuracy, F1, AUC, etc.
│   └── feature_importance.png # Top-N feature importance bar chart
│
├── app/
│   └── app.py                 # Streamlit prediction UI
│
├── main.py                    # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full ML pipeline

```bash
python main.py
```

This will:
- Auto-fetch the Titanic dataset from OpenML (first run only)
- Engineer features and preprocess the data
- Train four classifiers (RandomForest, GradientBoosting, LogisticRegression, XGBoost)
- Select the best model by 5-fold cross-validation accuracy
- Save `models/best_model.pkl` and `models/scaler.pkl`
- Save `reports/model_metrics.json` and `reports/feature_importance.png`

### 3. Launch the Streamlit app

```bash
streamlit run app/app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 📊 Models Trained

| Model               | Description                                     |
|---------------------|-------------------------------------------------|
| RandomForest        | 200 trees, max_depth=6                          |
| GradientBoosting    | 200 estimators, learning_rate=0.1, max_depth=4  |
| LogisticRegression  | max_iter=1000                                   |
| XGBoost             | 200 estimators, learning_rate=0.1 *(optional)*  |

---

## 🛠️ Engineered Features

| Feature        | Description                              |
|----------------|------------------------------------------|
| `Title`        | Encoded passenger title (Mr/Miss/Mrs/…)  |
| `FamilySize`   | `SibSp + Parch + 1`                      |
| `IsAlone`      | 1 if `FamilySize == 1`                   |
| `FarePerPerson`| `Fare / FamilySize`                      |

---

## 📈 Reported Metrics

Stored in `reports/model_metrics.json`:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Full classification report per class

---

## 🔮 Interactive Prediction

The Streamlit app lets you set passenger attributes (class, sex, age, family size, fare, embarkation port) via sliders and dropdowns, then displays the predicted survival outcome and probability in real-time.

---

## 📦 Requirements

- Python ≥ 3.9
- pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, joblib, streamlit

## Author 
Aniket Chaudhary
