![Demo GIF](https://user-images.githubusercontent.com/74038190/225813708-98b745f2-7d22-48cf-9150-083f1b00d6c9.gif)

# 📊 Customer Churn Prediction

## 📌 Overview

Customer churn is one of the most critical problems in the telecom/retail/financial industries. This project builds a **machine learning model** to predict whether a customer will churn (leave the service) or stay.
The solution includes **data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation**.

---

## ⚙️ Tech Stack

* **Python 3.8+**
* **Pandas**, **NumPy**
* **Matplotlib**, **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
│
├── data/                  # datasets (only sample data, not huge raw data)
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── src/                   # source code
│   ├── preprocessing.py   # data cleaning & feature engineering
│   ├── model.py           # ML models & training code
│   ├── utils.py           # helper functions
│   └── __init__.py
├── screenshots/           # all project images/screenshots
├── requirements.txt       # project dependencies
├── README.md              # project documentation
└── .gitignore             # ignored files (venv, pycache, etc.)
```

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dhananjaya-Verma/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

* Windows:

  ```bash
  venv\Scripts\activate
  ```
* Linux/Mac:

  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Exploratory Data Analysis (EDA)

Key insights generated from the dataset:

* Distribution of churn vs non-churn customers
* Impact of tenure, monthly charges, and contract type on churn
* Correlation heatmap of features

🖼️ **Sample Output (EDA Graphs)**
![EDA Distribution](screenshots/eda_distribution.png)
![Correlation Heatmap](screenshots/correlation_heatmap.png)

---

## 🏗️ Model Training

The following ML models were trained and evaluated:

* Logistic Regression
* Random Forest
* XGBoost
* Support Vector Machine (SVM)
* KNN

**Best performing model:** SVM (accuracy \~84%, AUC \~0.87)

🖼️ **Model Training Snapshot**
![Training Progress](screenshots/model_training.png)

---

## 📈 Results

| Model                  | Accuracy | Precision | Recall | F1-Score | AUC  |
| ---------------------- | -------- | --------- | ------ | -------- | ---- |
| Logistic Regression    | 0.63     | 0.60      | 0.57   | 0.58     | 0.63 |
| Support Vector Machine | 0.84     | 0.81      | 0.78   | 0.79     | 0.87 |
| KNN                    | 0.80     | 0.78      | 0.75   | 0.76     | 0.84 |
| Random Forest          | 0.76     | 0.73      | 0.70   | 0.71     | 0.76 |

---

🖼️ **Confusion Matrix**
![Confusion Matrix](screenshots/confusion_matrix.png)

🖼️ **ROC Curve**
![ROC Curve](screenshots/ROC_curves.png)

---

## 📌 How to Use

Run the prediction script:

```bash
python src/model.py
```

## 🎛️ Streamlit App
![Streamlit App](screenshots/streamlit_app.png)

I built a Streamlit dashboard with:

🔽 Dropdown menu to select ML model (Random Forest, Logistic Regression, SVM, KNN)

🎨 Colorful, interactive UI for predictions

📊 Model results & comparison

📈 Feature importance & SHAP explanations

Run the app locally:
```bash
streamlit run src/streamlit_app.py
```
---

## ✅ Future Improvements

* Hyperparameter tuning with Optuna/RandomizedSearchCV
* Add XGBoost / LightGBM models for comparison
* Deploy app to Streamlit Cloud / Heroku / AWS
* Handle class imbalance with SMOTE
