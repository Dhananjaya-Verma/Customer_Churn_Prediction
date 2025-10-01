import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards
import os
import shap

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------- MODEL PATHS ----------
MODEL_DIR = "models"
models = {
    "Random Forest": os.path.join(MODEL_DIR, "churn_pipeline.pkl"),
    "KNN": os.path.join(MODEL_DIR, "churn_pipeline_knn.pkl"),
    "SVM": os.path.join(MODEL_DIR, "churn_pipeline_svm.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "churn_pipeline_logistic_regression.pkl"),
}

# ---------- LOAD PIPELINE ----------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ---------- CUSTOM STYLING ----------
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        color: white;
        background: #0078D7;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background: #005A9E;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.image(
    "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png",
    width=200
)
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict Churn", "🔍 Model Insights", "ℹ️ About"])

# Sidebar model selection
st.sidebar.markdown("### 🤖 Choose Model")
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
pipeline = load_model(models[model_choice])

# ---------- HOME ----------
if page == "🏠 Home":
    st.title("📊 Customer Churn Prediction App")

    # Banner Image
    st.image(
        "https://images.unsplash.com/photo-1533750349088-cd871a92f312?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
        use_container_width=True,
        caption="Customer Churn & Retention Analytics"
    )

    st.write(f"""
    Welcome to the **Churn Prediction Dashboard**.  
    You are currently using the **{model_choice}** model.  
    
    ---
    ### 📌 Why Churn Prediction?
    - Retaining an existing customer is far cheaper than acquiring a new one.
    - By predicting which customers are likely to leave, companies can take **preventive actions**.
    - Churn prediction is widely used in **telecom, banking, SaaS, and subscription businesses**.
    """)

# ---------- PREDICT ----------
elif page == "📈 Predict Churn":
    st.title(f"📈 Predict Customer Churn ({model_choice})")

    st.subheader("Enter Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
        phone = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
        total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])

    # Build input dataframe
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    if st.button("🔮 Predict"):
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error(f"⚠️ Customer is **likely to churn** (Risk: {probability:.2f})")
            else:
                st.success(f"✅ Customer is **not likely to churn** (Risk: {1-probability:.2f})")

        with col2:
            st.metric("Churn Probability", f"{probability:.2%}")
            st.metric("Prediction", "Churn" if prediction == 1 else "No Churn")

        style_metric_cards(background_color="#FFFFFF", border_left_color="#0078D7", border_color="#CCCCCC")

elif page == "🔍 Model Insights":
    import shap

    st.title("🔍 Model Insights")
    st.write("Here are the most important features driving churn predictions:")

    try:
        clf = pipeline.named_steps.get("classifier", pipeline)

        # ---- Feature Names Extraction ----
feature_names = None
if "preprocessor" in pipeline.named_steps:
    preprocessor = pipeline.named_steps["preprocessor"]

    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]

    cat_transformer = preprocessor.transformers_[1][1]

    # Case 1: categorical transformer is a pipeline
    if hasattr(cat_transformer, "named_steps"):
        ohe = cat_transformer.named_steps.get("onehot")
    # Case 2: directly OneHotEncoder
    elif isinstance(cat_transformer, (type(preprocessor),)):
        ohe = cat_transformer
    else:
        ohe = None

    if ohe is not None and hasattr(ohe, "get_feature_names_out"):
        cat_expanded = ohe.get_feature_names_out(cat_features)
        feature_names = list(num_features) + list(cat_expanded)
    else:
        feature_names = list(num_features) + list(cat_features)


        # ---- Random Forest ----
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(8, 5))
            feat_imp.plot(kind="barh", ax=ax, color="#0078D7")
            ax.set_title("Top 15 Feature Importances (Random Forest)")
            ax.invert_yaxis()
            st.pyplot(fig)

        # ---- Logistic Regression / Linear SVM ----
        elif hasattr(clf, "coef_"):
            importances = clf.coef_[0]
            feat_imp = pd.Series(importances, index=feature_names).sort_values(key=abs, ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(8, 5))
            feat_imp.plot(kind="barh", ax=ax, color="#FF5733")
            ax.set_title("Top 15 Feature Coefficients (Linear Model)")
            ax.invert_yaxis()
            st.pyplot(fig)

        # ---- SHAP for non-linear models (KNN, SVM, etc.) ----
        else:
            st.info("ℹ️ Using SHAP values since this model has no built-in feature importance.")

            # sample few rows for SHAP
            df = pd.read_csv(r"D:\CSE(DataScience)\Customer_Churn_Prediction\Datasets\Telco-Customer-Churn-dataset.csv")
            X_sample = df.drop("Churn", axis=1).sample(100, random_state=42)

            # preprocess first
            X_transformed = pipeline.named_steps["preprocessor"].transform(X_sample)

            # use KernelExplainer instead of direct Explainer for KNN/SVM
            explainer = shap.KernelExplainer(clf.predict_proba, X_transformed[:50])
            shap_values = explainer.shap_values(X_transformed[:10], nsamples=100)

            st.subheader("📊 SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, X_transformed[:10], feature_names=feature_names, show=False)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not compute feature importance/SHAP values: {e}")


# ---------- ABOUT ----------
elif page == "ℹ️ About":
    st.title("ℹ️ About this Project")
    st.markdown("""
    This project predicts **customer churn** using multiple models:
    - Random Forest
    - KNN
    - SVM
    - Logistic Regression  

    **Tech Stack:**
    - Python (pandas, scikit-learn, joblib)
    - Machine Learning Pipelines
    - Streamlit for interactive UI
    - Deployed on Streamlit Cloud  

    **Author:** Dhananjaya Verma 
    """)

# ---------- FOOTER ----------
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #333333;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #d1d1d1;
    }
    </style>
    <div class="footer">
        📊 Customer Churn Prediction App | Built with ❤️ by <b>Dhananjaya Verma</b> |
        <a href="https://github.com/Dhananjaya-Verma/Customer-Churn-Prediction" target="_blank">GitHub</a> ·
        <a href="https://www.linkedin.com/in/dhananjaya-verma-661611224/" target="_blank">LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
