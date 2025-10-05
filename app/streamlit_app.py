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
    "Random Forest": os.path.join(MODEL_DIR, "churn_pipeline_random_forest.pkl"),
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

    /* Metric card custom styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #0078D7;
    }
    [data-testid="stMetricDelta"] {
        font-size: 16px;
    }
    [data-testid="stMetric"] {
        background: #f9f9f9;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        text-align: center;
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

    # 🔮 Predict button yahan hai
    if st.button("🔮 Predict"):
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)

        # ✅ Left column → normal Streamlit success/error box
        with col1:
            if prediction == 1:
                st.error(f"⚠️ Customer is **likely to churn** (Risk: {probability:.2f})")
            else:
                st.success(f"✅ Customer is **not likely to churn** (Risk: {1-probability:.2f})")

        # ✅ Right column → custom styled card (gray background)
        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: #f2f2f2;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
                    text-align: center;
                ">
                    <h4 style="margin:0; color:#0078D7;">Churn Probability</h4>
                    <p style="font-size:28px; margin:5px 0; color:#333;">{probability:.2%}</p>
                    <h4 style="margin:0; color:#0078D7;">Prediction</h4>
                    <p style="font-size:24px; margin:5px 0; color:#333;">
                        {"Churn" if prediction == 1 else "No Churn"}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

elif page == "🔍 Model Insights":
    st.title(f"🔍 Model Insights ({model_choice})")

    import shap
    import matplotlib.pyplot as plt

    try:
        # Extract classifier step
        if "model" in pipeline.named_steps:
            model = pipeline.named_steps["model"]
        elif "classifier" in pipeline.named_steps:
            model = pipeline.named_steps["classifier"]
        else:
            model = pipeline

        # Extract preprocessor
        preprocessor = pipeline.named_steps.get("preprocessor", None)

        # Get feature names
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = input_data.columns  # fallback

        importances = None
        method = None

        # Random Forest → feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            method = "Tree-based Feature Importances"

        # Logistic Regression → coefficients
        elif hasattr(model, "coef_"):
            importances = abs(model.coef_[0])
            method = "Logistic Regression Coefficients"

        # If still None → fallback to SHAP (for SVM, KNN, etc.)
        if importances is None:
            st.info("⚡ Using SHAP for feature importance (since this model doesn’t expose coefficients).")

            # Sample 200 rows from training data for efficiency
            X_sample = X.sample(n=min(200, len(X)), random_state=42)
            X_transformed = preprocessor.transform(X_sample) if preprocessor else X_sample

            explainer = shap.Explainer(model, X_transformed)
            shap_values = explainer(X_transformed)

            importances = np.abs(shap_values.values).mean(axis=0)
            method = "SHAP Values"

        # Create DataFrame
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)

        st.subheader(f"📊 Top 15 Features Driving Churn ({method})")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        importance_df.plot(
            kind="barh",
            x="Feature",
            y="Importance",
            legend=False,
            ax=ax,
            color="skyblue"
        )
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Top 15 Feature Importances")
        plt.gca().invert_yaxis()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")


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
