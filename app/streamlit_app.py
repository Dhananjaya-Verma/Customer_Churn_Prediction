import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------- LOAD PIPELINE ----------
@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.pkl")

pipeline = load_model()

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

# ---------- HOME ----------
if page == "🏠 Home":
    st.title("📊 Customer Churn Prediction App")

    # Banner Image (Royalty-free from Unsplash)
    st.image(
        "https://images.unsplash.com/photo-1533750349088-cd871a92f312?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
        use_container_width=True,
        caption="Customer Churn & Retention Analytics"
    )

    st.write("""
    Welcome to the **Churn Prediction Dashboard**.  
    This tool helps identify customers who are at risk of leaving a company (**customer churn**) 
    using **Machine Learning**.

    ---
    ### 📌 Why Churn Prediction?
    - Retaining an existing customer is far cheaper than acquiring a new one.
    - By predicting which customers are likely to leave, companies can take **preventive actions** 
      such as discounts, better service, or loyalty programs.
    - Churn prediction is widely used in **telecom, banking, SaaS, and subscription businesses**.

    ---
    ### 🚀 What This App Does
    - 📈 Predicts churn for an individual customer using demographic & service details.
    - 🔍 Provides insights into the **most important features** influencing churn.
    - 🧠 Built using a **RandomForestClassifier** inside a Scikit-learn pipeline.
    - 🎨 Interactive, user-friendly dashboard powered by **Streamlit**.

    ---
    Use the sidebar to:
    - 📈 **Predict Churn** → Test the model with custom customer details.
    - 🔍 **Model Insights** → Explore which features affect churn the most.
    - ℹ️ **About** → Learn about the project and tech stack.
    """)

# ---------- PREDICT ----------
elif page == "📈 Predict Churn":
    st.title("📈 Predict Customer Churn")

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
                st.success(f"✅ Customer is **not likely to churn** (Risk: {probability:.2f})")

        with col2:
            st.metric("Churn Probability", f"{probability:.2%}")
            st.metric("Prediction", "Churn" if prediction == 1 else "No Churn")

        style_metric_cards(background_color="#FFFFFF", border_left_color="#0078D7", border_color="#CCCCCC")

# ---------- INSIGHTS ----------
elif page == "🔍 Model Insights":
    st.title("🔍 Model Insights")
    st.write("Here are the most important features driving churn predictions:")

    try:
        clf = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][2]
        ohe = preprocessor.transformers_[1][1].named_steps["onehot"]
        cat_expanded = ohe.get_feature_names_out(cat_features)

        all_features = list(num_features) + list(cat_expanded)

        importances = clf.feature_importances_
        feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 5))
        feat_imp.plot(kind="barh", ax=ax)
        ax.set_title("Top 15 Feature Importances")
        ax.invert_yaxis()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")

# ---------- ABOUT ----------
elif page == "ℹ️ About":
    st.title("ℹ️ About this Project")
    st.markdown("""
    This project predicts **customer churn** using a **RandomForestClassifier** 
    trained on telecom customer data.  

    **Tech Stack:**
    - Python (pandas, scikit-learn, joblib)
    - Machine Learning Pipeline
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