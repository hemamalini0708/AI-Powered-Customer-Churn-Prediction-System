import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS STYLING ============
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title-text {
        color: #1a1a2e;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle-text {
        color: #4a4a68;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        color: #16213e;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #6a11cb;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============ FIND MODEL FILES ============
@st.cache_resource
def find_model_files():
    """Find model files in common locations and return a dict of paths."""
    search_paths = [
        Path.cwd(),                     # Current working directory (project root)
        Path.cwd() / "models",          # models folder (plural)
        Path.cwd() / "model",           # model folder (singular) ← ADD THIS
        Path.cwd() / "app",              # app folder
        Path(__file__).parent,           # directory of this script (app/)
        Path(__file__).parent / "model", # model folder inside app/
        Path.home(),                     # Home directory
        Path("/mnt/user-data/uploads"),  # Cloud storage
    ]

    required_files = {
        "customer_churn.pkl": None,
        "scalar.pkl": None,
        "one_hot.pkl": None,
        "ord_enc.pkl": None,
        "feature_columns.pkl": None
    }

    for search_path in search_paths:
        if not search_path.exists():
            continue
        for file_name in required_files.keys():
            file_path = search_path / file_name
            if file_path.exists() and required_files[file_name] is None:
                required_files[file_name] = str(file_path)

    return required_files

# Find files and check for missing ones
model_files = find_model_files()
missing_files = [name for name, path in model_files.items() if path is None]

if missing_files:
    st.error(f" Missing model files: {', '.join(missing_files)}")
    st.info("""
    **How to fix:**
    1. Place these files in the SAME folder as this script:
       - customer_churn.pkl
       - scalar.pkl
       - one_hot.pkl
       - ord_enc.pkl
       - feature_columns.pkl

    2. Or create a 'models' folder in your project and put files there

    **Current working directory:** """ + str(Path.cwd()))
    st.stop()

# ============ LOAD ALL MODELS AND ENCODERS ============
@st.cache_resource
def load_all_models(file_paths):
    """Load all pickle files using the provided paths."""
    try:
        # --- FIXED: removed "model/" prefix ---
        model = pickle.load(open(file_paths["customer_churn.pkl"], "rb"))
        scaler = pickle.load(open(file_paths["scalar.pkl"], "rb"))
        one_hot_encoder = pickle.load(open(file_paths["one_hot.pkl"], "rb"))
        ordinal_encoder = pickle.load(open(file_paths["ord_enc.pkl"], "rb"))
        feature_columns = pickle.load(open(file_paths["feature_columns.pkl"], "rb"))

        st.sidebar.success(" All models loaded successfully!")
        return model, scaler, one_hot_encoder, ordinal_encoder, feature_columns
    except Exception as e:
        st.error(f" Model loading failed: {e}")
        st.stop()

model, scaler, one_hot_encoder, ordinal_encoder, feature_columns = load_all_models(model_files)

# ============ HEADER SECTION ============
st.markdown("<h1 class='title-text'>📱 AI-Powered Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Predict whether a customer will churn based on their profile</p>",
            unsafe_allow_html=True)

# ============ CREATE TABS ============
tab1, tab2, tab3, tab4 = st.tabs([" Billing & Tenure", " Personal Details", " Services", " Contract & Payment"])

# ============ TAB 1: BILLING & TENURE ============
with tab1:
    st.markdown("<h3 class='section-header'>Billing & Tenure Information</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            label="Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
            help="How many months has the customer been with the company?"
        )

    with col2:
        total_charges = st.number_input(
            label="Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=0.01,
            help="Total amount paid by customer in dollars"
        )

# ============ TAB 2: PERSONAL DETAILS ============
with tab2:
    st.markdown("<h3 class='section-header'>Personal Details</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            label=" Gender",
            options=["Female", "Male"],
            help="Customer's gender"
        )

    with col2:
        partner = st.selectbox(
            label=" Partner",
            options=["No", "Yes"],
            help="Does customer have a partner?"
        )

    with col3:
        dependents = st.selectbox(
            label=" Dependents",
            options=["No", "Yes"],
            help="Does customer have dependents?"
        )

# ============ TAB 3: SERVICES ============
with tab3:
    st.markdown("<h3 class='section-header'>Services Subscribed</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        multiple_lines = st.selectbox(
            label=" Multiple Lines",
            options=["No", "Yes"],
        )

    with col2:
        internet_service = st.selectbox(
            label=" Internet Service Type",
            options=["DSL", "Fiber optic", "No"],
        )

    with col3:
        online_security = st.selectbox(
            label=" Online Security",
            options=["No", "Yes"],
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        online_backup = st.selectbox(
            label=" Online Backup",
            options=["No", "Yes"],
        )

    with col2:
        device_protection = st.selectbox(
            label="Device Protection",
            options=["No", "Yes"],
        )

    with col3:
        tech_support = st.selectbox(
            label=" Tech Support",
            options=["No", "Yes"],
        )

    col1, col2 = st.columns(2)

    with col1:
        streaming_tv = st.selectbox(
            label=" Streaming TV",
            options=["No", "Yes"],
        )

    with col2:
        streaming_movies = st.selectbox(
            label=" Streaming Movies",
            options=["No", "Yes"],
        )

# ============ TAB 4: CONTRACT & PAYMENT ============
with tab4:
    st.markdown("<h3 class='section-header'>Contract & Payment Information</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            label="Contract",
            options=["Month-to-month", "One year", "Two year"],
        )

    with col2:
        paperless_billing = st.selectbox(
            label="Paperless Billing",
            options=["No", "Yes"],
        )

    with col3:
        payment_method = st.selectbox(
            label=" Payment Method",
            options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    col1, col2 = st.columns(2)

    with col1:
        sim_service = st.selectbox(
            label=" SIM Service",
            options=["No", "Yes"],
        )

# ============ PREDICTION SECTION ============
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        label="Predict Churn",
        use_container_width=True,
        type="primary",
        help="Click to predict if customer will churn"
    )

# ============ PROCESS PREDICTION ============
if predict_button:
    with st.spinner("Analyzing customer data..."):
        try:
            # Create raw input data
            one_hot_data = pd.DataFrame({
                'gender_Male': [1 if gender == "Male" else 0],
                'Partner_Yes': [1 if partner == "Yes" else 0]
            })

            ordinal_data = pd.DataFrame({
                'Dependents': [1 if dependents == "Yes" else 0],
                'MultipleLines': [1 if multiple_lines == "Yes" else 0],
                'InternetService': [
                    0 if internet_service == "DSL" else (1 if internet_service == "Fiber optic" else 2)],
                'OnlineSecurity': [1 if online_security == "Yes" else 0],
                'OnlineBackup': [1 if online_backup == "Yes" else 0],
                'DeviceProtection': [1 if device_protection == "Yes" else 0],
                'TechSupport': [1 if tech_support == "Yes" else 0],
                'StreamingTV': [1 if streaming_tv == "Yes" else 0],
                'StreamingMovies': [1 if streaming_movies == "Yes" else 0],
                'Contract': [0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)],
                'PaperlessBilling': [1 if paperless_billing == "Yes" else 0],
                'PaymentMethod': [0 if payment_method == "Electronic check" else (
                    1 if payment_method == "Mailed check" else (
                        2 if payment_method == "Bank transfer (automatic)" else 3))],
                'sim': [1 if sim_service == "Yes" else 0]
            })

            # Numerical features transformation
            tenure_transformed = np.log1p(tenure)
            charges_transformed = np.log1p(total_charges)

            # Combine all features
            input_array = np.array([[
                tenure_transformed,
                charges_transformed,
                one_hot_data['gender_Male'].values[0],
                one_hot_data['Partner_Yes'].values[0],
                ordinal_data['Dependents'].values[0],
                ordinal_data['MultipleLines'].values[0],
                ordinal_data['InternetService'].values[0],
                ordinal_data['OnlineSecurity'].values[0],
                ordinal_data['OnlineBackup'].values[0],
                ordinal_data['DeviceProtection'].values[0],
                ordinal_data['TechSupport'].values[0],
                ordinal_data['StreamingTV'].values[0],
                ordinal_data['StreamingMovies'].values[0],
                ordinal_data['Contract'].values[0],
                ordinal_data['PaperlessBilling'].values[0],
                ordinal_data['PaymentMethod'].values[0],
                ordinal_data['sim'].values[0]
            ]])

            # Scale the input
            input_scaled = scaler.transform(input_array)

            # Get prediction
            prediction = model.predict(input_scaled)[0]
            prediction_probability = model.predict_proba(input_scaled)[0]

            retained_probability = prediction_probability[0] * 100
            churn_probability = prediction_probability[1] * 100

            # ============ DISPLAY RESULTS ============
            st.markdown("---")
            st.subheader(" Prediction Results")

            col1, col2 = st.columns([1, 1])

            with col1:
                if prediction == 0:
                    st.success("CUSTOMER RETAINED")
                    st.metric(label="Retention Probability", value=f"{retained_probability:.2f}%")
                else:
                    st.error(" CUSTOMER WILL CHURN")
                    st.metric(label="Churn Probability", value=f"{churn_probability:.2f}%")

            with col2:
                st.info(" Prediction Details")
                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(label="Retention", value=f"{retained_probability:.1f}%")

                with metric_col2:
                    st.metric(label="Churn Risk", value=f"{churn_probability:.1f}%")

            # Progress bars
            st.markdown("### Prediction Confidence")
            st.markdown("**Probability of Retention:**")
            st.progress(retained_probability / 100)
            st.markdown("**Probability of Churn:**")
            st.progress(churn_probability / 100)

            # ============ RECOMMENDATIONS ============
            st.markdown("---")
            st.subheader(" Recommendations")

            if prediction == 0:
                with st.success(""):
                    st.write("""
                    ** This customer is likely to stay with your company.**

                    **Suggestions:**
                    - Continue providing excellent service
                    - Monitor for any service quality issues
                    - Consider upselling additional services
                    - Maintain regular customer engagement
                    """)
            else:
                with st.warning(""):
                    st.write("""
                    ** This customer is at high risk of churning!**

                    **Recommended Actions:**
                    - Contact customer immediately for feedback
                    - Offer loyalty discounts or incentives
                    - Review contract terms and billing issues
                    - Provide enhanced customer support
                    - Consider service plan adjustments
                    - Schedule follow-up call from management
                    """)

            # ============ CUSTOMER PROFILE SUMMARY ============
            st.markdown("---")
            st.subheader(" Customer Profile Summary")

            summary_data = {
                "Category": [
                    "Tenure", "Total Charges", "Gender", "Partner", "Dependents",
                    "Internet Service", "Contract Type", "Payment Method"
                ],
                "Value": [
                    f"{tenure} months",
                    f"${total_charges:.2f}",
                    gender,
                    partner,
                    dependents,
                    internet_service,
                    contract,
                    payment_method
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f" Error during prediction: {str(e)}")
            st.write("Please check your input values and try again.")

# ============ SIDEBAR INFORMATION ============
st.sidebar.markdown("""
**📊 Model Details:**
- **Algorithm:** Gaussian Naive Bayes
- **Input Features:** 17 customer attributes
- **Output:** Churn probability (0-100%)
- **Training Data:** Telecom customer dataset

**📝 Field Descriptions:**
- **Tenure:** How long customer has been with company
- **Total Charges:** Sum of all monthly charges paid
- **Internet Service:** Type of internet connection
- **Contract:** Length of customer's contract
- **Services:** Additional services customer subscribed to

**✅ Status:** Model Loaded & Ready
""")
# ============ FOOTER ============
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #4a4a68;'>
        <p>Project Done BY Hema Malini using Machine Learning | Built with Streamlit</p>
        <p>Last Updated: 2026 | v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)