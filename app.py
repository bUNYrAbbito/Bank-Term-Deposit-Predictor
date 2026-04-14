import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= PAGE CONFIG ================= #
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="centered"
)

# ================= LOAD MODEL ================= #
@st.cache_resource
def load_model():
    return joblib.load("bank_pipeline.pkl")

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded")
except FileNotFoundError:
    st.error("⚠️  `bank_pipeline.pkl` not found. Run the notebook first to train and save the model.")
    st.stop()

# ================= TITLE ================= #
st.title("🏦 Bank Term Deposit Predictor")
st.caption("Predict whether a bank client will subscribe to a term deposit.")
st.divider()

# ================= INPUT UI ================= #
st.subheader("👤 Client Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 35)
    balance = st.number_input("Balance (€)", value=1000, step=100)
    duration = st.number_input("Last Call Duration (sec)", value=200, min_value=0)

with col2:
    job = st.selectbox("Job", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid',
        'management', 'retired', 'self-employed', 'services',
        'student', 'technician', 'unemployed', 'unknown'
    ])
    marital   = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['primary', 'secondary', 'tertiary'])

with col3:
    default = st.selectbox("Credit Default?", ['no', 'yes'])
    housing = st.selectbox("Housing Loan?",   ['yes', 'no'])
    loan    = st.selectbox("Personal Loan?",  ['yes', 'no'])

st.divider()
st.subheader("📞 Campaign Details")

col4, col5, col6 = st.columns(3)

with col4:
    contact  = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
    day      = st.slider("Last Contact Day", 1, 31, 15)
    month    = st.selectbox("Month", [
        'jan','feb','mar','apr','may','jun',
        'jul','aug','sep','oct','nov','dec'
    ])

with col5:
    campaign = st.number_input("Contacts This Campaign", value=2,  min_value=1)
    pdays    = st.number_input("Days Since Last Contact (-1=never)", value=-1, min_value=-1)
    previous = st.number_input("Previous Campaign Contacts", value=0, min_value=0)

with col6:
    poutcome = st.selectbox("Previous Campaign Outcome",
                             ['unknown', 'failure', 'success', 'other'])

st.divider()

# ================= PREDICT ================= #
if st.button("🔮 Predict Subscription", type="primary", use_container_width=True):

    # Feature engineering — same as notebook
    was_contacted_before = 1 if previous > 0 else 0
    duration_x_balance   = duration * balance
    age_group            = 'young' if age <= 30 else ('mid' if age <= 50 else 'senior')
    campaign_heavy       = 1 if campaign > 5 else 0
    pdays_clean          = 0 if pdays == -1 else pdays

    # Build DataFrame
    input_data = pd.DataFrame([{
        'age'                  : age,
        'job'                  : job,
        'marital'              : marital,
        'education'            : education,
        'default'              : default,
        'balance'              : balance,
        'housing'              : housing,
        'loan'                 : loan,
        'contact'              : contact,
        'day'                  : day,
        'month'                : month,
        'duration'             : duration,
        'campaign'             : campaign,
        'pdays'                : pdays_clean,
        'previous'             : previous,
        'poutcome'             : poutcome,
        'was_contacted_before' : was_contacted_before,
        'duration_x_balance'   : duration_x_balance,
        'age_group'            : age_group,
        'campaign_heavy'       : campaign_heavy,
    }])

    pred  = model.predict(input_data)[0]
    prob  = model.predict_proba(input_data)[0][1]

    # Show result
    st.divider()
    if pred == 1:
        st.success(f"✅  **WILL Subscribe** — Confidence: {prob:.1%}")
        st.progress(prob)
        st.info("💡 High conversion likelihood. Prioritise this client for outreach.")
    else:
        st.error(f"❌  **Will NOT Subscribe** — Probability: {prob:.1%}")
        st.progress(prob)
        st.info("💡 Low likelihood. Consider a different offer or timing.")

    col_a, col_b = st.columns(2)
    col_a.metric("Subscription Probability", f"{prob:.1%}")
    col_b.metric("Prediction", "YES ✅" if pred == 1 else "NO ❌")

    with st.expander("📊 Show Full Input Summary"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
