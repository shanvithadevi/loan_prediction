import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("loan_data.csv")

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Loan Eligibility Predictor")

st.write("Fill in applicant details below to check loan eligibility:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=70, value=30)
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education", ["Graduate", "Non-Graduate"])
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed"])
annual_income = st.number_input("Annual Income", min_value=10000, max_value=200000, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=50000, value=15000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680)
existing_loan = st.selectbox("Existing Loan", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

if st.button("Predict Loan Eligibility"):
    # Encode inputs
    applicant = pd.DataFrame({
        "Gender": [label_encoders["Gender"].transform([gender])[0]],
        "Age": [age],
        "Marital_Status": [label_encoders["Marital_Status"].transform([marital_status])[0]],
        "Education": [label_encoders["Education"].transform([education])[0]],
        "Employment_Status": [label_encoders["Employment_Status"].transform([employment_status])[0]],
        "Annual_Income": [annual_income],
        "Loan_Amount": [loan_amount],
        "Credit_Score": [credit_score],
        "Existing_Loan": [label_encoders["Existing_Loan"].transform([existing_loan])[0]],
        "Property_Area": [label_encoders["Property_Area"].transform([property_area])[0]]
    })

    prediction = model.predict(applicant)[0]
    result = label_encoders["Loan_Status"].inverse_transform([prediction])[0]

    st.success(f"✅ Loan Status Prediction: {result}")
