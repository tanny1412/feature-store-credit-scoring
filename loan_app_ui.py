import datetime
from collections import OrderedDict

# Import Streamlit for UI, pandas for data handling, and the model class
import streamlit as st
from credit_model import CreditScoringModel

# Set up the Streamlit page layout to be wide for better UI
st.set_page_config(layout="wide")

# Load the trained credit scoring model
model = CreditScoringModel()
# Ensure the model is trained before allowing predictions
if not model.is_model_trained():
    raise Exception("The credit scoring model has not been trained. Please run run.py.")


def get_loan_request():
    # Collect user input for loan application via Streamlit sidebar
    zipcode = st.sidebar.text_input("Zip code", "94109")
    date_of_birth = st.sidebar.date_input(
        "Date of birth", value=datetime.date(year=1986, day=19, month=3)
    )
    ssn_last_four = st.sidebar.text_input(
        "Last four digits of social security number", "3643"
    )
    # Combine DOB and SSN for unique identification
    dob_ssn = f"{date_of_birth.strftime('%Y%m%d')}_{str(ssn_last_four)}"
    age = st.sidebar.slider("Age", 0, 130, 25)
    income = st.sidebar.slider("Yearly Income", 0, 1000000, 120000)
    person_home_ownership = st.sidebar.selectbox(
        "Do you own or rent your home?", ("RENT", "MORTGAGE", "OWN")
    )
    employment = st.sidebar.slider(
        "How long have you been employed (months)?", 0, 120, 12
    )
    loan_intent = st.sidebar.selectbox(
        "Why do you want to apply for a loan?",
        (
            "PERSONAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "EDUCATION",
            "MEDICAL",
            "DEBTCONSOLIDATION",
        ),
    )
    amount = st.sidebar.slider("Loan amount", 0, 100000, 10000)
    interest = st.sidebar.slider("Preferred interest rate", 1.0, 25.0, 12.0, step=0.1)
    # Return all user input as an OrderedDict matching model input format
    return OrderedDict(
        {
            "zipcode": [int(zipcode)],
            "dob_ssn": [dob_ssn],
            "person_age": [age],
            "person_income": [income],
            "person_home_ownership": [person_home_ownership],
            "person_emp_length": [float(employment)],
            "loan_intent": [loan_intent],
            "loan_amnt": [amount],
            "loan_int_rate": [interest],
        }
    )


# Main application title
st.title("Loan Application")

# Section: User input
st.header("User input:")
loan_request = get_loan_request()

# Section: Model prediction result
st.header("Application Status (model prediction):")
# Make a prediction using the trained model and user input
result = model.predict(loan_request)

# Display the result to the user
if result == 0:
    st.success("Your loan has been approved!")
elif result == 1:
    st.error("Your loan has been rejected!")

   
