import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('combined_model.pkl')

df = pd.read_csv('cs-training.csv')

# Streamlit app UI
st.title('Credit Risk Probability Calculator')

# Add input widgets for user input
st.sidebar.header('Input Features')

RevolvingUtilizationOfUnsecuredLines = st.sidebar.number_input('Revolving Utilization of Unsecured Lines', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
NumberOfTime30_59DaysPastDueNotWorse = st.sidebar.number_input('Number of Times 30-59 Days Past Due Not Worse', min_value=0, max_value=20, value=0)
DebtRatio = st.sidebar.number_input('Debt Ratio', min_value=0.0, value=0.3, step=0.01)
MonthlyIncome = st.sidebar.number_input('Monthly Income', min_value=0.0, value=5000.0, step=100.0)
NumberOfOpenCreditLinesAndLoans = st.sidebar.number_input('Number of Open Credit Lines and Loans', min_value=0, max_value=50, value=5)
NumberOfTimes90DaysLate = st.sidebar.number_input('Number of Times 90 Days Late', min_value=0, max_value=20, value=0)
NumberRealEstateLoansOrLines = st.sidebar.number_input('Number of Real Estate Loans or Lines', min_value=0, max_value=10, value=1)
NumberOfTime60_89DaysPastDueNotWorse = st.sidebar.number_input('Number of Times 60-89 Days Past Due Not Worse', min_value=0, max_value=20, value=0)
NumberOfDependents = st.sidebar.number_input('Number of Dependents', min_value=0, max_value=10, value=1)

# Create a button to trigger prediction

predict_button = st.sidebar.button('Predict')


# Display prediction result
if predict_button:
    user_input = np.array([
        [RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30_59DaysPastDueNotWorse,
        DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines, NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents]
    ])

  # Make prediction
    predicted_prob = model.predict_proba(user_input)[0][1]
    st.subheader('Prediction Probability')
    st.write(f'Probability of Credit Risk: {predicted_prob:.4f}')

    # Interpretation
    if predicted_prob >= 0.5:
        st.write('Interpretation: Credit Risk')
    else:
        st.write('Interpretation: No Credit Risk')
        
        
# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.title('About')
    st.sidebar.info('This app is a Credit Risk Probability Calculator.')

