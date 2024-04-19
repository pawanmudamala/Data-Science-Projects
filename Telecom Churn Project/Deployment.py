#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pickle import load

# Load the trained model
model = load(open(r'C:\\Users\\pavan\\Desktop\\Telecom-Churn-Project\\finalized_model.sav', 'rb'))

# Define the Streamlit app
def main():
    st.title('Telecom Churn Prediction App')
    st.write('Enter customer information to predict churn.')

    # Create input fields for user input
    voice_plan = st.selectbox('Voice Plan', ['Yes', 'No'])
    voice_messages = st.number_input('Voice Messages')
    intl_plan = st.selectbox('International Plan', ['Yes', 'No'])
    intl_charge = st.number_input('International Charge')
    intl_mins = st.number_input('International Mins')
    day_mins = st.number_input('Day Mins')
    day_charge = st.number_input('Day Charge')
    eve_charge = st.number_input('Evening Charge')
    customer_calls = st.number_input('Customer Calls')
    eve_mins = st.number_input('Evening Mins')
    intl_calls = st.number_input('International Calls')

    # Create a button to make predictions
    if st.button('Predict Churn'):
        # Create a DataFrame with user input
        input_data = pd.DataFrame({
            'voice_plan': [voice_plan],
            'voice_messages': [voice_messages],
            'intl_plan': [intl_plan],
            'intl_charge': [intl_charge],
            'intl_mins': [intl_mins],
            'day_mins': [day_mins],
            'day_charge': [day_charge],
            'eve_charge': [eve_charge],
            'customer_calls': [customer_calls],
            'eve_mins': [eve_mins],
            'intl_calls': [intl_calls],
        })

        # Convert categorical variables to numerical
        input_data['voice_plan'] = input_data['voice_plan'].map({'Yes': 1, 'No': 0})
        input_data['intl_plan'] = input_data['intl_plan'].map({'Yes': 1, 'No': 0})

        # Make prediction
        prediction = model.predict(input_data)

        # Display prediction
        if prediction[0] == 0:
            st.write('Prediction: Not Churn')
        else:
            st.write('Prediction: Churn')

if __name__ == '__main__':
    main()


# In[ ]:




