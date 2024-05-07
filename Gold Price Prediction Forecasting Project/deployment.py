#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pmdarima as pm

def load_arima_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def forecast_price(model, n_periods):
    arima_model, forecast, conf_int = model
    forecast_values, conf_int_values = arima_model.predict(n_periods=n_periods, return_conf_int=True)
    return forecast_values, conf_int_values

def main():
    st.title('ARIMA Model for Price Forecasting')

    # Load the ARIMA model
    model_path = 'arima_model.pkl'
    arima_model = load_arima_model(model_path)

    # Add input widgets
    n_periods = st.sidebar.slider('Number of Periods', min_value=1, max_value=100, value=20)

    # Make prediction
    forecast, conf_int = forecast_price(arima_model, n_periods)

    # Display prediction
    st.write('Forecasted Prices:')
    for i, value in enumerate(forecast):
        st.write(f'Period {i+1}: {value:.2f} (95% CI: {conf_int[i][0]:.2f} - {conf_int[i][1]:.2f})')

if __name__ == '__main__':
    main()


# In[15]:


with open('sarima_model.pkl', 'rb') as f:
    loaded_object = pickle.load(f)


# In[ ]:




