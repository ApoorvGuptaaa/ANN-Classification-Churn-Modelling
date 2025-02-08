import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('one_hot_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App UI
st.set_page_config(page_title='Churn Prediction', layout='centered')
st.title(':chart_with_downwards_trend: Churn Prediction App')
st.write("### Predict if a customer is likely to churn based on their profile")

# Sidebar for User Input
st.sidebar.header("Customer Information")
geography = st.sidebar.selectbox(':earth_americas: Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox(':bust_in_silhouette: Gender', label_encoder_gender.classes_)
age = st.sidebar.slider(':calendar: Age', 18, 92, 30)
balance = st.sidebar.number_input(':moneybag: Balance', min_value=0.0, format='%.2f')
credit_score = st.sidebar.number_input(':credit_card: Credit Score', min_value=300, max_value=900, value=650)
estimated_salary = st.sidebar.number_input(':dollar: Estimated Salary', min_value=0.0, format='%.2f')
tenure = st.sidebar.slider(':clock1: Tenure', 0, 10, 5)
num_of_products = st.sidebar.slider(':package: Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.radio(':credit_card: Has Credit Card?', [0, 1])
is_active_member = st.sidebar.radio(':busts_in_silhouette: Is Active Member?', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate input data and encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict only if button is clicked
if st.sidebar.button('Predict Churn'):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display results in the main area
    st.subheader('Prediction Result')
    st.metric(label='Churn Probability', value=f'{prediction_proba:.2%}')

    if prediction_proba > 0.5:
        st.error('🚨 The customer is likely to churn!')
    else:
        st.success('✅ The customer is NOT likely to churn.')
