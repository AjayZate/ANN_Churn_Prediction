import streamlit as st
import pandas as pd
import numpy as np
import tensorflow 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
#from tensorflow.keras.models import load_model

#load the model
model = tensorflow.keras.models.load_model('model.h5')

#loading encoder and scalar
with open('lable_encoded_gender.pkl','rb') as file:
    labelencoder = pickle.load(file)
with open('onehot_encoded_Geoghraphy.pkl','rb') as file:
    onehotencoder = pickle.load(file)
with open('scalar.pkl','rb') as file:
    scalar = pickle.load(file)

##streamlit app interface
st.title('Customer Churn Prediction Model')

#accepting user inputs
geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender',labelencoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input("Balance")
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('NUmber Of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('IS Active Member',[0,1])


#prepare_input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[labelencoder.transform([gender])[0]],
    'Age':[age],
    'Balance':[balance],
    'CreditScore':[credit_score],
    'Tenure':[tenure],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#one_hot_encoded
geo_encoded = onehotencoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

st.write(input_data)

#scale the data
input_scaled = scalar.transform(input_data)

#predictions
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

if prediction_proba>0.5:
    st.wrire('Customer is LIKELY to churn')
else:
    st.write('Customer is NOT LIKELY to churn')