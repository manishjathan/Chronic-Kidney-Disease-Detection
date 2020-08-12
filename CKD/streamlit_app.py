import streamlit as st
import numpy as np
import pandas as pd
import pickle
from feature_transform import transformAttributes

## All Required functions
median_sod = 138.0


scaler = pickle.load(open('scaler','rb'))
logClf = pickle.load(open('logClf','rb'))
neigh = pickle.load(open('neigh','rb'))
kernel_svc = pickle.load(open('kernel_svc','rb'))
decisionTreeClassifier = pickle.load(open('decisionTreeClassifier','rb'))
randomForestClassifier = pickle.load(open('randomForestClassifier','rb'))
stackedClassifier = pickle.load(open('stackedClassifier','rb'))


print("Logistic Classifier : ",logClf)
print("KNN Classifier : ",neigh)
print("Kernel SVC : ",kernel_svc)
print("Decision Tree Classifier : ",decisionTreeClassifier)
print("Random Forest Classifier : ",randomForestClassifier)
print("Stacked Classifier : ",stackedClassifier)


st.title("Chronic Kidney Disease Detection")

age = st.number_input('Enter Age')
bp = st.number_input('Enter Blood pressure')
al = st.number_input('Enter Albumin')
su = st.number_input('Enter Sugar')
bgr = st.number_input('Enter Blood GLucose Random')
bu = st.number_input('Enter Blood Urea')
sc = st.number_input('Enter Serum Creatinine')
sod = st.number_input('Enter Sodium')
hemo = st.number_input('Enter Hemoglobin')
rc = st.number_input('Enter Red bLood cell count')
wc = st.number_input('Enter White blood cell count')

input_list = [age,bp,al,su,bgr,bu,sc,sod,hemo,rc,wc]

if st.button('Predict CKD'):
    transf_inp = transformAttributes(input_list)
    scaler_mean = scaler.mean_
    scaler_var = scaler.var_
    std_test_x = (np.array(transf_inp.iloc[0])-scaler_mean)/np.sqrt(scaler_var)
    std_test_df = pd.DataFrame(std_test_x.reshape(1,-1))
    
    st.write("Prediction from Logistic Classifier : ",logClf.predict_proba(std_test_df))
    st.write("Prediction from KNN Classifier : ",neigh.predict_proba(std_test_df))
    #st.write("Prediction from Kernel SVC Classifier : ",kernel_svc.predict_prob(std_test_df))
    st.write("Prediction from Decision Tree Classifier : ",decisionTreeClassifier.predict_proba(std_test_df))
    st.write("Prediction from Random Forest Classifier : ",randomForestClassifier.predict_proba(std_test_df))
    st.write("Prediction from Stacked Classifier : ",stackedClassifier.predict_proba(std_test_df))