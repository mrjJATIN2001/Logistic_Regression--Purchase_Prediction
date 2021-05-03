import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('logisticmodel.pkl', 'rb')) 
dataset= pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [1,2, 3]].values
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:, 0])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
def predict_note_authentication(UserID, Gender,Age,EstimatedSalary):
  output= model.predict(sc.transform([[Gender, Age,EstimatedSalary]]))
  print("Purchased", output)
  if output==[1]:
    prediction="Item will be purchased"
  else:
    prediction="Item will not be purchased"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Item Purchase Prediction using Logistic Algorithm")
    UserID = st.text_input("UserID","")
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Age = st.number_input('Insert a Age',18,60)
    #Age = st.text_input("Age","Type Here")
    EstimatedSalary = st.number_input("Insert EstimatedSalary",15000,150000)
    resul=""
    if st.button("Prediction"):
      result=predict_note_authentication(UserID, Gender,Age,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))  
    if st.button("About"):
      st.header("Developed by Jatin Tak")
      st.subheader("Student, Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment 7: Logistic Regression</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()