import pickle 
import streamlit as st

model = pickle.load(open('chronic-kidney.sav', 'rb'))
st.title("Prediksi :red[Ginjal] Kronis")
st.header("", divider='rainbow')

col1, col2, col3 = st.columns(3)
with col1: 
    Bp = st.number_input("Tekanan darah", 50,180,80)
with col2:
    Sg = st.number_input("Specific Gravity normal", 1.00, 1.030, 1.025, step=0.001, format="%0.3f")
with col3:  
    Al = st.number_input("Albumin", 0,5,0) 

with col1:
    Su = st.number_input("Gula",0,5,0) 
with col2:
    Rbc = st.number_input("Sel darah merah", 0,1,1) 
with col3:
    Bu = st.number_input("Darah urea",1.5,391.1,10.0) 
with col1:
    Sc = st.number_input("Serum creatine",0.4,76.0,1.2) 
with col2:
    Sod = st.number_input("Sodium",4.5,168.0,135.0) 
with col3:
    Pot = st.number_input("Pottasium",2.5,47.0,5.0) 

with col1:
    Hemo = st.number_input("Hemoglobin",3.1,17.8,15.0) 
with col2:
    Wbcc = st.number_input("Jumlah sel darah putih",2200,26400,10400) 
with col3:
    Rbcc = st.number_input("Jumlah sel darah merah", 2.1, 8.0,4.5) 
    
Htn = st.selectbox("Hipertensi",["Tidak", "Ya"])
if Htn == "Ya" : 
    Htn = 1 
else : 
    Htn = 0


if st.button("Prediksi gagal ginjal"):
    heart_disease_predict = model.predict([[Bp,Sg,Al,Su,Rbc,Bu,Sc,Sod,Pot,Hemo,Wbcc,Rbcc,Htn]])
    if(heart_disease_predict[0]==1):
        st.warning("Pasien terkena gagal ginjal kronis")
    else :
        st.success("Pasien tidak terkena gagal ginjal kronis")
