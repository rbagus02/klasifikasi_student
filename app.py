import streamlit as st
import pandas as pd
import joblib

# Judul Aplikasi
st.title("Prediksi Status Mahasiswa - Dashboard Klasifikasi")

# Load model dari file
model_path = 'model.pkl'
model = joblib.load(model_path)

# Load dataset final
data_path = 'dataset.csv'
df = pd.read_csv(data_path)

# Tampilkan dataset
st.subheader("Dataset")
st.dataframe(df.head())