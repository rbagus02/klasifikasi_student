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
df = pd.read_csv(data_path, sep=';')

# Tampilkan dataset
st.subheader("Dataset")
st.dataframe(df.head())

st.subheader("🎯 Tampilkan Data Mahasiswa dengan Status Enrolled")

# Tombol untuk memunculkan data dengan Status = 1 (Enrolled)
if st.button("Tampilkan Mahasiswa Enrolled"):
    enrolled_df = df[df['Status'] == 1].reset_index(drop=True)
    st.dataframe(enrolled_df.head(20))  # Tampilkan 20 baris pertama