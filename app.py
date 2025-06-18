import streamlit as st
import pandas as pd
import joblib
import random

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

# Filter data yang berstatus Enrolled
df_enrolled = df[df['Status'] == 1]

st.subheader("🎲 Tampilkan Satu Baris Mahasiswa Berstatus Enrolled (Acak)")

# Tombol untuk menampilkan satu baris acak
if st.button("Ambil 1 Mahasiswa Enrolled Secara Acak"):
    random_row = df_enrolled.sample(n=1, random_state=random.randint(0, 1000)).reset_index(drop=True)
    st.write("### 📋 Data Mahasiswa Terpilih:")
    st.dataframe(random_row)