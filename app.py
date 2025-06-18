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

from sklearn.preprocessing import MinMaxScaler

# Daftar fitur yang digunakan model
selected_features = [
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date',
    'Scholarship_holder',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_1st_sem_enrolled',
    'Admission_grade',
    'Debtor',
    'Gender',
    'Age_at_enrollment',
    'Status'
]

# Filter data berstatus Enrolled
df_enrolled = df[df['Status'] == 'Enrolled']

st.subheader("Klasifikasi Mahasiswa Enrolled (Acak)")

# Tombol untuk memunculkan 1 data Enrolled acak dan klasifikasinya
if st.button("Ambil & Klasifikasi Mahasiswa Enrolled Acak"):
    if not df_enrolled.empty:
        # Ambil 1 baris acak
        random_row = df_enrolled.sample(n=1, random_state=random.randint(0, 1000)).reset_index(drop=True)

        # Hanya ambil fitur yang diperlukan
        row_for_model = random_row[selected_features]

        # Normalisasi (fit berdasarkan seluruh dataset agar skala cocok)
        scaler = MinMaxScaler()
        scaler.fit(df[selected_features])  # normalisasi berdasarkan seluruh data
        row_normalized = scaler.transform(row_for_model.drop(columns='Status'))

        # Prediksi
        prediction = model.predict(row_normalized)[0]
        label_map = {0: 'Dropout', 2: 'Graduate'}
        pred_label = label_map.get(prediction, 'Tidak diketahui')

        # Tampilkan hasil
        st.write("### Data Mahasiswa:")
        st.dataframe(random_row)

        st.write("### Hasil Prediksi:")
        st.success(f"Prediksi Model: **{pred_label}**")
    else:
        st.warning("Tidak ada data mahasiswa dengan Status == 1 (Enrolled).")