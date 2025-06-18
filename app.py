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

# Fitur yang digunakan (pastikan sesuai model)
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
    'Status'  # Status masih string
]

# Fitur numerik saja untuk normalisasi
fit_features = [f for f in selected_features if f != 'Status']

# Filter hanya mahasiswa berstatus Enrolled
df_enrolled = df[df['Status'] == 'Enrolled']

st.subheader("🎲 Klasifikasi Mahasiswa Enrolled (Acak)")

if st.button("Ambil & Klasifikasi Mahasiswa Enrolled Acak"):
    if not df_enrolled.empty:
        # Ambil satu baris acak
        random_row = df_enrolled.sample(n=1, random_state=random.randint(0, 1000)).reset_index(drop=True)

        # Normalisasi fitur
        scaler = MinMaxScaler()
        scaler.fit(df[fit_features])
        row_normalized = scaler.transform(random_row[fit_features])

        # Prediksi
        prediction = model.predict(row_normalized)[0]

        # Tampilkan
        st.write("### 📋 Data Mahasiswa Terpilih:")
        st.dataframe(random_row)

        st.markdown(f"""
        ### 🧾 Ringkasan:
        - **Status Asli**: Enrolled  
        - **Prediksi Model (angka)**: {prediction}
        """)
    else:
        st.warning("⚠️ Tidak ada data dengan Status = 'Enrolled'.")