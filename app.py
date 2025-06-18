import streamlit as st
import pandas as pd
import joblib

# Judul Aplikasi
st.title("Prediksi Status Mahasiswa - Dashboard Klasifikasi")

# Load model dari file
model_path = 'model.pkl'
model = joblib.load(model_path)

# Load dataset final
data_path = 'dataset_app.csv'
df = pd.read_csv(data_path)

# Pisahkan fitur dan target
X = df.drop(columns='Status')
y = df['Status']

# Tampilkan dataset
st.subheader("📋 Dataset")
st.dataframe(df.head())

# Prediksi dengan model
st.subheader("📊 Hasil Klasifikasi")
y_pred = model.predict(X)

# Mapping label (opsional jika target berupa angka)
status_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
df_result = df.copy()
df_result['Status_Prediksi'] = y_pred
df_result['Status'] = df_result['Status'].map(status_map)
df_result['Status_Prediksi'] = df_result['Status_Prediksi'].map(status_map)

# Tampilkan hasil
st.write(df_result[['Status', 'Status_Prediksi']].head(20))

# Tampilkan statistik ringkas
correct = (df_result['Status'] == df_result['Status_Prediksi']).sum()
total = len(df_result)
accuracy = correct / total

st.success(f"🎯 Akurasi prediksi terhadap dataset: {accuracy:.2%} ({correct} dari {total} benar)")

# Download hasil
csv = df_result.to_csv(index=False).encode('utf-8')
st.download_button("💾 Unduh Hasil Klasifikasi CSV", data=csv, file_name="hasil_klasifikasi.csv", mime="text/csv")
