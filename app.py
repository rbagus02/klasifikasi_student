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

st.subheader("🔍 Prediksi Individu dari Status Enrolled")

# Filter data enrolled
enrolled_data = df[df['Status'] == 1].reset_index(drop=True)

# Tampilkan selectbox untuk memilih baris
selected_index = st.selectbox("Pilih Mahasiswa (baris) dengan Status Enrolled", enrolled_data.index)

# Ambil baris yang dipilih
selected_row = enrolled_data.loc[selected_index]
selected_features = selected_row.drop('Status').values.reshape(1, -1)

# Jalankan prediksi
predicted_label = model.predict(selected_features)[0]
label_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

# Tampilkan hasil
st.write("### Hasil Prediksi:")
st.write(f"Status Saat Ini: **Enrolled**")
st.write(f"Prediksi Kemungkinan: **{label_map[predicted_label]}**")