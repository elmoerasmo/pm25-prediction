import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set konfigurasi halaman
st.set_page_config(
    page_title="Prediksi PM2.5 Bandung",
    page_icon="🌬️",
    layout="centered"
)

# Judul Aplikasi
st.title("🌬️ Prakiraan Kualitas Udara (PM2.5) 3 Jam ke Depan")
st.write("Aplikasi ini memprediksi konsentrasi PM2.5 dalam 3 jam ke depan berdasarkan kondisi cuaca saat ini menggunakan model LightGBM.")

# Fungsi untuk memuat model dan fitur secara aman
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('model_lightgbm_pm25.joblib')
    features = joblib.load('features_list.joblib')
    return model, features

try:
    model, features = load_model_artifacts()
except Exception as e:
    st.error(f"Gagal memuat file model. Pastikan 'model_lightgbm_pm25.joblib' dan 'features_list.joblib' berada di direktori yang sama. Error: {e}")
    st.stop()

# --- Form Input Pengguna ---
st.header("📝 Input Kondisi Cuaca & Udara Saat Ini")

col1, col2 = st.columns(2)

with col1:
    stasiun_input = st.selectbox(
        "Pilih Stasiun Pemantau",
        ['Antapani', 'Cibeureum', 'Dago', 'Soekarno-Hatta', 'Ujungberung']
    )
    pm25_sekarang = st.number_input("PM2.5 Saat Ini (µg/m³)", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
    suhu = st.number_input("Suhu Udara (°C)", min_value=0.0, max_value=50.0, value=27.0, step=0.1)
    kelembaban = st.number_input("Kelembaban Udara (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

with col2:
    kecepatan_angin = st.number_input("Kecepatan Angin (m/s)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
    arah_angin = st.number_input("Arah Angin (Derajat °)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

st.subheader("⏰ Waktu Prediksi")
col3, col4, col5 = st.columns(3)
with col3:
    hour = st.slider("Jam (0-23)", 0, 23, 12)
with col4:
    day_of_week = st.selectbox(
        "Hari",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'][x]
    )
with col5:
    month = st.slider("Bulan (1-12)", 1, 12, 6)

# --- Proses Prediksi ---
if st.button("🎯 Lakukan Prediksi 3 Jam Kedepan", type="primary", use_container_width=True):
    stasiun_list = ['antapani', 'cibeureum', 'dago', 'soekarno-hatta', 'ujungberung']
    stasiun_clean = stasiun_input.strip().lower()
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Membuat struktur data input sesuai training
    input_data = {
        'pm25_ugm3': pm25_sekarang,
        'suhu_C': suhu,
        'kelembaban_pct': kelembaban,
        'kecepatan_angin_ms': kecepatan_angin,
        'arah_angin_deg': arah_angin,
        'curah_hujan_mm': curah_hujan,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'pm25_lag1': pm25_sekarang,
        'pm25_lag2': pm25_sekarang
    }

    # One-hot encoding untuk stasiun secara manual
    for s in stasiun_list:
        input_data[f'stasiun_{s}'] = 1 if s == stasiun_clean else 0

    # Konversi ke DataFrame sesuai urutan fitur model
    df_input = pd.DataFrame([input_data])[features]

    # Melakukan Prediksi
    prediksi_val = float(model.predict(df_input)[0])
    selisih = prediksi_val - pm25_sekarang

    # Menentukan kategori dan rekomendasi
    if prediksi_val <= 15.0:
        kategori = "BAIK"
        rekomendasi = "Kualitas udara sangat baik. Aman untuk beraktivitas di luar ruangan!"
        st.balloons()
        color = "green"
    elif prediksi_val <= 55.0:
        kategori = "SEDANG"
        rekomendasi = "Kualitas udara sedang. Secara umum aman bagi sebagian besar orang, namun kelompok sensitif (seperti penderita asma) disarankan membatasi aktivitas fisik berat di luar ruangan."
        color = "orange"
    else:
        kategori = "TIDAK SEHAT"
        rekomendasi = "PERINGATAN! Udara diprediksi tidak sehat dalam 3 jam ke depan. Warga yang sensitif wajib menggunakan masker bila berada di luar ruangan."
        color = "red"

    # Menampilkan Hasil
    st.markdown("---")
    st.subheader("📊 Hasil Analisis Prediksi")

    metric_cols = st.columns(2)
    metric_cols[0].metric(
        label="Prediksi Konsentrasi PM2.5",
        value=f"{prediksi_val:.2f} µg/m³",
        delta=f"{selisih:+.2f} µg/m³ dari saat ini",
        delta_color="inverse" # Merah jika naik, hijau jika turun
    )
    metric_cols[1].metric(
        label="Lokasi Stasiun",
        value=stasiun_input
    )

    # Tampilan kategori berwarna
    if color == "green":
        st.success(f"Kategori Kualitas Udara: **{kategori}**")
    elif color == "orange":
        st.warning(f"Kategori Kualitas Udara: **{kategori}**")
    else:
        st.error(f"Kategori Kualitas Udara: **{kategori}**")

    st.info(f"💡 **Rekomendasi:** {rekomendasi}")
