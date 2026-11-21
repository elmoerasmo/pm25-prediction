# Prediksi PM2.5 Bandung

**Machine Learning Project Kasus 06**

Proyek Machine Learning untuk memprediksi konsentrasi partikulat halus PM2.5 tiga jam ke depan pada lima stasiun pemantauan kualitas udara di Kota Bandung. Sistem ini dikembangkan sebagai sarana prakiraan kualitas udara berbasis regresi deret waktu (*time-series regression*) untuk membantu masyarakat dan instansi terkait melakukan tindakan antisipatif sebelum kondisi udara memburuk.

Hasil prediksi diintegrasikan ke dalam aplikasi web interaktif berbasis **Streamlit** dengan tampilan indikator visual dan rekomendasi kesehatan yang mudah dipahami.

---

## 1. Latar Belakang & Permasalahan

PM2.5 merupakan partikulat halus berdiameter 2,5 mikrometer atau lebih kecil yang berisiko tinggi masuk ke saluran pernapasan dan memicu gangguan kesehatan, terutama bagi kelompok rentan seperti penderita asma, anak-anak, dan lansia. Pemantauan kualitas udara secara *real-time* saja belum cukup karena informasi baru tersedia setelah kondisi udara memburuk.

Proyek ini menempatkan **Dinas Lingkungan Hidup dan Kebersihan (DLHK)** serta **Dinas Kesehatan Kota Bandung** sebagai klien utama, sementara sistem aplikasinya ditujukan langsung bagi masyarakat umum dan pengambil kebijakan:

- **Bagi Masyarakat:** Menentukan waktu terbaik beraktivitas di luar ruangan, kebutuhan penggunaan masker medis, maupun aktivasi penyaring udara (*air purifier*).
- **Bagi Pemerintah Daerah:** Sebagai sistem peringatan dini untuk memberlakukan rekayasa lalu lintas sementara hingga pembatasan operasional industri lokal saat polusi diprediksi mencapai ambang berbahaya.

Akurasi model menjadi krusial karena kesalahan prediksi berdampak langsung pada aspek sosial dan ekonomi:
- ***False Negative*:** Model memprediksi udara bersih padahal polusi tinggi, berisiko memicu lonjakan kasus ISPA akibat kelompok rentan terpapar polusi tanpa perlindungan.
- ***False Positive*:** Model memprediksi polusi berbahaya padahal udara aman, berpotensi menimbulkan kepanikan massal dan kerugian ekonomi akibat pembatalan aktivitas luar ruang secara keliru.

## 2. Tujuan Proyek

1. Mengembangkan model Machine Learning berbasis **LightGBM** untuk memprediksi konsentrasi PM2.5 tiga jam ke depan secara akurat.
2. Mengidentifikasi pola perubahan kualitas udara berdasarkan dinamika temporal dan kondisi meteorologi di Bandung.
3. Membandingkan performa model dengan metode *baseline* persistensi naif.
4. Mengevaluasi akurasi prediksi menggunakan metrik utama **Mean Absolute Error (MAE)** dalam satuan fisik asli (µg/m³).
5. Membangun aplikasi web interaktif berbasis Streamlit yang dilengkapi validasi input dan visualisasi peringatan dini.

## 3. Ruang Lingkup Proyek

| Aspek | Keterangan |
|---|---|
| **Kasus** | Prakiraan PM2.5 Bandung (Kasus 06) |
| **Jenis Pembelajaran** | Supervised Learning — Time-Series Regression |
| **Target Prediksi** | Konsentrasi PM2.5 (+3 jam) pada stasiun yang sama |
| **Lokasi Pemantauan** | Antapani, Cibeureum, Dago, Soekarno-Hatta, Ujungberung |
| **Periode Dataset** | Juli 2024 – Juni 2026 (87.650 baris pengamatan awal) |
| **Pengguna Utama** | Masyarakat umum (khususnya kelompok rentan) & Petugas DLHK/Dinkes |

## 4. Dataset & Data Preparation

Dataset utama bersumber dari pencatatan sensor stasiun pemantau kualitas udara di Kota Bandung (`data/raw.csv`) dengan ukuran awal mencapai **87.650 baris** pengamatan temporal.

### Atribut Dataset

| Kolom | Keterangan | Satuan |
|---|---|---|
| `waktu` | Waktu pengamatan | WIB |
| `stasiun` | Nama stasiun pemantauan | - |
| `pm25_ugm3` | Konsentrasi PM2.5 | µg/m³ |
| `suhu_C` | Suhu udara | °C |
| `kelembaban_pct` | Kelembapan udara | % |
| `kecepatan_angin_ms` | Kecepatan angin | m/s |
| `arah_angin_deg` | Arah embusan angin | Derajat |
| `curah_hujan_mm` | Intensitas curah hujan | mm |

### Penanganan Kualitas Data (Data Cleaning)

| Jenis Masalah | Jumlah Baris | Cara Menangani | Alasan Penanganan |
|---|---|---|---|
| **Ketidakkonsistenan Nama Stasiun** | 15.120 baris | Mengubah string menjadi huruf kecil (*lowercase*), menghapus spasi (*strip*), dan memetakan variasi penulisan (misal: `'soetta'`, `'soekarno hatta'` → `'soekarno-hatta'`). | Menghindari redundansi kategori saat proses *One-Hot Encoding* agar lokasi yang sama dikenali sebagai satu entitas. |
| **Nilai Sensor Kosong (*Missing Values*)** | 825 baris | Interpolasi linier dua arah (*linear interpolation, both direction*) yang dikelompokkan berdasarkan masing-masing stasiun. | Data meteorologi dan polutan bersifat kontinu secara temporal; interpolasi mempertahankan gradasi perubahan fisis dibanding rata-rata global. |
| **Nilai Target Masa Depan Kosong** | 15 baris (3 baris terakhir/stasiun) | Menghapus (*drop*) baris data yang tidak memiliki nilai target masa depan. | Target prediksi adalah pergeseran +3 jam, sehingga 3 pengamatan terakhir di tiap stasiun tidak memiliki label aktual untuk pelatihan terbimbing. |

### Temuan Exploratory Data Analysis (EDA)

1. **Pola Fluktuasi Harian & Mobilitas:** Konsentrasi PM2.5 melonjak tajam pada jam sibuk pagi hari (**07.00–08.00 WIB**) seiring tingginya emisi transportasi publik/kendaraan, melandai di siang hari, lalu kembali meningkat menjelang malam akibat fenomena inversi suhu udara.
2. **Pengaruh Faktor Meteorologi:** Kecepatan angin memiliki korelasi negatif yang konsisten terhadap konsentrasi PM2.5—angin kencang membantu mendispersi polutan, sedangkan kondisi angin tenang memicu akumulasi partikulat di dekat permukaan tanah.

## 5. Metodologi & Pembagian Dataset

### Rekayasa Fitur (Feature Engineering)
- **Fitur Waktu:** Ekstraksi komponen jam (`hour`), hari dalam seminggu, bulan, dan penanda akhir pekan untuk menangkap siklus mobilitas harian.
- **Fitur Historis (*Lag Features*):** Nilai konsentrasi PM2.5 pada 1 jam (`pm25_lag1`) dan 2 jam (`pm25_lag2`) sebelumnya untuk memanfaatkan autokorelasi temporal polutan.
- **Encoding Spasial:** Penerapan *One-Hot Encoding* pada variabel `stasiun` untuk membedakan tingkat kerawanan polusi spesifik di tiap wilayah.

### Pembagian Dataset (Temporal Split)

Pembagian dilakukan secara kronologis untuk mensimulasikan kondisi operasional nyata tanpa kebocoran data masa depan (*data leakage*):

| Dataset | Periode | Jumlah / Kegunaan |
|---|---|---|
| **Train** | Sebelum Maret 2026 (Juli 2024 – Feb 2026) | **72.991 sampel** — Melatih bobot internal model |
| **Validation** | Maret – April 2026 | Penyetelan parameter dan pemilihan model terbaik |
| **Test** | Mei – Juni 2026 | Evaluasi akhir kemampuan generalisasi model pada data baru |

### Algoritma & Evaluasi
- **Baseline:** Metode persistensi naif (memproyeksikan konsentrasi PM2.5 3 jam ke depan sama dengan konsentrasi saat ini).
- **Model Utama:** **Light Gradient Boosting Machine (`LGBMRegressor`)**, dipilih karena efisiensi memori tinggi pada data tabular skala besar serta kemampuannya menangkap hubungan non-linier kompleks antara cuaca, waktu, dan polutan.

## 6. Hasil Evaluasi Eksperimen

| Tahap Evaluasi | MAE (µg/m³) | RMSE (µg/m³) | Koefisien Determinasi ($R^2$) |
|---|---|---|---|
| **Validation Set** (Mar–Apr 2026) | **6,5412** | 12,3512 | 0,5420 |
| **Test Set** (Mei–Jun 2026) | **7,9332** | 48,6534 | 0,0320 |

**Analisis Performa:**
- Pada tahap validasi, model mampu menjelaskan lebih dari setengah variasi perubahan polusi ($R^2 = 0,5420$) dengan kesalahan rata-rata hanya **6,54 µg/m³**.
- Pada *test set*, terjadi peningkatan RMSE dan penurunan $R^2$ akibat munculnya beberapa lonjakan ekstrem anomali polusi pada data bulan Mei 2026. Namun, nilai **MAE tetap terjaga di bawah ambang batas 8,0 µg/m³** (`7,9332 µg/m³`), membuktikan bahwa deviasi rata-rata prediksi tetap akurat dan andal untuk penggunaan praktis sehari-hari.
- **Feature Importance:** Variabel polusi 1 jam sebelumnya (`pm25_lag1`) dan jam harian (`hour`) menempati peringkat teratas sebagai prediktor paling berpengaruh, selaras dengan sifat akumulasi polutan atmosfer dan siklus harian emisi kendaraan perkotaan.

## 7. Teknologi yang Digunakan

| Teknologi | Fungsi |
|---|---|
| **Python** | Bahasa pemrograman utama |
| **Pandas & NumPy** | Manipulasi data tabular dan komputasi numerik |
| **Matplotlib & Seaborn** | Eksplorasi visualisasi data (EDA) dan grafik evaluasi |
| **Scikit-learn** | Preprocessing (`StandardScaler`, encoding) dan metrik evaluasi |
| **LightGBM** | Algoritma regresi utama (`LGBMRegressor`) |
| **Joblib** | Serialisasi model dan objek *scaler* terlatih |
| **Streamlit** | Pengembangan dashboard dan antarmuka aplikasi web interaktif |
| **Git & GitHub** | *Version control* dan kolaborasi kode |

## 8. Arsitektur & Fitur Aplikasi

```text
Data Sensor & Input Cuaca (Real-time)
                │
                ▼
Validasi Input & Preprocessing (StandardScaler + Feature Engineering)
                │
                ▼
Inferensi Model LightGBM (model.joblib)
                │
                ▼
Dashboard Web Streamlit (Gauge Indikator & Rekomendasi Kesehatan)
