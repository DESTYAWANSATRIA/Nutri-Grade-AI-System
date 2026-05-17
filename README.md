# Nutri-Grade AI System
> Klasifikasi Minuman & Rekomendasi Konsumsi Cerdas Berbasis CNN dan Logika Fuzzy Mamdani

Sistem kecerdasan buatan *end-to-end* yang memadukan *Computer Vision* untuk mengenali 30 jenis produk minuman kemasan di pasaran dan Logika Fuzzy Mamdani untuk menghitung skor kelayakan konsumsi secara granular berdasarkan kandungan gula dan lemak jenuh.

**Tech Stack:** `Python`, `TensorFlow/Keras`, `Scikit-Fuzzy`, `Flask`

---

### 🎯 Latar Belakang & Business Impact
Tingginya angka diabetes dan kebingungan konsumen dalam membaca tabel Informasi Nilai Gizi (ING) memicu kebutuhan akan sistem identifikasi instan. Sistem ini dibangun untuk menerjemahkan data gizi yang kompleks menjadi skor rekomendasi yang manusiawi secara *real-time*, membantu konsumen mengambil keputusan sehat hanya dengan memindai kemasan minuman.

---

### 🔄 Diagram Alir Sistem

1.  **Pengumpulan Dataset**
2.  **Proses Paralel:**
    * **Jalur Citra:**
        1.  Dataset Citra Produk
        2.  Preprocessing Citra (resize, normalisasi, augmentasi)
        3.  Pelatihan Model CNN
        4.  Output: Klasifikasi Citra
    * **Jalur Gizi:**
        1.  Informasi Nilai Gizi
        2.  Preprocessing Data Gizi (Seleksi gula & lemak jenuh)
        3.  Normalisasi Gizi (Konversi ke / 100 ml)
        4.  Perhitungan Nutri-Grade (Penambahan kolom)
3.  **Pencocokan Data** (Menggabungkan hasil klasifikasi citra dengan data gizi)
4.  **Input untuk Fuzzy:** Kandungan Gula/100ml & Kandungan Lemak Jenuh/100ml
5.  **Sistem Fuzzy Mamdani** (Penentuan Rekomendasi Konsumsi)
6.  **Output Akhir:** Klasifikasi Produk, Nutri-Grade, & Skor Rekomendasi Konsumsi

---

### 🌐 Arsitektur Backend & API (Flask)
Model AI yang telah dilatih dibungkus menggunakan *framework* **Flask** untuk melayani permintaan dari antarmuka klien secara *real-time*. Sistem ini menerima *request* berupa citra masukan, memprosesnya melalui model hibrida (CNN + Fuzzy), dan mengembalikan hasil analisis dalam format JSON yang terstruktur.

**Endpoint:** `POST /predict`

**Contoh API Response (JSON):**
```json
{
  "status": "success",
  "produk": {
    "nama_kelas": "Yakult",
    "akurasi_prediksi": 0.98
  },
  "analisis_gizi": {
    "gula_per_100ml": 14.2,
    "lemak_jenuh_per_100ml": 0.0,
    "nutri_grade": "D"
  },
  "rekomendasi": {
    "skor_fuzzy": 38.5,
    "kategori": "Buruk",
    "pesan": "Berisiko tinggi, sebaiknya batasi konsumsi karena kandungan gula tinggi."
  }
}

---

### 📊 Pengumpulan & Pra-pemrosesan Data

-   **Dataset Primer:** Dataset terdiri dari 1.500 citra primer (30 kelas produk, masing-masing 50 citra) yang diambil secara manual di lingkungan ritel nyata (Indomaret, Alfamart, Superindo) dengan variasi pencahayaan dan latar belakang.
-   **Pra-pemrosesan & Augmentasi:** Tahapan pra-pemrosesan mencakup Resize citra ke 120x120 piksel, Normalisasi (rescale 1./255), dan Augmentasi data real-time (rotasi 20 derajat, zoom 20%, shift 20%, dan horizontal flip) untuk meningkatkan ketahanan model. Data dibagi 80:20 (1.200 latih, 300 validasi).

**Daftar 30 Kelas Produk yang Dikenali:**
1.  ABC Kopi Susu
2.  BearBrand
3.  Benecol Lychee 100ml
4.  Cimory Bebas Laktosa 250ml
5.  Cimory Susu Coklat Cashew
6.  Cimory Yogurt Strawberry
7.  Cola-Cola 390ml
8.  Fanta Strawberry 390ml
9.  Floridina 350ml
10. Fruit Tea Freeze 350ml
11. Garantea
12. Golda Cappucino
13. Hydro Coco Original 250ml
14. Ichitan Thai Green Tea
15. Larutan Penyegar rasa Jambu
16. Mizone 500ml
17. NU Green Tea Yogurt
18. Nutri Boost Orange Flavour 250ml
19. Oatside Cokelat
20. Pepsi Blue 330ml
21. Pocari Sweat 500 ml
22. Sprite 390ml
23. Teh Pucuk Harum
24. Tebs Sparkling 330ml
25. Teh Kotak 200ml
26. Tehbotol Sosro 250ml
27. Ultra Milk Coklat Ultrajaya 200ml
28. Ultramilk Fullcream 250ml
29. Yakult
30. You C 1000 Orange

<img width="1080" height="1920" alt="dataset_sample" src="https://github.com/user-attachments/assets/6db6462a-13f8-479d-93de-2ed893748dc3" />
*Contoh variasi citra dalam dataset yang diambil dengan latar belakang, sudut, dan pencahayaan berbeda.*

---

### 🧠 Arsitektur Convolutional Neural Network (CNN)

Menghindari model raksasa seperti VGG16 untuk mencegah overfitting pada dataset berukuran kecil, sistem ini menggunakan Arsitektur Custom yang efisien secara komputasi.

-   **Feature Extraction:** 3 Blok Konvolusi (32, 64, dan 128 filter) menggunakan kernel (3,3), aktivasi ReLU, yang masing-masing diikuti oleh MaxPooling2D (2,2).
-   **Classification:** Lapisan Flatten, diikuti Dense Layer (512 neuron, ReLU).
-   **Regularization:** Penggunaan Dropout (0.5) yang terbukti sangat krusial dalam menyaring fitur tidak relevan dan mencegah model menghafal data latih.
-   **Output:** Dense Layer dengan 30 neuron (Softmax) yang dioptimasi menggunakan Adam Optimizer dan Categorical Crossentropy.

<img width="1147" height="304" alt="cnn_architecture" src="https://github.com/user-attachments/assets/a085eb87-f1a5-4987-b7b7-a3417000dc11" />
*Visualisasi arsitektur model CNN yang digunakan untuk klasifikasi citra produk.*

> **[Evaluasi Model]** Sistem mencapai stabilitas konvergensi pada epoch ke-61 (dengan Early Stopping) dan menghasilkan akurasi global rata-rata sebesar 92% (Weighted Avg F1-Score) berdasarkan evaluasi Confusion Matrix.

---

### ⚖️ Sistem Pendukung Keputusan (Logika Fuzzy Mamdani)

-   **Masalah Batas Tajam (Sharp Boundary):** Sistem regulasi gizi statis sering memvonis produk secara kaku (misal: 9.9g gula = Grade C, 10.1g = Grade D).
-   **Solusi Fuzzy:** Sistem Fuzzy memetakan crisp input (Kandungan Gula 0-51g/100ml dan Lemak Jenuh 0-16g/100ml) ke dalam 3 himpunan linguistik (Rendah, Sedang, Tinggi) menggunakan kurva trapesium dan segitiga.
-   **Rule Base & Defuzzifikasi:** Memanfaatkan matriks 9 Aturan (Rule Base) berbasis prinsip 'worst-attribute dominance', lalu menggunakan metode Centroid untuk menghasilkan output Skor Rekomendasi 0-100.
-   **Granularitas Output:** Sistem memecah rekomendasi menjadi 7 tingkatan halus (Sangat Buruk, Buruk, Cukup Buruk, Sedang, Cukup Baik, Baik, Sangat Baik) sehingga penilaian menjadi lebih manusiawi.

#### Tahapan Proses Inferensi Fuzzy
1.  **Fuzzifikasi (Fuzzification)**
    Mengubah input numerik (crisp) menjadi nilai keanggotaan fuzzy. Misalnya, nilai Gula 8g/100ml akan dipetakan ke himpunan fuzzy: mungkin 80% 'SEDANG' dan 20% 'RENDAH'.
2.  **Aplikasi Fungsi Implikasi (Rule Evaluation)**
    Setiap aturan dalam Rule Base dievaluasi. Kekuatan aktivasi setiap aturan ditentukan oleh operator AND (mengambil nilai minimum dari keanggotaan input). Contoh: IF Gula 'SEDANG' (0.8) AND Lemak 'RENDAH' (0.9), maka kekuatan aturan adalah min(0.8, 0.9) = 0.8.
3.  **Agregasi (Aggregation)**
    Output dari semua aturan yang aktif digabungkan menjadi satu himpunan fuzzy tunggal untuk variabel output (Skor Rekomendasi). Proses ini menggunakan operator OR (mengambil nilai maksimum).
4.  **Defuzzifikasi (Defuzzification)**
    Himpunan fuzzy hasil agregasi diubah kembali menjadi nilai numerik (crisp) tunggal menggunakan metode Centroid (titik pusat massa), menghasilkan skor akhir antara 0-100.

<img width="1189" height="1790" alt="fuzzy_membership_functions" src="https://github.com/user-attachments/assets/245659a3-e70a-4371-bc20-b0278490aa22" />
*Visualisasi Fungsi Keanggotaan Fuzzy untuk variabel input (Gula, Lemak Jenuh) dan output (Skor Rekomendasi).*

#### Matriks Aturan Fuzzy (Rule Base)

| | **Lemak Jenuh** | | |
| :--- | :---: | :---: | :---: |
| | **RENDAH** | **SEDANG** | **TINGGI** |
| **Gula RENDAH** | SANGAT BAIK | BAIK | SEDANG |
| **Gula SEDANG** | CUKUP BAIK | CUKUP BURUK | BURUK |
| **Gula TINGGI** | BURUK | SANGAT BURUK | SANGAT BURUK |

#### Fungsi Keanggotaan Input: Kandungan Gula

| Himpunan Fuzzy | Rentang Nilai (g / 100 ml) | Kategori Nutri-Grade |
| :--- | :--- | :--- |
| Rendah | 0 – 5 | Mewakili Grade A & Grade B |
| Sedang | > 5 – 10 | Mewakili Grade C |
| Tinggi | > 10 | Mewakili Grade D |

#### Fungsi Keanggotaan Input: Kandungan Lemak Jenuh

| Himpunan Fuzzy | Rentang Nilai (g / 100 ml) | Kategori Nutri-Grade |
| :--- | :--- | :--- |
| Rendah | 0 – 1,2 | Mewakili Grade A & Grade B |
| Sedang | > 1,2 – 2,8 | Mewakili Grade C |
| Tinggi | > 2,8 | Mewakili Grade D |

#### Fungsi Keanggotaan Output: Skor Rekomendasi Konsumsi

| Himpunan Fuzzy | Rentang Parameter | Keterangan / Tingkat Rekomendasi |
| :--- | :--- | :--- |
| Sangat Buruk | 0 – 25 | Berisiko sangat tinggi (Level terendah Grade D) |
| Buruk | 15 – 45 | Berisiko tinggi, sebaiknya hindari (Grade D) |
| Cukup Buruk | 30 – 60 | Mendekati batas bahaya, mulai batasi konsumsi |
| Sedang | 45 – 75 | Perhatikan porsi konsumsi (Transisi Grade C) |
| Cukup Baik | 60 – 90 | Relatif aman, boleh dikonsumsi (Transisi Grade B) |
| Baik | 75 – 100 | Pilihan sehat (Grade A & B) |
| Sangat Baik | 90 – 100 | Pilihan paling sehat, tanpa risiko (Murni Grade A) |

#### Parameter Fungsi Keanggotaan Fuzzy

| Variabel | Himpunan Fuzzy | Bentuk Fungsi | Parameter |
| :--- | :--- | :--- | :--- |
| Kandungan Gula (g/100ml) | RENDAH | Trapesium turun | [0, 0, 4, 7.5] |
| Kandungan Gula (g/100ml) | SEDANG | Segitiga | [4, 7.5, 11] |
| Kandungan Gula (g/100ml) | TINGGI | Trapesium naik | [7.5, 12, 50, 50] |
| Kandungan Lemak Jenuh (g/100ml) | RENDAH | Trapesium turun | [0, 0, 1.0, 2.0] |
| Kandungan Lemak Jenuh (g/100ml) | SEDANG | Segitiga | [1.0, 2.0, 3.0] |
| Kandungan Lemak Jenuh (g/100ml) | TINGGI | Trapesium naik | [2.0, 4.0, 15, 15] |
| Skor Rekomendasi (Nilai 0-100) | SANGAT BURUK | Segitiga | [0, 0, 25] |
| Skor Rekomendasi (Nilai 0-100) | BURUK | Segitiga | [15, 30, 45] |
| Skor Rekomendasi (Nilai 0-100) | CUKUP BURUK | Segitiga | [30, 45, 60] |
| Skor Rekomendasi (Nilai 0-100) | SEDANG | Segitiga | [45, 60, 75] |
| Skor Rekomendasi (Nilai 0-100) | CUKUP BAIK | Segitiga | [60, 75, 90] |
| Skor Rekomendasi (Nilai 0-100) | BAIK | Segitiga | [75, 90, 100] |
| Skor Rekomendasi (Nilai 0-100) | SANGAT BAIK | Segitiga | [90, 100, 100] |

---

### 🎥 Visualisasi & Live Demo

<img width="1918" height="865" alt="web_interface" src="https://github.com/user-attachments/assets/6c29b28e-c912-422c-a34b-6cbbf2eee692" />
*Screenshot Interface Web Flask*

<img width="1006" height="470" alt="model_acurracy" src="https://github.com/user-attachments/assets/c285a753-568b-4007-bd23-aa6fb2e4ccaf" />
*Grafik Training Accuracy & Loss*

<img width="2188" height="1990" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c989f81d-1540-49a1-b687-09506389983d" />
*Visualisasi Confusion Matrix*

<br>

<p align="center">
  <a href="https://huggingface.co/spaces/destyawan/nutri-grade-ai">
    <strong>Coba Live Demo (Hugging Face Spaces)</strong>
  </a>
</p>
