from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi Folder Upload
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# 1. LOAD MODEL & DATA
# ==========================================
MODEL_PATH = 'best_model.keras'
CSV_PATH = 'Nutri_Grade.csv'

try:
    model = load_model(MODEL_PATH)
    df = pd.read_csv(CSV_PATH)

    # Data Cleaning (Mengubah koma menjadi titik)
    cols_to_clean = ['gula', 'total_lemak', 'lemak_jenuh', 'gula/100ml', 'lemak_jenuh/100ml']
    for col in cols_to_clean:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: float(str(x).replace(',', '.')))

    print("✅ System Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading System: {e}")

# ==========================================
# 2. SETUP FUZZY LOGIC (Nutri-Grade Base + 7 Tingkat Output)
# ==========================================
# Universe
gula = ctrl.Antecedent(np.arange(0, 51, 0.1), 'gula')
lemak = ctrl.Antecedent(np.arange(0, 16, 0.1), 'lemak')
rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')

# --- FUNGSI KEANGGOTAAN INPUT (Berbasis Nutri-Grade) ---
# Referensi Nutri-Grade (per 100ml):
# Grade A: Gula <= 1, Lemak <= 0.7
# Grade B: Gula 1-5, Lemak 0.7-1.2
# Grade C: Gula 5-10, Lemak 1.2-2.8
# Grade D: Gula > 10, Lemak > 2.8

# Gula (g/100ml)
# Rendah (<5g) -> Mencakup Grade A & B
# Sedang (5-10g) -> Mencakup Grade C
# Tinggi (>10g) -> Mencakup Grade D
gula['rendah'] = fuzz.trapmf(gula.universe, [0, 0, 4, 7.5])  # Overlap di 4-6
gula['sedang'] = fuzz.trimf(gula.universe, [4, 7.5, 11])  # Tengah di 7.5
gula['tinggi'] = fuzz.trapmf(gula.universe, [7.5, 12, 50, 50])  # Naik di 9

# Lemak Jenuh (g/100ml)
# Rendah (<1.2g) -> Mencakup Grade A & B
# Sedang (1.2-2.8g) -> Mencakup Grade C
# Tinggi (>2.8g) -> Mencakup Grade D
lemak['rendah'] = fuzz.trapmf(lemak.universe, [0, 0, 1.0, 2.0])
lemak['sedang'] = fuzz.trimf(lemak.universe, [1.0, 2.0, 3.0])
lemak['tinggi'] = fuzz.trapmf(lemak.universe, [2.0, 4.0, 15, 15])

# --- FUNGSI KEANGGOTAAN OUTPUT (7 TINGKAT) ---
# Skala 0 (Sangat Buruk) sampai 100 (Sangat Baik)
rekomendasi['sangat_buruk'] = fuzz.trimf(rekomendasi.universe, [0, 0, 25])
rekomendasi['buruk'] = fuzz.trimf(rekomendasi.universe, [15, 30, 45])
rekomendasi['cukup_buruk'] = fuzz.trimf(rekomendasi.universe, [30, 45, 60])
rekomendasi['sedang'] = fuzz.trimf(rekomendasi.universe, [45, 60, 75])
rekomendasi['cukup_baik'] = fuzz.trimf(rekomendasi.universe, [60, 75, 90])
rekomendasi['baik'] = fuzz.trimf(rekomendasi.universe, [75, 90, 100])
rekomendasi['sangat_baik'] = fuzz.trimf(rekomendasi.universe, [90, 100, 100])

# --- ATURAN LOGIKA (MAPPING 7 TINGKAT) ---
# Logika: Menggabungkan status Gula & Lemak ke 7 level rekomendasi

# 1. KELOMPOK GULA RENDAH (<5g) - Zona Aman
# Gula Rendah + Lemak Rendah -> Sangat Baik (Grade A/B murni)
rule1 = ctrl.Rule(gula['rendah'] & lemak['rendah'], rekomendasi['sangat_baik'])
# Gula Rendah + Lemak Sedang -> Baik (Turun dikit karena lemak)
rule2 = ctrl.Rule(gula['rendah'] & lemak['sedang'], rekomendasi['baik'])
# Gula Rendah + Lemak Tinggi -> Sedang (Lemak merusak nilai sehat)
rule3 = ctrl.Rule(gula['rendah'] & lemak['tinggi'], rekomendasi['sedang'])

# 2. KELOMPOK GULA SEDANG (5-10g) - Zona Waspada
# Gula Sedang + Lemak Rendah -> Cukup Baik (Masih oke diminum)
rule4 = ctrl.Rule(gula['sedang'] & lemak['rendah'], rekomendasi['cukup_baik'])
# Gula Sedang + Lemak Sedang -> Cukup Buruk (Mulai waspada)
rule5 = ctrl.Rule(gula['sedang'] & lemak['sedang'], rekomendasi['cukup_buruk'])
# Gula Sedang + Lemak Tinggi -> Buruk (Kombinasi jelek)
rule6 = ctrl.Rule(gula['sedang'] & lemak['tinggi'], rekomendasi['buruk'])

# 3. KELOMPOK GULA TINGGI (>10g) - Zona Bahaya
# Gula Tinggi + Lemak Rendah -> Buruk (Gula saja sudah bahaya)
rule7 = ctrl.Rule(gula['tinggi'] & lemak['rendah'], rekomendasi['buruk'])
# Gula Tinggi + Lemak Sedang -> Sangat Buruk
rule8 = ctrl.Rule(gula['tinggi'] & lemak['sedang'], rekomendasi['sangat_buruk'])
# Gula Tinggi + Lemak Tinggi -> Sangat Buruk (Kombo maut)
rule9 = ctrl.Rule(gula['tinggi'] & lemak['tinggi'], rekomendasi['sangat_buruk'])

rek_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
fuzzy_sim = ctrl.ControlSystemSimulation(rek_ctrl)

# Label Map
LABEL_MAP = {
            0: 'ABC Kopi Susu', 1: 'BearBrand', 2: 'Benecol Lychee 100ml',
            3: 'Cimory Bebas Laktosa 250ml', 4: 'Cimory Susu Coklat Cashew',
            5: 'Cimory Yogurt Strawberry', 6: 'Cola-Cola 390ml',
            7: 'Fanta Strawberry 390ml', 8: 'Floridina 350ml',
            9: 'Fruit Tea Freeze 350ml', 10: 'Garantea',
            11: 'Golda Cappucino', 12: 'Hydro Coco Original 250ml',
            13: 'Ichitan Thai Green Tea', 14: 'Larutan Penyegar rasa Jambu',
            15: 'Mizone 500ml', 16: 'NU Green Tea Yogurt',
            17: 'Nutri Boost Orange Flavour 250ml', 18: 'Oatside Cokelat',
            19: 'Pepsi Blue 330ml', 20: 'Pocari Sweat 500 ml',
            21: 'Sprite 390ml', 22: 'Teh Pucuk Harum',
            23: 'Tebs Sparkling 330ml', 24:'Teh Kotak 200ml',
            25: 'Tehbotol Sosro 250ml', 26: 'Ultra Milk Coklat Ultrajaya 200ml',
            27: 'Ultramilk Fullcream 250ml', 28: 'Yakult',
            29: 'You C 1000 Orange'
        }


# ==========================================
# 3. ROUTES
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # 1. Prediksi Gambar
                img = image.load_img(filepath, target_size=(120, 120))
                x = image.img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                pred = model.predict(x)
                pred_idx = np.argmax(pred)
                product_name = LABEL_MAP.get(pred_idx, "Unknown")

                # 2. Ambil Data Gizi
                row = df[df['nama_produk'] == product_name]

                result = {
                    'image': filepath, 'product': product_name,
                    'gula': 0, 'lemak': 0, 'grade': '-',
                    'skor': 0, 'saran': 'Data Tidak Ditemukan', 'color': 'secondary'
                }

                if not row.empty:
                    # Ambil Data PER 100ML
                    gula_val = float(row['gula/100ml'].values[0])
                    lemak_val = float(row['lemak_jenuh/100ml'].values[0])
                    grade_val = '-'
                    if 'nutri_grade' in row.columns:
                        grade_val = str(row['nutri_grade'].values[0]).strip().upper()

                    # 3. Hitung Fuzzy (Input diclip)
                    fuzzy_sim.input['gula'] = min(gula_val, 50)
                    fuzzy_sim.input['lemak'] = min(lemak_val, 15)

                    fuzzy_sim.compute()
                    skor = fuzzy_sim.output['rekomendasi']

                    # Interpretasi Skor (7 Tingkat)
                    # Kita kirim kode HEX warna, bukan nama class Bootstrap
                    if skor >= 90:
                        saran = "SANGAT BAIK (Aman Dikonsumsi)"
                        bg_color = "#198754"  # Hijau Tua (Bootstrap Success Dark)
                        text_color = "white"
                    elif skor >= 80:
                        saran = "BAIK (Pilihan Sehat)"
                        bg_color = "#20c997"  # Hijau Teal
                        text_color = "white"
                    elif skor >= 65:
                        saran = "CUKUP BAIK (Boleh Dikonsumsi)"
                        bg_color = "#ADFF2F"  # Green Yellow (Custom Pilihan Anda)
                        text_color = "black"  # Teks hitam biar kontras
                    elif skor >= 50:
                        saran = "SEDANG (Perhatikan Porsi)"
                        bg_color = "#ffc107"  # Kuning (Bootstrap Warning)
                        text_color = "black"
                    elif skor >= 35:
                        saran = "CUKUP BURUK (Mulai Batasi)"
                        bg_color = "#fd7e14"  # Oranye
                        text_color = "white"
                    elif skor >= 20:
                        saran = "BURUK (Sebaiknya Hindari)"
                        bg_color = "#dc3545"  # Merah (Bootstrap Danger)
                        text_color = "white"
                    else:
                        saran = "SANGAT BURUK (Berisiko Tinggi)"
                        bg_color = "#8B0000"  # Merah Marun Gelap
                        text_color = "white"

                    print(f"Produk: {product_name} | Gula: {gula_val} | Skor: {skor:.2f} ({saran})")

                    result.update({
                        'gula': gula_val,
                        'lemak': lemak_val,
                        'grade': grade_val,
                        'skor': round(skor, 1),
                        'saran': saran,
                        'bg_color': bg_color,  # Kirim HEX Background
                        'text_color': text_color  # Kirim Warna Teks
                    })

                return render_template('index.html', result=result)

            except Exception as e:
                print(f"Error: {e}")
                return render_template('index.html', result=None)

    return render_template('index.html', result=None)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
