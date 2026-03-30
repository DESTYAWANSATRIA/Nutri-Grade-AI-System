# Gunakan versi Python yang ringan
FROM python:3.9-slim

# Atur direktori kerja di dalam server
WORKDIR /app

# Salin file requirements.txt dan instal library-nya
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh file proyek Kakak (termasuk best_model.keras dan folder static)
COPY . .

# Hugging Face mewajibkan aplikasi berjalan di port 7860
EXPOSE 7860

# Perintah untuk menjalankan aplikasi Flask menggunakan Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:7860"]
