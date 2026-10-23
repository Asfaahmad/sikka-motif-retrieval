# Temu Kembali Citra Motif Tenun Ikat Sikka

Prototipe web Final Project mata kuliah Temu Kembali Citra, Program Studi Informatika,
Universitas Muhammadiyah Malang.

**Judul:** Evaluasi Ketahanan Temu Kembali Citra Motif Tenun Ikat Sikka terhadap Variasi Pencahayaan
Menggunakan Histogram HSV, ResNet-18, dan Normalisasi Grayscale-CLAHE

**Kelompok:** Ahmad Hamdan Hamidiy (202310370311194) · Asfa Ahmad (202310370311203)

## Fitur
- Kueri dari 5 foto demo, foto dataset, atau unggahan sendiri
- Simulasi cahaya pada kueri (gelap, terang, gamma, lampu kuning, lampu biru, kecerahan)
- Perbandingan tiga model berdampingan:
  - Model 1: histogram HSV 16×4×4
  - Model 2: ResNet-18 pralatih ImageNet
  - Model 3 (usulan): grayscale + CLAHE → ResNet-18
- Tanda ✓/✗ kesesuaian motif dan waktu pencarian

## Menjalankan secara lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Saat pertama dibuka, aplikasi menghitung fitur 432 foto galeri (sekitar 1 menit), lalu menyimpannya di cache.

## Dataset
Folder `data/` berisi dataset **Motif-Sikka v3**:
I. D. Reja, "Motif-Sikka: A curated image dataset of traditional Sikka ikat weaving motifs for textile
pattern analysis," Mendeley Data, V3, 2025, doi: 10.17632/b2k52v9mkc.3. Lisensi **CC BY 4.0**.

Folder `demo/` berisi lima foto dataset yang dimodifikasi pencahayaan atau sudutnya untuk keperluan demo.
