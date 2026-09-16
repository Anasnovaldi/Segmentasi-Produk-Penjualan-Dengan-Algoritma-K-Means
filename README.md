# Segmentasi-Produk-Penjualan-Dengan-Algoritma-K-Means
Repository ini merupakan bagian dari pembelajaran machine learning unsupervised dengan algoritma K-Means untuk menetapkan segmentasi produk.

Program sederhana untuk mengelompokkan produk retail ke dalam 3 kategori — **Best Seller Konsisten**, **Musiman**, dan **Produk Kurang Laku** — menggunakan algoritma **K-Means Clustering**.
 
## Cara Kerja
 
Program membaca data penjualan bulanan (Jan-Des) tiap produk, lalu menghitung fitur turunan:
- **Rata-rata penjualan** — seberapa laku produk secara umum
- **Rasio lonjakan** (penjualan tertinggi ÷ rata-rata) — untuk mendeteksi produk musiman
Kedua fitur ini di-scaling lalu dikelompokkan menggunakan K-Means (K=3), dengan jumlah cluster optimal divalidasi melalui **Elbow Method**.