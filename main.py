import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    while True:
        try:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"{"Program Clustering Produk Berdasarkan Penjualan":^80}")
            nama_file = input("Masukkan Nama File (e.g Penjualan.csv) : ")
            bulan = ["jan","feb","mar","apr","mei","jun","jul","agu","sep","okt","nov","des"]
            df = pd.read_csv(nama_file)

            # --- Bikin FITUR turunan dari data penjualan mentah ---
            kolom_bulan = bulan
            df["rata_rata_jual"] = df[kolom_bulan].mean(axis=1)
            df["std_jual"] = df[kolom_bulan].std(axis=1)             # variasi/fluktuasi penjualan
            df["max_jual"] = df[kolom_bulan].max(axis=1)
            df["rasio_lonjakan"] = df["max_jual"] / df["rata_rata_jual"]  # seberapa besar lonjakan vs rata-rata

            print(df[["produk","rata_rata_jual","std_jual","rasio_lonjakan"]])

            # Tentukan Fitur
            X = df[["rata_rata_jual","rasio_lonjakan"]]

            # Wajib Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Method
            inertia_list = []
            K_range = range(1,7)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertia_list.append(km.inertia_)

            # Visualisasi Cluster Yang Optimal
            plt.figure(figsize=(7,5))
            plt.plot(K_range, inertia_list, marker="o")
            plt.xlabel("Jumlah Cluster (K)")
            plt.ylabel("Inertia")
            plt.title("Elbow Method")
            plt.grid(True, alpha=0.3)
            plt.show()

            # Buat Model K means dengan K=3
            model = KMeans(n_clusters=3, random_state=42, n_init=10)
            df["cluster"] = model.fit_predict(X_scaled)

            # --- Auto-labeling cluster berdasarkan karakteristik ---
            ringkasan = df.groupby("cluster")[["rata_rata_jual","rasio_lonjakan"]].mean()
            print("\nRingkasan tiap cluster:")
            print(ringkasan)

            def beri_label(row):
                if row["rata_rata_jual"] > ringkasan["rata_rata_jual"].median() and row["rasio_lonjakan"] < 2:
                    return "Best Seller Konsisten"
                elif row["rasio_lonjakan"] >= 2:
                    return "Musiman"
                else:
                    return "Produk Kurang Laku"

            df["status"] = df.apply(beri_label, axis=1)

            print(f"\n{"="*80}")
            print(f"{"========================= Hasil Clustering Produk =========================":^80}")
            print("-"*80)
            print(df[["produk","rata_rata_jual","rasio_lonjakan","status"]].to_string(index=False))
            print("="*80)

            # Visualisasi Cluster Produk
            plt.figure(figsize=(10,6))
            colors = {"Best Seller Konsisten":"green", "Musiman":"orange", "Produk Kurang Laku":"red"}
            for status in df["status"].unique():
                subset = df[df["status"]==status]
                plt.scatter(subset["rata_rata_jual"], subset["rasio_lonjakan"],
                            label=status, color=colors[status], s=120, alpha=0.7)
                for _, row in subset.iterrows():
                    plt.annotate(row["produk"], (row["rata_rata_jual"], row["rasio_lonjakan"]), fontsize=8)

            plt.xlabel("Rata-rata Penjualan Bulanan")
            plt.ylabel("Rasio Lonjakan (Max/Rata-rata)")
            plt.title("Segmentasi Produk berdasarkan Pola Penjualan")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except FileNotFoundError:
            print(f"file bernama {nama_file} tidak ditemukan")

        option_close = input("Apakah Kamu Ingin Melanjutkan (yes/no) : ").lower()
        if option_close == "n" or option_close == "no":
            break