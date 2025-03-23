# 📌 Sistem Rekomendasi Restoran dengan Hybrid Filtering

## 📖 Deskripsi Proyek

Proyek ini adalah sistem rekomendasi restoran berbasis web yang dikembangkan menggunakan **Streamlit** dan **Supabase**. Model rekomendasi menggunakan **Hybrid Filtering**, yaitu gabungan antara:

1. **Content-Based Filtering** – Merekomendasikan restoran berdasarkan kesamaan fitur.
2. **Item-Based Collaborative Filtering** – Merekomendasikan restoran berdasarkan rating pengguna lain dengan selera serupa.

Sistem ini juga memiliki fitur evaluasi **Intra-List Similarity (ILS)** untuk menilai keanekaragaman rekomendasi yang dihasilkan.
Sistem ini juga memiliki fitur evaluasi **Mean Absolute Error (MAE)** untuk mengukur rata-rata kesalahan absolut antara nilai aktual dan prediksi.

---

## ⚙️ Instalasi dan Konfigurasi

### 1️⃣ **Clone Repository**

```bash
git https://github.com/ahmadseloabadi/recsys-restaurant-streamlit
cd recsys-restaurant-streamlit
```

### 2️⃣ **Buat Virtual Environment** _(Opsional, tapi direkomendasikan)_

```bash
python -m venv venv
source venv/bin/activate  # Untuk Mac/Linux
venv\Scripts\activate     # Untuk Windows
```

### 3️⃣ **Instal Dependensi**

```bash
pip install -r requirements.txt
```

### 4️⃣ **Konfigurasi Supabase**

1. **Buka Supabase** → [https://supabase.com](https://supabase.com)
2. **Buat proyek baru** → Dapatkan **Database Connection URL** dari **Database → Connection Info**
3. **Buat file `.env`** di root folder proyek:

```env
SUPABASE_URL="https://your-supabase-url.supabase.co"
SUPABASE_KEY="your-anon-key"
SUPABASE_DB_URL="postgresql://username:password@host:port/database"
```

### 5️⃣ **Jalankan Aplikasi**

```bash
streamlit run sisrek_rest.py
```

---

## 🗂️ Struktur Proyek

```
📂 proyek-rekomendasi-restoran
├── 📄 sisrek_rest.py  # Main script aplikasi Streamlit
├── 📄 setup_supabase.py  # Membuat tabel dan upload data
├── 📂 data
│    ├──📂 dataset
│    │  ├── places_to_eat.csv  # Data restoran
│    │  ├── ratings.csv  # Data rating pengguna
│    │  ├── users.csv  # Data pengguna
├── 📄 requirements.txt  # Daftar dependensi
├── 📄 .env  # Konfigurasi kredensial Supabase
└── 📄 README.md  # Dokumentasi proyek ini
```

---

## 🔑 Contoh Login Admin dan User

### **Admin Login:**

- **Username:** `admin`
- **Password:** `admin`

### **User Login:**

- **Username:** `user1`
- **Password:** `user1`

> **Catatan:** Password default dihasilkan sebagai `nama`. Jika ada perubahan, cek di database Supabase.

---

## 📊 Struktur Database di Supabase

### **1️⃣ Tabel `users`** (Data Pengguna)

| Kolom    | Tipe Data | Keterangan             |
| -------- | --------- | ---------------------- |
| user_id  | TEXT (PK) | ID pengguna            |
| nama     | TEXT      | Nama pengguna          |
| password | TEXT      | Password terenkripsi   |
| role     | TEXT      | Hak akses (admin/user) |

### **2️⃣ Tabel `places_to_eat`** (Data Restoran)

| Kolom                          | Tipe Data | Keterangan                       |
| ------------------------------ | --------- | -------------------------------- |
| restaurant_id                  | UUID (PK) | ID restoran unik                 |
| nama_restoran                  | TEXT      | Nama restoran                    |
| preferensi_makanan             | TEXT      | Jenis makanan utama              |
| harga_rata_rata                | INTEGER   | Harga rata-rata                  |
| rating_toko                    | FLOAT     | Rating rata-rata                 |
| jenis_suasana                  | TEXT      | Suasana restoran                 |
| variasi_makanan                | TEXT      | Jenis makanan yang tersedia      |
| keramaian_restoran             | INTEGER   | Seberapa ramai restoran          |
| disajikan_atau_ambil_sendiri   | TEXT      | Cara penyajian makanan           |
| all_you_can_eat_atau_ala_carte | TEXT      | Jenis menu (AYCE atau ala carte) |

### **3️⃣ Tabel `ratings`** (Data Rating Pengguna)

| Kolom         | Tipe Data | Keterangan                                                 |
| ------------- | --------- | ---------------------------------------------------------- |
| id            | UUID (PK) | ID rating unik                                             |
| user_id       | TEXT (FK) | ID pengguna (Foreign Key ke `users.user_id`)               |
| restaurant_id | UUID (FK) | ID restoran (Foreign Key ke `places_to_eat.restaurant_id`) |
| nama_restoran | TEXT      | Nama restoran                                              |
| rating        | INTEGER   | Rating dari pengguna                                       |

---

## 📈 Evaluasi Model (ILS - Intra-List Similarity)

Aplikasi ini menyediakan fitur evaluasi **ILS** untuk mengukur kemiripan dalam daftar rekomendasi.

### **Bagaimana Cara Menggunakan ILS?**

1. Pilih **3 pengguna** dari daftar yang tersedia.
2. Sistem akan menghitung ILS berdasarkan daftar rekomendasi masing-masing pengguna.
3. Nilai ILS lebih tinggi → Rekomendasi kurang beragam. Nilai lebih rendah → Rekomendasi lebih beragam.

---

## 🛠️ Pengembangan dan Kontribusi

Jika ingin berkontribusi dalam proyek ini:

1. Fork repositori ini.
2. Buat branch baru: `git checkout -b feature-nama-fitur`
3. Commit perubahan: `git commit -m "Menambahkan fitur baru"`
4. Push ke branch Anda: `git push origin feature-nama-fitur`
5. Ajukan Pull Request ke repositori utama.

---

🚀 **Selamat mencoba sistem rekomendasi restoran ini! Jika ada kendala, jangan ragu untuk menghubungi saya.**
