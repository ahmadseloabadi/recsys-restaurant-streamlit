import os
import psycopg2
import pandas as pd
import uuid
from dotenv import load_dotenv
from bcrypt import hashpw, gensalt

# Load environment variables
load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

def connect_db():
    return psycopg2.connect(SUPABASE_DB_URL)

def create_tables():
    queries = [
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            nama TEXT NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS places_to_eat (
            restaurant_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            nama_restoran TEXT NOT NULL,
            preferensi_makanan TEXT,
            harga_rata_rata INTEGER,
            rating_toko FLOAT,
            jenis_suasana TEXT,
            variasi_makanan TEXT,
            keramaian_restoran INTEGER,
            disajikan_atau_ambil_sendiri TEXT,
            all_you_can_eat_atau_ala_carte TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ratings (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
            nama TEXT,
            restaurant_id UUID REFERENCES places_to_eat(restaurant_id) ON DELETE CASCADE,
            nama_restoran TEXT,
            rating INTEGER
        );
        """
    ]
    
    try:
        conn = connect_db()
        cursor = conn.cursor()
        for query in queries:
            cursor.execute(query)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Semua tabel berhasil dibuat di Supabase!")
    except Exception as e:
        print(f"❌ Gagal membuat tabel: {e}")

def hash_password(password):
    return hashpw(password.encode(), gensalt()).decode()

def upload_data():
    conn = connect_db()
    cursor = conn.cursor()
    file_path = "data/dataset/restaurant_recommendation.xlsx"
    xls = pd.ExcelFile(file_path)
    
    places_to_eat_df = pd.read_excel(xls, sheet_name="places_to_eat")
    ratings_df = pd.read_excel(xls, sheet_name="ratings")
    users_df = pd.read_excel(xls, sheet_name="users")
    
    users_df["password"] = users_df.apply(lambda row: hash_password(row["nama"]), axis=1)
    
    places_to_eat_df["restaurant_id"] = [str(uuid.uuid4()) for _ in range(len(places_to_eat_df))]
    users_df["user_id"] = [str(uuid.uuid4()) for _ in range(len(users_df))]
    
    resto_id_mapping = dict(zip(places_to_eat_df.index.astype(str), places_to_eat_df["restaurant_id"]))
    user_id_mapping = dict(zip(users_df.index.astype(str), users_df["user_id"]))
    
    ratings_df = ratings_df.dropna(subset=["restaurant_id"])
    ratings_df["restaurant_id"] = ratings_df["restaurant_id"].astype(str).map(resto_id_mapping)
    ratings_df = ratings_df.dropna(subset=["restaurant_id"])  # Hapus yang tidak berhasil dipetakan

    ratings_df = ratings_df.dropna(subset=["user_id"])
    ratings_df["user_id"] = ratings_df["user_id"].astype(str).map(user_id_mapping)
    ratings_df = ratings_df.dropna(subset=["user_id"])  # Hapus yang tidak berhasil dipetakan
    
    for _, row in users_df.iterrows():
        cursor.execute("""
            INSERT INTO users (user_id, nama, password, role) VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING;
        """, (row["user_id"], row["nama"], row["password"], row["role"]))
    
    for _, row in places_to_eat_df.iterrows():
        cursor.execute("""
            INSERT INTO places_to_eat (restaurant_id, nama_restoran, preferensi_makanan, harga_rata_rata, rating_toko,
                                      jenis_suasana, variasi_makanan, keramaian_restoran,
                                      disajikan_atau_ambil_sendiri, all_you_can_eat_atau_ala_carte)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (row["restaurant_id"], row["Nama Restoran"], row["Preferensi Makanan"], row["Harga Rata-Rata Makanan di Toko (Rp)"],
               row["Rating Toko"], row["Jenis Suasana"], row["Variasi Makanan"], row["Keramaian Restoran"],
               row["Disajikan atau Ambil Sendiri"], row["All You Can Eat atau Ala Carte"]))
    
    for _, row in ratings_df.iterrows():
        if pd.isna(row["restaurant_id"]) and pd.isna(row["user_id"]):
            continue
        cursor.execute("""
            INSERT INTO ratings (user_id, nama ,restaurant_id, nama_restoran, rating)
            VALUES (%s,%s, %s, %s, %s);
        """, (row["user_id"],row["nama"], row["restaurant_id"], row["Nama Restoran"], row["rating"]))
    
    conn.commit()
    cursor.close()
    conn.close()
    print("🚀 Semua data berhasil diunggah ke Supabase!")

if __name__ == "__main__":
    create_tables()
    upload_data()
