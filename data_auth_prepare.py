import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from streamlit_authenticator.utilities.hasher import Hasher

# Load Credentials dari JSON
creds = Credentials.from_service_account_file("chrome-setup-454416-j9-96c1d1735f41.json", scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
client = gspread.authorize(creds)

# Buka Spreadsheet
spreadsheet = client.open("restaurant_recommendation")

def load_data(sheet_name):
    """Membaca data dari Google Sheets dan mengubahnya ke DataFrame"""
    sheet = spreadsheet.worksheet(sheet_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def update_data(sheet_name, df):
    """Menulis DataFrame ke Google Sheets"""
    sheet = spreadsheet.worksheet(sheet_name)
    sheet.clear()  # Hapus data lama
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def create_users_from_ratings():
    """Membuat data user dari tab ratings"""
    ratings_df = load_data("ratings")  # Load data ratings

    if ratings_df.empty:
        print("Tab 'ratings' kosong, tidak ada user yang dibuat.")
        return

    # Ambil user_id dan name dari data ratings (pastikan tidak ada duplikasi)
    users_df = ratings_df[['user_id', 'nama']].drop_duplicates()

    # Buat password default: name + user_id
    users_df['password'] = users_df['nama'] + users_df['user_id'].astype(str)

    # Hash password sebelum menyimpan
    users_df['password'] = Hasher(users_df['password'].tolist()).generate()

    # Tambahkan kolom role sebagai "user"
    users_df['role'] = "user"

    # **Tambahkan Admin Secara Manual (Jika Belum Ada)**
    admin_data = pd.DataFrame({
        'user_id': [0],
        'nama': ['admin'],
        'password': Hasher(["admin123"]).generate(),  # Password admin123 (hashed)
        'role': ['admin']
    })

    # Gabungkan data admin dan user
    users_df = pd.concat([admin_data, users_df], ignore_index=True)


    # Simpan ke tab "users" di Google Sheets
    update_data("users", users_df)
    print("Data user berhasil dibuat dan disimpan di Google Sheets!")

# Jalankan hanya sekali
create_users_from_ratings()
