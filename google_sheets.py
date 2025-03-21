import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import json
import streamlit as st
from streamlit_authenticator.utilities.hasher import Hasher

# setting fot local
# # Load Credentials dari JSON
# creds = Credentials.from_service_account_file("chrome-setup-454416-j9-96c1d1735f41.json", scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
# client = gspread.authorize(creds)

# setting for streamlit community
# Baca credentials dari Streamlit Secrets
creds_json = st.secrets["GOOGLE_CREDENTIALS"]
creds_dict = json.loads(creds_json)

creds = Credentials.from_service_account_info(creds_dict, scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
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
def load_users():
    """Memuat daftar user dari Google Sheets"""
    return load_data("users")

def register_user(user_id, name, password, role="user"):
    """Mendaftarkan user baru ke Google Sheets dengan password yang di-hash"""
    users_df = load_users()

    # Hash password sebelum menyimpan
    hashed_password = Hasher([password]).generate()[0]

    new_user = pd.DataFrame({
        "user_id": [user_id],
        "nama": [name],
        "password": [hashed_password],
        "role": [role]
    })

    # Tambahkan user baru ke database Google Sheets
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    update_data("users", users_df)
# Contoh Penggunaan
places_to_eat = load_data("places_to_eat")
ratings = load_data("ratings")
