import psycopg2
import os
import nltk
import pandas as pd
import numpy as np
import re

import streamlit as st
import streamlit_authenticator as stauth

from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict, OrderedDict

import plotly.express as px

import time
import base64

from bcrypt import hashpw, gensalt, checkpw
from dotenv import load_dotenv

def gif_load(filename):
    file_ = open(f"data/img/{filename}.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url
data_loc=gif_load('icon')
st.set_page_config(page_title="restaurant recomendation", page_icon=f"data:image/gif;base64,{data_loc}")

@st.cache_resource
def download_stopwords():
    nltk.download('stopwords')
    return stopwords.words('indonesian')
stopwords_list = download_stopwords()

porter = PorterStemmer()

# Memuat dataset

# Load environment variables
load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

def connect_db():
    return psycopg2.connect(SUPABASE_DB_URL)

def load_data(table_name):
    conn = connect_db()
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_restaurant_id(nama_restoran):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT restaurant_id FROM places_to_eat WHERE nama_restoran = %s;", (nama_restoran,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None

def load_users():
    return load_data("users")

def check_user_exists(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT EXISTS (SELECT 1 FROM users WHERE user_id = %s);", (user_id,))
    exists = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return exists

ratingss = load_data("ratings")
places_to_eat = load_data("places_to_eat")

restaurant=places_to_eat.copy()
rating=ratingss.copy()
ratings = rating.dropna(subset=['restaurant_id'])
ratings = rating.dropna(subset=['rating'])
# Konversi user_id menjadi string 
rating['user_id'] = rating['user_id'].astype(str)
ratings['user_id'] = ratings['user_id'].astype(str)
ratings['rating'] = ratings['rating'].astype(int)


def clean_text(text):
    text = text.lower() # lowercase text
    text = re.sub('\W+',' ', text) # hapus special characters
    text = re.sub('\d+',' ', text) # hapus number
    text = ' '.join(OrderedDict((w,w) for w in text.split()).keys())
    words = text.split()                  # Tokenization
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2] # stopwords
    clean_words=[porter.stem(word) for word in clean_words]                        # Stemming dengan Porter stemmer
    return " ".join(clean_words)

# Preprocessing Content-Based Filtering
tfidf_vectorizer = TfidfVectorizer()
places_to_eat['Features'] = places_to_eat['preferensi_makanan'] + " " + places_to_eat['jenis_suasana'] + " " + places_to_eat['variasi_makanan'] #pengambilan fitur
places_to_eat['prepro_Features']=places_to_eat['Features'].apply(clean_text) #implementasi text preprocessing
tfidf_matrix = tfidf_vectorizer.fit_transform(places_to_eat['prepro_Features']) #implementasi tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)



# Preprocessing Item-Based Collaborative Filtering
pivot_tables = ratings.pivot_table(index='user_id', columns='restaurant_id', values='rating').fillna(0)
pivot_table = ratings.pivot_table(index='user_id', columns='nama_restoran', values='rating').fillna(0)

# Menyiapkan data untuk Surprise
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(ratings[['user_id', 'restaurant_id', 'rating']], reader)


trainset, testset = train_test_split(data, test_size=0.20,random_state=42)

# Menggunakan KNNBasic untuk item-based collaborative filtering
sim_options = {
    'name': 'cosine',
    'user_based': False  # item-based
}

algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

# Menyiapkan data popularitas restoran
average_ratings = rating.groupby(['restaurant_id', 'nama_restoran']).agg({'rating': 'mean'}).reset_index()
average_ratings.columns = ['restaurant_id','nama_restoran', 'Average Rating']
average_ratings['Average Rating'] = average_ratings['Average Rating'].round().clip(1, 5)
average_ratings = average_ratings.sort_values(by='Average Rating', ascending=False)

@st.cache_data
def matrix_variasi():
    # Split 'variasi_makanan' into individual food types
    exploded_foods = places_to_eat[['restaurant_id', 'variasi_makanan']].copy()
    exploded_foods['variasi_makanan'] = exploded_foods['variasi_makanan'].str.split(', ')
    exploded_foods = exploded_foods.explode('variasi_makanan')

    # Use get_dummies to create a one-hot encoding of the food variations
    matrix_foods = pd.get_dummies(exploded_foods['variasi_makanan'])
    matrix_foods['restaurant_id'] = exploded_foods['restaurant_id']

    # Group by 'restaurant_id' and sum to combine the one-hot encodings for each restaurant
    binary_matrix = matrix_foods.groupby('restaurant_id').sum()
    return binary_matrix

@st.cache_data
def content_prepro(restaurant):#bagian data preprocessing pada content based filtering
    #menghapus kolom yang tidak di gunakan
    st.write('Tampilan dataset Restaurant')
    st.dataframe(restaurant)
    restaurant.drop(columns=["harga_rata_rata","keramaian_restoran",'disajikan_atau_ambil_sendiri','all_you_can_eat_atau_ala_carte'],axis=1,inplace=True)
    st.write('menghapus beberapa kolom yang tidak di gunakan seperti "harga_rata_rata","keramaian_restoran","disajikan_atau_ambil_sendiri","all_you_can_eat_atau_ala_carte",sehingga tamilan dataset restaurant sebagai berikut')
    st.dataframe(restaurant)

    restaurant['Features'] = restaurant['preferensi_makanan'] + " " + restaurant['jenis_suasana'] + " " + restaurant['variasi_makanan']

    st.write('Tampilan fitur yang di gunakan pada metode content based filtering')
    restaurant_features=restaurant[['preferensi_makanan','jenis_suasana','variasi_makanan','Features']]
    st.dataframe(restaurant_features)

    restaurant['prepro_Features']=restaurant['Features'].apply(clean_text)
    # Transformasi content_preprocessing menggunakan TF-IDF
    tfidf = tfidf_vectorizer.fit_transform(restaurant['Features'])

    # Mendapatkan daftar kata yang digunakan dalam TF-IDF
    feature_names = tfidf_vectorizer.get_feature_names_out()
    # Membuat DataFrame kosong untuk menyimpan nilai TF-IDF
    tfidf_df = pd.DataFrame(columns=['TF-IDF'])

    # Mengisi DataFrame dengan nilai TF-IDF yang tidak nol
    for i, doc in enumerate(restaurant['prepro_Features']):
        doc_tfidf = tfidf[i]
        non_zero_indices = doc_tfidf.nonzero()[1]
        tfidf_values = doc_tfidf[0, non_zero_indices].toarray()[0]
        tfidf_dict = {feature_names[idx]: tfidf_values[j] for j, idx in enumerate(non_zero_indices)}
        tfidf_df.loc[i] = [' '.join(f'({feature_name}, {tfidf_dict[feature_name]:.3f})' for feature_name in tfidf_dict)]
    cosine_sim = cosine_similarity(tfidf)
    # Menggabungkan DataFrame hasil dengan DataFrame utama
    restaurant = pd.concat([restaurant, tfidf_df], axis=1)
    st.write('Tampilan hasil text preprocessing dan pembobotan TF-IDF')
    text_prepro_tfidf=restaurant[['Features','prepro_Features','TF-IDF']]
    st.dataframe(text_prepro_tfidf)

    # Konversi matriks similarity ke array NumPy
    item_similarity_matrix_array = np.array(cosine_sim)

    datamatriks=pd.DataFrame(item_similarity_matrix_array)
    matrix=datamatriks.head(100)

    # Tampilkan matriks similarity
    st.write("Tampilan Matriks Similarity Antar Item:")
    st.dataframe(matrix)

    #save data to csv
    # text_prepro_tfidf.to_csv("data/prepro/item-user-CF.csv",index=False)
    # matrix.to_csv("data/prepro/similarity-CBF.csv",index=False)
    # restaurant_features.to_csv("data/prepro/restaurant-feat-CBF.csv",index=False)

@st.cache_data
def collab_prepro(rating): 
    #bagian data preprocessing pada collaborative filtering
    st.write('Tampilan dataset rating ')
    ratings = pd.DataFrame(rating)
    st.dataframe(ratings)
    st.write('Tampilan item user matrix')
    st.dataframe(pivot_tables.T)
    # pivot_tables.T.to_csv("data/prepro/item-user-CF.csv",index=False)
    # Dapatkan matriks similarity antar item dari model
    item_similarity_matrix = algo.sim
    # Konversi matriks similarity ke array NumPy
    item_similarity_matrix_array = np.array(item_similarity_matrix)
    datamatriks=pd.DataFrame(item_similarity_matrix_array)
    # Ganti nilai None dengan 0
    datamatriks = datamatriks.fillna(0)
    matrix=datamatriks.head(100)
    # matrix.to_csv("data/prepro/similarity-CF.csv",index=False)
    # Tampilkan matriks similarity
    st.write("Tampilan Matriks Similarity Antar Item:")
    st.dataframe(matrix)

def get_user_rated_restaurants(user_id,num_recommendations): #bagian menampilkanr restoran yg sudah di rating user
    if user_id not in pivot_table.index:
        user_new = pd.DataFrame(columns=['restaurant_id',"nama_restoran", "Rating Anda"])
        return user_new
    restoran_user_tertentu = rating[rating['user_id'] == user_id]
    df_user_rated = restoran_user_tertentu[['restaurant_id', 'nama_restoran', 'rating']][:num_recommendations]
    return df_user_rated

def content_based_filtering(restaurant_name, num_recommendations):
    idx = places_to_eat.index[places_to_eat['nama_restoran'] == restaurant_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    restaurant_indices = [i[0] for i in sim_scores]
    return restaurant_indices

def get_recommendations_content_based(restaurant_name, num_recommendations):
    restaurant_indices=content_based_filtering(restaurant_name, num_recommendations)
    recommendations_content=[(places_to_eat['restaurant_id'].iloc[i],places_to_eat['nama_restoran'].iloc[i], places_to_eat['rating_toko'].iloc[i]) for i in restaurant_indices]
    df_recommendations_content = pd.DataFrame(recommendations_content, columns=["restaurant_id","nama_restoran", "Rating"])
    st.write(f"Rekomendasi restaurant yang serupa {restaurant_name}:")
    return df_recommendations_content

def item_based_collaborative_filtering(user_id,num_recommendations):
    
    if user_id not in pivot_table.index:
        # Return popular restaurants if user_id not found
        popular_restaurants = average_ratings.head(num_recommendations)
        
        pop_rest=list(zip(popular_restaurants['nama_restoran'], popular_restaurants['Average Rating']))
        popular_rest = pd.DataFrame(pop_rest, columns=["nama_restoran", "rating"])
        return popular_rest
    # Dapatkan item yang belum dirating oleh user_id
    rated_restaurants = ratings[ratings['user_id'] == user_id]['restaurant_id'].unique()
    all_restaurants = ratings['restaurant_id'].unique()
    unrated_restaurants = set(all_restaurants) - set(rated_restaurants)
    # Prediksi rating untuk item-item yang belum dirating oleh user_id
    predictions = [algo.predict(user_id, restaurant) for restaurant in unrated_restaurants]
    # Urutkan prediksi berdasarkan nilai prediksi (rating)
    predictions.sort(key=lambda x: x.est, reverse=True)
    # Ambil sejumlah num_recommendations teratas
    top_predictions = predictions[:num_recommendations]
    # Susun hasil rekomendasi ke dalam DataFrame
    recommendations = []
    for pred in top_predictions:
        restaurant_name = rating.loc[rating['restaurant_id'] == pred.iid, 'nama_restoran'].values[0]
        recommendations.append({'restaurant_id':pred.iid,'nama_restoran': restaurant_name, 'rating': pred.est})
    recommendations=pd.DataFrame(recommendations)    
    recommendations['rating'] = recommendations['rating'].round().clip(1, 5)
    return recommendations

def get_recommendations_item_based(user_id, num_recommendations):

    result_name = rating[rating['user_id'] == str(user_id)]['nama'].unique()
    
    if user_id not in pivot_table.index:
        st.subheader(f"Sepertinya {result_name} User baru berikut restaurant paling populer")
        rest_rect=item_based_collaborative_filtering(user_id,num_recommendations)
        return rest_rect
    recommendations=item_based_collaborative_filtering(user_id,num_recommendations)
    st.write(recommendations)
    df_recommendations_item = pd.DataFrame(recommendations, columns=['restaurant_id',"nama_restoran", "rating"])
    
    st.subheader(f"Rekomendasi restaurant untuk {result_name[0]}")
    return df_recommendations_item

def get_recommendations_hybrid(user_id,num_recommendations):    

    if user_id not in pivot_table.index:
        restaurant_name = average_ratings['nama_restoran'].values[0]
    else:
        restaurant_name = ratings.loc[ratings['user_id'] == user_id].sort_values('rating', ascending=False)['nama_restoran'].values[0]
    
    num_recommendations=num_recommendations//2
   
    content_rec=content_based_filtering(restaurant_name,num_recommendations)
    collab_rec=item_based_collaborative_filtering(user_id,num_recommendations)
    
    recommendations_content = [(places_to_eat['restaurant_id'].iloc[i],places_to_eat['nama_restoran'].iloc[i], places_to_eat['rating_toko'].iloc[i], 'Content-based') for i in content_rec]
    df_recommendations_content = pd.DataFrame(recommendations_content, columns=['restaurant_id',"nama_restoran", "rating", "Metode"])
    
    recommendations_item = [(rest['restaurant_id'],rest['nama_restoran'], rest['rating'], 'Collaborative') for idx, rest in collab_rec.iterrows()]
    df_recommendations_item = pd.DataFrame(recommendations_item, columns=['restaurant_id',"nama_restoran", "rating", "Metode"])
    combined_recommendations = pd.concat([df_recommendations_content, df_recommendations_item]).drop_duplicates().reset_index(drop=True)

    return combined_recommendations

def intra_list_similarity(recommendations, feature_matrix):
    """Menghitung ILS tanpa recmetrics"""
    if len(recommendations) < 2:
        return 0  # Jika hanya 1 rekomendasi, ILS tidak relevan

    vectors = feature_matrix.loc[recommendations].values
    similarity_matrix = cosine_similarity(vectors)
    avg_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean()
    
    return avg_similarity

def evaluate_ils(user, num_recommendations, places_to_eat):
    """Evaluasi Intra-List Similarity (ILS) untuk rekomendasi"""
    ils_results = []
    

    # Menghitung matrix fitur dari "variasi_makanan"
    exploded_foods = places_to_eat[['restaurant_id', 'variasi_makanan']].copy()
    exploded_foods['variasi_makanan'] = exploded_foods['variasi_makanan'].str.split(', ')
    exploded_foods = exploded_foods.explode('variasi_makanan')

    # One-hot encoding kategori makanan
    matrix_foods = pd.get_dummies(exploded_foods['variasi_makanan'])
    matrix_foods['restaurant_id'] = exploded_foods['restaurant_id']

    # Gabungkan encoding per restoran
    binary_matrix = matrix_foods.groupby('restaurant_id').sum()

    for num_rec in num_recommendations:
        row_cbf = {'Metode': 'CBF', 'Num_Rec': num_rec}
        row_hybrid = {'Metode': 'Hybrid', 'Num_Rec': num_rec}
        
        for username in user:
            user_id = rating[rating['nama'] == str(username)]['user_id'].unique()

            recommendations = get_recommendations_hybrid(str(user_id[0]), num_rec)
            
            # Content-Based Recommendations
            content_restaurants = recommendations[recommendations['Metode'] == 'Content-based']['restaurant_id'].tolist()
            content_ils = intra_list_similarity(content_restaurants, binary_matrix)
            row_cbf[f'User_{username}'] = content_ils            

            # Hybrid Recommendations
            hybrid_restaurants = recommendations['restaurant_id'].tolist()
            hybrid_ils = intra_list_similarity(hybrid_restaurants, binary_matrix)
            row_hybrid[f'User_{username}'] = hybrid_ils
        
        # Hitung rata-rata ILS
        row_cbf['Average'] = np.mean([row_cbf[f'User_{username}'] for username in user])
        row_hybrid['Average'] = np.mean([row_hybrid[f'User_{username}'] for username in user])

        # Simpan hasil
        ils_results.append(row_cbf)
        ils_results.append(row_hybrid)

    # Simpan hasil evaluasi ke CSV
    ils_results_df = pd.DataFrame(ils_results)
    # ils_results_df.to_csv("data/evaluation/eval_ils_new.csv", index=False)

    return ils_results_df

def plot_ils(ils_results_df):
    # Baca data dari CSV
    
    ils_results_df['Average'] = ils_results_df['Average'].apply(lambda x: f'{x:.4f}')
    # Plot
    st.write('Tampilan Grafik Evaluasi Model ILS pada Hybrid Filtering dan Content Based Filtering ')
    fig = px.line(ils_results_df, x='Num_Rec', y='Average', color='Metode', markers=True,text='Average',
              labels={'Num_Rec': 'Jumlah Rekomendasi', 'Average': 'Rata-rata ILS'},
              title='Perbandingan Rata-rata ILS')
    fig.update_traces(textposition='top center')

    # Tampilkan plot di Streamlit
    st.plotly_chart(fig)


    ILSrelevan=ils_results_df.loc[(ils_results_df['Average']==ils_results_df['Average'].max())]
    ILSvariant=ils_results_df.loc[(ils_results_df['Average']==ils_results_df['Average'].min())]
    
    st.write(f"Dari gambar dan tabel diatas kita dapat melihat hasil pengujian ILS pada metode hybrid filtering dan content based filering pada {ils_results_df.columns.values[2:5]} dimana di dapatkan rata-rata nilai ILS dengan nilai IlS terendah di dapat pada metode {ILSvariant['Metode'].values} dengan jumlah rekomendasi sebanyak {ILSvariant['Num_Rec'].values} data rekomendasi dengan nilai rata-rata sebesar {ILSvariant['Average'].values[0]} dan nilai IlS tertinggi di dapat pada metode {ILSrelevan['Metode'].values} dengan jumlah rekomendasi sebanyak {ILSrelevan['Num_Rec'].values} dengan nilai rata-rata sebesar {ILSrelevan['Average'].values[0]}")

    with st.expander('kesimpulan'):
        st.write(f"kenaikan dan penurunan nilai rata-rata ILS menunjukan nilai keberagaman hasil rekomendasi yang di berikan suatu model , nilai ILS terendah didapatkan pada model {ILSvariant['Metode'].values} dengan jumlah item rekomendasi sebesar {ILSvariant['Num_Rec'].values} item rekomendasi dan  rata-rata nilai ILS sebesar {ILSvariant['Average'].values} yang manandakan variasi/keberagaman yang tinggi dalam hasil rekomendasi yang diberikan, dan sebaliknya nilai ILS tertinggi didapatkan pada model {ILSrelevan['Metode'].values} dengan jumlah item rekomendasi sebesar {ILSrelevan['Num_Rec'].values} item rekomendasi dan  rata-rata nilai ILS sebesar {ILSrelevan['Average'].values} yang manandakan relevansi/kemiripan yang tinggi dalam hasil rekomendasi yang diberikan.")

def MAE_evaluation(k_values):
    # -----silahkan unkomen untuk melakuakn nilai MAE yang baru -----

    # DataFrame untuk menyimpan hasil pengujian
    results_df = pd.DataFrame(columns=['k', 'MAE'])

    # Loop untuk menguji model dengan nilai k yang berbeda
    for nilai_k in k_values:
        # Menggunakan KNNBasic untuk item-based collaborative filtering
        sim_options = {
            'name': 'cosine',
            'user_based': False  # item-based
        }
        # Inisialisasi model KNN dengan nilai k yang diuji
        model = KNNBasic(k=nilai_k, sim_options=sim_options,verbose=False)

        # Latih model pada data pelatihan
        model.fit(trainset)

        # Buat prediksi pada data uji
        predictions = model.test(testset)

        # Hitung Mean Absolute Error (MAE) untuk evaluasi
        mae = accuracy.mae(predictions)
    
        # Tambahkan hasil pengujian ke DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({'k': [nilai_k], 'MAE': [mae]})], ignore_index=True)
    # results_df.to_csv("data/evaluation/eval_k_new.csv",index=False)
    # results_df = pd.read_csv("data/evaluation/eval_k_new.csv")
    return results_df

def plot_MAE(result_MAE):
    st.write("MAE adalah salah satu metode evaluasi yang umum digunakan dalam data science. MAE menghitung rata-rata dari selisih absolut antara nilai prediksi dan nilai aktual.Dengan kata lain, MAE menghitung berapa rata-rata kesalahan absolut dalam prediksi. Semakin kecil nilai MAE, semakin baik kualitas model tersebut.")
    st.write('Tampilan Grafik model evaluasi MAE Item-Based Collabborative filtering pada tabel di atas dapat')
    result_MAE['MAE'] = result_MAE['MAE'].apply(lambda x: f'{x:.4f}')
    Kbest=result_MAE.loc[(result_MAE['MAE']==result_MAE['MAE'].min())]
    Kbad=result_MAE.loc[(result_MAE['MAE']==result_MAE['MAE'].max())]

    # Buat plot menggunakan plotly.express
    fig = px.line(result_MAE, x='k', y='MAE', title='Grafik MAE terhadap K', markers=True,text='MAE')
    fig.update_traces(textposition='top center')

    # Tampilkan plot di Streamlit
    st.plotly_chart(fig)

    st.write(f"Dari gambar dan tabel diatas kita dapat melihat pengujian nilai K dengan hasil MAE dan didapatkan nilai MAE paling terendah pada pengujian ke {Kbest.index.values + 1} dengan nilai k sebesar {Kbest['k'].values[0]} dan nilai MAE sebesar {Kbest['MAE'].values[0]} sedangkan nilai k tertinggi di dapatkan pada pengujian ke {Kbad.index.values + 1} dengan nilai k sebesar {Kbad['k'].values[0]} dan nilai MAE sebesar {Kbad['MAE'].values[0]} ,dapat di simpulkan semakin kecil nilai MAE semakain baik pula model machine learning yang buat")

def register_user(nama, password):
    if check_user_exists(user_id):
        return False
    hashed_password = hashpw(password.encode(), gensalt()).decode()
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (user_id, nama, password, role) VALUES (gen_random_uuid(), %s, %s, 'user')
    """, (nama, hashed_password))
    conn.commit()
    cursor.close()
    conn.close()
    return True

def register():
    
    with st.form("register"):
        st.subheader("Registrasi Pengguna Baru")
        new_user_name = st.text_input("Username Baru")
        new_password = st.text_input("Password Baru", type='password')
        confirm_password = st.text_input("Konfirmasi Password", type='password')
        
        if st.form_submit_button("Registrasi"):
            register_status=register_user(new_user_name, new_password)
            if register_status:
                if new_password == confirm_password:
                    st.success("Registrasi berhasil! Silakan login.")
                else:
                    st.error("Password dan konfirmasi password tidak cocok.")
            else:
                st.error("Username sudah terdaftar. Silakan gunakan username lain.")

def login():
    users_df = load_users()
    if users_df.empty:
        st.error("Tidak dapat memuat data pengguna dari database!")
        return None, None, None, None
    credentials = {"usernames": {}}

    for _, row in users_df.iterrows():
        credentials["usernames"][row["nama"]] = {
            "user_id": row["user_id"],
            "name": row["nama"],
            "password": row["password"],
            "role": row["role"]
        }
    st.write("Data Credentials:", credentials["usernames"]) 
    global authenticator
    authenticator = stauth.Authenticate(
        credentials,
        "auth_cookie",
        "random_key",
        cookie_expiry_days=30
    )

    name,authentication_status, username = authenticator.login(fields={"Username": "Username"})

    if authentication_status:
        user_data = credentials["usernames"].get(username)

        if not user_data:
            st.error(f"User {username} tidak ditemukan dalam database!")
            st.stop()
        return authentication_status,credentials["usernames"][username]["user_id"] ,credentials["usernames"][username]["name"], credentials["usernames"][username]["role"]
    elif authentication_status is False:
        st.error("Username atau password salah.")
    elif authentication_status is None:
        st.warning("Silakan masukkan Username dan password Anda.")
    return None,None,None, None

# @st.experimental_fragment
def add_ratings(baris, i, j, menu):
    with st.popover('Rating'):
        with st.form(key=f"rating_form_{i}_{j}", border=False):
            user_rating = st.slider(
                f"Beri rating untuk {baris[2]}",
                min_value=1, max_value=5, value=3,
                key=f"rating_slider_{i}_{j}"
            )
            submitted = st.form_submit_button(f"Simpan rating")
            
            if submitted:
                if f'ratings_to_save_{menu}' not in st.session_state:
                    st.session_state[f'ratings_to_save_{menu}'] = []
                existing_index = next(
                    (index for index, (saved_row, _) in enumerate(st.session_state[f'ratings_to_save_{menu}'])
                     if saved_row == baris),
                    None
                )
                
                if existing_index is not None:
                    st.session_state[f'ratings_to_save_{menu}'][existing_index] = (baris, user_rating)
                    st.toast(f"Rating {user_rating} diperbarui untuk {baris[2]}!", icon='✅')
                else:
                    st.session_state[f'ratings_to_save_{menu}'].append((baris, user_rating))
                    st.toast(f"Rating {user_rating} untuk {baris[2]}!", icon='✅')
                time.sleep(2)
                st.rerun()
        
        if f'ratings_to_save_{menu}' in st.session_state:
            for index, (saved_row, saved_rating) in enumerate(st.session_state[f'ratings_to_save_{menu}']):
                if saved_row == baris:
                    if st.button(f"Hapus rating {saved_rating}", key=f"delete_button_{i}_{j}"):
                        st.session_state[f'ratings_to_save_{menu}'].pop(index)
                        st.toast(f"Rating {saved_rating} untuk {saved_row[2]} dihapus!", icon='❌')
                        time.sleep(2)
                        st.rerun()
                    break

def save_ratings(user_id,username, menu):
    if f'ratings_to_save_{menu}' in st.session_state and st.session_state[f'ratings_to_save_{menu}']:
        conn = connect_db()
        cursor = conn.cursor()
        for baris, user_rating in st.session_state[f'ratings_to_save_{menu}']:
            restaurant_id = get_restaurant_id(baris[2])  # Ambil UUID dari places_to_eat
            if restaurant_id:
                cursor.execute("""
                    INSERT INTO ratings (user_id,nama , restaurant_id, nama_restoran, rating)
                    VALUES (%s::uuid,%s , %s::uuid, %s, %s);
                """, (user_id,username, restaurant_id, baris[2], user_rating))
            else:
                st.warning(f"Restoran {baris[2]} tidak ditemukan di database.")
                continue
        conn.commit()
        cursor.close()
        conn.close()

        st.toast("Semua rating telah disimpan!", icon='✅')
        time.sleep(2)
        st.rerun()
    else:
        st.toast("Tidak ada rating untuk disimpan.", icon='⚠️')

def delete_rating(menu):
    st.session_state[f'ratings_to_save_{menu}'] = []
    st.toast("Semua rating telah dihapus!", icon='❌')
    time.sleep(2)
    st.rerun()

def get_ratings_id(username,nama_restoran):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM ratings WHERE nama = %s AND nama_restoran = %s;", (username,nama_restoran))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None

def delete_user_rating(username, restaurant_name):
    # Cari indeks rating yang cocok di DataFrame
    id = get_ratings_id(username,restaurant_name)
    
    if id :
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
                "DELETE FROM ratings WHERE nama = %s AND nama_restoran = %s;"
            , (username, restaurant_name))
        conn.commit()
        cursor.close()
        conn.close()
        st.toast(f"Rating untuk {restaurant_name} berhasil dihapus!", icon='❌')
    else:
        st.toast(f"Tidak ada rating untuk {restaurant_name} yang ditemukan.", icon='⚠️')

@st.dialog("Konfirmasi Penghapusan")
def confirm_delete(restaurant_name):
    st.write(f"Apakah Anda yakin ingin menghapus rating untuk {restaurant_name}?")
    col1, col2 = st.columns(2)
    is_hold=False
    with col1:
        if st.button("Ya, hapus", key="confirm_yes"):
            is_hold='ya'
    with col2:
        if st.button("Tidak", key="confirm_no"):
            is_hold='tidak'
    if is_hold==False:
        st.session_state.pop('confirm_delete')
    elif is_hold=='ya':
        delete_user_rating(username, restaurant_name)
        st.rerun()
    else:
        st.rerun()

def restaurant_data():
    # Mengambil nama_restoran yang sudah di-rating oleh user tertentu
    rated_restaurants = rating[rating['user_id'] == username]['nama_restoran'].unique()
    # Mengambil data restoran yang belum di-rating oleh user tersebut
    unrated_restaurants = places_to_eat[~places_to_eat['nama_restoran'].isin(rated_restaurants)]
    unrated_restaurants=unrated_restaurants[['restaurant_id',"nama_restoran", "rating_toko"]]
    pop_rest=list(zip(average_ratings['nama_restoran'], average_ratings['Average Rating']))
    popular_rest = pd.DataFrame(pop_rest, columns=["nama_restoran", "Rating"])

    all_restaurant = places_to_eat['nama_restoran'].values.tolist()
    return all_restaurant,popular_rest,unrated_restaurants

# Streamlit UI
st.write(f"<h4 style='text-align: center ;font-family: Arial, Helvetica, sans-serif;font-size: 34px;word-spacing: 2px;color: #000000;font-weight: 700;' >Sistem Rekomendasi Restoran di Jogja </h4>",unsafe_allow_html=True)
# Authentication

authentication_status,user_id,username, role = login()


if authentication_status:
    if  role == "admin":
        with st.sidebar :
            st.subheader(f'Selamat Datang ADMIN 👋')
            authenticator.logout('Logout', 'sidebar', key='admin')
            admin_menu=option_menu('Sistem Rekomendasi',['data preprocessing', 'prediksi', 'evaluasi'])
        if admin_menu=='data preprocessing':
            st.subheader('Content Based Filtering')
            content_prepro(restaurant)
            # st.write('tampilan dataset merge ')
            # st.dataframe)
            st.subheader('Item-Based Collaborative Filtering')
            collab_prepro(rating)
            
        if admin_menu=='prediksi':
            tab1, tab2 ,tab3= st.tabs(["Content-Based Filtering", "Item-Based Collaborative Filtering","hybrid filtering"])

            with tab1:
                st.header("Content-Based Filtering")
                selected_restaurant = st.selectbox('Pilih restoran yang Anda sukai:', places_to_eat['nama_restoran'])
                number_recommendation=st.number_input('Masukan jumlah data yang akan direkomendasi',value=10,min_value=1,max_value=10,key='content based filtering')
                if st.button('rekomendasi',key='content based'):
                    recommendations_content = get_recommendations_content_based(selected_restaurant,number_recommendation)
                    st.dataframe(recommendations_content,use_container_width=True)

            with tab2:
                st.header("Item-Based Collaborative Filtering")
                user = st.selectbox('Pilih user id:', rating['nama'].unique(),key='colaborative filtering')
                user_id = rating[rating['nama'] == str(user)]['user_id'].unique()
                
                number_recommendation=st.number_input('Masukan jumlah data yang akan direkomendasi',value=10,min_value=1,max_value=10,key='item-based collaboorative filtering')
                if st.button('rekomendasi'):
                    
                    user_rated_restaurants = get_user_rated_restaurants(str(user_id[0]),number_recommendation)
                    if user_rated_restaurants.size > 0:
                        st.subheader(f'Restauran yang sudah dirating oleh {user}')
                    st.dataframe(user_rated_restaurants,use_container_width=True)
                    recommendations_item = get_recommendations_item_based(str(user_id[0]),number_recommendation)
                    st.dataframe(recommendations_item,use_container_width=True)
            with tab3:
                st.header("Hybrid Filtering")
                
                user = st.selectbox('Pilih user id:', rating['nama'].unique(),key='hybrid filtering user select')
                user_id = rating[rating['nama'] == str(user)]['user_id'].unique()
                number_recommendation=st.number_input('Masukan jumlah data yang akan direkomendasi',value=10,min_value=2,max_value=20,key='hybrid filtering')
                recommendation_hybrid=get_recommendations_hybrid(str(user_id[0]),number_recommendation)
                
                if st.button('rekomendasi',key='hybrid'):
                    
                   if str(user_id) in pivot_table.index or str(user_id) not in pivot_table.index:
                        recommendations_item = get_recommendations_item_based(str(user_id),int(number_recommendation)) 
                                      
                        for row in recommendations_item.itertuples():
                            with st.expander(f"nama_restoran {row[2]} dengan rating {row[3]}"):
                                # Rekomendasi content-based berdasarkan restoran yang dipilih dari item-based
                                selected_from_item = row[2]
                                content_based_recommendations = get_recommendations_content_based(selected_from_item,number_recommendation)
                                
                                for rows in content_based_recommendations.itertuples():
                                    st.write(f"{rows[2]} - Rating: {rows[2]}")

        if admin_menu=='evaluasi':
            st.subheader('evaluasi model MAE pada item-based collaborative filtering')
            angka_list = list(range(1, 26))
            k_value =  st.multiselect("Pilih hingga 5 angka:", angka_list,default=[3, 5, 7,9,10], max_selections=5)

            eval_mae=MAE_evaluation(k_value)
            st.dataframe(eval_mae)
            plot_MAE(eval_mae)

            st.subheader('evaluasi model ILS pada hybrid fitering filtering dan content based filtering')
            st.write('Intra-list similarity (ILS) adalah metrik evaluasi yang digunakan dalam sistem rekomendasi untuk mengukur kesamaan antara item-item yang direkomendasikan dalam daftar rekomendasi yang diberikan kepada pengguna.semakin tinggi nilai ILS maka semakin mirip daftar item rekomendasi yang diberikan dan semakin rendah nilai ils maka semakin beragan daftar item rekomendasi yang diberikan')

            st.write('pengujian ILS yang dilakukan menggunakan "variasi_makanan" yang dimiliki pada setiap item sebagai parameter kemiripan pada hasil daftar rekomendasi yang diberikan berikut merupakan tampilan matriks variasi_makanan dari semua restoran berdasarkan restaurant_id')
            
            selected_users = st.multiselect("Pilih 3 User untuk Uji ILS:", rating["nama"].unique(),default=["Affan", "Afrien", "Alfian"], max_selections=3)
            num_recommendations_list =  st.multiselect("Pilih hingga 3 angka:", angka_list,default=[3, 8, 10], max_selections=3)
            matrix_var=matrix_variasi()
            st.write(matrix_var)
            eva_ils=evaluate_ils(selected_users, num_recommendations_list,places_to_eat)
            st.write(f'berikut merupakan hasil pengujian ILS pada user {user_id}')
            st.dataframe(eva_ils)
            plot_ils(eva_ils)
    else:

        with st.sidebar :
            st.write(f'Selamat Datang {str(username)} 👋')
            authenticator.logout('Logout', 'main', key='user')
            user_menu=option_menu('user menu',['Restaurant','Rekomendation', 'User Rated'])
        
        if user_menu=='Restaurant':
            all_restaurant,popular_rest,unrated_restaurants=restaurant_data()
            
            col1,col2 = st.columns([0.7, 0.3])
            with col1:
                option_restaurant = st.selectbox('Opsi pencarian', ["Restoran yang belum diberi rating"]+["Restoran paling populer"]+["Cari nama_restoran"]+all_restaurant,key='search all restaurant')
                # Tambahkan input untuk pencarian
                if option_restaurant=="Cari nama_restoran":
                    restaurant_df= pd.DataFrame()
                    search_query = st.text_input("Cari nama_restoran", "")
                else:
                    search_query=False
            with col2:
                if 'ratings_to_save_restaurant' in st.session_state and st.session_state['ratings_to_save_restaurant']:
                    st.write(" ")
                    cols1,cols2 = st.columns(2)
                    with cols1 :
                        if st.button('save all ratings'):
                            save_ratings(user_id, username)
                    with cols2 :
                        if st.button('delete all ratings'):
                            delete_rating('restaurant')

            # Filter hasil berdasarkan pencarian
            if search_query:
                restaurant_df= unrated_restaurants[unrated_restaurants['nama_restoran'].str.contains(search_query, case=False)]
            elif option_restaurant not in ["Restoran yang belum diberi rating", "Restoran paling populer","Cari nama_restoran"]:
                restaurant_df = unrated_restaurants[unrated_restaurants['nama_restoran'] == option_restaurant]
           
            if option_restaurant=="Restoran yang belum diberi rating":
                # st.subheader(f"Restoran yang belum {str(result_name[0])} beri rating")
                for i, row in enumerate(unrated_restaurants.itertuples()):
                        cols1,cols2 = st.columns([0.8, 0.2])
                        with cols1:
                            st.info(f" {row[2]} - Rating: {row[3]}",icon="🍽️")
                        with cols2:
                            add_ratings(row, i, None,"restaurant")
            elif option_restaurant=="Restoran paling populer":
                for i, row in enumerate(popular_rest.itertuples()):
                        cols1,cols2 = st.columns([0.8, 0.2])
                        with cols1:
                            st.info(f" {row[2]} - Rating: {row[3]}",icon="🍽️")
                        with cols2:
                            add_ratings(row, i, None,"restaurant")
            else:
                if search_query=="":
                    st.warning("Silahkan masukan nama_restoran",icon='🖊️')
                elif not restaurant_df.empty:
                    for i, row in enumerate(restaurant_df.itertuples()):
                            cols1,cols2 = st.columns([0.8, 0.2])
                            with cols1:
                                st.info(f" {row[2]} - Rating: {row[3]}",icon="🍽️")
                            with cols2:
                                add_ratings(row, i, None,"restaurant")
                else:
                    st.error("Maaf, restoran yang Anda cari tidak ditemukan. Silakan periksa pencarian anda dan coba lagi",icon='😔')
            
        if user_menu=='Rekomendation':
            # Membuat daftar restoran
            restaurant_names = places_to_eat['nama_restoran'].values.tolist()
            # Menambahkan pilihan 'None' di awal daftar
            data_loc=gif_load('restaurant')

            st.markdown(f'<img src="data:image/gif;base64,{data_loc}" alt="restaurant gif" style="width:704px; height:300px">',unsafe_allow_html=True,)
            col1,col2 = st.columns([0.7, 0.3])
            with col1:
                selected_restaurant = st.selectbox('Pilih restoran yang Anda sukai:', ["Rekomendasi untuk kamu"]+restaurant_names,key='serach on rekomendation')
            with col2:
                if 'ratings_to_save_rekomendasi' in st.session_state and st.session_state['ratings_to_save_rekomendasi']:
                    st.write(" ")
                    cols1,cols2 = st.columns(2)
                    with cols1 :
                        if st.button('save all ratings'):
                            save_ratings(user_id, username,"rekomendasi")
                    with cols2 :
                        if st.button('delete all ratings'):
                            delete_rating("rekomendasi")
            
            if selected_restaurant == 'Rekomendasi untuk kamu':
                recommendations_item = get_recommendations_item_based(user_id,10)                
                for i, row in enumerate(recommendations_item.itertuples()):
                    cols1,cols2 = st.columns([0.8, 0.2])
                    with cols1:
                        with st.expander(f":knife_fork_plate: nama_restoran {row[2]} dengan rating {row[3]}"):
                            # Rekomendasi content-based berdasarkan restoran yang dipilih dari item-based
                            selected_from_item = row[2]
                            content_based_recommendations = get_recommendations_content_based(selected_from_item, 10)

                            for j, rows in enumerate(content_based_recommendations.itertuples()):
                                col1,col2 = st.columns([0.8, 0.2])
                                with col1:
                                    st.info(f" {rows[2]} - Rating: {rows[3]}",icon="🍽️")
                                with col2:
                                    add_ratings(rows, i, j,"rekomendasi")  
                    with cols2:
                        add_ratings(row, i, None,"rekomendasi")
            else:
                recommendations_content = get_recommendations_content_based(selected_restaurant,10)
                for i,row in enumerate(recommendations_content.itertuples()):
                    cols1,cols2 = st.columns([0.8, 0.2])
                    with cols1:
                        with st.expander(f":knife_fork_plate: nama_restoran {row[2]} dengan rating {row[3]}"):
                            # Rekomendasi content-based berdasarkan restoran yang dipilih dari item-based
                            selected_from_item = row[2]
                            content_based_recommendations = get_recommendations_content_based(selected_from_item,10)
                            
                            for j,rows in enumerate(content_based_recommendations.itertuples()):
                                col1,col2 = st.columns([0.8, 0.2])
                                with col1:
                                    st.info(f" {rows[2]} - Rating: {rows[3]}",icon="🍽️")
                                with col2:
                                    add_ratings(rows, i, j,"rekomendasi")
                    with cols2:
                        add_ratings(row, i, None,"rekomendasi")
        
        if user_menu =='User Rated':
            user_ratings =rating[rating['user_id'] == user_id]
            rated_restaurants = user_ratings['nama_restoran'].values.tolist()
            user_rated = get_user_rated_restaurants(user_id,len(user_ratings))
            if user_id not in pivot_table.index:
                st.subheader("belum ada data Restoran yang anda rating")
            else:
                st.subheader(f"Restoran yang sudah pernah {username}  beri rating:")
                cols1,cols2 = st.columns([0.7, 0.3])
                with cols1:
                    select_restaurant = st.selectbox('Opsi pencarian:', ["Semua Restoran yang anda rating"]+["Cari nama_restoran"]+rated_restaurants,key='search user rated')
                    # Tambahkan input untuk pencarian
                    if select_restaurant=="Cari nama_restoran":
                        user_rated_df=pd.DataFrame()
                        search_query_rated = st.text_input("Cari nama_restoran","")
                    else:
                        search_query_rated=False
                with cols2:
                    if 'ratings_to_save_user_rated' in st.session_state and st.session_state['ratings_to_save_user_rated']:
                        st.write(" ")
                        cols1,cols2 = st.columns(2)
                        with cols1 :
                            if st.button('save all ratings'):
                                save_ratings(user_id, username,"user_rated")
                        with cols2 :
                            if st.button('delete all ratings'):
                                delete_rating("user_rated")
            
                if search_query_rated:
                    user_rated_df= user_rated[user_rated['nama_restoran'].str.contains(search_query_rated, case=False)]
                # Filter hasil berdasarkan pencarian
                elif select_restaurant not in ["Semua Restoran yang anda rating","Cari nama_restoran"]:
                    user_rated_df = user_rated[user_rated['nama_restoran'] == select_restaurant]
                # st.write('ini searc',search_query_rated)
                if select_restaurant =="Semua Restoran yang anda rating":
                    for i,row in enumerate(user_rated.itertuples()):
                        with st.expander(f" :knife_fork_plate: nama_restoran {row[2]} rating {row[3]}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Tombol untuk memicu konfirmasi penghapusan
                                    if st.button(f"Hapus data rating", key=f"delete_button_{row[2]}"):
                                        # Set session state untuk memicu pop-up konfirmasi
                                        st.session_state['confirm_delete'] = row[2]
                                        confirm_delete(row[2])
                                        # st.write(f"session deleted {st.session_state['confirm_delete']}")
                                with col2:
                                    add_ratings(row, i, None,"user_rated")
                else:                
                        if search_query_rated=="":
                            st.warning("Silahkan masukan nama_restoran",icon='🖊️')
                        elif not user_rated_df.empty:
                            for i,row in enumerate(user_rated_df.itertuples()):
                                with st.expander(f" :knife_fork_plate: nama_restoran {row[2]} rating {row[3]}"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        # Tombol untuk memicu konfirmasi penghapusan
                                        if st.button(f"Hapus data rating", key=f"delete_button_{row[2]}"):
                                            # Set session state untuk memicu pop-up konfirmasi
                                            st.session_state['confirm_delete'] = row[2]
                                            confirm_delete(row[2])
                                            # st.write(f"session deleted {st.session_state['confirm_delete']}")
                                    with col2:
                                        add_ratings(row, i, None,"user_rated")
                        else:
                            st.error("Maaf, restoran yang Anda cari tidak ditemukan. Silakan periksa pencarian anda dan coba lagi",icon='😔')

        # Tampilkan pop-up konfirmasi jika tombol hapus ditekan
        if 'confirm_delete' in st.session_state :
            confirm_delete(st.session_state['confirm_delete'])

else:
    
    on = st.toggle("belum memiliki akun?")
    if on:
        register()

    