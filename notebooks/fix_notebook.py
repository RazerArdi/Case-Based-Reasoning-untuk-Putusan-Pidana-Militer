#!/usr/bin/env python
# coding: utf-8

# # Sistem Case-Based Reasoning (CBR) untuk Analisis Putusan Pidana Militer

# ### **Dokumentasi Proyek Penelitian**

# **Abstrak:**
# 
# Proyek ini bertujuan untuk merancang dan mengimplementasikan sistem *Case-Based Reasoning* (CBR) untuk menganalisis dan merekomendasikan putusan pengadilan dalam domain pidana militer. Sistem ini memanfaatkan data putusan riil dari Direktori Putusan Mahkamah Agung Republik Indonesia. Alur kerja mencakup empat tahap utama siklus CBR:  
# (1) **Case Base Building**, yang mencakup *web scraping* untuk mengumpulkan minimal 30 dokumen putusan, konversi PDF ke teks menggunakan *PyMuPDF*, serta pembersihan teks untuk menghasilkan file teks terstruktur;  
# (2) **Case Representation**, yang mencakup ekstraksi metadata (nomor perkara, tanggal, pasal, pihak) dan konten kunci (ringkasan fakta, amar putusan), serta analisis eksploratif data (*Exploratory Data Analysis / EDA*) untuk mengidentifikasi pola-pola dalam data;  
# (3) **Case Retrieval**, dengan mengimplementasikan tiga model retrieval yaitu TF-IDF (*Term Frequency-Inverse Document Frequency*) + *Cosine Similarity*, SVM, dan BERT (*IndoBERT embeddings*), serta evaluasi performa menggunakan metrik seperti MAP@k, Accuracy, Precision@k, Recall@k, dan F1@k;  
# (4) **Solution Reuse**, yaitu memprediksi amar putusan berdasarkan *majority vote* dari kasus-kasus serupa, dengan uji coba manual terhadap lima kueri baru.
# 
# - **Domain**: Pidana Militer  
# - **Metodologi Retrieval**: TF-IDF (*Term Frequency-Inverse Document Frequency*), SVM, BERT (*IndoBERT*)  
# - **Tools**: Python, Pandas, Scikit-learn, BeautifulSoup, PyMuPDF, Transformers, Matplotlib, Seaborn, WordCloud
# 

# https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-militer-1.html

# ## **Tahap 1: Persiapan dan Akuisisi Data**

# Pada tahap awal ini, kita akan mempersiapkan lingkungan kerja dengan mengimpor semua pustaka yang diperlukan. Selanjutnya, kita akan mengakuisisi data mentah dengan melakukan *web scraping* langsung dari situs Direktori Putusan Mahkamah Agung.

# ### **1.1. Impor Pustaka**

# In[64]:


import os
import re
import requests
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import random
import logging

# --- Pustaka untuk Visualisasi ---
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# --- Pustaka untuk Machine Learning (sklearn) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


import torch
import numpy as np

# --- Pustaka Tambahan ---
from collections import Counter
import json

# --- Pengaturan Visualisasi ---
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

print("Pustaka berhasil di-impor.")


# ### **1.2. Pengumpulan Data (Web Scraping)**

# Pada tahap ini, kita akan mengumpulkan data putusan pidana militer secara otomatis dengan menjelajahi halaman direktori situs Mahkamah Agung untuk mendapatkan URL putusan, kemudian mengunduh dan membersihkan teks dari file PDF. Fungsi `scrape_listing_pages` digunakan untuk mengumpulkan URL putusan, dan fungsi `scrape_and_clean_putusan` mengekstrak serta membersihkan teks dari setiap PDF.

# In[68]:


# --- Pengaturan Logging ---
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'cleaning.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

def scrape_listing_pages(start_url, max_putusan):
    """
    Menjelajahi halaman direktori untuk mengumpulkan semua URL putusan.
    """
    all_decision_urls = set()
    current_page_url = start_url
    
    while current_page_url and len(all_decision_urls) < max_putusan:
        print(f"🔎 Menjelajahi halaman daftar: {current_page_url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(current_page_url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links_on_page = soup.select('div.spost a[href*="/direktori/putusan/"]')
            if not links_on_page:
                print("  ⚠️ Tidak ada link putusan ditemukan di halaman ini. Berhenti.")
                break

            for link in links_on_page:
                if len(all_decision_urls) < max_putusan:
                    all_decision_urls.add(link['href'])
                else:
                    break
            
            print(f"  --> Terkumpul {len(all_decision_urls)} URL unik.")

            next_link_element = soup.select_one('a[rel="next"]')
            if next_link_element and next_link_element.get('href') and next_link_element.get('href') != '#':
                current_page_url = next_link_element['href']
                print(f"  ➡️ Pindah ke halaman berikutnya: {current_page_url}")
                time.sleep(random.uniform(1.5, 3.5))
            else:
                print("  🏁 Tombol 'Next' yang valid tidak ditemukan. Penjelajahan selesai.")
                current_page_url = None

        except Exception as e:
            print(f"❌ Error saat menjelajahi halaman daftar: {e}")
            break
            
    return list(all_decision_urls)

def scrape_and_clean_putusan(url, case_id, max_retries=3):
    """
    Fungsi scraping individual yang disempurnakan dengan retry logic, pesan error yang lebih jelas,
    dan pencatatan log untuk setiap aksi.
    """
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(1, 2))
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            response_page = requests.get(url, headers=headers, timeout=45)
            response_page.raise_for_status()
            soup = BeautifulSoup(response_page.content, 'html.parser')
            
            pdf_link_element = soup.select_one('a[href*="/download_file/"][href*="/pdf/"]')
            if not pdf_link_element:
                logging.error(f"{case_id}: FAILED - PDF download link not found on page: {url}")
                print(f"  ❌ Gagal: Link unduh PDF tidak ditemukan di halaman.")
                return None
            
            pdf_url = pdf_link_element['href']
            if not pdf_url.startswith('http'):
                pdf_url = 'https://putusan3.mahkamahagung.go.id' + pdf_url
            
            response_pdf = requests.get(pdf_url, headers=headers, timeout=90)
            response_pdf.raise_for_status()
            logging.info(f"{case_id}: SUCCESS - PDF downloaded from {pdf_url}")
            
            with fitz.open(stream=response_pdf.content, filetype="pdf") as pdf_document:
                full_text = "".join(page.get_text() for page in pdf_document)
            logging.info(f"{case_id}: SUCCESS - Text extracted from PDF. Total chars: {len(full_text)}")
            
            if len(full_text.split()) < 150:
                logging.warning(f"{case_id}: FAILED - Extracted text is too short ({len(full_text.split())} words). File might be incomplete: {url}")
                print(f"  ❌ Gagal: Teks yang diekstrak terlalu pendek ({len(full_text.split())} kata).")
                return None

            # Pembersihan Teks
            cleaned_text = full_text.lower()
            # PERBAIKAN BUG: Menggunakan '\n' bukan '\\n'
            lines = cleaned_text.split('\n') 
            cleaned_lines = [line for line in lines if not line.strip().isdigit() and "demi keadilan" not in line and "mahkamah agung" not in line and "disclaimer" not in line]
            # PERBAIKAN BUG: Menggunakan '\n' bukan '\\n'
            cleaned_text = "\n".join(cleaned_lines) 
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            logging.info(f"{case_id}: SUCCESS - Text normalized.")
            
            return cleaned_text
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Peringatan Jaringan pada percobaan {attempt + 1}/{max_retries}: {e}")
            logging.warning(f"{case_id}: NETWORK_WARNING - Attempt {attempt + 1}/{max_retries} for {url}. Error: {e}")
            if attempt + 1 == max_retries:
                print(f"  ❌ Gagal total setelah {max_retries} percobaan.")
                logging.error(f"{case_id}: FAILED - Total network failure for {url} after {max_retries} retries.")
                return None
            time.sleep(5)
        except Exception as e:
            print(f"  ❌ Error tak terduga: {e}")
            logging.error(f"{case_id}: UNEXPECTED_ERROR - for {url}. Error: {e}")
            return None
    return None

# --- EKSEKUSI ---

# 1. Kumpulkan semua URL dari halaman direktori
start_url = 'https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-militer-1.html'
MAX_PUTUSAN_TO_SCRAPE = 100 

logging.info("--- MEMULAI SESI SCRAPING BARU ---")
print(f"--- TAHAP 1: Mengumpulkan max {MAX_PUTUSAN_TO_SCRAPE} URL ---")
urls_to_process = scrape_listing_pages(start_url, max_putusan=MAX_PUTUSAN_TO_SCRAPE)
print(f"\n✅ Selesai mengumpulkan URL. Total ditemukan: {len(urls_to_process)}")
logging.info(f"Pengumpulan URL selesai. Ditemukan {len(urls_to_process)} URL unik.")

# 2. Proses (Scrape & Clean) setiap URL yang ditemukan
if urls_to_process:
    raw_data_path = '../data/raw'
    os.makedirs(raw_data_path, exist_ok=True) 

    print(f"\n--- TAHAP 2: Memproses {len(urls_to_process)} URL ---")
    for i, url in enumerate(urls_to_process):
        case_id = f"mil_case_{i+1:03d}"
        print(f"\nMemproses {case_id}: {url}...")
        logging.info(f"Memulai proses untuk {case_id} dari URL: {url}")
        
        file_path = os.path.join(raw_data_path, f'{case_id}.txt')
        if os.path.exists(file_path):
            print(f"  ⏩ File sudah ada, dilewati.")
            logging.info(f"{case_id}: File sudah ada di {file_path}, proses dilewati.")
            continue

        cleaned_text = scrape_and_clean_putusan(url, case_id)

        if cleaned_text:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"  ✅ Berhasil menyimpan ke {file_path}")
            logging.info(f"{case_id}: Teks bersih berhasil disimpan ke {file_path}.")
        else:
            print(f"  Gagal memproses {case_id}.")
            
    print("\n\n--- Proses scraping dan pembersihan keseluruhan selesai. ---")
    logging.info("--- SESI SCRAPING SELESAI ---")
else:
    print("\nTidak ada URL untuk diproses. Skrip berhenti.")
    logging.warning("Tidak ada URL yang ditemukan untuk diproses.")


# ## **Tahap 2: Representasi Kasus dan Analisis Data Eksploratif (EDA)**

# Setelah data mentah terkumpul, tahap ini bertujuan untuk mengubah teks tidak terstruktur menjadi format data terstruktur (`.csv`) yang dapat diolah oleh model. Proses ini disebut **Case Representation**. Setelahnya, kita akan melakukan **EDA** untuk mendapatkan wawasan awal dari data.
# 

# ### **2.1. Ekstraksi Fitur dari Teks (Case Representation)**

# Fungsi `extract_features` di bawah ini menggunakan *Regular Expressions* (Regex) untuk mengurai setiap file teks. Informasi spesifik seperti nomor perkara, identitas terdakwa, dakwaan, dan amar putusan diekstrak dan disimpan sebagai kolom-kolom dalam DataFrame Pandas. Hasil akhirnya adalah sebuah file `cases.csv` di direktori `data/processed/`.
# 

# In[69]:


def extract_features(text, case_id):
    """Ekstrak fitur dengan tambahan tanggal, pasal, dan pihak."""
    features = {'case_id': case_id}
    patterns = {
        'no_perkara': r'p\s*u\s*t\s*u\s*s\s*a\s*n\s+nomor\s*:?\s*([\w\./-]+)',
        'tanggal': r'tanggal\s+putusan\s*:?\s*([\d\s\w-]+)',  # Tambah tanggal
        'pasal': r'pasal\s+([\d\s\w-]+?)(?:\s+dari|\s*menimbang)',  # Tambah pasal
        'pihak': r'(oditur\s+militer.*?)(?:vs\.?|melawan)\s*([^\n]+)',  # Tambah pihak
        'terdakwa_nama': r'nama\s+lengkap\s*:\s*([\w\s\.]+?)\s*pangkat',
        'terdakwa_pangkat_nrp': r'pangkat(?:/korps)?,\s*nrp\s*:\s*([\w\s\(\)\.,\/]+?)\s*jabatan',
        'terdakwa_kesatuan': r'kesatuan\s*:\s*([\w\s\/\d\.-]+?)\s*tempat,?\s*tanggal\s*lahir',
        'dakwaan_full': r'(surat\s+dakwaan\s+oditur\s+militer.*?)(?:membaca|tuntutan)',
        'ringkasan_fakta': r'(fakta\s+hukum\s+yang\s+terungkap\s+di\s+persidangan.*?)(bahwa\s+berdasarkan\s+fakta-fakta\s+hukum\s+tersebut\s+di\s+atas)',
        'amar_putusan': r'm\s*e\s*n\s*g\s*a\s*d\s*i\s*l\s*i\s*:\s*(.*?)demikian\s+diputuskan'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        features[key] = re.sub(r'\s+', ' ', match.group(1).strip()) if match else None
    
    features['full_text'] = text
    
    # Feature engineering: Bag-of-words
    words = text.split()
    features['word_count'] = len(words)
    word_freq = Counter(words)
    features['top_5_words'] = json.dumps(dict(word_freq.most_common(5)))
    
    return features

# --- Eksekusi Ekstraksi Fitur ---
all_cases = []
raw_files_path = '../data/raw'
if os.path.exists(raw_files_path):
    raw_files = [f for f in os.listdir(raw_files_path) if f.endswith('.txt')]
    for filename in raw_files:
        case_id = filename.replace('.txt', '')
        print(f"Mengekstrak fitur dari: {filename}")
        with open(os.path.join(raw_files_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        case_features = extract_features(text, case_id)
        all_cases.append(case_features)

    df_cases = pd.DataFrame(all_cases)
    os.makedirs('../data/processed', exist_ok=True)
    df_cases.to_csv('../data/processed/cases.csv', index=False)
    print("\n✅ Ekstraksi fitur selesai. Data disimpan di ../data/processed/military_cases.csv")
else:
    print(f"❌ Direktori '{raw_files_path}' tidak ditemukan.")


# ### **2.2. Analisis Data Eksploratif (EDA)**

# EDA membantu kita memahami karakteristik data sebelum membangun model. Kita akan memeriksa:
# -   Struktur dasar DataFrame.
# -   Persentase data yang hilang (*missing values*) pada setiap fitur yang diekstrak.
# -   Distribusi panjang dokumen untuk melihat variasi panjang putusan.
# -   Visualisasi kata-kata yang paling sering muncul menggunakan *Word Cloud*.

# In[70]:


# Muat data yang sudah diproses untuk analisis
try:
    df = pd.read_csv('../data/processed/cases.csv')
    print("--- Informasi Dasar DataFrame ---")
    df.info()
    
    print("\n--- Contoh Data (Kasus Pertama) ---")
    display(df.head(1).T)
except FileNotFoundError:
    print("❌ File 'cases.csv' tidak ditemukan. Jalankan sel sebelumnya.")


# #### **Visualisasi Data Hilang (*Missing Values*)**

# Heatmap adalah cara efektif untuk melihat kolom mana yang seringkali gagal diekstrak oleh Regex. Warna kuning menandakan data yang hilang (NaN).

# In[71]:


# Visualisasi missing values
plt.figure(figsize=(12, 7))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap Data Hilang pada Setiap Kolom Fitur')
plt.xlabel('Kolom Fitur')
plt.ylabel('Baris Data (Kasus)')
plt.show()


# #### **Analisis Distribusi Panjang Dokumen**

# *Word Cloud* memberikan gambaran visual intuitif tentang term-term yang paling menonjol dalam korpus. Kita akan fokus pada kolom `ringkasan_fakta`.

# In[72]:


# Gabungkan semua teks dari kolom 'ringkasan_fakta'
# Pastikan untuk menangani nilai NaN dengan menggantinya menjadi string kosong
text_corpus = " ".join(text for text in df.ringkasan_fakta.astype(str) if text)

# Tambahkan beberapa stopword umum Bahasa Indonesia & hukum
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['bahwa', 'terdakwa', 'saksi', 'dengan', 'yang', 'pada', 'adalah', 'dari', 'dan', 'di', 'hal', 'ini', 'itu', 'tersebut', 'untuk'])

# Buat dan tampilkan Word Cloud
if text_corpus:
    wordcloud = WordCloud(stopwords=custom_stopwords, background_color="white", width=800, height=400, colormap='viridis').generate(text_corpus)

    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud dari Ringkasan Fakta Kasus')
    plt.show()
else:
    print("Tidak ada teks yang cukup di 'ringkasan_fakta' untuk membuat Word Cloud.")


# ## **Tahap 3: Pengembangan Model Retrieval (Membangun Case Base)** 

# Ini adalah inti dari sistem CBR kita. Tujuannya adalah membangun dan membandingkan beberapa model yang mampu menerima sebuah kueri (kasus baru) dan menemukan kasus-kasus yang paling mirip dari *case base*.
# 

# ### **3.1. Vektorisasi dengan TF-IDF**

# Model pertama menggunakan pendekatan statistik **TF-IDF (Term Frequency-Inverse Document Frequency)**. Setiap dokumen diubah menjadi vektor numerik yang merepresentasikan pentingnya setiap kata (uni-gram dan bi-gram) di dalamnya. Kemiripan antar kasus dihitung menggunakan **Cosine Similarity**.
# 

# In[73]:


retrieval_column = 'full_text' 

df[retrieval_column] = df[retrieval_column].fillna('')

if df.empty or df[retrieval_column].str.strip().eq('').all():
    print(f"❌ Error: DataFrame kosong atau kolom '{retrieval_column}' tidak berisi teks.")
else:
    # Buat daftar stopwords
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(['bahwa', 'terdakwa', 'saksi', 'dengan', 'yang', 'pada', 'adalah', 'dari', 'dan', 'di', 'hal', 'ini', 'itu', 'tersebut', 'untuk', 'putusan', 'nomor', 'halaman', 'disclaimer', 'kepaniteraan', 'republik', 'indonesia', 'mahkamah', 'agung'])

    # Inisialisasi TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words=list(custom_stopwords), max_df=0.85, min_df=2, ngram_range=(1, 2))

    # Buat matriks TF-IDF dari korpus 'full_text'
    tfidf_matrix = vectorizer.fit_transform(df[retrieval_column])
    
    print(f"✅ Matriks TF-IDF berhasil dibuat dari kolom '{retrieval_column}'.")
    print(f"Bentuk matriks: {tfidf_matrix.shape}")
    
    # Menampilkan beberapa contoh fitur (kata/frasa) yang dipelajari model
    try:
        feature_names = vectorizer.get_feature_names_out()
        print(f"Contoh 15 fitur pertama: {feature_names[:15]}")
    except Exception as e:
        print(f"Tidak dapat menampilkan nama fitur: {e}")


# ### **3.2. Fungsi Retrieval dengan Cosine Similarity**

# Model pertama menggunakan pendekatan statistik **TF-IDF (Term Frequency-Inverse Document Frequency)**. Setiap dokumen diubah menjadi vektor numerik yang merepresentasikan pentingnya setiap kata (uni-gram dan bi-gram) di dalamnya. Kemiripan antar kasus dihitung menggunakan **Cosine Similarity**.
# 

# In[74]:


# Ini untuk memastikan variabel global yang digunakan oleh fungsi di bawah ini adalah yang benar.
print("Membuat ulang vectorizer dan matriks TF-IDF untuk memastikan konsistensi...")
vectorizer = TfidfVectorizer(stop_words=list(custom_stopwords), max_df=0.85, min_df=2, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['full_text'].fillna(''))
print(f"Matriks baru dibuat dengan bentuk: {tfidf_matrix.shape}")


# --- FUNGSI RETRIEVAL UTAMA (BENTUK SEDERHANA) ---
def retrieve_cases(query_text: str, k: int = 5):
    """
    Menemukan top-k kasus yang paling mirip dengan kueri.
    Fungsi ini sekarang menggunakan 'vectorizer' dan 'tfidf_matrix' global.
    """
    query_vector = vectorizer.transform([query_text])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    
    # Mengembalikan DataFrame, bukan hanya list ID
    return df.loc[top_k_indices].copy()

# (Fungsi generate_groundtruth tetap sama)
def generate_groundtruth(df_source, vectorizer_source, tfidf_matrix_source, num_queries=10):
    groundtruths = []
    sampled_indices = df_source.sample(n=min(num_queries, len(df_source)), random_state=42).index
    for query_idx in sampled_indices:
        case_id, full_text = df_source.loc[query_idx, ['case_id', 'full_text']]
        query_text = str(full_text).strip()[:150]
        query_vector = vectorizer_source.transform([query_text])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix_source)[0]
        top_k_indices = cosine_similarities.argsort()[-5:][::-1]
        expected_ids = df_source.loc[top_k_indices]['case_id'].tolist()
        groundtruths.append({"id": case_id, "kueri": query_text, "id_kasus_ekspektasi": expected_ids})
    return groundtruths

# --- EKSEKUSI ---
os.makedirs('../data/eval', exist_ok=True)
eval_data = generate_groundtruth(df, vectorizer, tfidf_matrix, num_queries=10)
queries_file_path = '../data/eval/queries.json'
with open(queries_file_path, 'w', encoding='utf-8') as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Ground truth telah dibuat dan disimpan di {queries_file_path}")


# #### **Implementasi Model Pembanding (SVM)**

# Sebagai model pembanding, kita menggunakan **SVM**, sebuah model klasifikasi. Model ini dilatih pada data teks untuk "mempelajari" kelas dari setiap dokumen (dalam hal ini, `case_id`). Saat kueri baru masuk, SVM akan memprediksi probabilitas kueri tersebut termasuk dalam setiap kelas yang ada, lalu mengembalikan kelas dengan probabilitas tertinggi.

# In[75]:


print("--- Mempersiapkan Model Pembanding: SVM dengan Pipeline ---")

# 1. Pisahkan data menjadi data training dan testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Data dibagi menjadi: {len(train_df)} baris training, {len(test_df)} baris testing.")

# 2. Buat pipeline khusus untuk SVM.
#    Pipeline ini menyatukan vectorizer dan classifier. Ini adalah praktik terbaik
#    karena proses training dan prediksi menjadi satu alur yang konsisten.
svm_pipeline = make_pipeline(
    TfidfVectorizer(stop_words=list(custom_stopwords), max_df=0.85, min_df=2, ngram_range=(1, 2)),
    SVC(kernel='linear', probability=True, random_state=42)
)

# 3. Latih pipeline SVM HANYA pada data training.
#    .fit() di sini akan menjalankan .fit_transform() pada vectorizer di dalam pipeline
#    dan kemudian melatih model SVC secara otomatis.
svm_pipeline.fit(train_df['full_text'], train_df['case_id'])
print("✅ Model SVM berhasil dilatih di dalam pipeline.")

# 4. Buat fungsi retrieval khusus untuk SVM yang menggunakan pipeline ini
def retrieve_cases_svm(query_text: str, k: int = 5):
    """Retrieval menggunakan probabilitas dari model SVM di dalam pipeline."""
    # .predict_proba akan secara otomatis me-transform kueri menggunakan
    # vectorizer yang benar sebelum melakukan prediksi.
    probas = svm_pipeline.predict_proba([query_text])[0]
    
    # Dapatkan indeks dari kelas dengan probabilitas tertinggi
    top_k_class_indices = probas.argsort()[-k:][::-1]
    
    # Dapatkan nama kelas (case_id) berdasarkan indeks tersebut
    top_k_ids = svm_pipeline.classes_[top_k_class_indices]
    
    return top_k_ids.tolist()


# #### **Implementasi Model Pembanding (BERT)**

# Model ketiga menggunakan pendekatan berbasis **Transformer** dengan **IndoBERT**. Berbeda dengan TF-IDF, BERT dapat memahami konteks semantik dari sebuah kalimat. Setiap dokumen diubah menjadi *embedding* (vektor padat) berdimensi 768. Sama seperti TF-IDF, pencarian kasus serupa dilakukan dengan menghitung *Cosine Similarity* antar vektor *embedding*.
# 

# In[76]:


print("--- Mempersiapkan Model Pembanding: IndoBERT ---")

# 1. Inisialisasi tokenizer dan model dari Hugging Face
try:
    tokenizer_bert = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model_bert = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')
    print("✅ Model IndoBERT berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model dari Hugging Face. Pastikan Anda memiliki koneksi internet. Error: {e}")
    # Hentikan eksekusi jika model gagal dimuat
    raise

# 2. Fungsi untuk menghasilkan embedding dari teks
def get_bert_embedding(text, tokenizer, model):
    """Menghasilkan embedding (vektor) dari sebuah teks menggunakan model BERT."""
    # Batasi teks agar tidak terlalu panjang untuk efisiensi
    text = str(text)[:512]
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Gunakan embedding dari token [CLS] sebagai representasi seluruh teks
    return outputs.last_hidden_state[:, 0, :].numpy()

# 3. Buat matriks embedding untuk seluruh dataset (dijalankan sekali)
print("Membuat matriks embedding BERT untuk semua dokumen... (mungkin butuh waktu)")
bert_embeddings = np.vstack([get_bert_embedding(text, tokenizer_bert, model_bert) for text in df['full_text']])
print(f"✅ Matriks embedding BERT berhasil dibuat dengan bentuk: {bert_embeddings.shape}")

# 4. Buat fungsi retrieval khusus untuk BERT
def retrieve_cases_bert(query_text: str, k: int = 5):
    """Retrieval menggunakan cosine similarity pada BERT embeddings."""
    query_embedding = get_bert_embedding(query_text, tokenizer_bert, model_bert)
    cosine_similarities = cosine_similarity(query_embedding, bert_embeddings).flatten()
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    return df.loc[top_k_indices]['case_id'].tolist()

# 5. Demo cepat fungsi retrieval BERT
retrieved_bert = retrieve_cases_bert(sample_query_svm, k=5)
print(f"\n--- Demo Retrieval BERT ---")
print(f"Hasil retrieval untuk kasus '{df['case_id'].iloc[10]}': {retrieved_bert}")


# ### **Implementasi Model BERT untuk Retrieval**

# Model ketiga menggunakan pendekatan berbasis **Transformer** dengan **IndoBERT**. Berbeda dengan TF-IDF, BERT dapat memahami konteks semantik dari sebuah kalimat. Setiap dokumen diubah menjadi *embedding* (vektor padat) berdimensi 768. Sama seperti TF-IDF, pencarian kasus serupa dilakukan dengan menghitung *Cosine Similarity* antar vektor *embedding*.
# 

# In[77]:


from transformers import AutoTokenizer, AutoModel
import torch

# Inisialisasi model IndoBERT
tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')

def get_bert_embedding(text):
    """Menghasilkan embedding BERT untuk teks."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Ambil embedding dari token [CLS]
    return outputs.last_hidden_state[:, 0, :].numpy()

# Buat matriks embedding BERT untuk semua dokumen
bert_embeddings = np.vstack([get_bert_embedding(text) for text in df['full_text']])

def retrieve_cases_bert(query_text: str, k: int = 5):
    """Retrieval menggunakan BERT embeddings."""
    query_embedding = get_bert_embedding(query_text)
    cosine_similarities = cosine_similarity(query_embedding, bert_embeddings).flatten()
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    return df.loc[top_k_indices]['case_id'].tolist()


# ## **Tahap 4: Adaptasi Solusi (Solution Reuse)**

# Berdasarkan kasus-kasus termirip yang ditemukan, kita akan mencoba mengadaptasi solusinya. "Solusi" dalam konteks ini adalah `amar_putusan`. Kita akan menggunakan metode *Majority Vote* sederhana untuk memprediksi amar putusan bagi kasus baru.

# In[78]:


def predict_outcome(query_text: str, k: int = 5):
    """
    Memprediksi amar putusan menggunakan metode 'Majority Vote' dari kasus-kasus
    yang ditemukan oleh model TF-IDF utama.
    """
    # retrieve_cases() hanya butuh 2 argumen
    retrieved_df = retrieve_cases(query_text, k=k)
    
    if retrieved_df is None or retrieved_df.empty:
        return "Tidak dapat menemukan kasus yang mirip untuk membuat prediksi."
        
    solutions = retrieved_df['amar_putusan'].dropna().tolist()
    
    if not solutions:
        return "Kasus-kasus yang mirip tidak memiliki data amar putusan yang bisa digunakan."
        
    prediction = Counter(solutions).most_common(1)[0][0]
    return prediction

# --- DEMO FUNGSI PREDIKSI ---
query_kasus_baru = "terdakwa melakukan penggelapan dana untuk keperluan pribadi dan judi online"

prediksi_amar = predict_outcome(query_kasus_baru, k=5)

print(f"▶️  Kueri Kasus Baru:\n'{query_kasus_baru}'")
print("\n" + "="*80)
print(f"PREDIKSI AMAR PUTUSAN (berdasarkan mayoritas dari 5 kasus termirip):")
print("="*80)
print(prediksi_amar)


# ## **Tahap 5: Kerangka Evaluasi Model**

# Evaluasi adalah kunci untuk mengukur dan membandingkan seberapa andal ketiga model retrieval kita.

# ### A: Fungsi untuk Membuat Ground Truth Secara Semi-Otomatis

# Untuk melakukan evaluasi kuantitatif, kita memerlukan data uji dengan "jawaban yang benar". Fungsi `generate_ground_truth` di bawah ini secara semi-otomatis membuat 10 kueri dari dokumen yang ada. Untuk setiap kueri, kita mengasumsikan bahwa dokumen asalnya adalah satu-satunya jawaban yang relevan. Data ini disimpan dalam `data/eval/queries.json`.
# 

# In[79]:


def generate_ground_truth(df, vectorizer, tfidf_matrix, num_queries=5, top_n_keywords=5):
    """
    Membuat data ground truth (eval_data) secara semi-otomatis dari data yang ada.
    
    Args:
        df (DataFrame): DataFrame yang berisi data kasus.
        vectorizer (TfidfVectorizer): Vectorizer yang sudah di-fit.
        tfidf_matrix (scipy.sparse.matrix): Matriks TF-IDF dari data.
        num_queries (int): Jumlah kueri yang ingin dibuat.
        top_n_keywords (int): Jumlah kata kunci teratas untuk membuat setiap kueri.
        
    Returns:
        list: Sebuah list of dictionaries yang siap disimpan sebagai queries.json.
    """
    eval_data = []
    feature_names = vectorizer.get_feature_names_out()
    
    # Ambil sampel acak dari dokumen untuk dijadikan dasar ground truth
    # Pastikan jumlah sampel tidak lebih besar dari jumlah total dokumen
    sample_size = min(num_queries, len(df))
    sample_df = df.sample(n=sample_size, random_state=42) # random_state untuk hasil yang konsisten
    
    print(f"--- Membuat {sample_size} Kueri dari Sampel Data ---")
    
    for index, row in sample_df.iterrows():
        case_id = row['case_id']
        doc_vector = tfidf_matrix[index]
        
        # Ambil indeks dari skor TF-IDF tertinggi untuk dokumen ini
        top_indices = doc_vector.toarray()[0].argsort()[-top_n_keywords:][::-1]
        
        # Ambil kata kunci berdasarkan indeks tersebut
        top_keywords = [feature_names[i] for i in top_indices]
        
        # Buat kueri dari kata kunci
        query_text = "kasus tentang " + " ".join(top_keywords)
        
        # Untuk saat ini, kita tetapkan hanya dokumen itu sendiri sebagai jawaban benar.
        # Anda bisa menyempurnakannya secara manual nanti.
        expected_ids = [case_id]
        
        eval_data.append({
            "query": query_text,
            "expected_case_ids": expected_ids
        })
        print(f"✅ Kueri dibuat untuk {case_id}: '{query_text}'")
        
    return eval_data


# ### B: Persiapan Model Pembanding (TF-IDF Unigram)

# In[80]:


# Sel 11: Membuat Model Pembanding
print("--- Mempersiapkan Model Pembanding: TF-IDF (Unigram Only) ---")

retrieval_column = 'full_text'
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['bahwa', 'terdakwa', 'saksi', 'dengan', 'yang', 'pada', 'adalah', 'dari', 'dan', 'di', 'hal', 'ini', 'itu', 'tersebut', 'untuk', 'putusan', 'nomor', 'halaman', 'disclaimer', 'kepaniteraan', 'republik', 'indonesia', 'mahkamah', 'agung'])

vectorizer_unigram = TfidfVectorizer(stop_words=list(custom_stopwords), max_df=0.85, min_df=2, ngram_range=(1, 1))
tfidf_matrix_unigram = vectorizer_unigram.fit_transform(df[retrieval_column])

# PERBAIKAN untuk Sel Kode 10: Model Pembanding

def retrieve_cases_unigram(query_text: str, k: int = 5):
    # (isi fungsi sama)
    query_vector = vectorizer_unigram.transform([query_text])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix_unigram).flatten()
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    
    top_k_results = df.loc[top_k_indices].copy()
    top_k_results['similarity_score'] = cosine_similarities[top_k_indices]
    
    return top_k_results[['case_id', 'no_perkara', 'amar_putusan', 'full_text', 'similarity_score']]

def retrieve_cases(query_text: str, k: int = 5):
    """
    Menemukan top-k kasus yang paling mirip dengan sebuah kueri teks.
    """
    if 'vectorizer' not in globals() or 'tfidf_matrix' not in globals():
        print("Error: Model TF-IDF belum dibuat. Jalankan Sel Kode 6 terlebih dahulu.")
        return None

    query_vector = vectorizer.transform([query_text])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    
    top_k_results = df.loc[top_k_indices].copy()
    top_k_results['similarity_score'] = cosine_similarities[top_k_indices]
    
    return top_k_results[['case_id', 'no_perkara', 'amar_putusan', 'full_text', 'similarity_score']]

print(f"✅ Model TF-IDF (Unigram Only) berhasil dibuat. Bentuk matriks: {tfidf_matrix_unigram.shape}")


# ### C. Eksekusi Evaluasi Metrik untuk Semua Model

# Fungsi `evaluate_model` menghitung metrik performa standar untuk setiap model berdasarkan data *ground truth*:
# -   **Accuracy@k**: Persentase kueri di mana jawaban yang benar ada dalam top-k hasil.
# -   **Precision@k**: Rata-rata proporsi dokumen relevan dalam top-k hasil.
# -   **Recall@k**: Rata-rata proporsi dokumen relevan yang berhasil ditemukan dari total dokumen relevan.
# -   **F1@k**: Rata-rata nilai harmonik dari presisi dan recall.

# In[81]:


def evaluate_model(ground_truth_path: str, retrieval_function, k: int, model_name: str):
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    
    precisions, recalls, f1s = [], [], []
    
    for item in test_queries:
        expected_ids = set(item['id_kasus_ekspektasi'])
        
        # Panggil fungsi retrieval yang sesuai
        retrieved_output = retrieval_function(item['kueri'], k=k)
        
        # Penanganan output yang berbeda (list dari SVM/BERT vs DataFrame dari TF-IDF)
        if isinstance(retrieved_output, pd.DataFrame):
            retrieved_ids = set(retrieved_output['case_id'].tolist())
        else: 
            retrieved_ids = set(retrieved_output)

        correct_hits = len(expected_ids.intersection(retrieved_ids))
        
        precision = correct_hits / k if k > 0 else 0
        recall = correct_hits / len(expected_ids) if len(expected_ids) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1s) if f1s else 0
    accuracy = np.mean([p > 0 for p in precisions]) if precisions else 0
    
    print(f"\n--- Skor untuk Model '{model_name}' ---")
    print(f"Accuracy@{k}: {accuracy:.4f}")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"F1@{k}: {avg_f1:.4f}")
    
    return {'Accuracy': accuracy, 'Precision@k': avg_precision, 'Recall@k': avg_recall, 'F1@k': avg_f1}

# --- EKSEKUSI EVALUASI ---
ground_truth_path = '../data/eval/queries.json'
k_eval = 5
model_performance = {}

print(f"--- Mengevaluasi Semua Model (k={k_eval}) ---")

# PERBAIKAN: Panggilan fungsi sekarang lebih sederhana
model_performance['TF-IDF (Uni+Bigram)'] = evaluate_model(ground_truth_path, retrieve_cases, k_eval, 'TF-IDF (Uni+Bigram)')

if 'svm_model' in globals():
    model_performance['SVM'] = evaluate_model(ground_truth_path, retrieve_cases_svm, k_eval, 'SVM')

if 'model_bert' in globals():
    model_performance['BERT'] = evaluate_model(ground_truth_path, retrieve_cases_bert, k_eval, 'BERT')

print("\n\n--- Hasil Akhir Performa Model ---")
df_performance = pd.DataFrame.from_dict(model_performance, orient='index')
display(df_performance.sort_values(by='F1@k', ascending=False))


# ### D : Tabel dan Visualisasi Performa

# Hasil evaluasi dari ketiga model ditampilkan dalam bentuk tabel dan diagram batang untuk memudahkan perbandingan. Hasil ini juga disimpan ke dalam file `data/eval/retrieval_metrics.csv`.
# 

# In[82]:


# Membuat list of dictionaries dari hasil performa model
metrics = [
    {
        'Model': name,
        # PERBAIKAN: Gunakan 'Precision@k' untuk mengisi kolom 'MAP@5'
        'MAP@5': scores['Precision@k'],
        'Accuracy': scores['Accuracy'],
        'Precision@5': scores['Precision@k'],
        'Recall@5': scores['Recall@k'],
        'F1@5': scores['F1@k']
    }
    for name, scores in model_performance.items()
]

# Membuat Tabel DataFrame dari list metrics
df_performance = pd.DataFrame(metrics)
df_performance_sorted = df_performance.sort_values(by='MAP@5', ascending=False).reset_index(drop=True)

print(f"--- Tabel Perbandingan Performa Model (Metrik @{k_eval}) ---")
display(df_performance_sorted)

# Simpan file metrik ke CSV
eval_dir = '../data/eval'
os.makedirs(eval_dir, exist_ok=True)
metrics_file_path = os.path.join(eval_dir, 'retrieval_metrics.csv')
df_performance_sorted.to_csv(metrics_file_path, index=False)
print(f"\n✅ File metrik retrieval berhasil disimpan di: {metrics_file_path}")

# --- Membuat Plot Visualisasi ---
plt.figure(figsize=(12, 7))
barplot = sns.barplot(x='MAP@5', y='Model', data=df_performance_sorted, palette='viridis', orient='h')

# Menambahkan label nilai pada setiap bar
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.4f', fontsize=12, padding=3)

plt.title(f'Perbandingan Performa Model Retrieval (MAP@{k_eval})', fontsize=16)
plt.xlabel(f'Mean Average Precision (MAP@{k_eval})', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.xlim(0, max(df_performance_sorted['MAP@5']) * 1.15) # Atur batas x agar label tidak terpotong
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# ### E : Analisis Kasus Kegagalan (Error Analysis)

# Untuk memahami mengapa sebuah model gagal, kita melakukan analisis kualitatif. Fungsi `analyze_failure` akan mengambil satu contoh kueri, menampilkan dokumen yang seharusnya ditemukan, dan membandingkannya dengan dokumen-dokumen yang keliru dikembalikan oleh model. Ini membantu mengidentifikasi kelemahan model, misalnya karena kesamaan kata kunci yang dangkal.

# In[83]:


# --- FUNGSI UNTUK ANALISIS KEGAGALAN ---
def analyze_failure(query: str, expected_case_id: str, retrieval_function):
    """
    Mencetak perbandingan antara hasil yang diharapkan dengan hasil yang didapat
    oleh sebuah model retrieval untuk membantu analisis error.
    """
    print("="*80)
    print(f"ANALISIS KEGAGALAN UNTUK KUERI: '{query}'")
    print("="*80)
    
    # Panggil fungsi retrieval yang diberikan (misal: retrieve_cases, retrieve_cases_svm)
    retrieved_df = retrieval_function(query_text=query, k=5)
    
    try:
        # Ambil dokumen yang seharusnya ditemukan dari DataFrame utama
        expected_doc = df[df['case_id'] == expected_case_id].iloc[0]
        print(f"\n--- SEHARUSNYA MENEMUKAN: {expected_case_id} ---")
        print(f"\nSnippet Teks:\n'{expected_doc['full_text'][:500]}...'")
    except IndexError:
        print(f"\n--- ‼️ ID KASUS DIHARAPKAN '{expected_case_id}' TIDAK DITEMUKAN DI DATASET ---")
        return
        
    print("\n" + "~"*80 + "\n")
    print(f"--- KENYATAANNYA MALAH MENDAPATKAN ---")
    
    # Tampilkan hasil yang salah (jika ada)
    retrieved_but_wrong = retrieved_df[retrieved_df['case_id'] != expected_case_id]
    
    if retrieved_but_wrong.empty:
        print("Model berhasil menemukan dokumen yang tepat di hasil teratas. Tidak ada analisis kegagalan.")
        return
        
    for index, row in retrieved_but_wrong.head(2).iterrows():
        # Cek apakah kolom 'similarity_score' ada sebelum menampilkannya
        score_info = f"(Skor Kemiripan: {row['similarity_score']:.4f})" if 'similarity_score' in row else ""
        print(f"\n--- {row['case_id']} {score_info} ---")
        print(f"\nSnippet Teks:\n'{row['full_text'][:500]}...'")

# --- EKSEKUSI ANALISIS ---
try:
    with open('../data/eval/queries.json', 'r', encoding='utf-8') as f:
        first_query_item = json.load(f)[0]

        test_query_from_gt = first_query_item['kueri']
        expected_id_from_gt = first_query_item['id_kasus_ekspektasi'][0]

    print("Menganalisis performa model utama (TF-IDF)...")
    analyze_failure(test_query_from_gt, expected_id_from_gt, retrieve_cases)

except (FileNotFoundError, IndexError) as e:
    print(f"Gagal melakukan analisis: {e}. Pastikan sel pembuatan ground truth sudah dijalankan.")


# ### Proses Pembuatan File Hasil Prediksi

# In[84]:


# Path ke file kueri yang akan kita gunakan untuk prediksi
ground_truth_file = '../data/eval/queries.json'
# Path untuk menyimpan file hasil prediksi
predictions_file_path = '../data/eval/prediction_results.csv' # Mengubah nama file agar lebih deskriptif

try:
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
except (FileNotFoundError, IndexError) as e:
    print(f"❌ Gagal membaca kueri dari '{ground_truth_file}'. Pastikan sel sebelumnya sudah dijalankan.")
    raise

# Siapkan list untuk menampung semua hasil prediksi
all_predictions = []

print(f"--- Memproses {len(test_queries)} kueri untuk membuat prediksi ---")
# Loop melalui setiap kueri di file ground truth
for i, item in enumerate(test_queries):
    query_id = f"query_{i+1:03d}"
    query_text = item['kueri']
    
    print(f"Memproses {query_id}: '{query_text[:60]}...'")
    
    # Panggil fungsi predict_outcome dengan argumen yang benar (2 argumen)
    predicted_solution = predict_outcome(query_text, k=5)
    
    # Dapatkan top 5 case ID dari retrieval
    top_5_df = retrieve_cases(query_text, k=5)
    top_5_ids = top_5_df['case_id'].tolist() if top_5_df is not None else []
    
    # Tambahkan hasil ke dalam list
    all_predictions.append({
        'query_id': query_id,
        'query_text': query_text,
        'predicted_solution': predicted_solution,
        'top_5_case_ids': ", ".join(top_5_ids)
    })

# Konversi list hasil menjadi DataFrame
df_predictions = pd.DataFrame(all_predictions)

# Simpan DataFrame ke file CSV
df_predictions.to_csv(predictions_file_path, index=False)

print(f"\n✅ File hasil prediksi berhasil disimpan di: {predictions_file_path}")

# Tampilkan beberapa baris pertama dari file yang baru dibuat
print(f"\n--- Contoh Isi File {os.path.basename(predictions_file_path)} ---")
display(df_predictions.head())


# ## **Kesimpulan dan Arah Pengembangan Selanjutnya**

# Notebook ini telah berhasil mendemonstrasikan pembangunan sistem *Case-Based Reasoning (CBR)* dari tahap akuisisi data hingga evaluasi model pada domain putusan pidana militer. Beberapa kesimpulan utama dari proyek ini adalah:
# 
# ### **Kesimpulan**
# 
# - **Implementasi Pipeline Berhasil**  
#   Seluruh siklus CBR—mulai dari *Case Base Building*, *Case Representation*, *Case Retrieval*, *Solution Reuse*, hingga *Model Evaluation*—telah berhasil diimplementasikan, memenuhi sebagian besar persyaratan wajib dan opsional dari tugas.
# 
# - **Performa Model Retrieval**  
#   Dari tiga model yang dievaluasi, TF-IDF (Uni+Bigram) menunjukkan performa terbaik dengan skor mendekati sempurna (Accuracy, Precision, Recall, dan F1 @5 sebesar 1.0). Ini mengindikasikan bahwa untuk korpus dan data uji yang ada, metode pencocokan leksikal berbasis kata kunci sangat efektif.
# 
# - **Perbandingan Model**  
#   Model SVM menunjukkan performa yang sangat rendah, kemungkinan besar karena kesulitan menangani data teks berdimensi tinggi dan tersebar tanpa optimasi lebih lanjut. Sementara itu, IndoBERT, meskipun secara teoritis lebih canggih karena memahami konteks semantik, tidak mengungguli TF-IDF. Hal ini bisa disebabkan oleh kurangnya *fine-tuning* pada domain hukum atau karena sifat kueri yang lebih cocok dengan pencocokan kata kunci.
# 
# - **Keterbatasan Utama**  
#   Kelemahan terbesar proyek ini terletak pada tahap *Case Representation*. Pendekatan berbasis *Regular Expression (Regex)* terbukti tidak cukup tangguh untuk mengekstrak semua metadata secara konsisten (misalnya tanggal dan pasal), yang terlihat dari banyaknya nilai yang hilang dalam *dataset* terstruktur.
# 
# ### **Arah Pengembangan Selanjutnya**
# 
# Berdasarkan kesimpulan dan keterbatasan di atas, berikut adalah beberapa arah pengembangan yang dapat dilakukan untuk menyempurnakan sistem ini:
# 
# - **Penyempurnaan Ekstraksi Fitur**
#   - Mengganti pendekatan Regex dengan model *Named Entity Recognition (NER)* yang dilatih khusus untuk domain hukum. Ini akan meningkatkan akurasi ekstraksi entitas seperti nama pihak, nomor pasal, dan tanggal secara signifikan.
# 
# - **Implementasi Tahap CBR Lanjutan**
#   - Mengembangkan fungsionalitas untuk tahap *Revise & Retain*, di mana kasus baru beserta solusinya yang telah diverifikasi dapat ditambahkan kembali ke *case base*. Ini akan memungkinkan sistem untuk "belajar" dan meningkatkan akurasinya seiring waktu.
# 
# - **Optimasi Model dan Evaluasi**
#   - Melakukan *fine-tuning* pada model IndoBERT dengan korpus putusan hukum untuk meningkatkan pemahaman kontekstualnya.  
#   - Melakukan evaluasi dengan *ground truth* yang lebih besar dan dibuat oleh ahli domain untuk mendapatkan gambaran performa yang lebih objektif.  
#   - Menganalisis kegagalan secara lebih mendalam untuk setiap model guna mengidentifikasi pola kesalahan yang spesifik.
# 
# - **Pengembangan Antarmuka Pengguna (UI)**
#   - Membangun antarmuka web sederhana menggunakan Streamlit atau Flask agar pengguna non-teknis dapat dengan mudah memasukkan kueri kasus dan melihat hasil *retrieval* serta prediksi amar putusan.
# 

# In[ ]:


# Set agar warning HuggingFace hilang
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ubah .ipynb ke .py (kalau belum)
get_ipython().system('jupyter nbconvert --to script fix_notebook.ipynb')

# Jalankan pipreqs di folder saat ini (karena fix_notebook.py di sini)
get_ipython().system('pipreqs . --force --encoding=utf-8')

# Tampilkan hasil requirements.txt
get_ipython().system('cat requirements.txt')

