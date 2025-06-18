<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stars][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Sistem Case-Based Reasoning untuk Analisis Putusan Pidana Militer</h3>

<a href="https://putusan3.mahkamahagung.go.id/public/frontend/images/logo.png" title="Lihat Logo Mahkamah Agung">
  <img 
    src="https://putusan3.mahkamahagung.go.id/public/frontend/images/logo.png" 
    alt="Logo Mahkamah Agung" 
    style="display: block;"
  >
</a>


  <p align="center">
    Sistem berbasis Python untuk menganalisis dan merekomendasikan putusan pengadilan pidana militer menggunakan pendekatan Case-Based Reasoning (CBR).
    <br />
    <a href="https://github.com/your_username/repo_name"><strong>Lihat Dokumentasi »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/repo_name">Lihat Demo</a>
    ·
    <a href="https://github.com/your_username/repo_name/issues/new?labels=bug&template=bug-report.md">Laporkan Bug</a>
    ·
    <a href="https://github.com/your_username/repo_name/issues/new?labels=enhancement&template=feature-request.md">Ajukan Fitur</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Daftar Isi</summary>
  <ol>
    <li>
      <a href="#tentang-proyek">Tentang Proyek</a>
      <ul>
        <li><a href="#dibuat-dengan">Dibuat dengan</a></li>
      </ul>
    </li>
    <li>
      <a href="#memulai">Memulai</a>
      <ul>
        <li><a href="#prasyarat">Prasyarat</a></li>
        <li><a href="#instalasi">Instalasi</a></li>
      </ul>
    </li>
    <li><a href="#penggunaan">Penggunaan</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#kontribusi">Kontribusi</a></li>
    <li><a href="#lisensi">Lisensi</a></li>
    <li><a href="#kontak">Kontak</a></li>
    <li><a href="#penghargaan">Penghargaan</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Tentang Proyek

Proyek ini merupakan implementasi sistem *Case-Based Reasoning* (CBR) untuk menganalisis dan merekomendasikan putusan pengadilan dalam domain pidana militer. Sistem ini memanfaatkan data putusan riil dari [Direktori Putusan Mahkamah Agung RI](https://putusan3.mahkamahagung.go.id/). Alur kerja sistem mencakup lima tahap utama:

1. **Membangun Case Base**: Mengunduh (scraping) dan membersihkan minimal 50 dokumen putusan pidana militer.
2. **Case Representation**: Mengekstrak metadata (nomor perkara, terdakwa, dll.) dan fitur teks (ringkasan fakta, amar putusan) ke format CSV.
3. **Case Retrieval**: Menggunakan pendekatan statistik TF-IDF dan *Cosine Similarity* untuk menemukan kasus serupa berdasarkan kueri.
4. **Solution Reuse**: Memprediksi amar putusan untuk kasus baru dengan metode *Majority Vote* dari top-5 kasus termirip.
5. **Evaluasi Model**: Mengukur performa retrieval dengan metrik *Precision@5* menggunakan data *ground truth* semi-otomatis.

Proyek ini dikembangkan sebagai bagian dari tugas mata kuliah Penalaran Komputer di Fakultas Teknik Informatika, Universitas Muhammadiyah Malang.

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

### Dibuat dengan

- ![Python][Python]
- ![Pandas][Pandas]
- ![Scikit-learn][Scikit-learn]
- ![BeautifulSoup][BeautifulSoup]
- ![Matplotlib][Matplotlib]
- ![Seaborn][Seaborn]
- ![WordCloud][WordCloud]

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

<!-- GETTING STARTED -->
## Memulai

Berikut adalah langkah-langkah untuk menyiapkan dan menjalankan proyek ini secara lokal.

### Prasyarat

Pastikan Anda memiliki perangkat lunak berikut:
- Python 3.10 atau lebih tinggi
  ```sh
  python --version
  ```
- pip (Python package manager)
  ```sh
  pip --version
  ```
- Jupyter Notebook
  ```sh
  jupyter notebook --version
  ```

### Instalasi

1. **Kloning Repositori**
   ```sh
   git clone https://github.com/your_username/repo_name.git
   cd repo_name
   ```

2. **Buat dan Aktifkan Virtual Environment (Direkomendasikan)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate     # Windows
   ```

3. **Instal Dependensi**
   ```sh
   pip install -r requirements.txt
   ```

   Contoh isi `requirements.txt`:
   ```
   requests
   beautifulsoup4
   PyMuPDF
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   wordcloud
   ```

4. **Ubah URL Remote Git (Opsional)**
   Untuk menghindari push ke repositori asli secara tidak sengaja:
   ```sh
   git remote set-url origin https://github.com/your_username/your_repo_name.git
   git remote -v  # Verifikasi perubahan
   ```

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

<!-- USAGE -->
## Penggunaan

Proyek ini dijalankan melalui Jupyter Notebook `fix_notebook.ipynb`. Berikut adalah langkah-langkah untuk menjalankan pipeline CBR secara end-to-end:

1. **Jalankan Notebook**
   ```sh
   jupyter lab
   ```
   Buka `fix_notebook.ipynb` di browser.

2. **Ikuti Tahapan dalam Notebook**
   - **Tahap 1**: Mengunduh dan membersihkan data putusan dari situs Mahkamah Agung, disimpan di `data/raw/`.
   - **Tahap 2**: Mengekstrak fitur dan menyimpan data terstruktur di `data/processed/military_cases.csv`.
   - **Tahap 3**: Membangun model retrieval berbasis TF-IDF untuk menemukan kasus serupa.
   - **Tahap 4**: Memprediksi amar putusan untuk kueri baru menggunakan *Majority Vote*.
   - **Tahap 5**: Mengevaluasi performa retrieval dengan *Precision@5* menggunakan data di `data/eval/queries_generated.json`.

3. **Contoh Kueri**
   Uji fungsi retrieval:
   ```python
   query = "seorang prajurit menggunakan dana kesatuan untuk kepentingan pribadi dan bermain judi online"
   hasil_retrieval = retrieve_cases(query, k=3)
   display(hasil_retrieval)
   ```

   Uji prediksi putusan:
   ```python
   query = "perwira keuangan menggelapkan uang untuk judi"
   prediksi_amar = predict_outcome(query, k=5)
   print(prediksi_amar)
   ```

4. **Struktur Direktori**
   ```plaintext
   repo_name/
   ├── data/
   │   ├── raw/                  # Teks mentah (*.txt)
   │   ├── processed/            # Data terstruktur (military_cases.csv)
   │   ├── eval/                 # Data ground truth (queries_generated.json)
   │   └── results/              # Hasil prediksi (predictions.csv)
   ├── notebooks/
   │   └── fix_notebook.ipynb    # Notebook utama
   ├── logs/                     # Log pembersihan (opsional)
   ├── requirements.txt          # Daftar dependensi
   └── README.md                 # File ini
   ```

Untuk detail lebih lanjut, lihat `fix_notebook.ipynb`.

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Membangun *Case Base* dengan *web scraping* (50+ dokumen)
- [x] Representasi kasus dalam format CSV
- [x] Implementasi retrieval berbasis TF-IDF dan *Cosine Similarity*
- [x] Prediksi amar putusan dengan *Majority Vote*
- [x] Evaluasi retrieval dengan *Precision@5* dan *ground truth* semi-otomatis
- [ ] Integrasi model NLP berbasis Transformer (IndoBERT)
- [ ] Menambahkan metrik evaluasi (*Recall@k*, *F1-Score*)
- [ ] Pengembangan antarmuka web dengan Flask/Django
- [ ] Implementasi tahap *Revise & Retain* untuk pembelajaran iteratif

Lihat [issues](https://github.com/your_username/repo_name/issues) untuk daftar fitur yang diusulkan dan masalah yang diketahui.

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

<!-- CONTRIBUTING -->
## Kontribusi

Kontribusi sangat dihargai untuk meningkatkan proyek ini. Jika Anda memiliki saran:

1. Fork repositori
2. Buat branch fitur (`git checkout -b feature/FiturBaru`)
3. Commit perubahan (`git commit -m 'Menambahkan FiturBaru'`)
4. Push ke branch (`git push origin feature/FiturBaru`)
5. Buka Pull Request

### Kontributor Utama
- [Nama Mahasiswa 1]
- [Nama Mahasiswa 2]

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

<!-- LICENSE -->
## Lisensi

Didistribusikan di bawah Lisensi MIT. Lihat `LICENSE.txt` untuk informasi lebih lanjut.

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

<!-- CONTACT -->
## Kontak

Fakultas Teknik Informatika, Universitas Muhammadiyah Malang  
Email: [informatika@umm.ac.id](mailto:informatika@umm.ac.id)  
Situs: [informatika.umm.ac.id](https://informatika.umm.ac.id)  
Link Proyek: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Penghargaan

- [Direktori Putusan Mahkamah Agung RI](https://putusan3.mahkamahagung.go.id)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Shields.io](https://shields.io)
- [Python Documentation](https://docs.python.org/3/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/your_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/your_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/your_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/your_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/your_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/your_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/your_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/your_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/your_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/your_username/repo_name/blob/main/LICENSE.txt
[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Pandas]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Scikit-learn]: https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[BeautifulSoup]: https://img.shields.io/badge/BeautifulSoup-4A4A55?style=for-the-badge&logo=python&logoColor=white
[Matplotlib]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white
[Seaborn]: https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white
[WordCloud]: https://img.shields.io/badge/WordCloud-4A4A55?style=for-the-badge&logo=python&logoColor=white