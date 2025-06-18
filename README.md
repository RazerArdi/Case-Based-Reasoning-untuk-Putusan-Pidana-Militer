<a id="readme-top"></a>

[][contributors-url]
[][forks-url]
[][stars-url]
[][issues-url]
[][license-url]

<br />
<div align="center">
<a href="https://putusan3.mahkamahagung.go.id" title="Mahkamah Agung RI">
<img src="https://putusan3.mahkamahagung.go.id/public/frontend/images/logo.png" alt="Logo Mahkamah Agung" width="220">
</a>
<h3 align="center">Sistem Case-Based Reasoning untuk Analisis Putusan Pidana Militer</h3>
<p align="center">
Sebuah sistem berbasis Python untuk menganalisis dan merekomendasikan putusan pengadilan pidana militer menggunakan pendekatan Case-Based Reasoning (CBR).
<br />
<a href="https://github.com/RazerArdi/Case-Based-Reasoning-untuk-Putusan-Pidana-Militer"><strong>Jelajahi Notebook »</strong></a>
<br />
<br />
<a href="https://github.com/RazerArdi/Case-Based-Reasoning-untuk-Putusan-Pidana-Militer/issues">Laporkan Bug</a>
·
<a href="https://github.com/RazerArdi/Case-Based-Reasoning-untuk-Putusan-Pidana-Militer/issues">Ajukan Fitur</a>
</p>
</div>

<details>
<summary>Daftar Isi</summary>
<ol>
<li><a href="#tentang-proyek">Tentang Proyek</a></li>
<li><a href="#tabel-pemenuhan-parameter-uas">Tabel Pemenuhan Parameter UAS</a></li>
<li><a href="#dibuat-dengan">Dibuat Dengan</a></li>
<li><a href="#memulai">Memulai</a></li>
<li><a href="#struktur-direktori">Struktur Direktori</a></li>
<li><a href="#roadmap-proyek">Roadmap Proyek</a></li>
<li><a href="#kontribusi">Kontribusi</a></li>
<li><a href="#lisensi">Lisensi</a></li>
<li><a href="#kontak">Kontak</a></li>
<li><a href="#penghargaan">Penghargaan</a></li>
</ol>
</details>

## Tentang Proyek

Proyek ini adalah implementasi dari sistem **Case-Based Reasoning (CBR)** sederhana yang dirancang untuk menganalisis putusan pengadilan dalam domain **Pidana Militer**. Sistem ini dibangun menggunakan Python dan memanfaatkan data putusan yang dipublikasikan di [Direktori Putusan Mahkamah Agung RI](https://putusan3.mahkamahagung.go.id/).

Siklus CBR yang diterapkan mencakup:

1. **Membangun Case Base**: Mengunduh dan membersihkan lebih dari 30 dokumen putusan pidana militer.
2. **Case Representation**: Mengekstrak metadata dan fitur teks penting, kemudian menyimpannya dalam format `.csv`.
3. **Case Retrieval**: Mengimplementasikan tiga model berbeda (TF-IDF + Cosine Similarity, SVM, dan IndoBERT) untuk menemukan kasus-kasus yang paling relevan dengan kueri baru.
4. **Solution Reuse**: Mengadaptasi "solusi" dari kasus-kasus termirip (top-k) untuk memprediksi amar putusan pada kasus baru menggunakan metode *Majority Vote*.
5. **Evaluasi Model**: Menganalisis dan membandingkan performa ketiga model retrieval menggunakan metrik standar seperti Akurasi, Presisi, Recall, dan F1-Score.

Proyek ini dikembangkan sebagai tugas Ujian Akhir Semester mata kuliah Penalaran Komputer di Fakultas Teknik Informatika, Universitas Muhammadiyah Malang.

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Tabel Pemenuhan Parameter UAS

Tabel berikut merangkum pemenuhan setiap parameter yang diwajibkan dalam dokumen UAS.

| Tahap / Parameter | Perintah UAS | Status Implementasi | Keterangan |
|--------------------|--------------|---------------------|------------|
| **Domain Kasus** | Pilih satu domain hukum spesifik. | ✅ **Terpenuhi** | Domain yang dipilih adalah **Pidana Militer**. |
| **Volume Data** | Minimal 30 dokumen putusan. | ✅ **Terpenuhi** | Berhasil mengunduh dan memproses **81 dokumen**. |
| **Tahap 1: Case Base** | Mengumpulkan, membersihkan, dan menyimpan data mentah (`.txt`). | ✅ **Terpenuhi** | Data mentah disimpan di direktori `data/raw/`. |
| ↳ *Opsional* | Mencatat log pembersihan (`cleaning.log`). | ✅ **Terpenuhi** | File `cleaning.log` dibuat di direktori `logs/`. |
| **Tahap 2: Representasi** | Ekstrak metadata & konten, simpan dalam format terstruktur (`.csv`). | ⚠️ **Terpenuhi Sebagian** | Disimpan ke `.csv`. Ekstraksi `no_perkara` & `amar_putusan` berhasil, namun `tanggal` & `pasal` sering gagal. |
| **Tahap 3: Retrieval** | Implementasi TF-IDF **atau** Embedding (BERT). | ✅ **Terpenuhi (Melampaui)** | Mengimplementasikan **tiga model**: TF-IDF, SVM, dan IndoBERT. |
| ↳ *Model* | Gunakan SVM/Naive Bayes **atau** model Transformer. | ✅ **Terpenuhi (Melampaui)** | Model SVM dan Transformer (IndoBERT) diimplementasikan. |
| **Tahap 4: Reuse** | Prediksi solusi (amar putusan) dengan *Majority Vote* dari top-k kasus. | ✅ **Terpenuhi** | Fungsi `predict_outcome` berhasil diimplementasikan. |
| **Tahap 5: Evaluasi** | Ukur metrik (Akurasi, Presisi, Recall, F1). | ✅ **Terpenuhi** | Metrik diukur dan dibandingkan untuk ketiga model. |
| ↳ *Opsional* | Analisis kegagalan (Error Analysis). | ✅ **Terpenuhi** | Analisis kegagalan sederhana untuk satu kueri dilakukan. |
| **Output: Kode** | Notebook dan/atau script Python per tahap CBR. | ✅ **Terpenuhi** | Seluruh pipeline diimplementasikan dalam `fix_notebook.ipynb`. |
| **Output: Repo** | Struktur direktori `/data`, `/notebooks`, dan `README.md`. | ✅ **Terpenuhi** | Struktur direktori proyek sesuai spesifikasi. |

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Dibuat Dengan

Daftar pustaka dan *framework* utama yang digunakan dalam proyek ini:

- [Python](https://www.python.org/)
- [Jupyter](https://jupyter.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch](https://pytorch.org/)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Memulai

Berikut adalah panduan untuk menyiapkan dan menjalankan proyek ini di lingkungan lokal Anda.

### Prasyarat

Pastikan perangkat lunak berikut telah terinstal:

- Python 3.10 atau versi lebih baru
- `pip` dan `venv` (biasanya sudah termasuk dalam instalasi Python)
- Jupyter Notebook atau Jupyter Lab

### Instalasi

1. **Kloning Repositori**

   ```sh
   git clone https://github.com/username/repo_name.git
   cd repo_name
   ```

2. **Buat dan Aktifkan Virtual Environment**

   ```sh
   # Untuk Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate

   # Untuk Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Instal Dependensi**

   Gunakan file `requirements.txt` untuk menginstal semua pustaka yang dibutuhkan.

   ```sh
   pip install -r requirements.txt
   ```

   Isi file `requirements.txt`:

   ```txt
   requests
   beautifulsoup4
   PyMuPDF
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   wordcloud
   torch
   transformers
   ipywidgets
   ```

4. **Jalankan Jupyter**

   ```sh
   jupyter lab
   ```

   Buka file `notebooks/fix_notebook.ipynb` dari antarmuka Jupyter dan jalankan setiap sel secara berurutan.

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Struktur Direktori

Proyek ini diorganisir dengan struktur sebagai berikut:

```
repo_name/
├── data/
│   ├── raw/                # Teks mentah (*.txt)
│   ├── processed/          # Data terstruktur (cases.csv)
│   └── eval/               # Ground truth & metrik
│       ├── queries.json
│       └── retrieval_metrics.csv
├── notebooks/
│   └── fix_notebook.ipynb  # Notebook utama berisi semua tahap CBR
├── logs/
│   └── cleaning.log        # Riwayat proses pembersihan data
├── requirements.txt        # Daftar dependensi
└── README.md
```

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Roadmap Proyek

- [x] **Tahap 1: Case Base** - *Web scraping* dan pembersihan data (81 dokumen).
- [x] **Tahap 2: Representasi** - Ekstraksi fitur dan penyimpanan ke CSV.
- [x] **Tahap 3: Retrieval** - Implementasi 3 model (TF-IDF, SVM, IndoBERT).
- [x] **Tahap 4: Reuse** - Prediksi amar putusan dengan *Majority Vote*.
- [x] **Tahap 5: Evaluasi** - Pengukuran metrik dan analisis performa.
- [x] **Tugas Opsional** - Analisis kegagalan dan pembuatan `cleaning.log`.
- [ ] **Pengembangan Lanjutan**:
    - [ ] Implementasi tahap *Revise & Retain*.
    - [ ] Optimasi parameter model.
    - [ ] Pengembangan antarmuka pengguna.

Lihat [issues](https://github.com/username/repo_name/issues) untuk daftar fitur yang diusulkan.

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Kontribusi

Kontribusi sangat dihargai. Untuk berkontribusi:

1. Fork proyek.
2. Buat branch fitur (`git checkout -b feature/FiturLuarBiasa`).
3. Commit perubahan (`git commit -m 'Menambahkan FiturLuarBiasa'`).
4. Push ke branch (`git push origin feature/FiturLuarBiasa`).
5. Buka Pull Request.

### Tim Pengembang

- **[Bayu Ardiyansyah]** - `[202210370311025]`
- **[Lutfi Indra Nur Praditya]** - `[202210370311482]`

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Lisensi

Didistribusikan di bawah Lisensi MIT. Lihat `LICENSE.txt` untuk informasi lebih lanjur.

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Kontak


- Email: [Bayu Ardiyansyah](mailto:bayuardi30@outlook.com)

Link Proyek: [https://github.com/RazerArdi/Case-Based-Reasoning-untuk-Putusan-Pidana-Militer](https://github.com/RazerArdi/Case-Based-Reasoning-untuk-Putusan-Pidana-Militer)

<p align="right">(<a href="#readme-top">kembali ke atas</a>)</p>

## Penghargaan

- [Direktori Putusan Mahkamah Agung RI](https://putusan3.mahkamahagung.go.id)
- [Template README by Othneil Drew](https://github.com/othneildrew/Best-README-Template)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Dokumentasi Scikit-learn](https://scikit-learn.org/stable/)
- [Shields.io](https://shields.io/)

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