import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
# Judul Aplikasi
st.title("Prediksi Harga Jual Mobil Bekas")

# Memuat dataset
# URL = 'dcleaning.csv'
df = pd.read_csv("dcleaning.csv")
@st.cache_data
def load_data():
    df = pd.read_csv("dcleaning.csv")
    return df

# Menampilkan dataset
st.write(df)

# Fungsi untuk menampilkan halaman utama
def show_about():
    st.title("H")
    # Path ke file CSV
    file_path = 'dcleaning.csv'

    # Periksa keberadaan file
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di lokasi yang benar.")
        st.stop()

    # Muat data CSV
    try:
        df = pd.read_csv(file_path)
        st.write(df)  # Tampilkan data jika berhasil dimuat
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")


    # Menampilkan informasi tentang tahun dan ukuran mesin
    st.subheader('Informasi Tahun dan Ukuran Mesin')
    tahun_counts = df['tahun'].value_counts()
    st.text(f'Proporsi tahun {tahun_counts.index[0]}: {tahun_counts.values[0] / sum(tahun_counts):.2%}')
    st.text(f'Proporsi tahun {tahun_counts.index[1]}: {tahun_counts.values[1] / sum(tahun_counts):.2%}')
    ukuran_mesin_counts = df['ukuran_mesin'].value_counts()
    st.text(f'Proporsi ukuran mesin besar: {ukuran_mesin_counts.values[0] / sum(ukuran_mesin_counts):.2%}')
    st.text(f'Proporsi ukuran mesin kecil: {ukuran_mesin_counts.values[1] / sum(ukuran_mesin_counts):.2%}')

    # Menampilkan grafik
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    tahun_counts.plot.bar(ax=ax[0])
    ax[0].set_title('Distribusi Tahun Mobil')
    ax[0].set_xlabel('Tahun')
    ax[0].set_ylabel('Jumlah Mobil')
    df['ukuran_mesin'].plot.hist(ax=ax[1])
    ax[1].set_title('Distribusi Ukuran Mesin')
    ax[1].set_xlabel('Ukuran Mesin')
    ax[1].set_ylabel('Jumlah Mobil')
    st.pyplot(fig)

def predict_cancellation(df):
    # Memasukkan input dari pengguna
    st.subheader('Prediksi Harga')
    # models = st.text_input('Masukkan Model Mobil')
    tahun = int(st.number_input('Tahun', 0, 10000, 0))
    jarak_tempuh = int(st.number_input('Jarak Tempuh (km)', 0, 100000, 0))
    pajak = int(st.number_input('Pajak (IDR)', 0, 100000000, 0))
    mpg = int(st.number_input('Penggunaan Bahan Bakar', 0, 50, 0))
    ukuran_mesin = int(st.number_input('Ukuran Mesin', 0, 5, 0))

    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'tahun': [tahun],
        'jarak_tempuh': [jarak_tempuh],
        'pajak': [pajak],
        'mpg': [mpg],
        'ukuran_mesin': [ukuran_mesin]
    })

    # Melakukan prediksi
    if st.button('Perkiraan Harga'):
        # Memuat model menggunakan pickle
        with open('gnbb.pkl', 'rb') as f:
            clf = pickle.load(f)

        # Melakukan prediksi
        predict = clf.predict(input_data)

        # Menampilkan hasil prediksi
        st.write('Perkiraan harga mobil:', predict)
    # Fungsi untuk menampilkan halaman tentang

def show_Distribusi(df):
    # Plot Scatter antara Tahun dan Harga
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tahun', y='harga', data=df)
    plt.title('Scatter Plot Tahun vs Harga')
    plt.xlabel('Tahun')
    plt.ylabel('Harga')
    plt.text(2010, 30000, 'Penjelasan: Scatter plot ini menunjukkan hubungan antara tahun pembuatan mobil dengan harga jualnya. Dapat dilihat bahwa tidak ada pola hubungan yang jelas antara kedua variabel ini.')
    st.pyplot()

    # Plot Box antara Transmisi dan Harga
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='transmisi', y='harga', data=df)
    plt.title('Box Plot Transmisi vs Harga')
    plt.xlabel('Transmisi')
    plt.ylabel('Harga')
    plt.text(1, 30000, 'Penjelasan: Box plot ini menunjukkan distribusi harga mobil berdasarkan jenis transmisi. Kita dapat melihat perbedaan distribusi harga antara mobil dengan transmisi manual dan otomatis.')
    st.pyplot()

    # Plot Scatter antara Jarak Tempuh dan Harga
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='jarak_tempuh', y='harga', data=df)
    plt.title('Scatter Plot Jarak Tempuh vs Harga')
    plt.xlabel('Jarak Tempuh (km)')
    plt.ylabel('Harga')
    plt.text(50000, 30000, 'Penjelasan: Scatter plot ini menunjukkan hubungan antara jarak tempuh mobil dengan harga jualnya. Dapat dilihat bahwa semakin tinggi jarak tempuhnya, harga mobil cenderung lebih rendah.')
    st.pyplot()

    # Plot Scatter antara Pajak dan Harga
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='pajak', y='harga', data=df)
    plt.title('Scatter Plot Pajak vs Harga')
    plt.xlabel('Pajak (IDR)')
    plt.ylabel('Harga')
    plt.text(50000, 30000, 'Penjelasan: Scatter plot ini menunjukkan hubungan antara besarnya pajak mobil dengan harga jualnya. Dapat dilihat bahwa tidak ada pola hubungan yang jelas antara kedua variabel ini.')
    st.pyplot()

    # Plot Scatter antara Konsumsi Bahan Bakar (MPG) dan Harga
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='mpg', y='harga', data=df)
    plt.title('Scatter Plot MPG vs Harga')
    plt.xlabel('Konsumsi Bahan Bakar (MPG)')
    plt.ylabel('Harga')
    plt.text(40, 30000, 'Penjelasan: Scatter plot ini menunjukkan hubungan antara konsumsi bahan bakar mobil (MPG) dengan harga jualnya. Dapat dilihat bahwa tidak ada pola hubungan yang jelas antara kedua variabel ini.')
    st.pyplot()

    # Plot Box antara Ukuran Mesin dan Harga
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='ukuran_mesin', y='harga', data=df)
    plt.title('Box Plot Ukuran Mesin vs Harga')
    plt.xlabel('Ukuran Mesin')
    plt.ylabel('Harga')
    plt.text(3, 30000, 'Penjelasan: Box plot ini menunjukkan distribusi harga mobil berdasarkan ukuran mesinnya. Kita dapat melihat perbedaan distribusi harga antara mobil dengan ukuran mesin kecil dan besar.')
    st.pyplot()

    # Plot Box antara Kategori Mesin dan Harga
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Kategori_Mesin', y='harga', data=df)
    plt.title('Box Plot Kategori Mesin vs Harga')
    plt.xlabel('Kategori Mesin')
    plt.ylabel('Harga')
    plt.text(1, 30000, 'Penjelasan: Box plot ini menunjukkan distribusi harga mobil berdasarkan kategori mesinnya. Kita dapat melihat perbedaan distribusi harga antara mobil dengan kategori mesin A dan B.')
    st.pyplot()

    show_Distribusi(df)

def show_Perbandingan(df):
    # Membuat subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot pertama: Transmisi
    sns.countplot(x='transmisi', data=df, palette='rocket', ax=axes[0, 0])
    axes[0, 0].set_title('Jumlah Mobil berdasarkan Jenis Transmisi', fontweight="bold", size=10)

    # Plot kedua: Kategori Mesin
    sns.countplot(data=df, x='Kategori_Mesin', palette='Set1_r', ax=axes[0, 1])
    axes[0, 1].set_title('Jumlah Mobil berdasarkan Kategori Mesin', fontweight="bold", size=10)

    # Plot ketiga: Kategori Pajak
    sns.countplot(data=df, x='pajak', ax=axes[1, 0]).set_title('Jumlah Mobil berdasarkan Kategori Pajak', fontsize=10)

    # Plot keempat: Scatterplot Jarak Tempuh vs Harga
    sns.scatterplot(x='jarak_tempuh', y='harga', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Scatter Plot Jarak Tempuh vs Harga', fontweight="bold", size=10)

    # Menampilkan plot di Streamlit
    st.pyplot(fig)

    # Tambahkan kode interaktif di Streamlit
    show_Perbandingan(df)

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def show_hubungan(df):
    st.title("Hubungan Nilai")
    st.write("Menu ini menampilkan hubungan antara fitur-fitur yang relevan dengan prediksi harga mobil dalam dataset Mobil Bekas.")
    
    # List nama kolom yang ingin digunakan untuk plotting
    plot_columns = ['Transmisi vs Harga', 'Jarak Tempuh vs Harga', 'Pajak vs Harga', 'MPG vs Harga', 'Ukuran Mesin vs Harga']

    for plot_option in plot_columns:
        st.subheader(plot_option)

        if plot_option == "Transmisi vs Harga":
            sns.boxplot(x='transmisi', y='harga', data=df)
            st.pyplot()

            st.text("Penjelasan: Box plot ini memperlihatkan distribusi harga mobil berdasarkan jenis transmisi. Kita dapat melihat perbedaan distribusi harga antara mobil dengan transmisi manual dan otomatis.")

        elif plot_option == "Jarak Tempuh vs Harga":
            sns.scatterplot(x='jarak_tempuh', y='harga', data=df)
            st.pyplot()

            st.text("Penjelasan: Scatter plot ini menunjukkan hubungan antara jarak tempuh mobil dengan harga jualnya. Semakin tinggi jarak tempuhnya, harga mobil cenderung lebih rendah.")

        elif plot_option == "Pajak vs Harga":
            sns.scatterplot(x='pajak', y='harga', data=df)
            st.pyplot()

            st.text("Penjelasan: Scatter plot ini menunjukkan hubungan antara besarnya pajak mobil dengan harga jualnya. Tidak ada pola hubungan yang jelas antara kedua variabel ini.")

        elif plot_option == "MPG vs Harga":
            sns.scatterplot(x='mpg', y='harga', data=df)
            st.pyplot()

            st.text("Penjelasan: Scatter plot ini menunjukkan hubungan antara konsumsi bahan bakar mobil (MPG) dengan harga jualnya. Tidak ada pola hubungan yang jelas antara kedua variabel ini.")

        elif plot_option == "Ukuran Mesin vs Harga":
            sns.scatterplot(x='ukuran_mesin', y='harga', data=df)
            st.pyplot()

            st.text("Penjelasan: Scatter plot ini menunjukkan hubungan antara ukuran mesin mobil dengan harga jualnya. Tidak ada pola hubungan yang jelas antara kedua variabel ini.")

    st.title("Korelasi")
    df_corr = df[['tahun', 'harga', 'transmisi', 'jarak_tempuh', 'pajak', 'mpg', 'ukuran_mesin']].corr()

    # Buat heatmap menggunakan seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Korelasi antar Fitur')
    st.pyplot(fig)

    # Tambahkan teks penjelasan
    text = 'Heatmap korelasi di atas menunjukkan hubungan antara fitur-fitur yang relevan dengan prediksi harga mobil. Nilai korelasi dapat membantu dalam pemahaman terhadap pengaruh masing-masing fitur terhadap harga mobil.'
    st.markdown(text)

    # Panggil fungsi show_hubungan() di luar definisi fungsi
    df = pd.DataFrame(df)  # Definisikan DataFrame Anda di sini
    show_hubungan(df)




def show_Komposisi(df):
    st.title("Komposisi")
    st.write("""
        Menu ini menampilkan komposisi data berdasarkan tahun dalam dataset Mobil Bekas
    """)
    st.write("Komposisi data diperlihatkan melalui diagram pie, yang membagi data berdasarkan tahun mobil bekas.")

    # Buat pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(df.groupby(by=["tahun"]).size(), labels=df["tahun"].unique(), autopct="%0.2f")
    ax.set_title('Komposisi Berdasarkan Tahun')
    st.pyplot(fig)

    st.write("Penjelasan: Diagram pie di atas memperlihatkan proporsi jumlah mobil bekas berdasarkan tahun pembuatannya. Setiap bagian dari diagram pie menunjukkan persentase mobil bekas yang berasal dari tahun tertentu, yang membantu dalam memahami distribusi data berdasarkan tahun.")

# Panggil fungsi show_Komposisi() di luar definisi fungsi
    show_Komposisi(df)


# Memuat data
df = load_data()

# Mengatur sidebar
df2 = pd.read_csv('dcleaning.csv')
nav_options = {
    "About": show_about,
    "Distribution": lambda: show_Distribusi(df),
    "Relations": lambda: show_hubungan(df),
    "Perbandingan": lambda: show_Perbandingan(df),
    "Komposisi": lambda: show_Komposisi(df),
    "Prediction": lambda: predict_cancellation(df2)
}

# Menampilkan sidebar
st.sidebar.title("Mobil Bekas")
selected_page = st.sidebar.radio("Menu", list(nav_options.keys()))

# Menampilkan halaman yang dipilih
nav_options[selected_page]()