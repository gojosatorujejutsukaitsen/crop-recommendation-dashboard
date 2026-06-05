import streamlit as st
import pandas as pd

st.sidebar.title("📋 Menu Dashboard")
st.sidebar.markdown("Halaman Utama")
st.set_page_config(
    page_title="Crop Recommendation Dashboard",
    page_icon="🌱",
    layout="wide"
)


# LOAD DATA
df = pd.read_csv("data/data/Crop_recommendation.csv")
df_clean = pd.read_csv("data/data/Crop_recommendation_clean.csv")

# HEADER
st.title("🌱 Crop Recommendation Dashboard")
st.subheader("Klasifikasi Jenis Tanaman Berdasarkan Kondisi Tanah dan Iklim")
st.caption(
    "Dashboard Sistem Rekomendasi Tanaman berbasis Machine Learning menggunakan algoritma Decision Tree dan Random Forest."
)

# KPI
col1, col2, col3, col4 = st.columns(4)

col1.metric("Jumlah Data", f"{len(df):,}")
col2.metric("Jenis Tanaman", df["label"].nunique())
col3.metric("Variabel", 7)
col4.metric("Data Setelah Cleaning", f"{len(df_clean):,}")

st.divider()

# EXECUTIVE SUMMARY
st.markdown("## 📖 Ringkasan Penelitian")

st.info("""
Penelitian ini bertujuan membangun sistem rekomendasi tanaman berdasarkan kondisi tanah dan iklim menggunakan algoritma Decision Tree dan Random Forest.

Dataset yang digunakan terdiri dari 2.200 observasi, 22 jenis tanaman, dan 7 variabel yang merepresentasikan unsur hara tanah serta faktor lingkungan.

Melalui pendekatan machine learning, sistem dapat memberikan rekomendasi jenis tanaman yang sesuai berdasarkan karakteristik lahan yang dimasukkan pengguna.
""")

st.divider()

# PREVIEW DATA
st.markdown("## 📄 Preview Dataset")

st.dataframe(df.head(20), use_container_width=True)

st.divider()

# INSIGHT
st.markdown("## 💡 Insight Utama")

st.success("""
• Dataset memiliki distribusi kelas yang sepenuhnya seimbang sehingga proses pelatihan model berlangsung tanpa bias akibat dominasi kelas tertentu.

• Curah hujan dan kelembapan terbukti menjadi faktor lingkungan yang paling berpengaruh dalam menentukan rekomendasi tanaman.

• Random Forest memberikan performa terbaik dengan akurasi 99,55% dan mampu mempertahankan F1-Score yang tinggi pada hampir seluruh jenis tanaman.

• Hasil penelitian menunjukkan bahwa kombinasi informasi tanah dan iklim dapat dimanfaatkan secara efektif untuk mendukung pengambilan keputusan dalam pemilihan tanaman.
""")
