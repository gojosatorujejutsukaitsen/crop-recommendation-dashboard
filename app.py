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

# KPI
col1, col2, col3, col4 = st.columns(4)

col1.metric("Jumlah Data", f"{len(df):,}")
col2.metric("Jenis Tanaman", df["label"].nunique())
col3.metric("Variabel", 7)
col4.metric("Data Setelah Cleaning", f"{len(df_clean):,}")

st.divider()

# EXECUTIVE SUMMARY
st.markdown("## Ringkasan Penelitian")

st.info("""
Penelitian ini membangun sistem rekomendasi tanaman berdasarkan kondisi tanah dan iklim menggunakan algoritma Decision Tree dan Random Forest.

Dataset terdiri dari 2.200 observasi dan 22 jenis tanaman dengan 7 variabel lingkungan yang mempengaruhi pertumbuhan tanaman.

Random Forest menjadi model terbaik dengan akurasi 99,55%.
""")

st.divider()

# PREVIEW DATA
st.markdown("## Preview Dataset")

st.dataframe(df.head(20), use_container_width=True)

st.divider()

# INSIGHT
st.markdown("## Insight Utama")

st.success("""
• Dataset mencakup 22 jenis tanaman yang mewakili berbagai kondisi agroklimat.

• Curah hujan dan kelembapan menjadi faktor paling dominan dalam menentukan rekomendasi tanaman.

• Random Forest menghasilkan performa hampir sempurna dengan akurasi 99,55%.
""")
