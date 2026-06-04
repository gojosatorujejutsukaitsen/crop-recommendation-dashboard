import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/data/Crop_recommendation.csv")

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Karakteristik Data",
    page_icon="📊",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("📊 Karakteristik Data")
st.markdown(
    """
    Analisis karakteristik dataset Crop Recommendation
    berdasarkan kondisi tanah dan iklim.
    """
)

# =========================
# KPI
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Observasi", "2.200")
col2.metric("Jenis Tanaman", "22")
col3.metric("Variabel", "7")
col4.metric("Distribusi Kelas", "Seimbang")

st.divider()

# =========================
# DISTRIBUSI TANAMAN
# =========================
st.subheader("🌾 Distribusi Jenis Tanaman")

label_count = (
    df["label"]
    .value_counts()
    .sort_values(ascending=True)
    .reset_index()
)

label_count.columns = ["Tanaman", "Jumlah"]

fig = px.bar(
    label_count,
    x="Jumlah",
    y="Tanaman",
    orientation="h",
    color="Jumlah",
    color_continuous_scale="Blues",
    text="Jumlah"
)

fig.update_layout(
    height=700,
    coloraxis_showscale=False,
    xaxis_title="Jumlah Observasi",
    yaxis_title="Jenis Tanaman"
)

st.plotly_chart(fig, use_container_width=True)

st.success("""
• Seluruh 22 jenis tanaman memiliki jumlah observasi yang identik (100 data per kelas), menghasilkan distribusi yang sepenuhnya seimbang.

• Keseimbangan ini mengurangi risiko model lebih sering memprediksi kelas tertentu akibat dominasi jumlah data.

• Dengan tidak adanya class imbalance, perbandingan performa antar kelas pada tahap klasifikasi dapat dievaluasi secara lebih objektif dan reliabel.

• Kondisi ini menjadi salah satu faktor yang mendukung tingginya performa model klasifikasi pada penelitian ini.
""")

# =========================
# NPK
# =========================
st.subheader("Karakteristik Nutrisi Tanah (N, P, K)")

npk = df[["N", "P", "K"]]

fig = px.box(
    npk,
    points="outliers",
    color_discrete_sequence=["#1F4E79"]
)

fig.update_layout(height=550)

st.plotly_chart(fig, use_container_width=True)

st.success("""
📌 Insight Utama

• Nitrogen (N) menunjukkan rentang nilai paling luas dibandingkan Fosfor (P) dan Kalium (K), mengindikasikan bahwa kebutuhan unsur hara nitrogen antar tanaman dalam dataset sangat beragam.

• Fosfor (P) memiliki nilai tengah yang relatif lebih tinggi dan distribusi yang cukup konsisten pada sebagian besar observasi.

• Kalium (K) menampilkan beberapa nilai ekstrem yang jauh di atas mayoritas data, menunjukkan adanya kelompok tanaman dengan kebutuhan kalium yang sangat tinggi.

• Variasi unsur hara yang cukup besar menunjukkan bahwa rekomendasi tanaman tidak hanya dipengaruhi kondisi lingkungan, tetapi juga karakteristik kesuburan tanah yang berbeda antar komoditas.
""")

st.divider()

# =========================
# IKLIM
# =========================
st.subheader("Karakteristik Lingkungan")

iklim = df[["temperature", "humidity", "ph", "rainfall"]]

fig = px.box(
    iklim,
    points="outliers",
    color_discrete_sequence=["#5DADE2"]
)

fig.update_layout(height=550)

st.plotly_chart(fig, use_container_width=True)

st.success("""
📌 Insight Utama

• Curah hujan (rainfall) memiliki variasi terbesar dibandingkan seluruh variabel lingkungan, menunjukkan bahwa kebutuhan air antar jenis tanaman dalam dataset sangat beragam.

• Kelembapan (humidity) juga memperlihatkan rentang distribusi yang luas sehingga berpotensi menjadi faktor pembeda penting dalam proses klasifikasi tanaman.

• Nilai pH relatif stabil dengan variasi yang lebih kecil dibandingkan variabel lingkungan lainnya, menandakan bahwa sebagian besar tanaman berada pada kisaran keasaman tanah yang tidak terlalu berbeda.

• Temuan ini konsisten dengan hasil pemodelan Random Forest yang menunjukkan bahwa curah hujan dan kelembapan merupakan dua variabel paling berpengaruh dalam menentukan rekomendasi tanaman.
""")

st.divider()

st.markdown("---")

st.markdown("## 🔗 Korelasi Antar Variabel")

corr = df.drop(columns=["label"]).corr()

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

plt.title("Heatmap Korelasi Variabel")

st.pyplot(fig)

st.success("""
📌 Insight Utama

• Hubungan terkuat ditemukan antara Fosfor (P) dan Kalium (K) dengan koefisien korelasi sebesar 0,74, menunjukkan bahwa kedua unsur hara tersebut cenderung meningkat secara bersamaan pada beberapa kondisi lahan.

• Sebagian besar pasangan variabel memiliki korelasi yang rendah (|r| < 0,30), menandakan bahwa setiap variabel menyimpan informasi yang relatif berbeda.

• Curah hujan (rainfall) menunjukkan korelasi yang sangat lemah dengan variabel lainnya, sehingga berpotensi memberikan informasi unik dalam proses klasifikasi tanaman.

• Rendahnya korelasi antar sebagian besar variabel mengindikasikan bahwa rekomendasi tanaman dipengaruhi oleh kombinasi berbagai faktor tanah dan lingkungan, bukan oleh satu faktor dominan saja.
""")

# =========================
# TABEL STATISTIK
# =========================
st.subheader("Ringkasan Statistik")

st.dataframe(
    df.describe().round(2),
    use_container_width=True
)

st.divider()

# =========================
# RINGKASAN
# =========================
st.success("""
📌 Ringkasan Karakteristik Data

1. Dataset terdiri dari 2.200 observasi dan 22 jenis tanaman dengan distribusi yang sepenuhnya seimbang.

2. Variabilitas terbesar ditemukan pada variabel curah hujan, sedangkan pH merupakan variabel yang paling stabil.

3. Kandungan Nitrogen menunjukkan variasi tertinggi di antara unsur hara tanah.

4. Struktur dataset yang seimbang dan beragam menjadi fondasi kuat bagi pembangunan model klasifikasi tanaman.
""")
