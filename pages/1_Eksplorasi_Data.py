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

st.info("""
📌 **Insight Utama**

• Seluruh 22 jenis tanaman memiliki jumlah observasi yang sama, yaitu 100 data.

• Distribusi kelas yang sepenuhnya seimbang menunjukkan tidak adanya masalah *class imbalance*.

• Kondisi ini membantu model klasifikasi mempelajari seluruh kelas secara adil tanpa dominasi kelas tertentu.

• Dataset yang seimbang meningkatkan reliabilitas evaluasi performa model klasifikasi.
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

st.info("""
📌 Insight Utama

• Kandungan Nitrogen memiliki variasi paling besar dibandingkan Fosfor dan Kalium.

• Kalium menunjukkan rentang distribusi yang lebih sempit sehingga relatif lebih stabil.

• Variasi nutrisi yang tinggi menunjukkan bahwa dataset mencakup beragam kondisi kesuburan tanah.
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

st.info("""
📌 Insight Utama

• Curah hujan memiliki variasi terbesar dibandingkan seluruh variabel lingkungan.

• Nilai pH relatif stabil dengan rata-rata mendekati kondisi netral.

• Mayoritas observasi berada pada lingkungan dengan kelembapan tinggi yang mencerminkan karakteristik wilayah tropis.
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

st.info("""
📌 Korelasi digunakan untuk melihat hubungan antar variabel.

• Nilai mendekati 1 menunjukkan hubungan positif yang kuat.

• Nilai mendekati -1 menunjukkan hubungan negatif yang kuat.

• Nilai mendekati 0 menunjukkan hubungan yang lemah.
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
