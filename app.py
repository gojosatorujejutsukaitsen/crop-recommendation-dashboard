import streamlit as st

st.set_page_config(
    page_title="Crop Recommendation Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Crop Recommendation Dashboard")

st.markdown("""
## Klasifikasi Jenis Tanaman Berdasarkan Kondisi Tanah dan Iklim

Menggunakan Decision Tree dan Random Forest
""")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Observasi", "2200")
col2.metric("Jenis Tanaman", "22")
col3.metric("Variabel", "7")
col4.metric("Deteksi Outlier", "611")

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy DT", "96.36%")
col2.metric("Accuracy RF", "99.55%")
col3.metric("F1 RF", "99.55%")
col4.metric("Best Model", "Random Forest")

st.divider()

st.success("""
Random Forest menjadi model terbaik dengan akurasi 99.55%.

Faktor iklim berkontribusi 52.25% terhadap klasifikasi tanaman,
sedangkan faktor tanah berkontribusi 47.75%.

Curah hujan dan kelembapan merupakan variabel paling berpengaruh.
""")
