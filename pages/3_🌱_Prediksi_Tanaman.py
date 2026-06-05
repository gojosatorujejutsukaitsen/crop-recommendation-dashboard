import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ====================================
# KONFIGURASI HALAMAN
# ====================================

st.set_page_config(
    page_title="Prediksi Tanaman",
    page_icon="🌱",
    layout="wide"
)

# ====================================
# LOAD MODEL
# ====================================

rf_model = joblib.load(
    "data/data/models/random_forest.pkl"
)

scaler = joblib.load(
    "data/data/models/scaler.pkl"
)

label_encoder = joblib.load(
    "data/data/models/label_encoder.pkl"
)

# ====================================
# HEADER
# ====================================

st.title("🌱 Prediksi Tanaman")

st.markdown("""
Masukkan karakteristik tanah dan kondisi lingkungan lahan untuk memperoleh rekomendasi tanaman yang paling sesuai berdasarkan model Random Forest.
""")

st.info("""
Gunakan nilai yang berada dalam rentang dataset agar hasil prediksi lebih representatif.

Sistem akan menganalisis kondisi lahan dan memberikan rekomendasi tanaman beserta tingkat keyakinan prediksi.
""")

col1, col2 = st.columns(2)

with col1:

    st.subheader("🧪 Kondisi Tanah")

    st.markdown("""
    **Nitrogen (N)**

    Kadar unsur Nitrogen dalam tanah yang berperan dalam pertumbuhan vegetatif tanaman.

    Rentang data: 0 – 140
    Contoh nilai: 90
    """)

    N = st.number_input(
        "Masukkan nilai Nitrogen (N)",
        min_value=0.0,
        max_value=140.0,
        value=90.0
    )

    st.markdown("""
    **Fosfor (P)**

    Kadar unsur Fosfor yang mendukung perkembangan akar, bunga, dan buah.

    Rentang data: 5 – 145
    Contoh nilai: 42
    """)

    P = st.number_input(
        "Masukkan nilai Fosfor (P)",
        min_value=5.0,
        max_value=145.0,
        value=42.0
    )

    st.markdown("""
    **Kalium (K)**

    Kadar unsur Kalium yang membantu ketahanan tanaman terhadap penyakit dan kondisi lingkungan.

    Rentang data: 5 – 205
    Contoh nilai: 43
    """)

    K = st.number_input(
        "Masukkan nilai Kalium (K)",
        min_value=5.0,
        max_value=205.0,
        value=43.0
    )

    st.markdown("""
    **pH Tanah**

    Tingkat keasaman tanah. Nilai sekitar 7 menunjukkan kondisi netral.

    Rentang data: 4.5 – 8.4
    """)

    ph = st.slider(
        "Pilih nilai pH",
        min_value=4.5,
        max_value=8.4,
        value=6.5
    )

# ====================================
# KOLOM KANAN - KONDISI LINGKUNGAN
# ====================================

with col2:

    st.subheader("🌦️ Kondisi Lingkungan")

    # TEMPERATURE
    st.markdown("""
    **🌡️ Suhu (Temperature)**

    Suhu lingkungan tempat tanaman akan dibudidayakan.

    Rentang data: 8 – 43 °C

    Contoh nilai: 25 °C
    """)

    temperature = st.slider(
        "Pilih suhu (°C)",
        min_value=8.0,
        max_value=43.0,
        value=25.0
    )

    # HUMIDITY
    st.markdown("""
    **💧 Kelembapan (Humidity)**

    Persentase kelembapan udara di sekitar lahan.

    Rentang data: 14 – 100 %

    Contoh nilai: 80 %
    """)

    humidity = st.slider(
        "Pilih kelembapan (%)",
        min_value=14.0,
        max_value=100.0,
        value=80.0
    )

    # RAINFALL
    st.markdown("""
    **🌧️ Curah Hujan (Rainfall)**

    Jumlah curah hujan yang diterima lahan.

    Rentang data: 20 – 300 mm

    Contoh nilai: 200 mm
    """)

    rainfall = st.slider(
        "Pilih curah hujan (mm)",
        min_value=20.0,
        max_value=300.0,
        value=200.0
    )


st.divider()

if st.button(
    "🔮 Prediksi Tanaman",
    use_container_width=True,
    type="primary"
):

    input_data = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })

    st.write("Input Data:")
    st.dataframe(input_data)


    input_scaled = scaler.transform(input_data)

    st.write("Hasil Scaling:")
    st.dataframe(
        pd.DataFrame(
            input_scaled,
            columns=input_data.columns
        )
    )

    hasil_prediksi = rf_model.predict(input_scaled)

    nama_tanaman = le.inverse_transform(
        hasil_prediksi
    )[0]

    st.success(
        f"🌱 Tanaman yang direkomendasikan: **{nama_tanaman.upper()}**"
    )
