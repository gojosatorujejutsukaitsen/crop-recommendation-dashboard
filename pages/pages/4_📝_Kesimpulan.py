import streamlit as st

st.set_page_config(
    page_title="Kesimpulan",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Kesimpulan & Rekomendasi")

st.markdown("""
Halaman ini merangkum temuan utama dari proses eksplorasi data,
pemodelan, dan pengembangan sistem rekomendasi tanaman berbasis
Machine Learning.
""")

st.divider()

# ==================================================
# RINGKASAN DATASET
# ==================================================

st.subheader("📊 Ringkasan Dataset")

st.success("""
• Dataset terdiri dari 2.200 observasi yang mencakup 22 jenis tanaman.

• Setiap kelas memiliki jumlah observasi yang sama sehingga distribusi data sepenuhnya seimbang.

• Variabel yang digunakan meliputi unsur hara tanah (N, P, K) serta faktor lingkungan berupa suhu, kelembapan, pH, dan curah hujan.
""")

# ==================================================
# TEMUAN EKSPLORASI DATA
# ==================================================

st.subheader("🔍 Temuan Utama Eksplorasi Data")

st.info("""
• Curah hujan dan kelembapan menunjukkan variasi yang tinggi antar tanaman sehingga menjadi faktor pembeda yang penting.

• Nitrogen memiliki rentang nilai paling luas di antara unsur hara tanah, menunjukkan kebutuhan nutrisi yang beragam pada setiap tanaman.

• Sebagian besar variabel memiliki korelasi yang rendah sehingga masing-masing memberikan informasi yang unik dalam proses klasifikasi.

• Hubungan terkuat ditemukan antara Fosfor (P) dan Kalium (K) dengan koefisien korelasi sebesar 0,74.
""")

# ==================================================
# HASIL PEMODELAN
# ==================================================

st.subheader("🌲 Hasil Evaluasi Model")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Decision Tree Accuracy",
        "96.36%"
    )

with col2:
    st.metric(
        "Random Forest Accuracy",
        "99.55%"
    )

st.success("""
• Random Forest menghasilkan performa terbaik dengan akurasi 99,55%.

• F1-Score pada hampir seluruh kelas mendekati nilai sempurna.

• Kesalahan klasifikasi antar tanaman sangat rendah berdasarkan confusion matrix.

• Random Forest dipilih sebagai model utama karena memiliki kemampuan generalisasi yang lebih baik dibandingkan Decision Tree tunggal.
""")

# ==================================================
# IMPLIKASI
# ==================================================

st.subheader("🌱 Implikasi Praktis")

st.success("""
Sistem yang dikembangkan mampu memberikan rekomendasi tanaman berdasarkan kondisi tanah dan lingkungan secara cepat dan objektif.

Informasi ini dapat membantu petani maupun pihak terkait dalam menentukan alternatif tanaman yang paling sesuai dengan karakteristik lahan yang dimiliki.
""")

# ==================================================
# KETERBATASAN
# ==================================================

st.subheader("⚠️ Keterbatasan Sistem")

st.warning("""
• Model hanya menggunakan variabel yang tersedia pada dataset.

• Faktor ekonomi seperti harga pasar dan biaya produksi belum dipertimbangkan.

• Sistem belum menggunakan data cuaca real-time maupun informasi lokasi geografis.
""")

# ==================================================
# PENGEMBANGAN
# ==================================================

st.subheader("🚀 Pengembangan Selanjutnya")

st.info("""
• Integrasi data cuaca real-time.

• Penambahan variabel karakteristik lahan yang lebih lengkap.

• Integrasi sistem informasi geografis (GIS).

• Pengembangan rekomendasi berbasis produktivitas dan keuntungan ekonomi.
""")

st.divider()

st.success("""
🎯 Kesimpulan Akhir

Berdasarkan hasil evaluasi, Random Forest berhasil memberikan performa klasifikasi yang sangat tinggi dengan akurasi 99,55%.

Temuan ini menunjukkan bahwa kombinasi informasi unsur hara tanah dan kondisi lingkungan dapat dimanfaatkan secara efektif untuk membangun sistem rekomendasi tanaman berbasis Machine Learning yang akurat dan mudah digunakan.
""")
