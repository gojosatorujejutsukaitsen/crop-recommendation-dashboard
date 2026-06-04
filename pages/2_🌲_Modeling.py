import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🌲 Modeling & Evaluasi Model")

st.markdown("""
Halaman ini menampilkan hasil pemodelan klasifikasi menggunakan
algoritma Decision Tree dan Random Forest.
""")

st.divider()

# KPI MODEL
col1, col2, col3 = st.columns(3)

col1.metric(
    "Decision Tree Accuracy",
    "96.36%"
)

col2.metric(
    "Random Forest Accuracy",
    "99.55%"
)

col3.metric(
    "Model Terbaik",
    "Random Forest"
)

st.divider()

st.info("""
Tahap ini membandingkan performa Decision Tree dan Random Forest
berdasarkan Accuracy, F1-Score, dan Cross Validation.
""")

st.subheader("📊 Perbandingan Kinerja Model")

hasil_model = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy (%)": [96.36, 99.55],
    "F1-Score (%)": [96.35, 99.55],
    "Cross Validation (%)": [96.20, 99.40]
})

st.dataframe(
    hasil_model,
    use_container_width=True,
    hide_index=True
)

# =========================
# VISUALISASI DI SINI
# =========================

fig = px.bar(
    hasil_model,
    x="Model",
    y="Accuracy (%)",
    color="Model",
    text="Accuracy (%)",
    title="Perbandingan Akurasi Model"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    showlegend=False,
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# INSIGHT DI BAWAH GRAFIK
# =========================

st.success("""
🎯 Random Forest memberikan performa terbaik dengan akurasi 99,55%.

🌳 Decision Tree tetap memiliki performa sangat baik dengan akurasi 96,36%.

📈 Selisih performa menunjukkan bahwa pendekatan ensemble pada Random Forest mampu meningkatkan kemampuan klasifikasi dibandingkan satu pohon keputusan tunggal.
""")

# ==================================================
# FEATURE IMPORTANCE RANDOM FOREST
# ==================================================

import joblib
import pandas as pd
import plotly.express as px

st.divider()

st.subheader("🌲 Feature Importance Random Forest")

rf_model = joblib.load(
    "data/data/models/random_forest.pkl"
)

fitur = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

importance_df = pd.DataFrame({
    "Fitur": fitur,
    "Importance": rf_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=True
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Fitur",
    orientation="h",
    color="Importance",
    color_continuous_scale="greens",
    title="Feature Importance Random Forest"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Variabel dengan nilai importance tertinggi merupakan faktor yang paling berpengaruh dalam menentukan rekomendasi tanaman. Semakin besar nilai importance, semakin besar kontribusinya terhadap keputusan model Random Forest.
""")


# ==================================================
# VISUALISASI DECISION TREE
# ==================================================

import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

st.divider()

st.subheader("🌳 Visualisasi Decision Tree")

dt_model = joblib.load(
    "data/data/models/decision_tree.pkl"
)

fig, ax = plt.subplots(figsize=(22, 10))

fitur = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

plot_tree(
    dt_model,
    feature_names=fitur,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)

st.pyplot(fig)

st.info("""
Visualisasi ini menampilkan 3 level pertama dari Decision Tree untuk memudahkan interpretasi proses klasifikasi tanaman.
""")
