import streamlit as st
import pandas as pd

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
