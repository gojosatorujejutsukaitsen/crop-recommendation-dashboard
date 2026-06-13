import streamlit as st

def tampilkan_footer():

    st.markdown("---")

    st.markdown(
        """
        <div style='text-align: center;
                    color: #808080;
                    font-size: 13px;
                    line-height: 1.7;
                    margin-top: 20px;
                    margin-bottom: 10px;'>

        <b>Crop Recommendation Dashboard</b><br><br>

        <b>Developed by:</b><br>
        Aditya Taufiqur Rahman • Aqmarine Ekstraktie Hakim • Atikah Fitriah Kaputri<br>
        Chevroline Nathalia Manalu • Nurhasanah • Yoshepine Lamria Simatupang<br><br>

        Program Studi S1 Statistika<br>
        Universitas Riau<br><br>

        © 2026 All Rights Reserved.
        </div>
        """,
        unsafe_allow_html=True
    )
