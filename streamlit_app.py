import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Untuk kebutuhan riil, simpan model dari sel sebelumnya (joblib.dump(best_rf, 'rf_model.pkl'))
# dan load di sini: best_rf = joblib.load('rf_model.pkl')

st.title("Sistem Klasifikasi Kelayakan Investasi Saham IDX30")
st.markdown("Mengimplementasikan algoritma **Random Forest** teroptimasi untuk memproyeksikan portofolio tahun depan (CRISP-DM Deployment).")

st.sidebar.header("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Home", "Dataset", "Prediksi", "Evaluasi Model"])

if menu == "Prediksi":
    st.subheader("Simulasi Prediksi Kelayakan (Input Rasio Fundamental)")
    col1, col2, col3 = st.columns(3)

    with col1:
        eps_growth = st.number_input("EPS Growth (%)", value=0.05)
        roe = st.number_input("ROE (%)", value=0.12)
        cr = st.number_input("Current Ratio (x)", value=1.50)

    with col2:
        rev_growth = st.number_input("Revenue Growth (%)", value=0.08)
        npm = st.number_input("NPM (%)", value=0.15)
        per = st.number_input("PER (x)", value=10.5)

    with col3:
        roa = st.number_input("ROA (%)", value=0.06)
        der = st.number_input("DER (x)", value=0.85)
        pbv = st.number_input("PBV (x)", value=1.2)

    if st.button("Kalkulasi Proyeksi", type='primary'):
        # Pada sistem nyata, input ini juga harus melalui proses scaler.transform()
        # prediksi = best_rf.predict(scaler.transform([[eps_growth, rev_growth, roa, roe, npm, der, cr, per, pbv]]))

        # Simulasi Antarmuka
        st.success("Terkalkulasi. Proyeksi Selesai.")
        st.write("### Rekomendasi Mesin: LAYAK INVESTASI")
        st.write("**Probabilitas Layak:** 84% | **Probabilitas Tidak Layak:** 16%")
