import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Dashboard Skripsi - IDX30", layout="wide", page_icon="📈")

st.title("📈 Dashboard Klasifikasi Investasi Saham IDX30")
st.markdown("**Implementasi Algoritma Random Forest & Metodologi CRISP-DM**")
st.divider()

# PERUBAHAN 1: Ganti nama file yang dibaca
@st.cache_data
def load_data():
    return pd.read_csv('Semua_Prediksi_IDX30_2026.csv')

@st.cache_resource
def load_model():
    return joblib.load('model_rf_idx30.pkl')

try:
    df = load_data()
    model = load_model()
    
    tab1, tab2, tab3 = st.tabs(["📋 Hasil Prediksi 30 Saham", "📊 Visualisasi Interaktif", "🤖 Uji Model Manual"])
    
    # ================= TAB 1: HASIL PREDIKSI KESELURUHAN =================
    with tab1:
        st.subheader(f"Daftar Keseluruhan Saham IDX30 ({len(df)} Saham)")
        
        # Opsi interaktif untuk memfilter tampilan tabel
        filter_status = st.radio("Tampilkan berdasarkan status:", ["Semua Saham", "Hanya Layak", "Hanya Tidak Layak"], horizontal=True)
        
        if filter_status == "Hanya Layak":
            df_tampil = df[df['Status'] == '✅ Layak']
        elif filter_status == "Hanya Tidak Layak":
            df_tampil = df[df['Status'] == '❌ Tidak Layak']
        else:
            df_tampil = df
            
        st.dataframe(
            df_tampil.style.format({
                'EPS': '{:,.2f}', 'PER': '{:.2f}', 'PBV': '{:.2f}',
                'ROE': '{:.4f}', 'ROA': '{:.4f}', 'DER': '{:.4f}',
                'Probabilitas_Layak': '{:.2%}'
            }),
            use_container_width=True, hide_index=True
        )

    # ================= TAB 2: VISUALISASI =================
    with tab2:
        st.subheader("Perbandingan Probabilitas Kelayakan Saham")
        chart_data = df.set_index('TICKER')['Probabilitas_Layak']
        st.bar_chart(chart_data)
        
    # ================= TAB 3: UJI MODEL MANUAL =================
    with tab3:
        st.subheader("Simulasi Prediksi Berdasarkan Rasio Fundamental")
        st.write("Masukkan rasio fundamental fiktif untuk melihat prediksi kelayakannya.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            eps_input = st.number_input("EPS (Earning Per Share)", value=150.0)
            per_input = st.number_input("PER (Price Earning Ratio)", value=15.0)
        with col2:
            pbv_input = st.number_input("PBV (Price to Book Value)", value=2.5)
            roe_input = st.number_input("ROE (Return on Equity)", value=0.15, format="%.4f")
        with col3:
            roa_input = st.number_input("ROA (Return on Asset)", value=0.08, format="%.4f")
            der_input = st.number_input("DER (Debt to Equity Ratio)", value=1.2, format="%.4f")
            
        if st.button("Jalankan Prediksi", type="primary"):
            input_data = np.array([[eps_input, per_input, pbv_input, roe_input, roa_input, der_input]])
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            st.divider()
            if pred == 1:
                st.success(f"✅ **HASIL: LAYAK INVESTASI** (Probabilitas: {prob:.2%})")
            else:
                st.error(f"❌ **HASIL: TIDAK LAYAK INVESTASI** (Probabilitas Layak: {prob:.2%})")

except FileNotFoundError:
    st.error("Pastikan file 'Semua_Prediksi_IDX30_2026.csv' dan 'model_rf_idx30.pkl' sudah diunggah ke repositori GitHub.")
