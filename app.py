import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Skripsi - IDX30", layout="wide", page_icon="📈")

# Header
st.title("📈 Dashboard Klasifikasi Investasi Saham IDX30")
st.markdown("**Implementasi Algoritma Random Forest & Metodologi CRISP-DM**")
st.divider()

# Memuat Data dan Model
@st.cache_data
def load_data():
    return pd.read_csv('Rekomendasi_IDX30_2026.csv')

@st.cache_resource
def load_model():
    return joblib.load('model_rf_idx30.pkl')

try:
    df = load_data()
    model = load_model()
    
    # Membuat Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Hasil Rekomendasi 2026", "📊 Visualisasi Interaktif", "🤖 Uji Model (Manual Input)"])
    
    # ================= TAB 1: HASIL REKOMENDASI =================
    with tab1:
        st.subheader("Daftar Saham Direkomendasikan")
        
        # Fitur Interaktif: Slider Filter Probabilitas
        min_prob = st.slider("Filter Berdasarkan Minimal Probabilitas Layak:", min_value=0.50, max_value=1.00, value=0.50, step=0.05)
        
        # Menerapkan filter
        df_filtered = df[df['Probabilitas_Layak'] >= min_prob]
        
        st.info(f"Menampilkan **{len(df_filtered)}** saham dengan probabilitas di atas **{min_prob:.0%}**.")
        
        st.dataframe(
            df_filtered.style.format({
                'EPS': '{:,.2f}', 'PER': '{:.2f}', 'PBV': '{:.2f}',
                'ROE': '{:.4f}', 'ROA': '{:.4f}', 'DER': '{:.4f}',
                'Probabilitas_Layak': '{:.2%}'
            }),
            use_container_width=True, hide_index=True
        )

    # ================= TAB 2: VISUALISASI =================
    with tab2:
        st.subheader("Perbandingan Probabilitas Kelayakan Saham")
        
        # Fitur Interaktif: Bar chart
        chart_data = df.set_index('TICKER')['Probabilitas_Layak']
        st.bar_chart(chart_data)
        
    # ================= TAB 3: UJI MODEL MANUAL =================
    with tab3:
        st.subheader("Simulasi Prediksi Berdasarkan Rasio Fundamental")
        st.write("Masukkan rasio fundamental fiktif atau dari saham di luar dataset untuk melihat prediksi kelayakannya.")
        
        # Form Input Interaktif
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
            # Format input sesuai fitur yang dilatih
            input_data = np.array([[eps_input, per_input, pbv_input, roe_input, roa_input, der_input]])
            
            # Prediksi
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            st.divider()
            if pred == 1:
                st.success(f"✅ **HASIL: LAYAK INVESTASI** (Probabilitas: {prob:.2%})")
            else:
                st.error(f"❌ **HASIL: TIDAK LAYAK INVESTASI** (Probabilitas Layak: {prob:.2%})")

except FileNotFoundError:
    st.error("Pastikan file 'Rekomendasi_IDX30_2026.csv' dan 'model_rf_idx30.pkl' sudah diunggah ke repositori GitHub.")
