import streamlit as st
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Skripsi - IDX30", layout="wide")

# Header
st.title("📊 Dashboard Rekomendasi Investasi Saham IDX30 (Proyeksi 2026)")
st.markdown("""
**Implementasi Algoritma Random Forest & CRISP-DM**  
Dashboard ini menampilkan daftar saham IDX30 yang diprediksi **Layak Investasi** untuk tahun 2026 berdasarkan analisis rasio fundamental (EPS, PER, PBV, ROE, ROA, DER).
""")

st.divider()

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    # Pastikan nama file sesuai dengan yang di-upload ke GitHub
    return pd.read_csv('Rekomendasi_IDX30_2026.csv')

try:
    df = load_data()
    
    st.subheader(f"🏆 Hasil Prediksi: {len(df)} Saham Direkomendasikan")
    st.write("Tabel di bawah ini diurutkan berdasarkan probabilitas kelayakan tertinggi dari model Random Forest.")
    
    # Menampilkan DataFrame dengan format yang rapi
    st.dataframe(
        df.style.format({
            'EPS': '{:,.2f}',
            'PER': '{:.2f}',
            'PBV': '{:.2f}',
            'ROE': '{:.4f}',
            'ROA': '{:.4f}',
            'DER': '{:.4f}',
            'Probabilitas_Layak': '{:.2%}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
except FileNotFoundError:
    st.error("File 'Rekomendasi_IDX30_2026.csv' tidak ditemukan. Pastikan file tersebut sudah diunggah ke repositori GitHub.")
