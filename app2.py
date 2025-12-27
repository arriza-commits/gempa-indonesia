import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Peta Zonasi Gempa", layout="wide")

st.title("🌏 Analisis Zonasi Gempa Bumi Indonesia")
st.markdown("""
Aplikasi ini menggunakan Machine Learning (**K-Means Clustering**) untuk mengelompokkan 
daerah rawan gempa berdasarkan data spasial (Latitude & Longitude).
""")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Membaca data TSV
    try:
        df = pd.read_csv('katalog_gempa_v2.tsv', sep='\t')
        # Ambil kolom penting & buang data kosong
        df_clean = df[['latitude', 'longitude', 'magnitude', 'depth']].dropna()
        return df_clean
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("File 'katalog_gempa_v2.tsv' tidak ditemukan. Pastikan file sudah di-upload ke GitHub!")
else:
    # --- SIDEBAR FILTER ---
    st.sidebar.header("Pengaturan Peta")
    
    # Slider untuk Filter Magnitudo
    min_mag = st.sidebar.slider("Filter Magnitudo (Min):", min_value=3.0, max_value=9.0, value=4.5)
    
    # Slider untuk Jumlah Klaster (Zona)
    n_clusters = st.sidebar.slider("Jumlah Zona (Klaster):", min_value=2, max_value=10, value=5)

    # Filter Data berdasarkan input User
    df_filtered = df[df['magnitude'] >= min_mag].copy()
    
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"Peta Sebaran (Jumlah Data: {len(df_filtered)})")
        
        if len(df_filtered) > 0:
            # --- PROSES MACHINE LEARNING ---
            X = df_filtered[['latitude', 'longitude']]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_filtered['cluster'] = kmeans.fit_predict(X)

            # --- VISUALISASI FOLIUM ---
            # Inisialisasi Peta
            m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles='CartoDB dark_matter')
            
            # Warna untuk tiap klaster
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                      '#00FFFF', '#FFA500', '#800080', '#008000', '#800000']
            
            # Batasi data yang ditampilkan di peta agar tidak lemot (Maks 1000 titik)
            df_plot = df_filtered.tail(1000)
            
            for idx, row in df_plot.iterrows():
                cluster_id = int(row['cluster'])
                color_idx = cluster_id % len(colors)
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=row['magnitude'], # Ukuran = Magnitudo
                    color=colors[color_idx],
                    fill=True,
                    fill_color=colors[color_idx],
                    fill_opacity=0.7,
                    popup=f"Mag: {row['magnitude']} | Depth: {row['depth']} km"
                ).add_to(m)

            # Tampilkan peta di Streamlit
            st_folium(m, width=None, height=500)
            
            # Legend sederhana
            st.caption(f"Peta persebaran {n_clusters} zona gempa bumi.")
        else:
            st.warning("Tidak ada data gempa dengan magnitudo sebesar itu.")

    with col2:
        st.subheader("Statistik Data")
        st.write(df_filtered.describe())
        
        st.write("Data Mentah (Sampel):")
        st.dataframe(df_filtered[['latitude', 'longitude', 'magnitude', 'depth']].head(10))