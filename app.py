import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# 1. Judul dan Deskripsi Web
st.set_page_config(page_title="Peta Zonasi Gempa", layout="wide")
st.title("🌏 Analisis Zonasi Gempa Bumi Indonesia")
st.markdown("Web App ini mengelompokkan wilayah gempa menggunakan algoritma **K-Means Clustering**.")

# 2. Load Data (dengan Cache agar cepat)
@st.cache_data
def load_data():
    # Pastikan file tsv ada di satu folder dengan app.py
    df = pd.read_csv('katalog_gempa_v2.tsv', sep='\t')
    df_clean = df[['latitude', 'longitude', 'magnitude', 'depth']].dropna()
    # Filter awal agar data tidak terlalu berat
    df_clean = df_clean[df_clean['magnitude'] >= 3.0]
    return df_clean

# Memanggil fungsi load data
df = load_data()

# 3. Sidebar (Fitur Interaktif)
st.sidebar.header("Filter Data")
min_mag = st.sidebar.slider("Minimal Magnitudo:", 3.0, 9.0, 4.5)
num_clusters = st.sidebar.slider("Jumlah Klaster (Zona):", 2, 10, 5)

# Filter data berdasarkan input user
df_filtered = df[df['magnitude'] >= min_mag].copy()

st.write(f"Menampilkan **{len(df_filtered)}** data gempa dengan Magnitudo >= {min_mag}")

# 4. Machine Learning (K-Means)
if len(df_filtered) > 0:
    # Clustering
    X = df_filtered[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_filtered['cluster'] = kmeans.fit_predict(X)

    # 5. Visualisasi Peta
    m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles='CartoDB dark_matter')
    
    # Warna-warni klaster
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
              '#00FFFF', '#FFA500', '#800080', '#008000', '#800000']
    
    # Ambil sampel agar rendering tidak berat di web (maks 2000 titik)
    limit_data = 2000
    df_plot = df_filtered.tail(limit_data)

    for idx, row in df_plot.iterrows():
        cluster_id = int(row['cluster'])
        color_idx = cluster_id % len(colors) # Agar warna looping jika klaster banyak
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['magnitude'],
            color=colors[color_idx],
            fill=True,
            fill_color=colors[color_idx],
            fill_opacity=0.7,
            popup=f"Mag: {row['magnitude']} | Depth: {row['depth']}km"
        ).add_to(m)

    # Menampilkan Peta di Streamlit
    st_folium(m, width=1000, height=600)

    # Menampilkan Tabel Data di bawah peta
    with st.expander("Lihat Data Mentah"):
        st.dataframe(df_filtered)
else:
    st.warning("Tidak ada data gempa dengan kriteria tersebut.")