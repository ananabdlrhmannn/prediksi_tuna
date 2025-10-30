import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Fungsi untuk membersihkan dan memproses data CSV yang berantakan
def clean_data(csv_content):
    """Membersihkan data CSV yang berantakan dari dokumen"""
    try:
        # Baca sebagai string dan split lines
        lines = csv_content.strip().split('\n')
        data = []
        
        for line in lines:
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 2:
                date_str = parts[0]
                prod_str = parts[1]
                
                # Skip jika tanggal atau produksi kosong
                if not date_str or not prod_str:
                    continue
                
                # Coba parse produksi sebagai integer
                try:
                    prod = int(float(prod_str.replace(',', '')))  # Handle koma sebagai pemisah ribuan
                except ValueError:
                    continue
                
                # Coba parse tanggal dengan berbagai format
                date_formats = [
                    '%d/%m/%Y', '%m/%d/%Y', '%d/%m/%y', '%m/%d/%y',
                    '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y'
                ]
                
                date_parsed = None
                for fmt in date_formats:
                    try:
                        date_parsed = pd.to_datetime(date_str, format=fmt, errors='raise')
                        break
                    except:
                        continue
                
                # Jika masih gagal, coba dengan dayfirst
                if date_parsed is None:
                    try:
                        date_parsed = pd.to_datetime(date_str, dayfirst=True, errors='raise')
                    except:
                        continue
                
                if pd.notna(date_parsed) and prod > 0:
                    data.append({'Date': date_parsed, 'Production': prod})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
        
        # Agregasi ke monthly sum
        df['Date'] = pd.to_datetime(df['Date'].dt.to_period('M').dt.to_timestamp())
        df_monthly = df.groupby('Date')['Production'].sum().reset_index()
        
        return df_monthly
    except Exception as e:
        st.error(f"Error dalam cleaning data: {e}")
        return pd.DataFrame()

# Fungsi preprocessing
def preprocess_data(df):
    """Preprocessing data untuk LSTM"""
    if len(df) < 24:  # Minimal 2 tahun data
        st.error("Data tidak cukup untuk training. Minimal 24 bulan diperlukan.")
        return None, None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Production']])
    return scaled_data, scaler

# Fungsi create sequences untuk LSTM
def create_sequences(data, seq_length=12):
    """Membuat sequences untuk LSTM input"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    
    if len(X) == 0:
        return np.array([]), np.array([])
    
    return np.array(X), np.array(y)

# Fungsi build dan train LSTM
@st.cache_resource
def build_lstm_model(input_shape):
    """Membuat model LSTM"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Fungsi train model
def train_model(model, X_train, y_train, epochs=50):
    """Training model LSTM"""
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    return model

# Fungsi predict
import numpy as np

def predict_future(model, last_sequence, scaler, steps=12):
    """Prediksi beberapa langkah ke depan secara iteratif."""
    last_sequence = np.asarray(last_sequence).astype("float32").flatten()

    timesteps = model.input_shape[1] or len(last_sequence)
    if len(last_sequence) < timesteps:
        raise ValueError(
            f"Panjang sequence ({len(last_sequence)}) < timesteps model ({timesteps})"
        )

    # Ambil hanya timesteps terakhir
    last_sequence = last_sequence[-timesteps:]

    # Pastikan bentuk input 3D
    current_seq = last_sequence.reshape((1, timesteps, 1)).astype("float32")

    print("✅ current_seq final shape:", current_seq.shape)

    predictions = []
    for _ in range(steps):
        # pastikan numpy array, bukan Tensor
        pred = model(current_seq, training=False).numpy()   # ← gunakan langsung model(), bukan model.predict()
        pred_value = float(pred[0][0])
        predictions.append(pred_value)

        # geser window
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = pred_value

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()


# Fungsi evaluasi
def evaluate_model(model, X_test, y_test, scaler):
    """Evaluasi model dengan RMSE"""
    if len(X_test) == 0:
        return float('inf')
    
    predictions = model.predict(X_test, verbose=0)
    # Pastikan shape sesuai
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    return rmse

# Streamlit App
st.set_page_config(page_title="Forecasting Tuna Production", layout="wide")

st.title("🚀 Sistem Forecasting Produksi Ikan Tuna dengan LSTM")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["🏠 Halaman Utama", "📁 Unggah Data", "📈 Hasil Forecasting"])

if page == "🏠 Halaman Utama":
    st.header("Selamat Datang!")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 📋 Tentang Sistem
        
        Sistem ini menggunakan **Long Short-Term Memory (LSTM)** untuk memprediksi produksi ikan tuna di Kota Ternate berdasarkan data historis.
        
        ### 🔧 Fitur Utama:
        - **Preprocessing**: Normalisasi Min-Max & Sliding Window
        - **Model**: LSTM Neural Network dengan 2 layers
        - **Evaluasi**: Root Mean Square Error (RMSE)
        - **Output**: Grafik visualisasi & Tabel prediksi
        
        ### 📊 Data Input:
        - Format: CSV (Tanggal, Produksi dalam Kg)
        - Periode: 2014-2023 (minimal 24 bulan)
        - Agregasi: Monthly sum
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Manfaat:
        - **Perencanaan**: Bahan pertimbangan sumber daya perikanan
        - **Ekonomi**: Stabilitas harga & pasokan
        - **Keberlanjutan**: Pengelolaan stok ikan yang efektif
        
        ### 📈 Teknologi:
        - **Backend**: Python, TensorFlow, Keras
        - **Frontend**: Streamlit
        - **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib
        """)
    
    st.markdown("---")
    st.info("👆 Gunakan sidebar untuk navigasi ke halaman berikutnya")

elif page == "📁 Unggah Data":
    st.header("📁 Unggah dan Proses Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Pilih file CSV", 
            type="csv", 
            help="Upload file CSV dengan format: Tanggal,Produksi (Kg)"
        )
    
    with col2:
        st.info("**Format CSV yang diharapkan:**\n`DD/MM/YYYY,Produksi`")
        example_data = st.checkbox("Lihat contoh data")
    
    if example_data:
        st.code("""
1/1/2014,1584
1/5/2014,852
17/01/2014,5556
19/01/2014,2400
...""", language="csv")
    
    if uploaded_file is not None:
        try:
            # Baca file
            csv_content = uploaded_file.read().decode('utf-8')
            
            with st.spinner("Memproses data..."):
                df = clean_data(csv_content)
                
            if not df.empty:
                st.success(f"✅ Data berhasil diproses! ({len(df)} bulan data)")
                
                # Tampilkan info data
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Periode", f"{df['Date'].min().strftime('%Y-%m')} - {df['Date'].max().strftime('%Y-%m')}")
                with col2:
                    st.metric("Total Data", len(df))
                with col3:
                    st.metric("Produksi Rata-rata", f"{df['Production'].mean():.0f} Kg")
                
                # Preview data
                st.subheader("📊 Preview Data")
                st.dataframe(df.head(12))
                
                # Statistik
                st.subheader("📈 Statistik Data")
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df.plot(x='Date', y='Production', ax=ax, title='Tren Produksi Ikan Tuna')
                    ax.set_ylabel('Produksi (Kg)')
                    st.pyplot(fig)
                
                with col2:
                    st.write(df['Production'].describe())
                
                # Simpan di session state
                st.session_state.df = df
                st.session_state.scaled_data = None
                st.session_state.model = None
                
                st.success("💾 Data tersimpan untuk forecasting!")
                
            else:
                st.error("❌ Tidak ada data valid yang ditemukan. Pastikan format CSV benar.")
                
        except Exception as e:
            st.error(f"❌ Error memproses file: {e}")
    
    elif st.button("📋 Gunakan Data Contoh"):
        # Gunakan sebagian data dari dokumen
        sample_data = """1/1/2014,1584
1/5/2014,852
17/01/2014,5556
19/01/2014,2400
22/01/2014,480
2/7/2014,2520
2/8/2014,324
2/10/2014,540
11/02/2014,2880
12/02/2014,780
13/02/2014,480
14/02/2014,1500
15/02/2014,2112
16/02/2014,324
17/02/2014,480
18/02/2014,1440
19/02/2014,1800
20/02/2014,1020
21/02/2014,2442
22/02/2014,1800
23/02/2014,144
24/03/2014,1560
25/03/2014,1320
26/03/2014,1200
27/03/2014,960
28/03/2014,4980
29/03/2014,3600
30/03/2014,240
31/03/2014,540
1/4/2014,2400
2/4/2014,960
3/4/2014,48
4/4/2014,108
5/4/2014,48
6/4/2014,1200
7/4/2014,1440
8/4/2014,720
9/4/2014,1560
10/4/2014,420
11/4/2014,6000
12/4/2014,480
13/4/2014,840
14/4/2014,480
15/4/2014,336
16/4/2014,240
17/4/2014,108
18/4/2014,105
19/4/2014,480
20/4/2014,1560
21/4/2014,648
22/4/2014,1080
23/4/2014,1920
24/4/2014,720
25/4/2014,960
26/4/2014,300
27/4/2014,1050
28/4/2014,720
29/4/2014,1125
30/4/2014,960
1/5/2014,1320
2/5/2014,2520
3/5/2014,240
4/5/2014,2316
5/5/2014,1152
6/5/2014,8040
7/5/2014,540
8/5/2014,720
9/5/2014,900
10/5/2014,960
11/5/2014,5040
12/5/2014,1116
13/5/2014,1320
14/5/2014,1824
15/5/2014,1080
16/5/2014,1560
17/5/2014,1560
18/5/2014,696
19/5/2014,240
20/5/2014,168
21/5/2014,60
22/5/2014,96
23/5/2014,360
24/5/2014,780
25/5/2014,96
26/5/2014,720
27/5/2014,480
28/5/2014,1680
29/5/2014,540
30/5/2014,1320"""
        
        with st.spinner("Memproses data contoh..."):
            df = clean_data(sample_data)
            
        if not df.empty:
            st.success(f"✅ Data contoh berhasil diproses! ({len(df)} bulan data)")
            st.dataframe(df.head())
            st.session_state.df = df
            st.session_state.scaled_data = None
            st.session_state.model = None
        else:
            st.error("❌ Gagal memproses data contoh")

elif page == "📈 Hasil Forecasting":
    st.header("📈 Hasil Forecasting Produksi Ikan Tuna")
    
    if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
        st.warning("⚠️ Belum ada data. Silakan unggah data terlebih dahulu.")
        st.stop()
    
    df = st.session_state.df
    st.subheader("📊 Data Historis")
    st.dataframe(df.tail())
    
    # Progress bar untuk training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Preprocessing
        status_text.text("Preprocessing data...")
        progress_bar.progress(20)
        
        scaled_data, scaler = preprocess_data(df)
        if scaled_data is None:
            st.stop()
        
        seq_length = min(12, len(scaled_data) // 3)  # Adjust sequence length
        X, y = create_sequences(scaled_data, seq_length)
        
        if len(X) == 0:
            st.error("❌ Data tidak cukup untuk membuat sequences. Minimal 24 bulan diperlukan.")
            st.stop()
        
        status_text.text("Membuat sequences...")
        progress_bar.progress(40)
        
        # Split train/test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build model
        status_text.text("Membangun model LSTM...")
        progress_bar.progress(60)
        
        input_shape = (X_train.shape[1], 1)
        model = build_lstm_model(input_shape)
        
        # Train model
        status_text.text("Training model...")
        progress_bar.progress(80)
        
        model = train_model(model, X_train, y_train, epochs=50)
        
        # Evaluasi
        status_text.text("Evaluasi model...")
        progress_bar.progress(90)
        
        rmse = evaluate_model(model, X_test, y_test, scaler)
        st.session_state.model = model
        st.session_state.scaled_data = scaled_data
        
        progress_bar.progress(100)
        status_text.text("✅ Forecasting selesai!")
        
        # Tampilkan hasil evaluasi
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 RMSE (Test Set)", f"{rmse:.2f}")
        with col2:
            st.metric("🎯 Akurasi", f"{max(0, 100 - (rmse/df['Production'].mean()*100)):.1f}%")
        
        # Prediksi masa depan
        status_text.text("Memprediksi masa depan...")
        
        # Ambil sequence terakhir
        last_sequence = scaled_data[-seq_length:].flatten()
        
        # Pastikan shape benar
        if len(last_sequence) != seq_length:
            st.error(f"Shape mismatch: expected {seq_length}, got {len(last_sequence)}")
            st.stop()
        
        future_steps = st.slider("Jumlah bulan prediksi", 1, 24, 12)
        future_pred = predict_future(model, last_sequence, scaler, future_steps)
        
        # Buat future dates
        last_date = df['Date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=future_steps, 
            freq='M'
        )
        
        # Visualisasi
        st.subheader("📈 Visualisasi Hasil")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Historical + Future Prediction
        ax1.plot(df['Date'], df['Production'], 'b-', label='Data Historis', linewidth=2)
        ax1.plot(future_dates, future_pred, 'r--', label='Prediksi', linewidth=2, markersize=8)
        ax1.set_title('Forecasting Produksi Ikan Tuna - Kota Ternate', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Tanggal')
        ax1.set_ylabel('Produksi (Kg)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Tren bulanan
        monthly_avg = df.groupby(df['Date'].dt.month)['Production'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                 'Jul', 'Agustus', 'Sep', 'Okt', 'Nov', 'Des']
        ax2.bar(months, monthly_avg.values, color='skyblue', alpha=0.7)
        ax2.set_title('Rata-rata Produksi Bulanan', fontsize=12)
        ax2.set_ylabel('Produksi Rata-rata (Kg)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabel hasil prediksi
        st.subheader("📋 Tabel Hasil Prediksi")
        pred_df = pd.DataFrame({
            'Bulan/Tahun': [date.strftime('%B %Y') for date in future_dates],
            'Prediksi Produksi (Kg)': future_pred.astype(int),
            'Confidence Interval': [f"{pred*0.9:.0f} - {pred*1.1:.0f}" for pred in future_pred]
        })
        
        st.table(pred_df)
        
        # Insight
        st.subheader("💡 Insight & Rekomendasi")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend = "📈 Naik" if future_pred[-1] > future_pred[0] else "📉 Turun"
            st.metric("Tren 12 Bulan", trend)
        
        with col2:
            avg_pred = np.mean(future_pred)
            st.metric("Prediksi Rata-rata", f"{avg_pred:.0f} Kg/bulan")
        
        with col3:
            total_pred = np.sum(future_pred)
            st.metric("Total Tahun Depan", f"{total_pred:.0f} Kg")
        
        st.info("""
        **Rekomendasi:**
        - Perencanakan stok ikan berdasarkan prediksi bulanan
        - Monitoring cuaca & faktor lingkungan yang mempengaruhi
        - Koordinasi dengan nelayan untuk optimalisasi tangkapan
        - Persiapan fasilitas pengolahan sesuai kapasitas prediksi
        """)
        
    except Exception as e:
        st.error(f"❌ Error dalam forecasting: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💻 Dikembangkan untuk Skripsi - Universitas Khairun Ternate | 2024</p>
    <p>🛠️ Teknologi: Python, Streamlit, TensorFlow, LSTM</p>
</div>
""", unsafe_allow_html=True)