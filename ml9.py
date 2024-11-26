import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from wordcloud import WordCloud
import pickle

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Judul Aplikasi
st.title("🚗 Aplikasi Prediksi Harga Mobil 🚗")
st.markdown("Prediksi harga mobil berdasarkan spesifikasi menggunakan **Machine Learning**.")

# Navigasi Sidebar
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman", ["Dataset", "Visualisasi", "Prediksi Harga"], index=0
)

# Load dataset
df = pd.read_csv("CarPrice_Assignment.csv")
st.sidebar.write(f"**Total Data**: {len(df):,}")

# Fungsi untuk menyimpan model
def save_model(model, filename="model_prediksi_harga_mobil.sav"):
    pickle.dump(model, open(filename, "wb"))

# Fungsi untuk memuat model
def load_model(filename="model_prediksi_harga_mobil.sav"):
    return pickle.load(open(filename, "rb"))

# Styling CSS untuk meningkatkan tampilan
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #FF7373;
        }
        .stNumberInput>div>input {
            border: 2px solid #FF4B4B;
            border-radius: 8px;
            font-size: 16px;
        }
        .stDataFrame {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Halaman Dataset
if menu == "Dataset":
    st.header("📊 Dataset Mobil")
    with st.expander("Lihat Data"):
        st.dataframe(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistik Deskriptif")
        st.write(df.describe())
    with col2:
        st.subheader("Data Kosong")
        st.write(df.isnull().sum())

# Halaman Visualisasi
elif menu == "Visualisasi":
    st.header("📈 Visualisasi Data Mobil")

    st.subheader("Distribusi Harga Mobil")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["price"], kde=True, color="blue", ax=ax)
    ax.set_title("Distribusi Harga Mobil", fontsize=16, fontweight="bold")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Nama Mobil")
        car_counts = df["CarName"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        car_counts.plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title("Top 10 Mobil Berdasarkan Jumlah", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Hubungan Highway MPG dan Harga")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df["highwaympg"], y=df["price"], color="purple", ax=ax)
        ax.set_title("Highway MPG vs Harga Mobil", fontsize=16, fontweight="bold")
        st.pyplot(fig)

    st.subheader("Word Cloud Nama Mobil")
    all_cars = " ".join(df["CarName"].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_cars)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)

# Halaman Prediksi Harga
elif menu == "Prediksi Harga":
    st.header("🔮 Prediksi Harga Mobil")

    # Split data untuk pelatihan model
    X = df[["highwaympg", "curbweight", "horsepower"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model
    model = LinearRegression()
    model.fit(X_train, y_train)
    save_model(model)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Masukkan Spesifikasi Mobil:")
        highway_mpg = st.number_input(
            "Highway MPG", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="MPG di jalan tol"
        )
        curbweight = st.number_input(
            "Curb Weight", min_value=0, max_value=5000, value=2500, step=10, help="Berat kendaraan"
        )
        horsepower = st.number_input(
            "Horsepower", min_value=0, max_value=500, value=150, step=5, help="Tenaga mesin"
        )

        if st.button("Prediksi Harga"):
            loaded_model = load_model()
            input_data = pd.DataFrame(
                {"highwaympg": [highway_mpg], "curbweight": [curbweight], "horsepower": [horsepower]}
            )
            predicted_price = loaded_model.predict(input_data)[0]
            st.success(f"Harga mobil yang diprediksi: **${predicted_price:,.2f}**")

    with col2:
        st.write("### Evaluasi Model:")
        model_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, model_pred)
        mse = mean_squared_error(y_test, model_pred)
        rmse = np.sqrt(mse)

        st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
