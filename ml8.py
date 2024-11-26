import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sidebar
st.sidebar.title("Select Page")
menu = st.sidebar.selectbox("Menu", ["Home", "Dataset", "Chart"])

# Switch-case implementation
if menu == "Home":
    st.image("image_path.jpg", use_column_width=True)  # Ganti dengan path gambar Anda
    st.title("Welcome to the Loan Application Dashboard")
    st.write("This is the home page showcasing an overview of the application.")

elif menu == "Dataset":
    # Load dataset
    try:
        df = pd.read_csv("seattle-weather.csv")  # Ganti dengan path file CSV Anda
        st.title("Dataset")
        st.dataframe(df.head(10))
    except FileNotFoundError:
        st.error("Dataset not found! Please provide the correct path.")

elif menu == "Chart":
    # Load dataset and generate chart
    try:
        df = pd.read_csv("seattle-weather.csv")  # Ganti dengan path file CSV Anda
        st.title("Applicant Income VS Loan Amount")

        # Plot
        fig, ax = plt.subplots()
        ax.bar(df["temp_max"], df["temp_min"])
        ax.set_xlabel("temp_max")
        ax.set_ylabel("temp_min")
        st.pyplot(fig)
    except FileNotFoundError:
        st.error("Dataset not found! Please provide the correct path.")
    except KeyError:
        st.error("Dataset columns not found! Ensure 'ApplicantIncome' and 'LoanAmount' exist.")

else:
    st.error("Invalid menu selection.")
