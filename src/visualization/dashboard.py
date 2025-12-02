# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.utils import load_scaler
from src.models.lstm_model import predict_lstm
from src.models.ensemble_model import EnsembleModel


def load_data(path="data/processed/final_dataset.csv"):
    return pd.read_csv(path)


def show_overview(df):
    st.header("📊 Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())


def show_trend_plot(df):
    st.header("📈 Player Trait Trend (Yearly)")
    if "year" in df.columns and "transfer_value" in df.columns:
        yearly = df.groupby("year")["transfer_value"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=yearly, x="year", y="transfer_value", marker="o", ax=ax)
        ax.set_title("Average Transfer Value Over Years")
        st.pyplot(fig)
    else:
        st.warning("Year or transfer_value column missing.")


def show_prediction_section(model, df):
    st.header("🔮 Predict Transfer Value")
    numeric_cols = df.select_dtypes(include="number").columns

    inputs = {}
    for col in numeric_cols:
        inputs[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].median()))

    input_df = pd.DataFrame([inputs])

    st.write("Input:", input_df)

    # prediction available only for ensemble
    if st.button("Predict"):
        val = model.predict(
            lstm_preds_test=[10],  # placeholder
            xgb_preds_test=[20]    # placeholder
        )
        st.success(f"Predicted Transfer Value: €{round(val[0], 2)}M")


def main():
    st.title("⚽ TransferIQ — Player Transfer Value Dashboard")

    df = load_data()
    model = EnsembleModel()  # placeholder, should load trained model later

    menu = ["Overview", "Yearly Trends", "Predict"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Overview":
        show_overview(df)

    elif choice == "Yearly Trends":
        show_trend_plot(df)

    elif choice == "Predict":
        show_prediction_section(model, df)


if __name__ == "__main__":
    main()
