# plot_trends.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_value_distribution(df: pd.DataFrame, column: str):
    """
    Plots distribution of numerical feature (e.g., transfer_value).
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, color="royalblue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_yearly_trend(df: pd.DataFrame, year_col: str, value_col: str):
    """
    Plots transfer values over years.
    """
    yearly = df.groupby(year_col)[value_col].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=yearly, x=year_col, y=value_col, marker="o", linewidth=2)
    plt.title(f"Average {value_col} Over Years")
    plt.xlabel("Year")
    plt.ylabel(f"Avg {value_col}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Shows correlation heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
