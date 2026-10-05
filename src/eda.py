"""
eda.py
-------
Generates the exploratory data analysis visualisations required by the
assessment: target distribution, correlation heatmap, feature
distributions, boxplots and categorical bar charts. Figures are saved to
reports/figures/ so they can be embedded in the academic report and shown
during the demonstration.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import FIGURES_DIR, get_logger
from data_preprocessing import load_data, clean_data
from feature_engineering import engineer_features, create_target

FIG_DIR = FIGURES_DIR
logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(df["G3"], bins=21, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Distribution of Final Grade (G3)")
    axes[0].set_xlabel("Final Grade (0-20)")
    axes[0].set_ylabel("Number of Students")

    counts = df["pass_fail"].map({1: "PASS", 0: "FAIL"}).value_counts()
    axes[1].bar(counts.index, counts.values, color=["#55A868", "#C44E52"])
    axes[1].set_title("PASS / FAIL Class Distribution")
    axes[1].set_ylabel("Number of Students")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 3, str(v), ha="center")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_target_distribution.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False,
                linewidths=0.3, cbar_kws={"shrink": 0.7})
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame):
    features = ["studytime", "absences", "failures", "age"]
    fig, axes = plt.subplots(1, len(features), figsize=(16, 3.5))
    for ax, col in zip(axes, features):
        sns.histplot(df[col], bins=15, ax=ax, color="#8172B2")
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_feature_distributions.png", dpi=150)
    plt.close()


def plot_boxplots(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["studytime", "absences", "failures"]):
        sns.boxplot(x="pass_fail", y=col, data=df, ax=ax,
                    hue="pass_fail", legend=False,
                    palette={0: "#C44E52", 1: "#55A868"})
        ax.set_xticklabels(["FAIL", "PASS"])
        ax.set_title(f"{col} by Outcome")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_boxplots_by_outcome.png", dpi=150)
    plt.close()


def plot_categorical_bars(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    pass_rate_sex = df.groupby("sex")["pass_fail"].mean() * 100
    axes[0].bar(pass_rate_sex.index, pass_rate_sex.values, color="#4C72B0")
    axes[0].set_title("Pass Rate (%) by Sex")
    axes[0].set_ylabel("Pass Rate (%)")

    pass_rate_school = df.groupby("higher")["pass_fail"].mean() * 100
    axes[1].bar(pass_rate_school.index, pass_rate_school.values, color="#DD8452")
    axes[1].set_title("Pass Rate (%) by 'Wants Higher Education'")
    axes[1].set_ylabel("Pass Rate (%)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_categorical_bar_charts.png", dpi=150)
    plt.close()


def run_eda():
    df = clean_data(load_data())
    df = engineer_features(df)
    df = create_target(df)

    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    plot_categorical_bars(df)

    logger.info(f"EDA figures saved to: {FIG_DIR}")
    return df


if __name__ == "__main__":
    run_eda()
