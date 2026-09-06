import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(".")


# USE PREPROCESSED DATASET
DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing"
    / "manual_imputed"
    / "dataset.csv"
)


# SAVE OUTPUTS HERE
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_check_new"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


TARGET_COLUMN = "Alzheimer’s Diagnosis"


NUMERIC_FEATURES = [
    "Age",
    "BMI",
    "Education Level",
    "Cognitive Test Score",
]


CATEGORICAL_FEATURES = [
    "Country",
    "Gender",
    "Physical Activity Level",
    "Smoking Status",
    "Alcohol Consumption",
    "Diabetes",
    "Hypertension",
    "Cholesterol Level",
    "Family History of Alzheimer’s",
    "Depression Level",
    "Sleep Quality",
    "Dietary Habits",
    "Air Pollution Exposure",
    "Employment Status",
    "Marital Status",
    "Genetic Risk Factor (APOE-ε4 allele)",
    "Social Engagement Level",
    "Income Level",
    "Stress Levels",
    "Urban vs Rural Living",

    # ENGINEERED FEATURES
    "Age Group",
    "BMI Category",
    "Cognitive Score Group",
]


def ensure_output_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    return pd.read_csv(DATASET_PATH)


def clean_numeric_columns(df):
    df = df.copy()

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def safe_filename(name):
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("’", "")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace("ε", "epsilon")
        .replace("-", "_")
    )


def save_basic_tables(df):
    pd.DataFrame({
        "column_name": df.columns
    }).to_csv(
        TABLES_DIR / "column_names.csv",
        index=False
    )

    pd.DataFrame({
        "column_name": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isnull().sum().values,
        "missing_percent": (df.isnull().mean() * 100).values,
    }).to_csv(
        TABLES_DIR / "column_summary.csv",
        index=False
    )

    df.describe(include="all").T.to_csv(
        TABLES_DIR / "descriptive_statistics.csv"
    )


def save_target_distribution(df):
    counts = df[TARGET_COLUMN].value_counts(dropna=False)

    counts.to_csv(
        TABLES_DIR / "target_distribution.csv"
    )

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Alzheimer’s Diagnosis Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "target_distribution.png",
        dpi=300
    )

    plt.close()


def save_numeric_histograms(df):
    for col in NUMERIC_FEATURES:

        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        plt.figure(figsize=(7, 4))

        if col in ["Age", "Education Level"]:
            bins = range(
                int(values.min()),
                int(values.max()) + 2
            )

            plt.hist(
                values,
                bins=bins,
                edgecolor="black"
            )

        else:
            plt.hist(
                values,
                bins=20,
                edgecolor="black"
            )

        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"{safe_filename(col)}_distribution.png",
            dpi=300
        )

        plt.close()


def save_numeric_boxplots_by_target(df):
    for col in NUMERIC_FEATURES:

        if col not in df.columns:
            continue

        plot_df = df[
            [TARGET_COLUMN, col]
        ].copy()

        plot_df[col] = pd.to_numeric(
            plot_df[col],
            errors="coerce"
        )

        plot_df = plot_df.dropna(
            subset=[TARGET_COLUMN, col]
        )

        if plot_df.empty:
            continue

        classes = ["No", "Yes"]

        data = []
        labels = []

        for cls in classes:

            values = plot_df.loc[
                plot_df[TARGET_COLUMN].astype(str) == cls,
                col
            ].dropna().values

            if len(values) > 0:
                data.append(values)
                labels.append(cls)

        if not data:
            continue

        plt.figure(figsize=(7, 4))

        plt.boxplot(data)

        plt.xticks(
            range(1, len(labels) + 1),
            labels
        )

        plt.title(
            f"{col} by Alzheimer’s Diagnosis"
        )

        plt.xlabel("Diagnosis")
        plt.ylabel(col)


        plt.savefig(
            FIGURES_DIR / f"{safe_filename(col)}_vs_diagnosis.png",
            dpi=300
        )

        plt.close()


def save_categorical_target_rates(df):
    for col in CATEGORICAL_FEATURES:

        if col not in df.columns:
            continue

        rate_table = pd.crosstab(
            df[col],
            df[TARGET_COLUMN],
            normalize="index"
        )

        rate_table.to_csv(
            TABLES_DIR / f"{safe_filename(col)}_diagnosis_rates.csv"
        )

        if "Yes" not in rate_table.columns:
            continue

        yes_rate = rate_table["Yes"].sort_values(
            ascending=False
        )

        plt.figure(figsize=(10, 7))

        plt.bar(
            yes_rate.index.astype(str),
            yes_rate.values
        )

        plt.title(
            f"Alzheimer’s Diagnosis Rate by {col}"
        )

        plt.xlabel(col)
        plt.ylabel("Diagnosis Rate")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"{safe_filename(col)}_yes_rate.png",
            dpi=300
        )

        plt.close()


def save_correlation_heatmap(df):
    existing_numeric = [
        col for col in NUMERIC_FEATURES
        if col in df.columns
    ]

    if len(existing_numeric) < 2:
        return

    numeric_df = df[
        existing_numeric
    ].copy()

    corr = numeric_df.corr()

    corr.to_csv(
        TABLES_DIR / "numeric_correlation.csv"
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(
        corr,
        vmin=-1,
        vmax=1
    )

    plt.colorbar()

    plt.xticks(
        range(len(existing_numeric)),
        existing_numeric,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(existing_numeric)),
        existing_numeric
    )

    plt.title(
        "Numeric Feature Correlation"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "numeric_correlation.png",
        dpi=300
    )

    plt.close()


def main():
    ensure_output_dirs()

    df = load_dataset()
    df = clean_numeric_columns(df)

    print("\nDataset shape:", df.shape)

    save_basic_tables(df)
    save_target_distribution(df)
    save_numeric_histograms(df)
    save_numeric_boxplots_by_target(df)
    save_categorical_target_rates(df)
    # save_correlation_heatmap(df)

    print("\nNew dataset visuals completed.")
    print(f"Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()