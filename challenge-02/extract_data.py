import pandas as pd


def extract_data(
    file_path: str,
    file_type: str = "parquet",
    columns=["title", "text", "verified_purchase", "rating"],
) -> pd.DataFrame:
    if file_type not in ["parquet", "csv"]:
        raise ValueError("file_type must be either 'parquet' or 'csv'")

    if file_type == "parquet":
        df = pd.read_parquet(file_path)

    if file_type == "csv":
        df = pd.read_csv(file_path)

    return df[columns]
