import pandas as pd


def export_data(df: pd.DataFrame, file_type: str = "parquet") -> None:
    if file_type not in ["parquet", "csv"]:
        raise ValueError("file_type must be either 'parquet' or 'csv'")

    if file_type == "parquet":
        df.to_parquet("data/transformed_amazon_reviews_2023.parquet", index=False)

    if file_type == "csv":
        df.to_csv("data/transformed_amazon_reviews_2023.csv", index=False)

    return None
