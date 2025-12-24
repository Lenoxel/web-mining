import pandas as pd


def export_data(file_path: str, df: pd.DataFrame, file_type: str = "parquet") -> None:
    if file_type not in ["parquet", "csv"]:
        raise ValueError("file_type must be either 'parquet' or 'csv'")

    if file_type == "parquet":
        df.to_parquet(file_path, index=False)

    if file_type == "csv":
        df.to_csv(file_path, index=False)

    return None
