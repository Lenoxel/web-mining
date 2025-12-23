from extract_data import extract_data
from transform_data import transform_data
from load_data import export_data

if __name__ == "__main__":
    # reviews_df = extract_data("data/amazon_reviews_2023.parquet", file_type="parquet")

    # transformed_reviews_df = transform_data(reviews_df)

    # export_data(transformed_reviews_df, file_type="parquet")

    transformed_reviews_df = extract_data(
        "data/transformed_amazon_reviews_2023.parquet",
        file_type="parquet",
        columns=["text", "rating"],
    )
    print(transformed_reviews_df.count())
    print(transformed_reviews_df.head(50))
    print(transformed_reviews_df.tail(50))
