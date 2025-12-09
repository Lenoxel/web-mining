from extract_data import extract_data
from transform_data import transform_data
from load_data import load_data

if __name__ == "__main__":
    file_path = "products_data.html"

    data = extract_data(file_path)

    transformed_data = transform_data(data)

    load_data(transformed_data, file_type="csv")
