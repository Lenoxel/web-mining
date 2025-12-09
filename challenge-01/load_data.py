import csv


def load_data(data: list[dict], file_type: str):
    if file_type == "csv":
        print("Loading data into CSV file: ofertas_calculadas.csv")

        keys = data[0].keys()

        with open(
            "ofertas_calculadas.csv", "w", newline="", encoding="utf-8"
        ) as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

        print("Data successfully loaded into ofertas_calculadas.csv")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
