import re


def transform_data(data: list[dict]) -> list[dict]:
    print("Transforming data...")

    transformed_data = []

    for item in data:
        price_numeric = re.sub(r"[^\d,]", "", item["price"]).replace(",", ".")
        item["price"] = round(float(price_numeric), 2)

        discount_str = item["discount"]
        if discount_str != "No discount":
            discount_numeric = re.sub(r"[^\d,]", "", discount_str).replace(",", ".")
            item["discount"] = float(discount_numeric)
        else:
            item["discount"] = 0.0

        rating_str = item["rating"]
        rating_parts = re.split(r"/", rating_str)

        rating_normalized = float(rating_parts[0]) / float(rating_parts[1]) * 100
        item["rating"] = rating_normalized

        transformed_data.append(item)

        date_str = item["date"]
        item["date"] = date_str.replace("/", "-")

        net_price = round(item["price"] - (item["price"] * item["discount"] / 100), 2)
        item["net_price"] = net_price

    print("Data transformation complete.")

    return transformed_data
