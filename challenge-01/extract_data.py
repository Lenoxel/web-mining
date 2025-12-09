from bs4 import BeautifulSoup


def extract_data(file_path: str) -> list[dict]:
    print(f"Extracting data from {file_path}...")

    with open(file_path, "r") as file:
        data = file.read()

        if not data:
            print("No data found in the file.")
            return

        soup = BeautifulSoup(data, "html.parser")

        cards = soup.find_all(class_="product-card")

        extracted_data = []

        for card in cards:
            price = (
                card.find(class_="price").get_text(strip=True)
                if card.find(class_="price")
                else None
            )

            if not price:
                id = card.get("id", "unknown")

                print(f"Price not found for product with id {id}, skipping...")

                continue

            title = card.find(name="h2").get_text(strip=True)

            rating = card.find(class_="rating").get_text(strip=True)

            discount_rate = (
                card.find(class_="discount-rate").get_text(strip=True)
                if card.find(class_="discount-rate")
                else "No discount"
            )

            date = card.get("data-date", "unknown")

            url_image = card.find(name="img")["src"]

            product_info = {
                "title": title,
                "price": price,
                "rating": rating,
                "discount": discount_rate,
                "date": date,
                "image_url": url_image,
            }

            extracted_data.append(product_info)

        print(f"Extracted {len(extracted_data)} products.")

        return extracted_data
