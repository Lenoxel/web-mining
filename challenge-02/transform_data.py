import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk
import emoji
import pandas as pd
from pandarallel import pandarallel

pd.set_option("display.max_colwidth", None)

pandarallel.initialize(progress_bar=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


def handle_not_verified_purchase(df: pd.DataFrame) -> pd.DataFrame:
    print(f'{"*" * 10} Handling "not verified purchase" reviews{"*" * 10}')

    verified_purchase_possible_values = df["verified_purchase"].unique()
    print(
        f"Possible values for 'verified_purchase': {verified_purchase_possible_values}"
    )

    reviews_with_verified_purchase = df[df["verified_purchase"] == True]

    average_rating_verified = reviews_with_verified_purchase["rating"].mean()
    print(f"Average rating for verified purchases: {average_rating_verified:.2f}")

    reviews_with_verified_purchase_count = reviews_with_verified_purchase.shape[0]
    print(
        f"Number of reviews with verified purchase: {reviews_with_verified_purchase_count}"
    )

    reviews_without_verified_purchase = df[df["verified_purchase"] == False]

    average_rating_unverified = reviews_without_verified_purchase["rating"].mean()
    print(f"Average rating for unverified purchases: {average_rating_unverified:.2f}")

    reviews_without_verified_purchase_count = reviews_without_verified_purchase.shape[0]
    print(
        f"Number of reviews without verified purchase: {reviews_without_verified_purchase_count}"
    )

    overall_average_rating = df["rating"].mean()
    print(f"Overall average rating: {overall_average_rating:.2f}")

    print("\n")

    reviews_without_verified_purchase_rating_distribution = (
        reviews_without_verified_purchase["rating"].value_counts().sort_index()
    )
    percentages_unverified = (
        reviews_without_verified_purchase_rating_distribution
        / reviews_without_verified_purchase_count
        * 100
    ).round(2)
    print(
        "Rating distribution for unverified purchases (count and percentage):\n"
        f"{reviews_without_verified_purchase_rating_distribution}\n"
        f"{percentages_unverified}"
    )

    reviews_with_verified_purchase_rating_distribution = (
        reviews_with_verified_purchase["rating"].value_counts().sort_index()
    )
    percentages_verified = (
        reviews_with_verified_purchase_rating_distribution
        / reviews_with_verified_purchase_count
        * 100
    ).round(2)

    print("\n")

    print(
        "Rating distribution for verified purchases (count and percentage):\n"
        f"{reviews_with_verified_purchase_rating_distribution}\n"
        f"{percentages_verified}"
    )

    print(
        "\nConclusion: Reviews from unverified purchases tend to have higher ratings compared to verified purchases. But they are fewer in number and, considering the overall average rating, they do not significantly skew the overall ratings. That said, their presence should still be taken into account when analyzing review data."
    )

    return df


def handle_rating(df: pd.DataFrame) -> pd.DataFrame:
    print("\n")

    print(f'{"*" * 10} Handling "rating" column{"*" * 10}')

    rating_possible_values = df["rating"].unique()
    print(f"Possible values for 'rating': {rating_possible_values}")

    rating_counts = df["rating"].value_counts().sort_index()
    print(f"Rating counts:\n{rating_counts}")

    average_rating = df["rating"].mean()
    print(f"Average rating: {average_rating:.2f}")

    print("\n")

    df = df[df["rating"] != 3].copy()

    df["sentiment"] = df["rating"].parallel_apply(
        lambda x: "positive" if x > 3 else "negative"
    )

    print(
        "\nConclusion: Neutral ratings (3 stars) have been removed from the dataset to focus on clear positive and negative sentiments. A new 'sentiment' column has been added to categorize reviews based on their ratings."
    )

    return df


def compare_positive_negative_reviews(df: pd.DataFrame) -> pd.DataFrame:
    print("\n")

    print(f'{"*" * 10} Comparing positive and negative reviews{"*" * 10}')

    sentiment_counts = df["sentiment"].value_counts()
    print(f"Sentiment counts:\n{sentiment_counts}")

    average_ratings_by_sentiment = df.groupby("sentiment")["rating"].mean()
    print(f"Average ratings by sentiment:\n{average_ratings_by_sentiment}")

    print("\n")

    positive_reviews = df[df["sentiment"] == "positive"]
    negative_reviews = df[df["sentiment"] == "negative"]

    positive_review_lengths = positive_reviews["text"].str.len()
    negative_review_lengths = negative_reviews["text"].str.len()

    average_length_positive = positive_review_lengths.mean()
    median_length_positive = positive_review_lengths.median()

    average_length_negative = negative_review_lengths.mean()
    median_length_negative = negative_review_lengths.median()

    longest_positive_review = positive_reviews.loc[positive_review_lengths.idxmax()]
    longest_negative_review = negative_reviews.loc[negative_review_lengths.idxmax()]

    print(
        f"Average review length for positive reviews: {average_length_positive:.2f} characters"
    )
    print(
        f"Median review length for positive reviews: {median_length_positive:.2f} characters"
    )
    print(
        f"Longest positive review - length: {len(longest_positive_review['text'])} characters)"
    )
    print(
        f"Average review length for negative reviews: {average_length_negative:.2f} characters"
    )
    print(
        f"Median review length for negative reviews: {median_length_negative:.2f} characters"
    )
    print(
        f"Longest negative review - length: {len(longest_negative_review['text'])} characters"
    )

    plt.figure(figsize=(10, 6))
    sns.violinplot(x="sentiment", y=df["text"].str.len(), data=df)
    plt.title("Distribution of Review Lengths by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Review Length (characters)")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x=df["text"].str.len(),
        hue="sentiment",
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Histogram of Review Lengths by Sentiment")
    plt.xlabel("Review Length (characters)")
    plt.ylabel("Density")
    plt.show()

    print(
        "\nConclusion: Negative reviews tend to have a slightly longer median length compared to positive reviews, but the average lengths are quite similar. This suggests that there is no significant difference in the verbosity of positive versus negative reviews."
    )

    return df


def handle_not_english_reviews(df: pd.DataFrame) -> pd.DataFrame:
    print("\n")

    print(f'{"*" * 10} Handling "not English" reviews{"*" * 10}')

    # Implement logic here

    return df


def handle_text(df: pd.DataFrame) -> pd.DataFrame:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import SnowballStemmer

    print("\n")

    print(f'{"*" * 10} Handling "text" column{"*" * 10}')

    empty_texts = df[df["text"].str.strip() == ""]
    num_empty_texts = empty_texts.shape[0]
    print(f"Number of empty texts: {num_empty_texts}")

    empty_titles = df[df["title"].str.strip() == ""]
    num_empty_titles = empty_titles.shape[0]
    print(f"Number of empty titles: {num_empty_titles}")

    print(
        f"Once there are {num_empty_texts} empty texts and {num_empty_titles} empty titles, they will be combined onto the 'text' column."
    )

    df["text"] = df["text"].fillna("")
    df["title"] = df["title"].fillna("")

    df["text"] = df["title"] + " " + df["text"]

    print("Dropping 'title' column after merging with 'text' column.")
    df.drop(columns=["title"], inplace=True)

    print("Converting text to lowercase...")
    df["text"] = df["text"].str.lower()

    print("Removing HTMl tags from text...")
    df["text"] = df["text"].str.replace(r"<.*?>", " ", regex=True)

    print("Demojizing emojis in text...")
    df["text"] = df["text"].parallel_apply(
        lambda text: emoji.demojize(text, language="en")
    )

    print("Removing punctuation and special characters from text...")
    df["text"] = df["text"].str.replace(r"[\.\/\-]", " ", regex=True)

    df["text"] = df["text"].str.translate(str.maketrans("", "", string.punctuation))

    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    print("Tokenizing text, removing stop words, and applying stemming...")

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    def process_tokens(text):
        tokens = word_tokenize(text)

        processed_tokens = [
            stemmer.stem(token)
            for token in tokens
            if token not in stop_words and len(token) > 2
        ]

        return " ".join(processed_tokens)

    df["text"] = df["text"].parallel_apply(process_tokens)

    return df


def transform_data(df):
    print("-" * 40)
    print("Transforming data...")

    print("\n")

    df = handle_not_verified_purchase(df)

    df = handle_rating(df)

    df = compare_positive_negative_reviews(df)

    df = handle_not_english_reviews(df)

    df = handle_text(df)

    print("\n")

    print("Data transformation complete.")
    print("-" * 40)

    return df
