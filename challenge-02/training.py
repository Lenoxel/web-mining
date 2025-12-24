from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_classifier_model(classifier_model_name: "str"):
    classifier_model_map = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "SupportVectorMachine": LinearSVC(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
    }

    return classifier_model_map.get(classifier_model_name, MultinomialNB())


def train_model(
    reviews_df: pd.DataFrame,
    classifier_model_name: "str" = "MultinomialNB",
):
    print("-" * 40)
    print("Training the model...")

    print("\n")

    classifier_model = get_classifier_model(classifier_model_name)
    print(f"Using classifier model: {classifier_model.__class__.__name__}")

    print("\n")

    print(f"Splitting data into training and testing sets...")

    X = reviews_df["text"]
    y = reviews_df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(min_df=5, max_features=20000, ngram_range=(1, 2))),
            ("tfidf", TfidfTransformer()),
            ("clf", classifier_model),
        ]
    )

    print("Fitting the model...")

    pipeline.fit(X_train, y_train)

    print("Evaluating the model...")

    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))

    print("\n")

    print("Model training completed.")
    print("-" * 40)

    return y_test, y_pred, pipeline


def plot_confusion_matrix(y_test, y_pred, classifier_model_name: "str"):
    confusion_matrix_generated = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_generated,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {classifier_model_name}")

    plt.show()
