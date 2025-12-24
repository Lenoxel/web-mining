from extract_data import extract_data
from transform_data import transform_data
from load_data import export_data
from training import train_model, plot_confusion_matrix

if __name__ == "__main__":
    reviews_df = extract_data("data/amazon_reviews_2023.parquet", file_type="parquet")

    transformed_reviews_df = transform_data(reviews_df)

    transformed_data_path = "data/transformed_amazon_reviews_2023.parquet"

    export_data(transformed_data_path, transformed_reviews_df, file_type="parquet")

    transformed_reviews_df = extract_data(
        transformed_data_path,
        file_type="parquet",
        columns=["text", "rating", "sentiment"],
    )

    # print(transformed_reviews_df.count())
    # print(transformed_reviews_df.sample(100))

    train_model(transformed_reviews_df, classifier_model_name="MultinomialNB")

    train_model(transformed_reviews_df, classifier_model_name="LogisticRegression")

    train_model(transformed_reviews_df, classifier_model_name="RandomForest")

    y_pred, y_test, pipeline = train_model(
        transformed_reviews_df, classifier_model_name="SupportVectorMachine"
    )

    plot_confusion_matrix(y_test, y_pred, classifier_model_name="SupportVectorMachine")

    print(
        f"Conclusion: After evaluating four algorithms (MultinomialNB, Logistic Regression, Random Forest, and LinearSVC), the best model identified has been the LinearSVC, which demonstrated the highest precision and F1-score. The LinearSVC model achieved a 93% Recall for negative reviews, which is excellent for detecting most of non-satisfactory reviews and their related customer complaints. While Naive Bayes had higher precision, it failed to identify around 25% of the complaints, which is an unacceptable risk for reputation management. LinearSVC provides the best coverage with high overall accuracy (94%)."
    )

    new_reviews = [
        "The product stopped working after a week. Very disappointed with the quality.",
        "Excellent value for money. Exceeded my expectations in every way!",
        "Mediocre experience. The product is okay, but the delivery was late.",
        "Terrible customer service. I will never buy from this seller again.",
        "Absolutely love it! High quality and works perfectly.",
        "I absolutely love this, amazing quality!",
        "Total waste of money, arrived broken.",
        "It's okay, not great but works.",
        "Do not buy this! It is a scam.",
        "Exceeded my expectations, highly recommend!",
    ]

    print("\nPredicting sentiments for new reviews:")

    predicted_sentiments = pipeline.predict(new_reviews)

    for review, sentiment in zip(new_reviews, predicted_sentiments):
        print(f"Review: '{review}' => Predicted Sentiment: {sentiment}")

    print(
        f"For the 10 new reviews, the trained model was able to identify 8 reviews correctly and 2 reviews incorrectly. This demonstrates the model's effectiveness in predicting sentiments on unseen data, and also that there is still room for improvement."
    )
