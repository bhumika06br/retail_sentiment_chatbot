import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("Loading dataset...")

df = pd.read_csv("data/amazon_reviews.csv")

df = df[["Review_text", "Own_Rating"]]

df = df.dropna()

df = df[df["Review_text"].str.strip() != ""]

print("Total reviews:", len(df))


X = df["Review_text"]
y = df["Own_Rating"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=50000
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced"
        )
    )
])


print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


joblib.dump(model, "models/sentiment_model.pkl")

print("\nModel saved successfully!")
print("Location: models/sentiment_model.pkl")