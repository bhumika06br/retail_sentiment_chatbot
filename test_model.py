import joblib

print("Loading sentiment model...")

model = joblib.load("models/sentiment_model.pkl")

print("Model loaded successfully!\n")


test_messages = [
    "I absolutely love this product! The quality is amazing.",
    "The product is okay and works as expected.",
    "The headphones stopped working after two days.",
    "I am extremely disappointed with this product.",
    "The package arrived today and everything is fine.",
    "I want a refund because this product is completely useless."
]


for message in test_messages:

    prediction = model.predict([message])[0]

    confidence = model.predict_proba([message]).max()

    print("-----------------------------------")
    print("Customer:", message)
    print("Sentiment:", prediction)
    print("Confidence:", f"{confidence * 100:.2f}%")