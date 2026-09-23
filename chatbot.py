import joblib
import subprocess


# Load sentiment model
model = joblib.load("models/sentiment_model.pkl")


def analyze_sentiment(message):

    sentiment = model.predict([message])[0]
    confidence = model.predict_proba([message]).max()

    return sentiment, confidence


def detect_severity(message):

    message_lower = message.lower()

    high_severity_words = [
        "charged twice",
        "fraud",
        "scam",
        "legal",
        "lawsuit",
        "three times",
        "multiple times",
        "nobody helped",
        "not resolved",
        "still not resolved",
        "support ignored"
    ]

    medium_severity_words = [
        "refund",
        "broken",
        "damaged",
        "not working",
        "stopped working",
        "missing",
        "wrong product",
        "late delivery",
        "payment problem",
        "defective",
        "faulty"
    ]

    for word in high_severity_words:
        if word in message_lower:
            return "High"

    for word in medium_severity_words:
        if word in message_lower:
            return "Medium"

    return "Low"


def generate_response(message, sentiment, confidence, severity):

    prompt = f"""
You are an empathetic retail customer support assistant.

Customer message:
{message}

Detected sentiment: {sentiment}
Sentiment confidence: {confidence:.2f}
Issue severity: {severity}

Respond naturally and professionally.

Rules:
- Acknowledge the customer's concern.
- Be empathetic.
- Keep the response concise.
- Do not invent company policies.
- Do not promise refunds or compensation.
- If severity is High, clearly recommend escalation to a human support agent.
- If severity is Medium, suggest contacting customer support for assistance.
- If severity is Low, provide a helpful response.
- Do not mention that you are an AI.
"""

    result = subprocess.run(
        ["ollama", "run", "llama3.2:3b", prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    return result.stdout.strip()


print("Retail Sentiment Chatbot")
print("=========================\n")


while True:

    message = input("Customer: ")

    if message.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    sentiment, confidence = analyze_sentiment(message)

    severity = detect_severity(message)

    response = generate_response(
        message,
        sentiment,
        confidence,
        severity
    )

    print("\nSentiment:", sentiment)
    print("Confidence:", f"{confidence * 100:.2f}%")
    print("Severity:", severity)

    if severity == "High":
        print("Escalation: Recommended")
    else:
        print("Escalation: Not required")

    print("\nChatbot:", response)
    print("\n" + "=" * 50 + "\n")