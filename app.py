import streamlit as st
import joblib
import requests
import json
import os
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RetailSense AI",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# FILE FOR SAVING CHAT HISTORY
# =========================================================

HISTORY_FILE = "chat_history.json"


# =========================================================
# LOAD SENTIMENT MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/sentiment_model.pkl")


model = load_model()


# =========================================================
# LOAD SAVED HISTORY
# =========================================================

def load_history():

    if os.path.exists(HISTORY_FILE):

        try:
            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except Exception:
            return []

    return []


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(history):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

def analyze_sentiment(message):

    sentiment = model.predict([message])[0]

    confidence = model.predict_proba([message]).max()

    return sentiment, confidence


# =========================================================
# SEVERITY DETECTION
# =========================================================

def detect_severity(message):

    message_lower = message.lower()

    high_severity_words = [
        "fraud",
        "scam",
        "lawsuit",
        "legal",
        "charged twice",
        "charged three times",
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


# =========================================================
# GENERATE LLAMA RESPONSE
# =========================================================

def generate_response(
    message,
    sentiment,
    confidence,
    severity
):

    prompt = f"""
You are a professional retail customer support assistant.

Customer message:
{message}

Internal analysis:
Sentiment: {sentiment}
Confidence: {confidence:.2f}
Severity: {severity}

Write the final response that will be shown directly to the customer.

Rules:

1. Use natural, grammatically correct English.
2. Be polite, warm and empathetic.
3. Directly address the customer's concern.
4. Keep the response to 2 or 3 sentences.
5. Do not unnecessarily repeat the customer's message.
6. Do not use strange symbols or corrupted characters.
7. Do not use emojis.
8. Do not mention sentiment, confidence or severity.
9. Do not mention AI, Llama, models or algorithms.
10. Do not invent company policies.
11. Do not promise refunds, replacements or compensation.
12. Do not claim that an action has already been taken.
13. If the customer is happy, thank them for their feedback.
14. If the customer is unhappy, acknowledge their frustration.
15. If severity is Medium, recommend contacting customer support.
16. If severity is High, recommend speaking with a human support agent.

Return ONLY the customer-facing response.
"""

    try:

        result = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        result.raise_for_status()

        response = result.json()["response"].strip()

        response = response.strip('"').strip("'")

        return response

    except requests.exceptions.ConnectionError:

        return (
            "I'm sorry, but I cannot connect to the AI assistant "
            "right now. Please make sure Ollama is running and try again."
        )

    except Exception:

        return (
            "I'm sorry, but I'm having trouble generating a response "
            "right now. Please contact customer support for assistance."
        )


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "history" not in st.session_state:

    st.session_state.history = load_history()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🤖 RetailSense AI")

st.sidebar.write("AI-Powered Retail Customer Support")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "💬 Chat",
        "📜 History"
    ]
)

st.sidebar.divider()

st.sidebar.write("### AI Pipeline")

st.sidebar.write("1. Customer Message")

st.sidebar.write("2. Sentiment Analysis")

st.sidebar.write("3. Severity Detection")

st.sidebar.write("4. Llama Response")

st.sidebar.write("5. Conversation History")


# =========================================================
# CHAT PAGE
# =========================================================

if page == "💬 Chat":

    st.title("🤖 RetailSense AI")

    st.subheader(
        "AI-Powered Retail Customer Sentiment Analysis & Support"
    )

    st.info(
        "RetailSense AI analyzes customer sentiment, "
        "detects issue severity and generates empathetic responses."
    )

    st.write("### 💬 Customer Support Chat")


    # -----------------------------------------
    # DISPLAY CURRENT CHAT
    # -----------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])


    # -----------------------------------------
    # CUSTOMER INPUT
    # -----------------------------------------

    customer_message = st.chat_input(
        "Type your message here..."
    )


    # -----------------------------------------
    # PROCESS MESSAGE
    # -----------------------------------------

    if customer_message:

        # Show customer message

        with st.chat_message("user"):

            st.write(customer_message)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": customer_message
            }
        )


        # -----------------------------------------
        # SENTIMENT
        # -----------------------------------------

        sentiment, confidence = analyze_sentiment(
            customer_message
        )


        # -----------------------------------------
        # SEVERITY
        # -----------------------------------------

        severity = detect_severity(
            customer_message
        )


        # -----------------------------------------
        # LLAMA RESPONSE
        # -----------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Analyzing and generating response..."
            ):

                response = generate_response(
                    customer_message,
                    sentiment,
                    confidence,
                    severity
                )

            st.write(response)


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


        # -----------------------------------------
        # SAVE TO HISTORY
        # -----------------------------------------

        history_record = {

            "timestamp": datetime.now().strftime(
                "%d-%m-%Y %I:%M %p"
            ),

            "customer_message": customer_message,

            "sentiment": sentiment,

            "confidence": round(
                confidence * 100,
                2
            ),

            "severity": severity,

            "bot_response": response
        }


        st.session_state.history.append(
            history_record
        )

        save_history(
            st.session_state.history
        )


    # =================================================
    # LATEST ANALYSIS
    # =================================================

    if st.session_state.messages:

        st.divider()

        st.write("### 📊 Latest Customer Analysis")


        # Find latest customer message

        last_customer_message = None

        for message in reversed(
            st.session_state.messages
        ):

            if message["role"] == "user":

                last_customer_message = message["content"]

                break


        if last_customer_message:

            sentiment, confidence = analyze_sentiment(
                last_customer_message
            )

            severity = detect_severity(
                last_customer_message
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Sentiment",
                    sentiment
                )


            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%"
                )


            with col3:

                st.metric(
                    "Severity",
                    severity
                )


            if severity == "High":

                st.error(
                    "🚨 Human escalation recommended"
                )

            elif severity == "Medium":

                st.warning(
                    "⚠️ Customer support assistance recommended"
                )

            else:

                st.success(
                    "✅ No escalation required"
                )


# =========================================================
# HISTORY PAGE
# =========================================================

elif page == "📜 History":

    st.title("📜 Conversation History")

    st.subheader(
        "Previous Customer Interactions"
    )


    history = st.session_state.history


    # -----------------------------------------
    # NO HISTORY
    # -----------------------------------------

    if not history:

        st.info(
            "No conversations have been recorded yet."
        )


    # -----------------------------------------
    # HISTORY EXISTS
    # -----------------------------------------

    else:

        st.write(
            f"Total conversations: **{len(history)}**"
        )

        st.divider()


        # Display newest first

        for index, record in enumerate(
            reversed(history)
        ):

            with st.expander(
                f"💬 {record['timestamp']}  |  "
                f"{record['sentiment']}  |  "
                f"{record['severity']}"
            ):

                st.write("### 👤 Customer")

                st.write(
                    record["customer_message"]
                )


                st.write("### 🤖 RetailSense AI")

                st.write(
                    record["bot_response"]
                )


                st.divider()


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Sentiment",
                        record["sentiment"]
                    )


                with col2:

                    st.metric(
                        "Confidence",
                        f"{record['confidence']}%"
                    )


                with col3:

                    st.metric(
                        "Severity",
                        record["severity"]
                    )


        # -----------------------------------------
        # CLEAR HISTORY
        # -----------------------------------------

        st.divider()

        if st.button(
            "🗑️ Clear Conversation History"
        ):

            st.session_state.history = []

            save_history([])

            st.success(
                "Conversation history cleared."
            )

            st.rerun()