# 🛍️ RetailSense AI — Retail Customer Sentiment Analysis Chatbot

An AI-powered retail customer support chatbot that analyzes customer sentiment, identifies issue severity, generates empathetic responses using Generative AI, and recommends human escalation for serious issues.

---

## 📌 Problem Statement

Retail customers may express different emotions and problems while interacting with customer support.

The goal of this project is to build an AI chatbot that can:

- Analyze customer sentiment in real time
- Classify messages as Positive, Neutral, or Negative
- Generate contextually appropriate and empathetic responses
- Identify the severity of customer issues
- Recommend human escalation for serious cases
- Maintain a history of customer interactions

The target is to achieve **80%+ sentiment classification accuracy**.

---

## 💡 Proposed Solution

RetailSense AI combines a traditional Machine Learning model with Generative AI.

### Machine Learning

We use:

**TF-IDF + Logistic Regression**

for measurable three-class sentiment classification:

- Positive
- Neutral
- Negative

### Generative AI

We use:

**Llama 3.2 3B**

through **Ollama** to generate natural and empathetic responses.

### Severity Detection

A rule-based severity layer identifies whether an issue is:

- 🟢 Low
- 🟡 Medium
- 🔴 High

High-severity issues can be recommended for human escalation.

---

## 🔄 System Workflow

```text
Customer Message
       ↓
Text Processing
       ↓
TF-IDF Feature Extraction
       ↓
Logistic Regression
       ↓
Sentiment Classification

       ↓
Severity Detection
       ↓
Low / Medium / High
       ↓
Llama 3.2 3B
       ↓
Empathetic Response
       ↓
Customer
       ↓
Interaction History