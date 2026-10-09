# AI-Powered Customer Support Ticket Analyzer

An academic Natural Language Processing (NLP) system developed for a 7th-semester college assignment. The system automatically classifies customer support tickets, performs sentiment analysis, determines operational priority, extracts domain entities, generates concise issue summaries, and drafts professional support responses using a Large Language Model (LLM) API with strict schema validation.

---

## 📌 Problem Statement

Customer support centers receive hundreds or thousands of unstructured inbound queries daily through emails, web forms, and chat portals. Manual triage—reading each ticket to determine its department, urgency, and customer mood—is slow, error-prone, and causes delayed resolution times. 

While classical NLP pipelines (such as Bag-of-Words or TF-IDF with SVM) can classify text into fixed classes, they fail to extract nuanced context, summarize problems, and draft context-aware responses simultaneously without complex multi-model pipelines.

---

## 🎯 Objectives

1. **Automate Ticket Triage**: Extract structured intelligence from raw, unstructured natural language text in real-time.
2. **Multi-Task NLP Execution**: Simultaneously perform:
   - **Intent / Category Classification** (`Billing`, `Technical`, `Account`, `Refund`, `Subscription`, `Shipping`, `Other`).
   - **Sentiment Analysis** (`Positive`, `Neutral`, `Negative`).
   - **Urgency / Priority Assessment** (`Low`, `Medium`, `High`, `Critical`).
   - **Named Entity Extraction** (Keywords, product names, error codes, amounts).
   - **Abstractive Text Summarization** (1-2 sentence core problem overview).
   - **Response Generation** (Polite, actionable draft for human agent review).
3. **Structured & Type-Safe Output**: Ensure the LLM output is not an unpredictable conversational paragraph, but a validated JSON object conforming to a strict Pydantic schema.
4. **Clean Software Engineering**: Maintain separation of concerns across configuration files, external prompt templates, backend logic, and frontend presentation.

---

## ⚙️ System Architecture & Workflow

The application is built around a deterministic, verifiable NLP pipeline:

```
┌─────────────────────────────────┐
│     Raw Customer Support        │
│          Ticket Text            │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      1. Input Validation        │  (Checks non-empty, min length)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     2. Prompt Construction      │  (Loads prompts/ticket_analysis.txt
└────────────────┬────────────────┘   and injects ticket text)
                 │
                 ▼
┌─────────────────────────────────┐
│        3. LLM API Call          │  (Calls Google Gemini API with low
└────────────────┬────────────────┘   temperature for determinism)
                 │
                 ▼
┌─────────────────────────────────┐
│      4. JSON Output Parsing     │  (Strips markdown code fences,
└────────────────┬────────────────┘   decodes JSON string)
                 │
                 ▼
┌─────────────────────────────────┐
│   5. Pydantic Schema Validation │  (Validates types, Enums, fields;
└────────────────┬────────────────┘   raises clear errors on mismatch)
                 │
                 ▼
┌─────────────────────────────────┐
│   6. Streamlit User Dashboard   │  (Displays metrics, entities,
└─────────────────────────────────┘   summary, and response draft)
```

---

## 🛠️ Technologies Used

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Standard language for NLP and AI research |
| **User Interface** | Streamlit | Lightweight, rapid frontend for interactive machine learning demos |
| **LLM Client** | Google GenAI SDK (`google-genai`) | Official Google SDK for Gemini models with native JSON support |
| **Data Validation** | Pydantic v2 | Enforces strict schema, field constraints, and Enum validation |
| **Configuration** | PyYAML | Keeps non-sensitive parameters cleanly decoupled from source code |
| **Environment Mgmt**| Python-Dotenv | Safely loads sensitive API keys from `.env` without hardcoding |

---

## 📂 Project Structure

```
customer-ticket-analyzer/
│
├── app.py                      # Streamlit frontend dashboard
├── requirements.txt            # Minimal required dependencies
├── README.md                   # Comprehensive academic documentation & viva guide
├── .env.example                # Sample environment file template
├── .gitignore                  # Prevents secrets and cache files from committing
├── evaluate.py                 # Evaluation benchmark script (accuracy calculation)
│
├── config/
│   └── config.yaml             # Non-secret model & application configuration
│
├── data/
│   └── sample_tickets.json     # 25 labeled benchmark tickets for model evaluation
│
├── prompts/
│   └── ticket_analysis.txt     # External prompt engineering template
│
└── src/
    ├── __init__.py             # Python package marker
    ├── llm.py                  # Modular LLM client with error handling
    ├── analyzer.py             # NLP workflow pipeline orchestrator
    └── schemas.py              # Pydantic models and Enums for validation
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure Python 3.10 or higher is installed on your computer.

### 2. Clone or Navigate to the Project Directory
```bash
cd customer-ticket-analyzer
```

### 3. (Optional but Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Your API Key
Copy the example environment file to `.env`:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux / Git Bash
cp .env.example .env
```

Open `.env` in any text editor and insert your API key:
```env
GEMINI_API_KEY=yourActualGeminiApiKeyHere
```

> **Tip for Students:** Google provides generous free tier access for developers and students:
> 1. Visit **Google AI Studio**: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
> 2. Sign in with any Google account and click **Create API Key**.
> 3. Paste the key into your `.env` file under `GEMINI_API_KEY`.
> 4. Model selection (default: `gemini-2.5-flash`) can be configured in `config/config.yaml`.

---

## 💻 Running the Application

### Launch the Streamlit Web Interface
Run the following command in your terminal:
```bash
streamlit run app.py
```
Your default web browser will automatically open at `http://localhost:8501`.

---

## 🧪 Model Evaluation

To demonstrate that this is a rigorous NLP system with measurable performance, an evaluation script and a benchmark dataset of 25 labeled tickets (`data/sample_tickets.json`) are included.

### Run Evaluation Benchmark
```bash
# Evaluate all 25 benchmark tickets
python evaluate.py

# Or run a quick evaluation on the first 5 tickets (saves tokens/time)
python evaluate.py --limit 5
```

### Sample Evaluation Output:
```text
================================================================================
      AI-POWERED CUSTOMER SUPPORT TICKET ANALYZER - EVALUATION BENCHMARK
================================================================================
Evaluating 25 labeled customer support tickets...
--------------------------------------------------------------------------------
ID   | Expected Category | Predicted Category | Status
--------------------------------------------------------------------------------
1    | Billing           | Billing            | ✅ MATCH
2    | Billing           | Billing            | ✅ MATCH
...
25   | Other             | Other              | ✅ MATCH
================================================================================
                           EVALUATION SUMMARY
================================================================================
Total Evaluated Tickets : 25
Correct Classifications : 24
Incorrect / Errors      : 1
Overall Accuracy        : 96.00%
--------------------------------------------------------------------------------
```

---

## 📋 Example Input and Output

### Input Ticket:
```text
"I was charged twice for my subscription this month and I want a refund for the extra payment."
```

### Validated JSON Output:
```json
{
  "category": "Billing",
  "sentiment": "Negative",
  "priority": "High",
  "entities": [
    "subscription",
    "duplicate payment",
    "refund"
  ],
  "summary": "Customer reports being billed twice for their subscription and is requesting a refund for the duplicate charge.",
  "suggested_response": "Hello, thank you for reaching out. We apologize for the duplicate charge on your account. We have initiated a review of the duplicate transaction and will process your refund promptly within 3-5 business days."
}
```

---

## 🎓 Viva Questions & Answers Guide

### 1. What problem does this project solve?
> **Answer**: Customer support centers receive large volumes of unorganized, unstructured text inquiries. This system automates triage by classifying the category, detecting customer sentiment, gauging priority, extracting keywords, and drafting an initial reply—reducing first-response time and human workload.

### 2. Why is this considered an NLP project rather than just a chatbot?
> **Answer**: Chatbots produce open-ended, free-form conversational text. This project treats the LLM as an **NLP information extraction and classification engine**. It takes unstructured natural language as input and executes 5 classical NLP tasks (text classification, sentiment analysis, named entity extraction, summarization, and conditional text generation) mapped strictly to a schema validated by Pydantic.

### 3. Why is the prompt stored in a separate file (`prompts/ticket_analysis.txt`) instead of hardcoded in Python?
> **Answer**: Decoupling the prompt follows software engineering best practices. It allows prompt engineers or domain experts to refine prompt instructions, few-shot examples, or guidelines without touching application code or risking syntax errors in Python.

### 4. Why is structured JSON output preferred over free-form text?
> **Answer**: Real-world enterprise systems (CRM platforms like Salesforce, Zendesk, or internal databases) require structured data to trigger automated routing, SLA alerts, and analytics. Free-form text cannot be easily queried, stored in databases, or routed via automated backend rules.

### 5. How do you prevent hallucination or bad outputs from the LLM?
> **Answer**: Three layers of defense are used:
> 1. **Low Temperature (0.1)** in `config/config.yaml` to ensure deterministic, focused outputs.
> 2. **Explicit Negative Constraints** in the prompt ("Rely strictly on provided ticket text; do not invent facts").
> 3. **Pydantic Schema Validation** in `src/schemas.py` which rejects invalid categories, malformed structures, or missing fields.

### 6. How does the application handle API failures and network errors?
> **Answer**: In `src/llm.py`, exceptions from the Google Gemini API (`errors.APIError`, quota limits HTTP 429, invalid credentials HTTP 401/403, and network timeouts) are captured and wrapped into sanitized `LLMServiceError` messages. Sensitive details like API keys are never leaked to the UI or logs.

---

## 🔮 Future Enhancements

1. **Human-in-the-Loop Feedback Loop**: Allow agents to accept or correct model classifications, saving corrections to a dataset for periodic accuracy tracking.
2. **Multi-Language Support**: Extend the prompt guidelines to detect incoming language and draft responses in the customer's native language.
3. **Email / Webhook Integration**: Ingest tickets automatically via IMAP/REST webhooks from Zendesk, Freshdesk, or Gmail.

---

## ⚠️ Academic Disclaimer

This project is developed as an **educational prototype** for a university NLP assignment. It is designed to assist customer support representatives by drafting preliminary analyses and responses, and is **not** intended to replace human customer service personnel or make fully automated financial/legal decisions without human oversight.
