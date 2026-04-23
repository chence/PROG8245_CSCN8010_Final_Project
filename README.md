# MediChat

MediChat is a multilingual medical information chatbot built for the PROG8245 machine learning and NLP final project. The current version combines traditional ML intent classification, retrieval-based controlled response generation, multilingual handling, audio transcription, lightweight AI agent behaviors, a Gradio web UI, SQLite chat logging, and DVC pipeline management.

## Core Features

- Traditional ML model comparison:
  - `TF-IDF + Multinomial Naive Bayes`
  - `TF-IDF + TruncatedSVD + Logistic Regression`
  - `TF-IDF + dense PCA + Logistic Regression`
- Chat-style Gradio interface with multi-turn conversation
- Text and audio input
- OpenAI-first language detection plus translate-to-English processing flow
- Controlled generation grounded in retrieved medical context
- Confidence and retrieval thresholding for out-of-scope fallback
- Routing, clarification, and session summary agents
- SQLite-based session, message, and summary logging
- Enriched curated knowledge base with stronger medication and mild symptom coverage
- DVC-ready prepare, train, and evaluate pipeline

## Updated Architecture

```text
User text/audio
  -> speech-to-text (optional, OpenAI)
  -> language detection
  -> translation to English when needed
  -> routing agent
  -> clarification agent when needed
  -> dialogue manager builds contextual query
  -> baseline ML classifier predicts intent
  -> TF-IDF retrieval over curated medical knowledge base
  -> controlled response generation from allowed context only
  -> translate response back to user language when needed
  -> session summary agent updates summary
  -> persist turn in SQLite
  -> display in Gradio chat UI
```

## Project Structure

```text
PROG8245_Final_Project/
├── app.py
├── data/
│   ├── raw/
│   │   ├── medical_intent_dataset.csv
│   │   ├── medical_knowledge_base.json
│   │   └── intent_responses.json
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── dataset_summary.json
│   └── medichat.sqlite3
├── documentation/
│   ├── architecture.md
│   ├── architecture_diagram.svg
│   ├── development_plan.md
│   ├── user_manual.md
│   ├── use_cases.md
│   ├── model_comparison.md
│   └── database_design.md
├── models/
│   ├── baseline_nb.joblib
│   ├── svd_logreg.joblib
│   ├── pca_logreg.joblib
│   └── *.metadata.json
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dialogue_manager.py
│   ├── retrieval.py
│   ├── translation.py
│   ├── speech_to_text.py
│   ├── response_generator.py
│   ├── database.py
│   └── utils.py
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add `OPENAI_API_KEY` to `.env` if you want:

- OpenAI-based language detection
- translation
- audio transcription
- OpenAI summary generation
- OpenAI response phrasing

The project still runs without the key, but language detection quality, translation, speech support, and summary quality will degrade gracefully.

You can also choose which trained classifier the app uses by setting:

```bash
MEDICHAT_MODEL_NAME=svd_logreg
```

Supported model names are:

- `baseline_nb`
- `svd_logreg`
- `pca_logreg`

## Reproducible ML Pipeline

Run the full pipeline:

```bash
dvc repro
```

Or run the stages directly:

```bash
python -m src.data_processing
python -m src.train
python -m src.evaluate
```

Outputs:

- processed train/test data in `data/processed/`
- trained models in `models/`
- evaluation JSON, CSV, markdown, and confusion matrices in `documentation/`

## Run the Web App

Train the models first, then launch:

```bash
python app.py
```

The default app model is `svd_logreg`, but you can change it with `MEDICHAT_MODEL_NAME` in `.env`.

The UI supports:

- text questions
- microphone audio
- multilingual flow
- multi-turn history
- session summaries
- turn-by-turn metadata display
- lightweight agent routing and clarification

## Controlled Generation Design

MediChat does not generate unrestricted medical advice. Each turn follows this controlled workflow:

1. Detect language and translate to English when needed.
2. Route the message through greeting, urgent safety, summary, clarification, or normal QA logic.
3. If needed, ask a clarification question before classification.
4. Classify the intent with a traditional ML model.
5. Retrieve supporting snippets from a curated medical knowledge base.
6. Reject low-confidence or low-relevance queries with a safe fallback.
7. Generate the final answer using only retrieved context.
8. Update the stored session summary.

This keeps the project aligned with the course requirement for explainable ML while still making the demo conversational.

## Recommended Demo

The strongest single-turn demo example is:

`Can I take ibuprofen when I have a mild fever, and what should I check on the label?`

Why this works well:

- it is common and easy to understand
- it fits the supported medication scope
- it matches the enriched knowledge base well
- it typically produces a grounded and safe answer

A strong follow-up is:

`Summarize this conversation`

This lets you immediately demonstrate the session summary agent.

## Safety Scope

- MediChat is not a diagnosis system.
- It provides general educational information only.
- It must not be used for emergencies, medication decisions, or personalized treatment.
- Severe or worsening symptoms should be referred to licensed medical professionals.

## Notes on the Dataset and Knowledge Base

The repository currently includes a balanced respiratory-health intent dataset in `data/raw/medical_intent_dataset.csv`, which keeps the project lightweight and reproducible for a student environment. The training code is organized so the dataset can be swapped for a larger medical intent classification CSV later if the team decides to extend the scope.

The retrieval layer is supported by a curated knowledge base in `data/raw/medical_knowledge_base.json`. It now includes broader coverage for:

- mild fever
- cough and congestion
- sore throat
- over-the-counter medication safety
- ibuprofen and acetaminophen label-reading scenarios
