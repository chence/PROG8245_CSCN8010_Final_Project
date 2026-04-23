# Architecture

## System Overview

MediChat uses a layered architecture that combines classical machine learning, retrieval, multilingual preprocessing, and lightweight AI agent behaviors.

The current version keeps the original explainable ML pipeline, but adds three controlled agent-style capabilities:

- `Routing agent`: decides whether a message should go to normal QA, urgent safety guidance, lightweight greeting handling, or conversation summary.
- `Clarification agent`: detects vague or underspecified questions and asks a follow-up question before the classifier is used.
- `Session summary agent`: creates a short summary of each conversation and stores it for history and turn details.

## High-Level Runtime Flow

```text
Text / Audio Input
    ->
Gradio Web UI
    ->
Speech-to-Text Module (optional for audio)
    ->
Language Detection + Translation Layer
    ->
Routing Agent
    ->
Clarification Agent or Standard QA Flow
    ->
Dialogue Manager
    ->
Intent Classification Model
    ->
Knowledge Retrieval
    ->
Controlled Response Generator
    ->
Translation Back to User Language
    ->
SQLite Logging + Session Summary Agent + Chat Display
```

## Main Modules

### `app.py`
Hosts the Gradio web UI, captures text or microphone input, updates the conversation window, displays turn details, and connects the front end to the runtime engine.

### `src/config.py`
Centralizes model paths, dataset paths, OpenAI settings, thresholds, and documentation output locations.

### `src/data_processing.py`
Loads the raw dataset, validates required columns, cleans text, performs a stratified split, and writes reproducible processed train/test files.

### `src/train.py`
Trains the three required classical ML pipelines:

- `baseline_nb`
- `svd_logreg`
- `pca_logreg`

### `src/evaluate.py`
Evaluates all trained models on the same test split and saves:

- JSON metrics
- CSV comparison table
- Markdown comparison table
- one confusion matrix image per model

### `src/translation.py`
Uses OpenAI as the default language detector when available, with local fallback behavior. It also translates non-English text into English for internal processing and translates the response back to the user language.

### `src/speech_to_text.py`
Transcribes microphone or uploaded audio into text using OpenAI audio transcription.

### `src/dialogue_manager.py`
Provides multi-turn context support and now contains two lightweight agent behaviors:

- follow-up query building for short contextual questions
- clarification planning for vague or underspecified user input

It also generates session summaries after each turn.

### `src/predict.py`
Coordinates the runtime inference flow. This is the orchestration layer where the routing agent decides whether to:

- handle a greeting
- trigger urgent safety guidance
- return a conversation summary
- ask a clarification question
- continue through the normal ML + retrieval path

### `src/retrieval.py`
Uses TF-IDF similarity over a curated medical knowledge base. Retrieval is filtered by predicted intent so the final response stays grounded in supported context.

### `src/response_generator.py`
Creates the final safe response. If OpenAI is available, the prompt limits the model to retrieved context and forbids diagnosis. If OpenAI is unavailable, a local deterministic fallback is used.

### `src/database.py`
Stores sessions, messages, metadata, and session summaries in SQLite. This supports persistent chat history, simple analytics, and summary display in the UI.

## Agent-Enhanced Runtime Behavior

### 1. Routing Agent

The routing agent runs before the classifier and checks whether the message should bypass the standard classification path.

Examples:

- `hello` -> lightweight greeting route
- `summarize this conversation` -> session summary route
- `I have chest pain and shortness of breath` -> urgent safety route
- ordinary medical questions -> standard QA route

This keeps the system more flexible without replacing the core ML pipeline.

### 2. Clarification Agent

The clarification agent checks whether the user question is too vague for safe or useful answering.

Examples:

- `I feel sick`
- `Can I take this?`
- `Is this serious?`

Instead of forcing the classifier to guess too early, the system asks for more detail first.

### 3. Session Summary Agent

After each completed turn, the system creates a short summary of the ongoing session. This summary is stored in SQLite and can be used in:

- the history sidebar
- the turn details panel
- future routing decisions
- presentation/demo explanations

When OpenAI is available, the summary is model-generated. Otherwise, a deterministic local summary is built from recent turns.

## Runtime Flow in Detail

1. The user sends text or records audio in the Gradio UI.
2. If audio is present, the speech-to-text module transcribes it.
3. The language detector identifies the source language.
4. Non-English text is translated to English for internal processing.
5. The routing agent decides whether to use summary mode, urgent safety mode, greeting mode, clarification mode, or standard QA mode.
6. If the message is vague, the clarification agent returns a follow-up question.
7. Otherwise, the dialogue manager builds a contextual query using recent history.
8. The baseline classifier predicts the intent and confidence.
9. The retriever selects top medical knowledge snippets for that intent.
10. If confidence or retrieval score is too low, the system returns a safe unsupported fallback.
11. Otherwise, the controlled response generator creates a short grounded answer.
12. The answer is translated back to the user language when needed.
13. The user turn, assistant turn, route metadata, and summary are stored in SQLite.
14. The UI updates the conversation, history, and turn details.

## Data and Storage

### Primary runtime data

- `data/raw/medical_intent_dataset.csv`
- `data/raw/medical_knowledge_base.json`
- `data/raw/intent_responses.json`
- `models/*.joblib`
- `data/medichat.sqlite3`

The knowledge base has been expanded to better support common demo-friendly questions about:

- mild fever
- cough and congestion
- sore throat
- over-the-counter medication safety
- ibuprofen and acetaminophen label reading

### SQLite tables

- `sessions`
- `messages`
- `session_summaries`

## Why This Fits the Course

- Keeps traditional ML front and center for classification and evaluation.
- Adds multilingual and audio handling without removing the explainable pipeline.
- Uses retrieval-grounded response generation instead of unrestricted chatbot behavior.
- Demonstrates practical AI enhancement through lightweight agent behaviors.
- Remains small enough to explain clearly in a class presentation.
