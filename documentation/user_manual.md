# User Manual

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional:

- Add `OPENAI_API_KEY` to `.env` for:
  - OpenAI-based language detection
  - translation
  - speech-to-text
  - conversation summary generation

## Train and Evaluate

Run the reproducible pipeline:

```bash
dvc repro
```

Or run manually:

```bash
python -m src.data_processing
python -m src.train
python -m src.evaluate
```

## Launch the Chatbot

```bash
.venv/bin/python app.py
```

Open the local Gradio link in your browser.

## How to Use the App

1. Type a general medical information question and click `Send`.
2. Or record audio from the microphone and stop recording to send it.
3. Review the chatbot answer in the chat window.
4. Check the `Turn details` panel for:
   - agent route
   - route reason
   - detected language
   - predicted intent
   - classifier confidence
   - retrieval score
   - whether the answer was treated as supported
   - clarification trigger when applicable
   - session summary
5. Use the `History` sidebar to reopen previous sessions.
6. Ask `Summarize this conversation` to demonstrate the session summary agent.

## Agent Features

MediChat now includes three lightweight agent-style features:

- `Routing agent`
  - Handles greetings
  - Detects urgent safety messages
  - Detects requests for conversation summaries
  - Sends normal medical questions to the standard QA pipeline
- `Clarification agent`
  - Detects vague inputs such as `I feel sick`
  - Asks for more detail before classification when needed
- `Session summary agent`
  - Builds a short summary after each turn
  - Stores the summary in SQLite
  - Shows the summary in `Turn Details`

## Multilingual Behavior

- OpenAI is the default language detector when an API key is configured.
- English input returns English only.
- Non-English input is processed in English internally.
- The user receives:
  - an answer in the original language
  - an English translation shown in the same response

If OpenAI services are unavailable, the app falls back to local detection and reduced translation capability.

## Example Questions

- `I have a sore throat and mild fever.`
- `What should I do at home for a cough?`
- `Should I see a doctor if breathing feels harder?`
- `Can I combine cold medicine with pain reliever?`
- `Can I take ibuprofen when I have a mild fever, and what should I check on the label?`
- `Tengo tos y fiebre leve, que hago?`

## Recommended Demo Flow

Use this sequence for a reliable classroom or presentation demo:

1. Ask:
   `Can I take ibuprofen when I have a mild fever, and what should I check on the label?`
2. Show the response and point out:
   - medication intent
   - grounded retrieval
   - safety-focused wording
3. Then ask:
   `Summarize this conversation`
4. Show the returned summary and the matching `Turn Details` panel.
5. If you want to demonstrate clarification, ask:
   `I feel sick`
6. If you want to demonstrate urgent routing, ask:
   `I have chest pain and shortness of breath`

## Limitations

- The medical scope is intentionally narrow and educational.
- The project does not diagnose illness.
- OpenAI-based language detection, translation, transcription, and summary generation depend on API availability.
- The retrieval knowledge base is curated, but it is still a controlled course-project resource rather than a full medical reference.
- Medication answers remain general and educational, not personalized.
