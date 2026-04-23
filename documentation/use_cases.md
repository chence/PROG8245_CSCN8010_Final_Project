# Use Cases

## Overview

This document describes representative use cases for MediChat after the addition of lightweight AI agent functionality.

The system supports:

- multilingual text questions
- multilingual voice questions
- classical ML intent classification
- retrieval-grounded safe responses
- clarification for vague questions
- urgent safety redirection
- conversation summarization

## Use Case 1: Ask a Standard Symptom Question

**Actor:** User

**Goal:** Get general information about a symptom.

**Preconditions:**

- The web app is running.
- A trained model is available.

**Main flow:**

1. The user enters a symptom question in the chat box.
2. The system detects the language.
3. If needed, the system translates the question to English.
4. The routing agent sends the message to the normal QA flow.
5. The dialogue manager builds contextual query text.
6. The classifier predicts intent.
7. The retriever selects relevant knowledge snippets.
8. The response generator produces a safe answer.
9. The answer is translated back if needed.
10. The conversation and summary are stored in SQLite.

**Example input:** `I have a cough and mild fever. What can I do at home?`

**Expected result:** The user receives a self-care or symptom-related response grounded in supported knowledge.

## Use Case 2: Ask a Medication Question

**Actor:** User

**Goal:** Ask for general medication-related information.

**Main flow:**

1. The user asks about medication safety or usage.
2. The routing agent keeps the message on the standard QA path.
3. The classifier predicts `Medication Question`.
4. Retrieval brings back medication-related context.
5. The system returns a general informational answer with a safety reminder.

**Example input:** `Can I take ibuprofen when I have a mild fever, and what should I check on the label?`

**Expected result:** A general medication safety answer is returned, without diagnosis or prescribing.

## Recommended Demo Example

**Best single-turn presentation example:** `Can I take ibuprofen when I have a mild fever, and what should I check on the label?`

Why this works well:

- it is very common and easy for the audience to understand
- it fits the supported `Medication Question` intent well
- it matches several knowledge-base keywords such as `ibuprofen`, `fever`, `label`, and `OTC`
- it stays in a safe educational scope instead of asking for diagnosis
- it usually produces a clean, confident, grounded answer

**Expected demo behavior:**

1. The system detects the language.
2. The routing agent keeps the message on the standard QA path.
3. The classifier predicts a medication-related intent.
4. Retrieval finds medication safety entries about ibuprofen, fever, and label reading.
5. The response explains that MediChat can only provide general safety guidance, reminds the user to read the label, avoid overlapping ingredients, and check with a pharmacist or clinician when unsure.

**Recommended follow-up for a stronger demo:** `Summarize this conversation`

This follow-up lets you demonstrate the session summary agent immediately after a successful grounded answer.

## Use Case 3: Clarify a Vague Medical Question

**Actor:** User

**Goal:** Get help even when the initial question is incomplete.

**Main flow:**

1. The user sends a vague question.
2. The clarification agent checks whether the message is too ambiguous.
3. Instead of forcing classification, the system asks a clarifying follow-up question.
4. The user provides more detail.
5. The system then continues through the normal QA flow.

**Example input:** `I feel sick`

**Expected result:** The system asks for more detail such as symptom, duration, or severity.

## Use Case 4: Clarify an Underspecified Medication Question

**Actor:** User

**Goal:** Help the user reformulate a medication question before classification.

**Main flow:**

1. The user sends an underspecified medication message.
2. The clarification agent detects that the medication name or concern is missing.
3. The system asks what medication the user means and what concern they have.

**Example input:** `Can I take this?`

**Expected result:** The system asks a clarification question instead of guessing.

## Use Case 5: Handle an Urgent Safety Message

**Actor:** User

**Goal:** Be redirected safely when symptoms may be urgent.

**Main flow:**

1. The user sends a message containing urgent symptom keywords.
2. The routing agent detects an emergency-style message.
3. The system bypasses the normal classifier and retriever.
4. The system returns urgent safety guidance and tells the user to contact emergency or urgent care services.

**Example input:** `I have chest pain and shortness of breath`

**Expected result:** Immediate safety guidance is returned instead of normal symptom education.

## Use Case 6: Summarize the Current Conversation

**Actor:** User

**Goal:** Review what has happened in the session so far.

**Main flow:**

1. The user asks for a summary.
2. The routing agent detects a summary request.
3. The session summary agent retrieves or generates the session summary.
4. The system returns a short recap of the user concern and latest guidance.

**Example input:** `Summarize this conversation`

**Expected result:** A brief summary of recent concerns and guidance is returned.

## Use Case 7: Ask a Question in a Non-English Language

**Actor:** User

**Goal:** Use the chatbot in another language.

**Main flow:**

1. The user sends a question in a non-English language.
2. The system detects the source language.
3. The question is translated to English for internal processing.
4. Standard routing and QA continue.
5. The response is translated back to the original language.
6. The UI can also show the English translation for transparency.

**Example input:** `我发烧了，应该怎么办？`

**Expected result:** The user receives a response in Chinese, with the internal English translation available in the UI.

## Use Case 8: Ask a Question by Voice

**Actor:** User

**Goal:** Interact with the chatbot using speech instead of typing.

**Main flow:**

1. The user records audio in the Gradio interface.
2. The speech-to-text component transcribes the audio.
3. The transcribed text enters the same multilingual and routing pipeline as typed text.
4. The answer is returned in the chat window.

**Example input:** Voice recording asking about cough, fever, or self-care.

**Expected result:** The system transcribes the audio and answers as if it were typed input.

## Use Case 9: Reopen a Previous Session

**Actor:** User

**Goal:** Continue or review an earlier conversation.

**Main flow:**

1. The user selects a previous session from the history sidebar.
2. The app loads stored messages from SQLite.
3. The stored session summary and latest metadata appear in the turn details panel.
4. The user can continue the conversation with context.

**Expected result:** Previous turns are restored and the session remains usable.

## Use Case 10: Out-of-Scope or Unsupported Question

**Actor:** User

**Goal:** Receive a safe fallback when the system does not have enough support.

**Main flow:**

1. The user asks a question outside the supported knowledge base.
2. The classifier or retriever confidence is too low.
3. The system returns a safe unsupported response instead of hallucinating.

**Example input:** Highly specific diagnosis or personalized treatment request.

**Expected result:** The user is told the question is unsupported and is reminded to seek professional care.

## Use Case 11: Instructor / Evaluator Demo

**Actor:** Instructor or evaluator

**Goal:** Demonstrate the project architecture and capabilities during presentation.

**Main flow:**

1. The evaluator opens the app.
2. They test the recommended medication demo question about ibuprofen and mild fever.
3. They test a vague question such as `I feel sick` to trigger clarification.
4. They test an urgent question such as `I have chest pain and shortness of breath` to trigger safety routing.
5. They ask for a conversation summary.
6. They inspect the history and turn details panels.

**Expected result:** The evaluator can see traditional ML, retrieval, multilingual handling, voice input, and agent-style orchestration working together.

------
## Show Case
1, input English, this response 
- hello
2, speak English, test ai agent
- I feel sick
3, speak Chinese 
- 我发烧了，喉咙痛，咳嗽
4, input Chinese
- 可以吃布洛芬吗？
5, input Chinese, upsupport
- 霍尔木兹海峡何时开放？