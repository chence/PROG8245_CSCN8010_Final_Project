# AI Agent Flow

## Overview

MediChat does not use a fully autonomous agent architecture. Instead, it uses a controlled and lightweight agent-style design on top of the classical ML pipeline.

This approach was chosen to keep the system:

- explainable
- safe
- easy to demonstrate in class
- compatible with the existing ML + retrieval structure

The current AI agent layer has three parts:

1. `Routing agent`
2. `Clarification agent`
3. `Session summary agent`

## Why This Is Called an Agent Layer

These components behave like simple agents because they make decisions about what should happen next in the conversation.

They do not:

- autonomously browse the web
- call arbitrary external tools
- plan long multi-step tasks
- replace the classifier or retriever

They do:

- inspect the incoming message
- choose an appropriate conversation path
- ask follow-up questions when needed
- summarize the conversation state

## Full Runtime Flow

```text
User input
  -> speech-to-text if audio
  -> language detection
  -> translation to English if needed
  -> routing agent decides the path
  -> clarification agent may ask for more detail
  -> dialogue manager builds contextual query
  -> ML classifier predicts intent
  -> retriever finds knowledge snippets
  -> response generator writes safe answer
  -> translation back to user language if needed
  -> session summary agent updates summary
  -> SQLite stores messages and summary
  -> UI shows answer and turn details
```

## 1. Routing Agent

### Location

- [src/predict.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/predict.py:60)

### Purpose

The routing agent decides which high-level path the user message should follow before standard classification begins.

### Supported routes

- `standard_qa`
- `urgent_safety`
- `lightweight_chat`
- `session_summary`

### What it looks for

#### Greeting messages

Examples:

- `hello`
- `hi`
- `good morning`

Behavior:

- returns a lightweight greeting response
- does not run the classifier or retriever

#### Urgent safety phrases

Examples:

- `chest pain`
- `shortness of breath`
- `can't breathe`
- `stroke`
- `heart attack`

Behavior:

- bypasses normal QA
- returns urgent safety guidance
- avoids pretending to assess emergencies

#### Summary requests

Examples:

- `summarize this conversation`
- `what have we talked about`
- `recap this chat`

Behavior:

- returns the current session summary
- does not use retrieval-based QA

#### Normal medical questions

Examples:

- `I have a sore throat and mild fever`
- `Can I take ibuprofen when I have a mild fever?`

Behavior:

- stays on the normal QA path

## 2. Clarification Agent

### Location

- [src/dialogue_manager.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/dialogue_manager.py:48)

### Purpose

The clarification agent detects questions that are too vague or underspecified for safe, useful answering.

Instead of forcing the classifier to guess early, it asks the user for more detail.

### Typical triggers

#### Ambiguous follow-up references

Examples:

- `this`
- `that`
- `it`

Behavior:

- asks what the user is referring to

#### Vague symptom statements

Examples:

- `I feel sick`
- `not feeling well`
- `help me`
- `is this serious`

Behavior:

- asks for symptom, duration, or worsening details

#### Underspecified medication questions

Examples:

- `Can I take this?`
- `What medicine should I use?`

Behavior:

- asks which medication the user means
- asks whether the concern is safety, side effects, or usage

### Why this helps

- improves conversation quality
- reduces wrong early classifications
- makes the system feel more interactive
- keeps the response grounded in actual user context

## 3. Session Summary Agent

### Location

- [src/dialogue_manager.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/dialogue_manager.py:107)
- [src/database.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/database.py:48)

### Purpose

The session summary agent creates a short summary of the current conversation after each completed turn.

The summary is stored in SQLite and reused in the UI.

### What it summarizes

- the user’s recent concern
- the latest guidance returned by the assistant
- the most recent intent when useful

### Output destination

- `session_summaries` table in SQLite
- turn details panel in the UI
- summary response route when the user explicitly asks for a recap

### Summary generation modes

#### OpenAI mode

If `OPENAI_API_KEY` is configured:

- the summary is generated with OpenAI
- the result is shorter and more natural

#### Local fallback mode

If OpenAI is unavailable:

- the summary is built from recent user and assistant turns
- the result is still deterministic and usable

## Relationship to the Classical ML Pipeline

The agent layer sits around the classifier instead of replacing it.

### The classifier still does the core intent work

- `Medication Question`
- `Self-Care Advice`
- `Symptom Inquiry`
- `Seek Medical Help`

### The retriever still grounds answers

- uses TF-IDF
- filters by intent
- selects curated medical context

### The response generator still enforces safe scope

- uses only allowed context
- avoids diagnosis
- falls back safely when support is weak

So the agent layer is best understood as:

- an orchestration layer
- not a replacement for the ML model

## Example Conversation Paths

### Path A: Standard QA

Input:

`Can I take ibuprofen when I have a mild fever, and what should I check on the label?`

Flow:

1. Language detection
2. Routing agent -> `standard_qa`
3. Clarification not needed
4. Dialogue manager builds contextual query
5. Classifier predicts medication intent
6. Retriever finds ibuprofen and label-related entries
7. Response generator returns safe medication guidance
8. Session summary agent updates the summary

### Path B: Clarification

Input:

`I feel sick`

Flow:

1. Language detection
2. Routing agent -> `standard_qa`
3. Clarification agent triggers
4. Assistant asks for more detail
5. User provides symptoms
6. Then the normal QA path can continue

### Path C: Urgent Safety

Input:

`I have chest pain and shortness of breath`

Flow:

1. Language detection
2. Routing agent -> `urgent_safety`
3. System returns urgent warning guidance
4. No classifier or normal retrieval answer is used

### Path D: Conversation Summary

Input:

`Summarize this conversation`

Flow:

1. Routing agent -> `session_summary`
2. Stored session summary is returned
3. UI displays the recap

## What Appears in Turn Details

The app exposes part of the agent flow through the right sidebar.

Examples:

- `Agent route`
- `Route reason`
- `Clarification trigger`
- `Session summary`
- classifier confidence
- retrieval score

This helps the team explain the system during demos.

## Good Demo Script

A clean demo sequence is:

1. Ask:
   `Can I take ibuprofen when I have a mild fever, and what should I check on the label?`
2. Show:
   - standard route
   - medication intent
   - grounded retrieval
3. Ask:
   `Summarize this conversation`
4. Show:
   - session summary route
   - stored summary
5. Ask:
   `I feel sick`
6. Show:
   - clarification trigger
7. Ask:
   `I have chest pain and shortness of breath`
8. Show:
   - urgent safety route

## Benefits of This Design

- safer than an unrestricted agent
- easier to explain than a multi-agent autonomous system
- preserves course focus on classical ML
- still demonstrates practical AI-enhanced orchestration
- strong for classroom demos because the behavior is visible in the UI

## Summary

MediChat’s AI agent flow is a controlled orchestration layer around the original ML pipeline.

It improves the project by adding:

- better message routing
- better handling of vague questions
- better conversation summaries

without requiring the team to abandon the explainable classifier + retrieval design.
