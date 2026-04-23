from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.config import get_config
from src.database import ChatDatabase
from src.utils import compact_text


@dataclass
class DialogueContext:
    session_id: str
    history: list[dict[str, Any]]
    summary: str = ""


@dataclass
class ClarificationPlan:
    should_clarify: bool
    question: str = ""
    reason: str = ""


class DialogueManager:
    def __init__(self, database: ChatDatabase | None = None) -> None:
        self.database = database or ChatDatabase()

    def get_context(self, session_id: str | None, language: str | None = None) -> DialogueContext:
        active_session = self.database.ensure_session(session_id, language=language)
        history = self.database.get_recent_messages(active_session, limit=8)
        summary = self.database.get_session_summary(active_session)
        return DialogueContext(session_id=active_session, history=history, summary=summary)

    def build_query(self, english_text: str, history: list[dict[str, Any]]) -> str:
        recent_user_turns = [
            item.get("english_text") or item.get("original_text", "")
            for item in history
            if item.get("role") == "user"
        ][-2:]
        if len(english_text.split()) >= 6 or not recent_user_turns:
            return english_text
        prior_context = " ".join(recent_user_turns)
        return compact_text(f"{prior_context} Follow-up question: {english_text}", max_chars=400)

    def plan_clarification(self, english_text: str, history: list[dict[str, Any]]) -> ClarificationPlan:
        normalized = compact_text(english_text or "", max_chars=240).strip()
        if not normalized:
            return ClarificationPlan(
                should_clarify=True,
                question="Could you share the symptom, medication, or health concern you want help with?",
                reason="empty_input",
            )

        lower = normalized.lower()
        cleaned_for_tokens = lower
        for char in "?.,!;:()[]{}":
            cleaned_for_tokens = cleaned_for_tokens.replace(char, " ")
        tokens = [token for token in cleaned_for_tokens.replace("/", " ").split() if token]
        prior_user_turns = [
            item.get("english_text") or item.get("original_text", "")
            for item in history
            if item.get("role") == "user"
        ]

        follow_up_tokens = {"this", "that", "it", "them", "those", "these"}
        vague_symptom_phrases = {
            "i feel sick",
            "not feeling well",
            "feel bad",
            "help me",
            "what should i do",
            "is this serious",
        }
        medication_keywords = {"medication", "medicine", "drug", "tablet", "pill", "dose", "dosage"}
        symptom_keywords = {"pain", "fever", "cough", "rash", "vomit", "dizzy", "symptom", "hurt", "sore"}
        known_medication_names = {
            "ibuprofen",
            "advil",
            "motrin",
            "acetaminophen",
            "paracetamol",
            "tylenol",
            "naproxen",
            "aspirin",
        }
        mentions_specific_medication = any(token in known_medication_names for token in tokens)

        if len(tokens) <= 2 and any(token in follow_up_tokens for token in tokens):
            return ClarificationPlan(
                should_clarify=True,
                question="What does 'this' refer to? Please mention the symptom, medicine, or concern directly.",
                reason="ambiguous_follow_up",
            )

        if any(phrase in lower for phrase in vague_symptom_phrases):
            return ClarificationPlan(
                should_clarify=True,
                question="Could you share the main symptom, how long it has been happening, and whether it is getting worse?",
                reason="vague_symptom_description",
            )

        if any(keyword in tokens for keyword in medication_keywords) and len(tokens) < 6 and not mentions_specific_medication:
            return ClarificationPlan(
                should_clarify=True,
                question="Which medication are you asking about, and what is your concern: safety, side effects, or how to use it?",
                reason="underspecified_medication_question",
            )

        if any(keyword in tokens for keyword in symptom_keywords) and len(tokens) < 4 and not prior_user_turns:
            return ClarificationPlan(
                should_clarify=True,
                question="Please add a little more detail about the symptom, such as when it started and how severe it feels.",
                reason="underspecified_symptom_question",
            )

        return ClarificationPlan(should_clarify=False)

    def summarize_session(self, session_id: str) -> str:
        messages = self.database.get_messages(session_id)
        summary = self._generate_summary(messages)
        if summary:
            self.database.upsert_session_summary(session_id, summary)
        return summary

    def _generate_summary(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return ""

        openai_summary = self._generate_summary_openai(messages)
        if openai_summary:
            return compact_text(openai_summary, max_chars=280)

        recent_user_turns = [
            compact_text(item.get("english_text") or item.get("original_text", ""), max_chars=100)
            for item in messages
            if item.get("role") == "user"
        ][-2:]
        last_assistant = next(
            (item for item in reversed(messages) if item.get("role") == "assistant"),
            None,
        )
        summary_parts: list[str] = []
        if recent_user_turns:
            summary_parts.append(f"Recent concern: {' | '.join(recent_user_turns)}")
        if last_assistant and last_assistant.get("intent"):
            summary_parts.append(f"Last intent: {last_assistant['intent']}")
        if last_assistant:
            guidance = compact_text(
                last_assistant.get("english_text") or last_assistant.get("original_text", ""),
                max_chars=140,
            )
            if guidance:
                summary_parts.append(f"Latest guidance: {guidance}")
        return " ".join(summary_parts)

    def _generate_summary_openai(self, messages: list[dict[str, Any]]) -> str:
        client = self._get_client()
        if client is None:
            return ""

        recent_lines = [
            f"{item['role'].title()}: {compact_text(item.get('english_text') or item.get('original_text', ''), max_chars=180)}"
            for item in messages[-8:]
        ]
        if not recent_lines:
            return ""

        config = get_config()
        try:
            response = client.chat.completions.create(
                model=config.openai_model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize this medical information chatbot conversation in at most two short sentences. "
                            "Mention the user's main concern and the latest guidance. Do not diagnose."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n".join(recent_lines),
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def _get_client(self) -> OpenAI | None:
        config = get_config()
        if not config.openai_api_key:
            return None
        try:
            return OpenAI(api_key=config.openai_api_key)
        except Exception:
            return None
