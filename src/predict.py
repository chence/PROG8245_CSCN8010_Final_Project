from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import get_config
from src.dialogue_manager import DialogueManager
from src.response_generator import SAFETY_NOTICE, generate_controlled_response
from src.retrieval import KnowledgeRetriever, RetrievedContext
from src.translation import TranslationResult, detect_language, translate_text
from src.utils import load_json

UNSUPPORTED_MESSAGE = (
    "This question is not supported."
    # "I can only answer supported, general medical information questions in this course project. "
    # "Please rephrase the question or ask a healthcare professional for personalized advice."
)


@dataclass
class PredictionArtifacts:
    model: Any
    responses: dict[str, str]
    retriever: KnowledgeRetriever
    dialogue_manager: DialogueManager


@dataclass
class RouteDecision:
    route: str
    reason: str


class MediChatEngine:
    def __init__(self, model_name: str | None = None) -> None:
        config = get_config()
        selected_model = model_name or config.default_model_name
        model_path = config.model_artifact_path(selected_model)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run the training pipeline before launching the app."
            )
        self.config = config
        self.model_name = selected_model
        self.artifacts = PredictionArtifacts(
            model=joblib.load(model_path),
            responses=load_json(config.responses_path),
            retriever=KnowledgeRetriever(str(config.knowledge_base_path)),
            dialogue_manager=DialogueManager(),
        )

    def classify(self, english_text: str) -> tuple[str, float]:
        probabilities = self.artifacts.model.predict_proba([english_text])[0]
        best_index = int(np.argmax(probabilities))
        return str(self.artifacts.model.classes_[best_index]), float(probabilities[best_index])

    def should_use_grounded_response(self, confidence: float, retrieval_score: float) -> tuple[bool, str]:
        standard_supported = (
            confidence >= self.config.confidence_threshold
            and retrieval_score >= self.config.retrieval_threshold
        )
        if standard_supported:
            return True, "standard"

        # Rescue borderline predictions when retrieval is clearly strong.
        soft_confidence_threshold = max(0.30, self.config.confidence_threshold - 0.08)
        soft_retrieval_threshold = max(self.config.retrieval_threshold, self.config.retrieval_threshold * 2.5)
        soft_supported = (
            confidence >= soft_confidence_threshold
            and retrieval_score >= soft_retrieval_threshold
        )
        if soft_supported:
            return True, "soft_retrieval_rescue"

        return False, "unsupported"

    def route_message(self, english_text: str, context: Any) -> RouteDecision:
        lowered = (english_text or "").strip().lower()
        summary_triggers = {
            "summarize this chat",
            "summarize this conversation",
            "summary of this chat",
            "summary of this conversation",
            "what have we talked about",
            "recap this chat",
        }
        urgent_keywords = {
            "chest pain",
            "shortness of breath",
            "trouble breathing",
            "can't breathe",
            "severe bleeding",
            "passed out",
            "unconscious",
            "stroke",
            "heart attack",
            "seizure",
            "suicidal",
        }
        greeting_inputs = {"hi", "hello", "hey", "good morning", "good evening"}

        if context.history and any(trigger in lowered for trigger in summary_triggers):
            return RouteDecision(route="session_summary", reason="user_requested_conversation_summary")
        if any(keyword in lowered for keyword in urgent_keywords):
            return RouteDecision(route="urgent_safety", reason="urgent_symptom_keywords_detected")
        if lowered in greeting_inputs:
            return RouteDecision(route="lightweight_chat", reason="brief_greeting_detected")
        return RouteDecision(route="standard_qa", reason="default_medical_qa_flow")

    def process_message(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        source_language = detect_language(user_text)
        to_english: TranslationResult
        if source_language == "en":
            to_english = TranslationResult(
                text=user_text.strip(),
                translated=False,
                source_language="en",
                target_language="en",
                provider="local",
            )
        else:
            to_english = translate_text(user_text, "en", source_language=source_language)

        context = self.artifacts.dialogue_manager.get_context(session_id, language=source_language)
        route_decision = self.route_message(to_english.text, context)
        clarification = self.artifacts.dialogue_manager.plan_clarification(to_english.text, context.history)

        contextual_query = to_english.text
        intent = "agent_routed"
        confidence = 1.0
        retrieved = RetrievedContext(score=0.0, entries=[])
        supported = True

        if route_decision.route == "session_summary":
            intent = "session_summary"
            english_response = context.summary or self.artifacts.dialogue_manager.summarize_session(context.session_id)
            if not english_response:
                english_response = (
                    "There is not enough conversation history to summarize yet. "
                    "Send a few questions first, then ask for a summary again."
                )
        elif route_decision.route == "urgent_safety":
            intent = "urgent_safety"
            english_response = (
                "Your message may describe an urgent symptom. MediChat cannot assess emergencies. "
                "Please contact emergency services or an urgent care professional right away, especially if symptoms "
                "are severe, sudden, or getting worse.\n\n"
                f"{SAFETY_NOTICE}"
            )
        elif route_decision.route == "lightweight_chat":
            intent = "lightweight_chat"
            english_response = (
                "Hello. You can ask about symptoms, self-care, medication safety, or when to seek medical help. "
                "Please share your question in as much detail as you can."
            )
        elif clarification.should_clarify:
            intent = "clarification_needed"
            english_response = clarification.question
        else:
            contextual_query = self.artifacts.dialogue_manager.build_query(to_english.text, context.history)
            intent, confidence = self.classify(contextual_query)
            retrieved = self.artifacts.retriever.retrieve(contextual_query, intent=intent)

            supported, support_mode = self.should_use_grounded_response(confidence, retrieved.score)
            if supported:
                fallback_message = self.artifacts.responses.get(intent, UNSUPPORTED_MESSAGE)
                english_response = generate_controlled_response(
                    intent=intent,
                    user_question=to_english.text,
                    context_items=retrieved.entries,
                    conversation_history=context.history,
                    fallback_message=fallback_message,
                )
            else:
                intent = "unsupported"
                english_response = f"{UNSUPPORTED_MESSAGE}\n\n{SAFETY_NOTICE}"
        if "support_mode" not in locals():
            support_mode = "agent_override" if supported else "unsupported"

        translated_response = (
            translate_text(english_response, source_language, source_language="en").text
            if source_language != "en"
            else english_response
        )

        self.artifacts.dialogue_manager.database.log_message(
            context.session_id,
            role="user",
            original_text=user_text,
            english_text=to_english.text,
            language=source_language,
            metadata={
                "contextual_query": contextual_query,
                "route": route_decision.route,
                "route_reason": route_decision.reason,
            },
        )
        self.artifacts.dialogue_manager.database.log_message(
            context.session_id,
            role="assistant",
            original_text=translated_response,
            english_text=english_response,
            language=source_language,
            intent=intent,
            confidence=confidence,
            metadata={
                "retrieval_score": retrieved.score,
                "retrieved_titles": [entry["title"] for entry in retrieved.entries],
                "supported": supported,
                "support_mode": support_mode,
                "route": route_decision.route,
                "route_reason": route_decision.reason,
                "clarification_reason": clarification.reason,
            },
        )
        session_summary = self.artifacts.dialogue_manager.summarize_session(context.session_id)

        return {
            "session_id": context.session_id,
            "language": source_language,
            "english_text": to_english.text,
            "intent": intent,
            "confidence": confidence,
            "retrieval_score": retrieved.score,
            "supported": supported,
            "response": translated_response,
            "english_response": english_response,
            "english_translation": english_response if source_language != "en" else "",
            "retrieved_context": retrieved.entries,
            "support_mode": support_mode,
            "route": route_decision.route,
            "route_reason": route_decision.reason,
            "clarification_reason": clarification.reason,
            "session_summary": session_summary,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single MediChat prediction.")
    parser.add_argument("text")
    parser.add_argument("--model-name", default=get_config().default_model_name)
    args = parser.parse_args()
    engine = MediChatEngine(model_name=args.model_name)
    print(engine.process_message(args.text))
