from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.config import get_config


SAFETY_NOTICE = (
    "This is general medical information only and not a diagnosis. "
    "If symptoms are severe, worsening, or urgent, contact a licensed clinician or emergency service."
)


def _get_client() -> OpenAI | None:
    config = get_config()
    if not config.openai_api_key:
        return None
    return OpenAI(api_key=config.openai_api_key)


def _local_response(
    intent: str,
    user_question: str,
    context_items: list[dict[str, Any]],
    fallback_message: str,
) -> str:
    if not context_items:
        return f"{fallback_message}\n\n{SAFETY_NOTICE}"

    question_lower = user_question.lower()
    if intent == "Medication Question":
        if "ibuprofen" in question_lower:
            return (
                "Yes, in general, ibuprofen is commonly used as an over-the-counter option for pain or fever relief for many adults. "
                "Please follow the package label and ask a pharmacist or clinician if you are unsure whether it is appropriate for you.\n\n"
                f"{SAFETY_NOTICE}"
            )
        if "acetaminophen" in question_lower or "paracetamol" in question_lower or "tylenol" in question_lower:
            return (
                "Yes, in general, acetaminophen is commonly used as an over-the-counter option for pain or fever relief. "
                "Please follow the package label carefully and avoid taking more than one product with the same active ingredient.\n\n"
                f"{SAFETY_NOTICE}"
            )

    bullet_points = []
    for item in context_items[:3]:
        bullet_points.append(f"- {item['title']}: {item['content']}")

    lead = {
        "Medication Question": "Here is general medication safety information based on the supported knowledge base:",
        "Seek Medical Help": "Here is general guidance on when to seek professional care:",
        "Self-Care Advice": "Here are safe self-care suggestions for mild symptoms:",
        "Symptom Inquiry": "Here is general information related to your symptoms:",
    }.get(intent, "Here is the supported information I found:")

    return (
        f"{lead}\n"
        f"{chr(10).join(bullet_points)}\n\n"
        "If your question goes beyond this supported scope, MediChat should defer to a healthcare professional.\n\n"
        f"{SAFETY_NOTICE}"
    )


def generate_controlled_response(
    *,
    intent: str,
    user_question: str,
    context_items: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]],
    fallback_message: str,
) -> str:
    client = _get_client()
    if client is None:
        return _local_response(intent, user_question, context_items, fallback_message)

    history_text = "\n".join(
        f"{item['role'].title()}: {item.get('english_text') or item.get('original_text', '')}"
        for item in conversation_history[-6:]
    )
    context_text = "\n".join(
        f"- [{item['intent']}] {item['title']}: {item['content']}"
        for item in context_items
    )
    config = get_config()
    try:
        response = client.chat.completions.create(
            model=config.openai_model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MediChat, a medical information chatbot for a course project. "
                        "Use only the supplied context. Do not diagnose, prescribe, or invent facts. "
                        "If the context is insufficient, say the question is unsupported. "
                        "Keep the answer concise, conversational, and safe. "
                        "If the supplied context clearly supports a common over-the-counter medicine such as ibuprofen "
                        "or acetaminophen as a general option, start with a short direct sentence such as "
                        "'Yes, in general...' and then add one short safety sentence before the reminder."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Intent: {intent}\n"
                        f"User question: {user_question}\n\n"
                        f"Recent conversation:\n{history_text or 'No prior turns.'}\n\n"
                        f"Allowed medical context:\n{context_text or 'No supported context.'}\n\n"
                        f"Fallback if insufficient: {fallback_message}\n\n"
                        "Write a short answer grounded only in the allowed medical context. "
                        "End with one sentence reminding the user that the system is informational, not diagnostic."
                    ),
                },
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _local_response(intent, user_question, context_items, fallback_message)
