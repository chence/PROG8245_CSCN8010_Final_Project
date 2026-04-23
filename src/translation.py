from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langdetect import DetectorFactory, detect
from openai import OpenAI

from src.config import get_config


load_dotenv()
DetectorFactory.seed = 0

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "pt-br": "Brazilian Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
}

LANGUAGE_DETECTION_METHOD = os.getenv("MEDICHAT_LANGUAGE_DETECTION_METHOD", "openai").lower()


@dataclass
class TranslationResult:
    text: str
    translated: bool
    source_language: str
    target_language: str
    provider: str


def _get_client() -> OpenAI | None:
    config = get_config()
    api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def detect_language_openai(text: str) -> str:
    client = _get_client()
    if client is None:
        print("---OpenAI client not available, falling back to langdetect")
        return detect_language_langdetect(text)

    config = get_config()
    try:
        response = client.chat.completions.create(
            model=config.openai_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a language detector. Respond with ONLY the ISO 639-1 language code "
                        "(e.g., 'en', 'zh', 'hi', 'fr', 'es', 'de', 'pt', 'tr', 'ja', 'ko', 'ru')."
                    ),
                },
                {
                    "role": "user",
                    "content": f"What language is this text in? Text: {text}",
                },
            ],
        )
        detected = response.choices[0].message.content.strip().lower()
        # print(f"---OpenAI detected language: {detected}")
        return "zh" if detected.startswith("zh") else detected
    except Exception as exc:
        print(f"---OpenAI language detection error: {exc}, falling back to langdetect")
        return detect_language_langdetect(text)


def detect_language_langdetect(text: str) -> str:
    sample = (text or "").strip()
    if not sample:
        return "en"

    try:
        chinese_count = sum(1 for char in sample if "\u4e00" <= char <= "\u9fff")
        korean_count = sum(
            1
            for char in sample
            if ("\uac00" <= char <= "\ud7af") or ("\u1100" <= char <= "\u11ff")
        )
        japanese_count = sum(
            1
            for char in sample
            if ("\u3040" <= char <= "\u309f") or ("\u30a0" <= char <= "\u30ff")
        )
        devanagari_count = sum(1 for char in sample if "\u0900" <= char <= "\u097f")
        arabic_count = sum(1 for char in sample if "\u0600" <= char <= "\u06ff")

        total_cjk = chinese_count + korean_count + japanese_count
        if total_cjk > 0:
            if chinese_count > korean_count and chinese_count > japanese_count:
                return "zh"
            if korean_count > chinese_count and korean_count > japanese_count:
                return "ko"
            if japanese_count > chinese_count and japanese_count > korean_count:
                return "ja"
            if chinese_count > 0:
                return "zh"

        if devanagari_count > 0:
            return "hi"

        if arabic_count > 0:
            return "ar"

        english_medical_words = {
            "what",
            "where",
            "when",
            "how",
            "why",
            "is",
            "are",
            "the",
            "a",
            "and",
            "or",
            "have",
            "i",
            "me",
            "my",
            "do",
            "can",
            "will",
            "should",
            "could",
            "would",
            "treat",
            "diabetes",
            "symptoms",
            "help",
            "pain",
            "fever",
            "cough",
            "sore",
            "medical",
            "health",
            "medicine",
            "doctor",
            "hospital",
            "patient",
            "disease",
        }
        text_lower = sample.lower()
        english_word_count = sum(1 for word in english_medical_words if word in text_lower)

        detected = detect(sample).lower()
        if detected.startswith("zh"):
            return "zh"

        if detected == "en":
            if english_word_count >= 1:
                return "en"
            return detected

        return detected
    except Exception as exc:
        print(f"---Langdetect error: {exc}")
        return "en"


def detect_language(text: str) -> str:
    if LANGUAGE_DETECTION_METHOD == "openai":
        return detect_language_openai(text)
    return detect_language_langdetect(text)


def translate_text(
    text: str,
    target_language: str,
    source_language: str | None = None,
) -> TranslationResult:
    source = source_language or detect_language(text)
    target = target_language.lower()
    if not text.strip() or source == target:
        return TranslationResult(
            text=text,
            translated=False,
            source_language=source,
            target_language=target,
            provider="local",
        )

    client = _get_client()
    if client is None:
        return TranslationResult(
            text=text,
            translated=False,
            source_language=source,
            target_language=target,
            provider="unavailable",
        )

    try:
        config = get_config()
        target_name = LANGUAGE_NAMES.get(target, target)
        response = client.chat.completions.create(
            model=config.openai_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a translation engine. Return only the translated text.",
                },
                {
                    "role": "user",
                    "content": f"Translate this text to {target_name}: {text}",
                },
            ],
        )
        translated = response.choices[0].message.content.strip()
        # print(f"---Translation to {target_name}: {translated}")
        return TranslationResult(
            text=translated,
            translated=True,
            source_language=source,
            target_language=target,
            provider="openai",
        )
    except Exception as exc:
        print(f"---Translation error ({target}): {exc}")
        return TranslationResult(
            text=text,
            translated=False,
            source_language=source,
            target_language=target,
            provider="error",
        )
