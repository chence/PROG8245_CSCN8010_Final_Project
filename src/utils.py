"""
Utility functions for language detection and translation.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langdetect import detect
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

# Language detection method: 'langdetect' (fast, local) or 'openai' (accurate, requires API)
LANGUAGE_DETECTION_METHOD = 'langdetect'  # Change to 'openai' if you prefer


def detect_language_openai(text: str) -> str:
    """
    Detect language using OpenAI API for higher accuracy.
    Returns ISO 639-1 language code (e.g., 'en', 'zh', 'hi', 'tr', 'pt')
    """
    if client is None:
        print("---OpenAI client not available, falling back to langdetect")
        return detect_language_langdetect(text)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a language detector. Respond with ONLY the ISO 639-1 language code (e.g., 'en', 'zh', 'hi', 'fr', 'es', 'de', 'pt', 'tr', 'ja', 'ko', 'ru'). No other text."
                },
                {
                    "role": "user",
                    "content": f"What language is this text in? Text: {text}"
                }
            ],
            temperature=0.1,
        )
        detected = response.choices[0].message.content.strip().lower()
        print(f"---OpenAI detected language: {detected}")
        return detected
    except Exception as e:
        print(f"---OpenAI language detection error: {e}, falling back to langdetect")
        return detect_language_langdetect(text)


def detect_language_langdetect(text: str) -> str:
    """
    Detect language using langdetect library with heuristics.
    Uses CJK and Devanagari character detection for improved accuracy.
    """
    try:
        # Check for Chinese characters (CJK Unified Ideographs)
        chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        # Check for Korean characters (Hangul)
        korean_count = sum(1 for c in text if '\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff')
        # Check for Japanese characters (Hiragana + Katakana)
        japanese_count = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
        # Check for Devanagari (Hindi, Sanskrit, etc.)
        devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097f')
        # Check for Arabic
        arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
        
        total_cjk = chinese_count + korean_count + japanese_count
        
        # If we have CJK characters, use heuristics
        if total_cjk > 0:
            if chinese_count > korean_count and chinese_count > japanese_count:
                return 'zh'
            elif korean_count > chinese_count and korean_count > japanese_count:
                return 'ko'
            elif japanese_count > chinese_count and japanese_count > korean_count:
                return 'ja'
        
        # If we have Devanagari characters, return Hindi
        if devanagari_count > 0:
            return 'hi'
        
        # If we have Arabic characters, return Arabic
        if arabic_count > 0:
            return 'ar'
        
        # Check for common English medical words before relying on langdetect
        english_medical_words = {'what', 'where', 'when', 'how', 'why', 'is', 'are', 'the', 'a', 'and', 'or', 'have', 'i', 'me', 'my', 'do', 'can', 'will', 'should', 'could', 'would', 'treat', 'diabetes', 'symptoms', 'help', 'pain', 'fever', 'cough', 'sore', 'medical', 'health', 'medicine', 'doctor', 'hospital', 'patient', 'disease'}
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_medical_words if word in text_lower)
        
        # Fall back to langdetect for non-CJK, non-Devanagari, non-Arabic text
        detected = detect(text)
        
        # If langdetect detected English, verify with medical word check
        if detected == 'en':
            if english_word_count >= 1:  # At least one English medical word
                return 'en'
            else:
                # If no English words found but detected as 'en', likely false positive
                return detected
        
        # For other languages, trust langdetect's detection
        return detected
    except Exception as e:
        print(f"---Langdetect error: {e}")
        return 'en'


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Uses the method specified by LANGUAGE_DETECTION_METHOD constant.
    
    Args:
        text: Input text to detect language for
        
    Returns:
        ISO 639-1 language code (e.g., 'en', 'zh', 'hi', 'tr', 'pt')
    """
    if LANGUAGE_DETECTION_METHOD == 'openai':
        return detect_language_openai(text)
    else:
        return detect_language_langdetect(text)


def translate_text(text: str, dest: str) -> str:
    """
    Translate text to the destination language using OpenAI API.
    
    Args:
        text: Text to translate
        dest: Destination language code (e.g., 'en', 'zh', 'hi', 'tr', 'pt')
        
    Returns:
        Translated text, or original text if translation fails or client is unavailable
    """
    if client is None:
        return text
    try:
        # Map language codes to full language names for better translation
        # Handles both simple codes (zh, en) and full locale codes (zh-cn, en-us)
        language_map = {
            'en': 'English',
            'zh': 'Chinese',
            'zh-cn': 'Simplified Chinese',
            'zh-tw': 'Traditional Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'pt': 'Portuguese',
            'pt-br': 'Brazilian Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish',
        }
        dest_lang = language_map.get(dest.lower(), dest)

        prompt = f"Translate this text to {dest_lang}: {text}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        translated = response.choices[0].message.content
        print(f"---Translation to {dest_lang}: {translated}")
        return translated
    except Exception as e:
        print(f"---Translation error ({dest}): {e}")
        return text
