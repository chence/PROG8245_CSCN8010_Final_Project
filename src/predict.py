from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processing import DEFAULT_RESPONSES_PATH, load_artifact, load_response_templates
from src.utils import detect_language, translate_text

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

# Confidence threshold for intent prediction (0.0 - 1.0)
# If the model's confidence is below this, the chatbot will return a fallback message
CONFIDENCE_THRESHOLD = 0.5

# Fallback message for out-of-scope or low-confidence queries
FALLBACK_MESSAGE = "Sorry, this question is currently not supported by MediChat."


def predict_intent_with_confidence(
    text: str,
    model_path: str = 'models/baseline_nb.joblib',
    threshold: float = CONFIDENCE_THRESHOLD,
    use_similarity_check: bool = False
) -> tuple[str | None, float]:
    """
    Predict intent with confidence score using predict_proba().
    
    Args:
        text: Input text (should be in English)
        model_path: Path to the trained model
        threshold: Confidence threshold (0-1). If prediction confidence < threshold,
                   returns None to trigger fallback message
        use_similarity_check: If True, also checks TF-IDF vector similarity with
                             training data to detect out-of-scope questions
    
    Returns:
        Tuple of (intent_label, confidence_score)
        If confidence < threshold, returns (None, confidence_score)
    """
    model = load_artifact(model_path)
    
    # Get prediction probabilities for all classes
    proba = model.predict_proba([text])[0]
    
    # Get the highest probability and its corresponding class index
    max_confidence = np.max(proba)
    predicted_idx = np.argmax(proba)
    predicted_intent = model.classes_[predicted_idx]
    
    print(f"---Model confidence: {max_confidence:.4f} (threshold: {threshold})")
    print(f"---All class probabilities: {dict(zip(model.classes_, proba))}")
    
    # Optional: Cosine similarity check with training data
    if use_similarity_check:
        try:
            # Get TF-IDF vectorizer from the pipeline
            vectorizer = model.named_steps['tfidf']
            
            # Transform the input text
            user_vector = vectorizer.transform([text])
            
            # If we have access to training vectors (would need to save them),
            # compute average cosine similarity. For now, we'll use a simple heuristic:
            # Check if the TF-IDF vector is sparse (few non-zero features = out of domain)
            sparsity = 1 - (user_vector.nnz / (user_vector.shape[0] * user_vector.shape[1]))
            print(f"---TF-IDF sparsity: {sparsity:.2%} (higher = more out-of-domain)")
        except Exception as e:
            print(f"---Similarity check failed: {e}")
    
    # Return None if confidence below threshold (triggers fallback response)
    if max_confidence < threshold:
        print(f"---Confidence {max_confidence:.4f} below threshold {threshold}")
        return None, max_confidence
    
    return predicted_intent, max_confidence


def predict_intent(
    text: str,
    model_path: str = 'models/baseline_nb.joblib',
    responses_path: str = str(DEFAULT_RESPONSES_PATH),
    confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> Dict[str, str | float]:
    """
    Main prediction function with confidence-based fallback mechanism.
    
    Args:
        text: Input text for prediction
        model_path: Path to the trained model
        responses_path: Path to response templates
        confidence_threshold: Confidence threshold for accepting predictions
    
    Returns:
        Dictionary containing language, english_text, intent, response, 
        english_response, and confidence_score
    """
    print(f"\n===Predicting text: {text}")

    model = load_artifact(model_path)
    responses = load_response_templates(responses_path)

    language = detect_language(text)
    print(f"---Detected language: {language}")
    
    english_text = translate_text(text, 'en') if language != 'en' else text
    print(f"---English translation: {english_text}")
    
    # Use confidence-based prediction instead of direct predict()
    intent, confidence = predict_intent_with_confidence(
        english_text,
        model_path=model_path,
        threshold=confidence_threshold,
        use_similarity_check=False
    )
    
    # If intent is None (confidence below threshold), use fallback message
    if intent is None:
        english_response = FALLBACK_MESSAGE
    else:
        english_response = responses.get(
            intent,
            'I can only support basic non-emergency medical information at the moment.'
        )
    
    final_response = translate_text(english_response, language) if language != 'en' else english_response

    return {
        'language': language,
        'english_text': english_text,
        'intent': intent if intent is not None else 'unknown',
        'response': final_response,
        'english_response': english_response,
        'confidence_score': float(confidence),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a MediChat intent label from text.')
    parser.add_argument('text', help='Input text for prediction.')
    parser.add_argument('--model-path', default='models/baseline_nb.joblib',
                        help='Path to the trained model')
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD,
                        help=f'Confidence threshold for accepting predictions (default: {CONFIDENCE_THRESHOLD})')
    args = parser.parse_args()

    result = predict_intent(args.text, model_path=args.model_path, confidence_threshold=args.threshold)
    
    # Pretty print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Detected Language: {result['language']}")
    print(f"Intent: {result['intent']}")
    print(f"Confidence Score: {result['confidence_score']:.4f}")
    print()
    
    # Show Original → English → Response flow for non-English input
    if result['language'] != 'en':
        print("📝 Original Input:")
        print(f"   {args.text}")
        print()
        print("🔤 English Translation:")
        print(f"   {result['english_text']}")
        print()
        print("💬 English Response:")
        print(f"   {result['english_response']}")
        print()
        print("🌐 Final Response (translated):")
        print(f"   {result['response']}")
    else:
        print("💬 Response:")
        print(f"   {result['response']}")
    
    print("="*60)
