from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path = BASE_DIR
    raw_data_path: Path = BASE_DIR / "data/raw/medical_intent_dataset.csv"
    processed_dir: Path = BASE_DIR / "data/processed"
    knowledge_base_path: Path = BASE_DIR / "data/raw/medical_knowledge_base.json"
    responses_path: Path = BASE_DIR / "data/raw/intent_responses.json"
    models_dir: Path = BASE_DIR / "models"
    documentation_dir: Path = BASE_DIR / "documentation"
    database_path: Path = BASE_DIR / "data/medichat.sqlite3"
    train_split_path: Path = BASE_DIR / "data/processed/train.csv"
    test_split_path: Path = BASE_DIR / "data/processed/test.csv"
    prepared_summary_path: Path = BASE_DIR / "data/processed/dataset_summary.json"
    evaluation_json_path: Path = BASE_DIR / "documentation/evaluation_metrics.json"
    comparison_csv_path: Path = BASE_DIR / "documentation/model_comparison.csv"
    comparison_md_path: Path = BASE_DIR / "documentation/model_comparison.md"
    confidence_threshold: float = float(os.getenv("MEDICHAT_CONFIDENCE_THRESHOLD", "0.45"))
    retrieval_threshold: float = float(os.getenv("MEDICHAT_RETRIEVAL_THRESHOLD", "0.1"))
    top_k_context: int = int(os.getenv("MEDICHAT_TOP_K_CONTEXT", "3"))
    default_model_name: str = os.getenv("MEDICHAT_MODEL_NAME", "svd_logreg")
    openai_model: str = os.getenv("MEDICHAT_OPENAI_MODEL", "gpt-4o-mini")
    openai_transcription_model: str = os.getenv(
        "MEDICHAT_TRANSCRIPTION_MODEL",
        "gpt-4o-mini-transcribe",
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    def model_artifact_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.joblib"

    def metadata_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.metadata.json"

    def confusion_matrix_path(self, model_name: str) -> Path:
        return self.documentation_dir / f"confusion_matrix_{model_name}.png"


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
