# PROG8245 Final Project - MediChat

MediChat is a multilingual, audio/text-ready medical chatbot prototype for the PROG8245 final project. The current implementation focuses on intent classification for narrow-domain respiratory health questions using classical NLP methods required by the course brief: TF-IDF vectorization, optional dimensionality reduction, and machine learning classifiers.

# Group Info
- Group Name: Group1, Group8
- Group Members:
    - Ce Chen, 9007166
    - Zhuoran Zhang, 9048508
    - Haibo Yuan, 9010929
    - Abdallah Mohamed, 9089339

# GitHub Link
https://github.com/chence/PROG8245_CSCN8010_Final_Project_Dry_Run.git

 # Mp4 Link
https://github.com/chence/PROG8245_CSCN8010_Final_Project_Dry_Run/blob/master/mp4/FinalProjectDryRun.mp4

## Project Goal
The goal is to build a safe educational chatbot that can classify a user's medical question into an intent category and return an appropriate general response. The project is intentionally limited to general information only and does not provide diagnosis.

## Current Scope
- Baseline model: TF-IDF + Naive Bayes
- Advanced models prepared: SVD + Logistic Regression, PCA + Logistic Regression
- Multilingual text support through language detection and translation fallback
- Gradio web interface scaffold in `app.py`
- DVC-ready repository structure for reproducible data and model tracking

## Folder Structure
```text
PROG8245_Final_Project/
├── data/
│   ├── raw/                       # Raw dataset files
│   │   ├── medical_intent_dataset.csv
│   │   └── intent_responses.json
│   └── processed/                 # Processed dataset files
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── baseline_nb.joblib         # Trained baseline model
│   └── baseline_nb.metadata.json  # Model metadata
├── notebooks/
│   └── FinalProjectDryRun_completed.ipynb
├── documentation/
│   ├── architecture.md
│   ├── development_plan.md
│   ├── evaluation_metrics.json
│   ├── pitch.md
│   └── user_manual.md
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── predict.py                 # Intent prediction & main logic
│   ├── train.py                   # Model training pipeline
│   ├── evaluate.py                # Model evaluation
│   └── utils.py                   # Language detection & translation utilities
├── app.py                         # Gradio web interface
├── dvc.yaml                       # DVC pipeline configuration
├── params.yaml                    # Training parameters
├── requirements.txt
├── README.md
└── .env                           # API keys (add manually, not in repo)
```

## Quick Start
### 1. Create environment and install packages
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Train the baseline model
```bash
python -m src.train --model-name baseline_nb --model-out models/baseline_nb.joblib
```

### 3. Evaluate the model
```bash
python -m src.evaluate --model-path models/baseline_nb.joblib
```

### 4. Run the web app
```bash
python app.py
```

### 5. Test Predict (Optional, test only)
```bash
python -m src.predict "QUESTION"
```

## DVC Workflow
Initialize Git and DVC if you want to continue tracking experiments:
```bash
git init
dvc init
dvc repro
```

This project already includes a starter `dvc.yaml` pipeline and `params.yaml` file.

## Recommended Team Development Order
1. Set up GitHub + DVC repository
2. Confirm the baseline model works end-to-end
3. Add multilingual translation logic
4. Replace the small demo dataset with the HPAI-BSC medical-specialities dataset
5. Compare baseline, SVD, and PCA models
6. Add audio input and improve the UI

## Safety Note
MediChat is an academic prototype. It provides general educational support only. It must not be used for emergency care, diagnosis, or medication decisions.
