# DVC Pipeline Guide

## Overview

This project uses DVC to make the machine learning workflow reproducible and easy to rerun.

The pipeline has three stages:

1. `prepare`
2. `train`
3. `evaluate`

These stages are defined in [dvc.yaml](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/dvc.yaml), and their parameters are stored in [params.yaml](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/params.yaml).

## Why DVC Is Used Here

DVC helps the team:

- keep the ML workflow reproducible
- rerun only the stages affected by a change
- track which data, code, and parameters produced each model
- demonstrate an end-to-end pipeline during the presentation

## Pipeline Structure

### Stage 1: `prepare`

Purpose:

- load the raw dataset
- clean and validate it
- create train/test splits
- write dataset summary information

Command:

```bash
./.venv/bin/python -m src.data_processing --data-path ${prepare.data_path} --train-out ${prepare.train_out} --test-out ${prepare.test_out} --summary-out ${prepare.summary_out} --test-size ${prepare.test_size}
```

Main inputs:

- [src/data_processing.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/data_processing.py)
- [src/config.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/config.py)
- [src/utils.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/utils.py)
- raw dataset from `prepare.data_path`

Outputs:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/dataset_summary.json`

### Stage 2: `train`

Purpose:

- train the three ML models
- save model artifacts and metadata

Command:

```bash
./.venv/bin/python -m src.train --train-path ${train.train_path} --model-dir ${train.model_dir} --tfidf-max-features ${train.tfidf_max_features} --tfidf-ngram-max ${train.tfidf_ngram_max} --svd-components ${train.svd_components} --pca-components ${train.pca_components} --logistic-max-iter ${train.logistic_max_iter}
```

Main inputs:

- [src/train.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/train.py)
- processed training split
- training parameters from `params.yaml`

Outputs:

- `models/baseline_nb.joblib`
- `models/svd_logreg.joblib`
- `models/pca_logreg.joblib`
- `models/baseline_nb.metadata.json`
- `models/svd_logreg.metadata.json`
- `models/pca_logreg.metadata.json`
- `models/training_summary.json`

### Stage 3: `evaluate`

Purpose:

- evaluate the trained models on the test split
- save metrics and confusion matrices

Command:

```bash
./.venv/bin/python -m src.evaluate --test-path ${evaluate.test_path} --model-dir ${evaluate.model_dir}
```

Main inputs:

- [src/evaluate.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/evaluate.py)
- processed test split
- trained model artifacts

Outputs:

- `documentation/evaluation_metrics.json`
- `documentation/model_comparison.csv`
- `documentation/model_comparison.md`
- `documentation/confusion_matrix_baseline_nb.png`
- `documentation/confusion_matrix_svd_logreg.png`
- `documentation/confusion_matrix_pca_logreg.png`

## Parameters

The main pipeline parameters are stored in [params.yaml](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/params.yaml).

### `prepare`

- `data_path`
- `train_out`
- `test_out`
- `summary_out`
- `test_size`

### `train`

- `train_path`
- `model_dir`
- `tfidf_max_features`
- `tfidf_ngram_max`
- `svd_components`
- `pca_components`
- `logistic_max_iter`

### `evaluate`

- `test_path`
- `model_dir`

If you change one of these values, DVC can detect that the corresponding stage is outdated.

## Most Common Commands

### Check pipeline status

```bash
dvc status
```

This shows whether any stage outputs are out of date relative to their dependencies or parameters.

### Reproduce the full pipeline

```bash
dvc repro
```

This reruns only the stages that need updating.

### Reproduce from a specific stage

```bash
dvc repro train
```

This starts at `train` and continues downstream if needed.

### Reproduce only evaluation

```bash
dvc repro evaluate
```

This is useful when evaluation artifacts are missing or stale.

## Recommended Workflow

### Case 1: You changed training data

Examples:

- raw CSV changed
- train/test split settings changed
- cleaning logic changed in `src/data_processing.py`

Recommended command:

```bash
dvc repro
```

Why:

- `prepare` must rerun
- `train` must rerun because train data changed
- `evaluate` must rerun because model outputs changed

### Case 2: You changed training code or model parameters

Examples:

- [src/train.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/train.py) changed
- `params.yaml` training values changed

Recommended command:

```bash
dvc repro
```

Why:

- `train` must rerun
- `evaluate` must rerun

### Case 3: You changed only evaluation logic

Examples:

- [src/evaluate.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/evaluate.py) changed

Recommended command:

```bash
dvc repro evaluate
```

Why:

- models do not need retraining
- only evaluation outputs need regeneration

### Case 4: You changed app behavior or knowledge base only

Examples:

- `app.py`
- `src/predict.py`
- `src/dialogue_manager.py`
- `src/translation.py`
- `data/raw/medical_knowledge_base.json`

Recommended command:

```bash
.venv/bin/python app.py
```

Why:

- these changes affect runtime behavior
- they do not change training artifacts
- DVC retraining is not needed

## Example Session

### Full pipeline run

```bash
source .venv/bin/activate
dvc status
dvc repro
```

### Launch the app after the pipeline

```bash
.venv/bin/python app.py
```

## How DVC Decides What to Rerun

DVC checks:

- declared code dependencies in `dvc.yaml`
- declared data dependencies in `dvc.yaml`
- declared parameters in `params.yaml`
- expected outputs from each stage

If any dependency or parameter changes, the affected stage is marked as outdated.

## Files Controlled by the Pipeline

### Input side

- [data/raw/medical_intent_dataset.csv](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/data/raw/medical_intent_dataset.csv)
- [src/data_processing.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/data_processing.py)
- [src/train.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/train.py)
- [src/evaluate.py](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/src/evaluate.py)
- [params.yaml](/Users/chrischen/work/PROG8245/PROG8245_Final_Project/params.yaml)

### Output side

- processed datasets in `data/processed/`
- model artifacts in `models/`
- evaluation artifacts in `documentation/`

## Good Presentation Talking Points

- The project is not just a chatbot UI; it also has a reproducible ML pipeline.
- DVC separates data preparation, training, and evaluation into explicit stages.
- The team can show `dvc repro` as evidence of reproducibility.
- Runtime improvements such as knowledge-base updates or agent logic do not always require retraining.

## Summary

Use DVC when you change:

- training data
- training code
- evaluation code
- pipeline parameters

Do not retrain with DVC when you change only:

- UI behavior
- agent routing logic
- translation defaults
- the retrieval knowledge base for runtime responses

In those cases, restarting the app is enough.
