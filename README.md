# Casting Defect Classifier: End-to-End ML Pipeline

## 1. Overview
This repository contains a complete, end-to-end Machine Learning pipeline designed for a manufacturing quality-assurance team. It automatically flags defective cast products from images captured on the production line. 

This project fulfills the EC-1 Mini-Project requirements for Machine Learning Engineering (PCAM* ZC412) under "Flavor B"[cite: 1]. It takes raw image data through feature engineering, model comparison, packaging, deployment, and live monitoring[cite: 1].

## 2. System Architecture
Our pipeline connects the trained model to the outside world via a REST API, while seamlessly logging data for drift detection and retraining[cite: 1].

```text
[ Factory Camera ] 
        │
        ▼ (Image Upload)
┌────────────────────────────────────────────────────────┐
│                   REST API (FastAPI)                   │ 
│  Endpoint: /predict                                    │
└───────┬────────────────────────────────────────┬───────┘
        │                                        │
        ▼                                        ▼ (Logs to CSV)
┌──────────────────────┐               ┌─────────────────────────┐
│ Packaged Model       │               │ Live Monitoring         │
│ (PyTorch ResNet18)   │               │ - Accuracy Drift Checks │
│ - Image Translation  │               │ - Confidence Tracking   │
│ - Defect Prediction  │               │ - Retraining Triggers   │
└──────────────────────┘               └─────────────────────────┘
```

## 3. Technology Stack
* **Data Versioning:** DVC (Data Version Control)[cite: 1]
* **Model Training:** PyTorch (Custom CNN & ResNet18 Transfer Learning)[cite: 1]
* **Experiment Tracking:** MLflow[cite: 1]
* **Model Serving / API:** FastAPI & Uvicorn[cite: 1]
* **Monitoring:** Custom CSV logging & Python visualization[cite: 1]

## 4. Local Setup & Installation
To run this project on your local machine, follow these steps to set up an isolated Python environment:

**Step 1: Create a virtual environment**
`python -m venv .venv`

**Step 2: Activate the environment**
* On Windows: `.\.venv\Scripts\activate`
* On Mac/Linux: `source .venv/bin/activate`

**Step 3: Install dependencies**
`pip install -r requirements.txt`

## 5. Running the Application (API Service)
To start the API server and test the model's predictions:

**Step 1: Start the FastAPI Server**
`uvicorn src.serving.app:app --reload`[cite: 6]

**Step 2: Access the UI**
Open your web browser and navigate to: `http://127.0.0.1:8000/docs`[cite: 6]

**Step 3: Test a Prediction**
Expand the green `POST /predict` box, click **"Try it out"**, upload a test image (e.g., from the `data/raw/test/` folder), and click **"Execute"**[cite: 1, 6]. The server will return a JSON response showing the defect label and the model's confidence score.

## 6. Project Milestones & Reports
* **Week 1 (Data Engineering):** Versioned raw and processed datasets via DVC, handling edge cases and data validation[cite: 1].
* **Week 2 (Model Comparison):** Evaluated a custom CNN against a ResNet18 Transfer Learning model. ResNet18 was selected as the champion model with an accuracy of 97.48% (See `model_comparison_report.md`)[cite: 1, 11].
* **Week 3 (Deployment):** Packaged the model and deployed it as a robust REST API capable of handling invalid uploads[cite: 1].
* **Week 4 (Drift & Monitoring):** The API automatically logs all traffic to `prediction_log.csv`[cite: 5, 8]. We simulated environmental drift (blur, lighting, rotation) to establish strict thresholds for when the model fails (See `drift_degradation_summary.csv`)[cite: 1, 7]. The exact business rules and automated pipeline for retraining the model are documented in `retraining_trigger_design.md`[cite: 1].
