# AI Job Impact Classifier

Continuation of Practical Task 6:
- trained a machine learning model;
- deployed it with FastAPI;
- containerized it with Docker.

This version extends the solution into a fuller ML system by adding:
- a Streamlit frontend for user interaction;
- MLflow experiment tracking and model registry integration.

## Project Structure

```text
practice6/
|- main.py
|- train.py
|- streamlit_app.py
|- model.joblib
|- requirements.txt
|- Dockerfile
|- README.md
|- data/
|  |- ai_job_impact.csv
```

## Features

- FastAPI endpoint for job-status prediction.
- Streamlit web UI that sends data to FastAPI and visualizes prediction probabilities.
- MLflow integration in training pipeline:
  - experiment creation;
  - parameter logging;
  - metric logging (accuracy, precision, recall, F1);
  - artifact logging (`model.joblib`, classification report, sklearn model);
  - model registration with name + version in MLflow Model Registry.

## Setup

```bash
pip install -r requirements.txt
```

## Train + Track with MLflow

Run training with MLflow tracking and model registration:

```bash
python train.py --tracking-uri sqlite:///mlflow.db --experiment-name ai-job-impact-experiment --model-name ai-job-impact-classifier
```

What this does:
- creates/uses experiment `ai-job-impact-experiment`;
- logs model parameters and evaluation metrics;
- logs model artifacts;
- registers model as `ai-job-impact-classifier` with an incremented version.

Optional: open MLflow UI in another terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open: http://localhost:5000

## MLflow Results (ai-job-impact-experiment2)

| Run ID | Run Name | Status | Start Time | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | n_estimators | test_size | random_state |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 06efa532141b49bc974a20b726fda4cb | random-forest-10-trees | FINISHED | 2026-04-24 16:28:19 | 0.9350 | 0.9158 | 0.9350 | 0.9216 | 10 | 0.5 | 42 |
| 0e747265ea574ed0b6da104e70c76976 | random-forest-10-trees | FINISHED | 2026-04-24 16:27:15 | 0.9375 | 0.9000 | 0.9375 | 0.9178 | 10 | 0.2 | 42 |
| ef803bfb20744b4897f1c8ef9f11340c | random-forest-200-trees | FINISHED | 2026-04-24 16:13:03 | 0.9325 | 0.9078 | 0.9325 | 0.9187 | 200 | 0.2 | 42 |

## Run FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- Health check: http://localhost:8000/
- Swagger docs: http://localhost:8000/docs

## Run Streamlit Frontend

In a separate terminal:

```bash
streamlit run streamlit_app.py
```

Then open the shown URL (typically http://localhost:8501), set FastAPI base URL if needed, and submit a profile.

## Docker (API)

```bash
docker build -t ai-job-api .
docker run -p 8000:8000 ai-job-api
```

## API Endpoints

### GET /

```json
{"message": "ML API is running"}
```

### POST /predict

Example request:

```json
{
  "Age": 35,
  "Gender": "Male",
  "Education_Level": "Bachelor",
  "Industry": "IT",
  "Job_Role": "Data Analyst",
  "Years_Experience": 8,
  "AI_Adoption_Level": "High",
  "Automation_Risk": "Medium",
  "Upskilling_Required": "Yes",
  "Salary_Before_AI": 75000,
  "Salary_After_AI": 80000,
  "Work_Hours_Per_Week": 40,
  "Remote_Work": "Yes",
  "Job_Satisfaction": 7,
  "Productivity_Change_%": 12.5
}
```

Example response:

```json
{
  "predicted_status": "Modified",
  "probabilities": {
    "Modified": 0.615,
    "Replaced": 0.085,
    "Unchanged": 0.3
  },
  "top_risk_factor": "High automation risk combined with high AI adoption in your industry"
}
```
