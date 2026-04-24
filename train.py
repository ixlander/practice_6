import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models import infer_signature

TARGET = "Job_Status"
DROP_COLS = ["Employee_ID"]

NUM_COLS = [
    "Age", "Years_Experience", "Salary_Before_AI", "Salary_After_AI",
    "Work_Hours_Per_Week", "Job_Satisfaction", "Productivity_Change_%",
]
CAT_COLS = [
    "Gender", "Education_Level", "Industry", "Job_Role",
    "AI_Adoption_Level", "Automation_Risk", "Upskilling_Required", "Remote_Work",
]

def train(
    data_path: str = "data/ai_job_impact.csv",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "ai-job-impact-experiment",
    model_name: str = "ai-job-impact-classifier",
    n_estimators: int = 200,
    test_size: float = 0.2,
    random_state: int = 42,
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")

    run_name = f"random-forest-{n_estimators}-trees"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "algorithm": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "test_size": test_size,
                "random_state": random_state,
                "class_weight": "balanced",
                "data_path": data_path,
            }
        )

        print(f"loading dataset: {data_path}")
        df = pd.read_csv(data_path).drop(columns=DROP_COLS, errors="ignore")
        print(f"rows: {len(df):,}  |  Columns: {df.shape[1]}")
        print(f"target distribution:{df[TARGET].value_counts().to_string()}")

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ])

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight="balanced",
            )),
        ])

        print("training RandomForest classifier...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "f1_weighted": f1_weighted,
            }
        )

        print(f"accuracy : {accuracy:.4f}")
        print("classification report:")
        report_text = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        print(report_text)
        mlflow.log_dict(report_dict, "classification_report.json")

        artifact = {
            "pipeline": pipeline,
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "classes": pipeline.classes_.tolist(),
        }
        joblib.dump(artifact, "model.joblib")
        print("model saved as model.joblib")

        mlflow.log_artifact("model.joblib", artifact_path="joblib")

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2),
        )

        model_uri = f"runs:/{run.info.run_id}/model"

    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(
        "model registered in MLflow Model Registry as "
        f"'{model_name}' version {registered_model.version}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ai_job_impact.csv")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="ai-job-impact-experiment")
    parser.add_argument("--model-name", default="ai-job-impact-classifier")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(
        data_path=args.data,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        random_state=args.random_state,
    )