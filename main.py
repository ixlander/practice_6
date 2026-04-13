import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="AI Job Impact Classifier",
    description=(
        "Predicts how AI adoption will affect an employee's job status "
        "(Unchanged / Modified / Replaced) based on their profile and industry context."
    ),
    version="1.0.0",
)

MODEL_PATH = "model.joblib"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"'{MODEL_PATH}' not found — run train.py first.")

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
CLASSES: list[str] = artifact["classes"]

class EmployeeFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=80, example=35)
    Gender: Literal["Male", "Female", "Other"] = Field(..., example="Male")
    Education_Level: Literal["High School", "Bachelor", "Master", "PhD"] = Field(..., example="Bachelor")
    Industry: Literal["IT", "Healthcare", "Finance", "Education", "Manufacturing", "Retail", "Marketing"] = Field(..., example="IT")
    Job_Role: str = Field(..., example="Data Analyst")
    Years_Experience: int = Field(..., ge=0, le=50, example=8)
    AI_Adoption_Level: Literal["Low", "Medium", "High"] = Field(..., example="High")
    Automation_Risk: Literal["Low", "Medium", "High"] = Field(..., example="Medium")
    Upskilling_Required: Literal["Yes", "No"] = Field(..., example="Yes")
    Salary_Before_AI: int = Field(..., ge=0, example=75000)
    Salary_After_AI: int = Field(..., ge=0, example=80000)
    Work_Hours_Per_Week: int = Field(..., ge=1, le=100, example=40)
    Remote_Work: Literal["Yes", "No"] = Field(..., example="Yes")
    Job_Satisfaction: int = Field(..., ge=1, le=10, example=7)
    Productivity_Change_pct: float = Field(..., alias="Productivity_Change_%", example=12.5)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
                "Productivity_Change_%": 12.5,
            }
        }


class PredictionResponse(BaseModel):
    predicted_status: str
    probabilities: dict[str, float]
    top_risk_factor: str

