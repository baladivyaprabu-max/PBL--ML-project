import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database.db import (
    init_db,
    insert_prediction,
    get_predictions,
    get_latest_prediction,
    get_prediction_count
)

app = FastAPI(
    title="AI-Powered Water Demand Prediction System API",
    description="REST API for water demand forecasting, analytics, and model performance metrics",
    version="1.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
DATA_PATH = os.path.join(BASE_DIR, "data", "merged_water_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "water_demand_xgb_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "model_metrics.json")
DEFAULTS_PATH = os.path.join(BASE_DIR, "ml", "country_defaults.json")

# In-memory cached resources
model = None
metrics_data = {}
country_defaults_data = {}
df_data = None

@app.on_event("startup")
def startup_event():
    global model, metrics_data, country_defaults_data, df_data
    init_db()
    
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
            
    if os.path.exists(DEFAULTS_PATH):
        with open(DEFAULTS_PATH, "r") as f:
            country_defaults_data = json.load(f)
            
    if os.path.exists(DATA_PATH):
        df_data = pd.read_csv(DATA_PATH)
        print(f"Loaded merged data with {len(df_data)} rows.")

class PredictInput(BaseModel):
    country: Optional[str] = "Global Average"
    year: Optional[int] = 2024
    population: float = Field(..., gt=0, description="Total population count")
    per_capita_water_use: float = Field(..., gt=0, description="Per capita water use in liters/day")
    agricultural_water_use: float = Field(..., ge=0, le=100, description="Agricultural water use percentage")
    industrial_water_use: float = Field(..., ge=0, le=100, description="Industrial water use percentage")
    household_water_use: float = Field(..., ge=0, le=100, description="Household water use percentage")
    rainfall_impact: float = Field(..., ge=0, description="Annual precipitation in mm")
    groundwater_depletion_rate: float = Field(..., ge=0, description="Groundwater depletion rate percentage")

def determine_demand_status(val: float) -> tuple[str, str, str]:
    if val < 450:
        return (
            "Low Demand",
            "low",
            "Water demand is below typical baseline. Municipal reservoirs and water infrastructure are operating well within capacity."
        )
    elif val <= 550:
        return (
            "Moderate Demand",
            "moderate",
            "Water demand is in the normal balanced operating range. Standard conservation and aquifer monitoring are advised."
        )
    elif val <= 650:
        return (
            "High Demand",
            "high",
            "Elevated consumption detected. Recommended to increase water recycling, optimize agricultural irrigation, and track storage levels."
        )
    else:
        return (
            "Critical Demand",
            "critical",
            "Urgent: Water demand exceeds sustainable thresholds. Prioritize groundwater replenishment and industrial efficiency protocols."
        )

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "dataset_rows": len(df_data) if df_data is not None else 0
    }

@app.get("/api/countries")
def get_countries():
    """Returns list of distinct countries in the dataset along with baseline stats."""
    if df_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    countries = sorted(df_data["Country"].unique().tolist())
    country_list = []
    
    for c in countries:
        c_df = df_data[df_data["Country"] == c]
        country_list.append({
            "name": c,
            "record_count": len(c_df),
            "avg_consumption": round(float(c_df["Total Water Consumption (Billion Cubic Meters)"].mean()), 2),
            "avg_population": round(float(c_df["Population"].mean())),
            "avg_rainfall": round(float(c_df["Rainfall Impact (Annual Precipitation in mm)"].mean()), 1)
        })
    return {"countries": country_list}

@app.get("/api/country-defaults/{country}")
def get_country_defaults(country: str):
    """Returns pre-filled real feature averages for a specific country."""
    if country in country_defaults_data:
        return country_defaults_data[country]
    # Fallback to dataset mean
    if df_data is not None:
        means = df_data.mean(numeric_only=True)
        return {
            "Country": country,
            "LatestYear": 2024,
            "Population": round(float(means.get("Population", 100000000))),
            "Per Capita Water Use (Liters per Day)": round(float(means.get("Per Capita Water Use (Liters per Day)", 276.0)), 1),
            "Agricultural Water Use (%)": round(float(means.get("Agricultural Water Use (%)", 50.0)), 1),
            "Industrial Water Use (%)": round(float(means.get("Industrial Water Use (%)", 27.8)), 1),
            "Household Water Use (%)": round(float(means.get("Household Water Use (%)", 24.8)), 1),
            "Rainfall Impact (Annual Precipitation in mm)": round(float(means.get("Rainfall Impact (Annual Precipitation in mm)", 1544.8)), 1),
            "Groundwater Depletion Rate (%)": round(float(means.get("Groundwater Depletion Rate (%)", 2.57)), 2),
            "Avg Total Water Consumption": round(float(means.get("Total Water Consumption (Billion Cubic Meters)", 501.22)), 2)
        }
    raise HTTPException(status_code=404, detail="Country defaults not found")

@app.get("/api/dashboard")
def get_dashboard():
    """Returns overview statistics and chart data for the dashboard."""
    if df_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    total_countries = int(df_data["Country"].nunique())
    total_records = int(len(df_data))
    avg_consumption = round(float(df_data["Total Water Consumption (Billion Cubic Meters)"].mean()), 2)
    latest_pred = get_latest_prediction()
    
    # 1. Consumption by Year
    year_group = df_data.groupby("Year").agg({
        "Total Water Consumption (Billion Cubic Meters)": ["mean", "sum"],
        "Rainfall Impact (Annual Precipitation in mm)": "mean"
    }).reset_index()
    year_group.columns = ["year", "avg_consumption", "total_consumption", "avg_rainfall"]
    year_data = [
        {
            "year": int(r["year"]),
            "avg_consumption": round(float(r["avg_consumption"]), 2),
            "total_consumption": round(float(r["total_consumption"]), 2),
            "avg_rainfall": round(float(r["avg_rainfall"]), 1)
        }
        for _, r in year_group.iterrows()
    ]
    
    # 2. Population vs Water Consumption (per country latest/mean)
    pop_consumption = []
    country_grouped = df_data.groupby("Country").agg({
        "Population": "mean",
        "Total Water Consumption (Billion Cubic Meters)": "mean",
        "Per Capita Water Use (Liters per Day)": "mean"
    }).reset_index()
    
    for _, row in country_grouped.iterrows():
        pop_consumption.append({
            "country": row["Country"],
            "population_millions": round(float(row["Population"]) / 1_000_000, 2),
            "consumption": round(float(row["Total Water Consumption (Billion Cubic Meters)"]), 2),
            "per_capita": round(float(row["Per Capita Water Use (Liters per Day)"]), 1)
        })
        
    # 3. Sectoral usage breakdown
    sector_usage = [
        {"sector": "Agricultural", "percentage": round(float(df_data["Agricultural Water Use (%)"].mean()), 1), "color": "#10b981"},
        {"sector": "Industrial", "percentage": round(float(df_data["Industrial Water Use (%)"].mean()), 1), "color": "#0284c7"},
        {"sector": "Household", "percentage": round(float(df_data["Household Water Use (%)"].mean()), 1), "color": "#f59e0b"}
    ]
    
    # 4. Country comparison (sorted by consumption)
    country_comparison = []
    c_summary = df_data.groupby("Country").agg({
        "Total Water Consumption (Billion Cubic Meters)": "mean",
        "Rainfall Impact (Annual Precipitation in mm)": "mean",
        "Groundwater Depletion Rate (%)": "mean"
    }).reset_index().sort_values("Total Water Consumption (Billion Cubic Meters)", ascending=False)
    
    for _, row in c_summary.iterrows():
        country_comparison.append({
            "country": row["Country"],
            "consumption": round(float(row["Total Water Consumption (Billion Cubic Meters)"]), 2),
            "rainfall": round(float(row["Rainfall Impact (Annual Precipitation in mm)"]), 1),
            "depletion": round(float(row["Groundwater Depletion Rate (%)"]), 2)
        })
        
    return {
        "summary": {
            "total_countries": total_countries,
            "total_records": total_records,
            "average_water_consumption": avg_consumption,
            "latest_prediction": latest_pred,
            "year_range": f"{int(df_data['Year'].min())} - {int(df_data['Year'].max())}"
        },
        "consumption_by_year": year_data,
        "population_vs_consumption": pop_consumption,
        "sector_usage": sector_usage,
        "country_comparison": country_comparison
    }

@app.get("/api/historical-data")
def get_historical_data(
    country: Optional[str] = Query(None, description="Filter by Country name"),
    year_start: Optional[int] = Query(None, description="Start year"),
    year_end: Optional[int] = Query(None, description="End year"),
    limit: int = Query(500, ge=1, le=1000)
):
    """Returns records from the merged dataset with optional filtering."""
    if df_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    filtered = df_data.copy()
    if country and country != "All":
        filtered = filtered[filtered["Country"] == country]
    if year_start:
        filtered = filtered[filtered["Year"] >= year_start]
    if year_end:
        filtered = filtered[filtered["Year"] <= year_end]
        
    filtered = filtered.sort_values(["Year", "Country"]).head(limit)
    
    records = []
    for _, r in filtered.iterrows():
        records.append({
            "Country": r["Country"],
            "Year": int(r["Year"]),
            "Total Water Consumption": round(float(r["Total Water Consumption (Billion Cubic Meters)"]), 2),
            "Population": int(r["Population"]),
            "Per Capita Water Use": round(float(r["Per Capita Water Use (Liters per Day)"]), 2),
            "Agricultural Water Use": round(float(r["Agricultural Water Use (%)"]), 2),
            "Industrial Water Use": round(float(r["Industrial Water Use (%)"]), 2),
            "Household Water Use": round(float(r["Household Water Use (%)"]), 2),
            "Rainfall Impact": round(float(r["Rainfall Impact (Annual Precipitation in mm)"]), 2),
            "Groundwater Depletion Rate": round(float(r["Groundwater Depletion Rate (%)"]), 2),
            "Water Scarcity Level": str(r.get("Water Scarcity Level", "Moderate"))
        })
        
    return {
        "total_results": len(records),
        "data": records
    }

@app.get("/api/model-metrics")
def get_model_metrics():
    """Returns ML model training parameters, performance metrics, and feature importances."""
    if not metrics_data:
        raise HTTPException(status_code=500, detail="Model metrics not available")
    return metrics_data

@app.post("/api/predict")
def predict_water_demand(payload: PredictInput):
    """Predicts total water demand using the trained Random Forest model and stores in SQLite."""
    if model is None:
        raise HTTPException(status_code=500, detail="ML model is not loaded")
        
    features_ordered = [
        payload.population,
        payload.per_capita_water_use,
        payload.agricultural_water_use,
        payload.industrial_water_use,
        payload.household_water_use,
        payload.rainfall_impact,
        payload.groundwater_depletion_rate
    ]
    feature_names = [
        "Population",
        "Per Capita Water Use (Liters per Day)",
        "Agricultural Water Use (%)",
        "Industrial Water Use (%)",
        "Household Water Use (%)",
        "Rainfall Impact (Annual Precipitation in mm)",
        "Groundwater Depletion Rate (%)"
    ]
    
    df_features = pd.DataFrame([features_ordered], columns=feature_names)
    pred_value = float(model.predict(df_features)[0])
    pred_value = max(0.0, round(pred_value, 2))
    
    status, status_level, insights = determine_demand_status(pred_value)
    
    # Save record in database
    db_record = insert_prediction({
        "country": payload.country,
        "year": payload.year,
        "population": payload.population,
        "per_capita_water_use": payload.per_capita_water_use,
        "agricultural_water_use": payload.agricultural_water_use,
        "industrial_water_use": payload.industrial_water_use,
        "household_water_use": payload.household_water_use,
        "rainfall_impact": payload.rainfall_impact,
        "groundwater_depletion_rate": payload.groundwater_depletion_rate,
        "predicted_demand": pred_value,
        "demand_status": status
    })
    
    return {
        "success": True,
        "id": db_record["id"] if db_record else None,
        "created_at": db_record["created_at"] if db_record else None,
        "country": payload.country,
        "year": payload.year,
        "predicted_demand": pred_value,
        "unit": "Billion Cubic Meters (BCM)",
        "demand_status": status,
        "status_level": status_level,
        "insights": insights,
        "inputs": payload.dict()
    }

@app.get("/api/predictions")
def list_predictions(limit: int = Query(100, ge=1, le=500)):
    """Returns stored prediction history from SQLite."""
    rows = get_predictions(limit=limit)
    return {
        "count": len(rows),
        "total_in_db": get_prediction_count(),
        "predictions": rows
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
