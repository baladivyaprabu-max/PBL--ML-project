import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ML_DIR = os.path.join(BASE_DIR, "ml")

def train_and_export():
    water_path = os.path.join(DATA_DIR, "cleaned_global_water_consumption.csv")
    pop_path = os.path.join(DATA_DIR, "world-population.csv")
    
    if not os.path.exists(water_path) or not os.path.exists(pop_path):
        raise FileNotFoundError(f"Missing CSV files in {DATA_DIR}")
        
    print("Loading datasets...")
    df_water = pd.read_csv(water_path)
    df_pop = pd.read_csv(pop_path)
    
    # Strip whitespace from columns
    df_water.columns = df_water.columns.str.strip()
    df_pop.columns = df_pop.columns.str.strip()
    
    # Rename country to Country in population
    if "country" in df_pop.columns:
        df_pop = df_pop.rename(columns={"country": "Country"})
        
    # Harmonize country names
    country_mapping = {
        "United Kingdom": "UK",
        "United States": "USA"
    }
    df_pop["Country"] = df_pop["Country"].replace(country_mapping)
    
    # Build complete Country-Year grid for smooth population interpolation
    water_countries = sorted(df_water["Country"].unique())
    all_years = pd.DataFrame(
        [(c, y) for c in water_countries for y in range(2000, 2025)],
        columns=["Country", "Year"]
    )
    
    pop_filtered = df_pop[df_pop["Country"].isin(water_countries)][["Country", "Year", "Population"]].copy()
    pop_filtered["Year"] = pd.to_numeric(pop_filtered["Year"], errors="coerce")
    pop_filtered["Population"] = pd.to_numeric(pop_filtered["Population"], errors="coerce")
    
    merged_pop = pd.merge(all_years, pop_filtered, on=["Country", "Year"], how="left")
    merged_pop["Population"] = merged_pop.groupby("Country")["Population"].transform(
        lambda s: s.interpolate(method="linear").bfill().ffill()
    )
    
    # Merge datasets
    df_merged = pd.merge(df_water, merged_pop, on=["Country", "Year"], how="inner")
    
    # Save merged data
    merged_data_path = os.path.join(DATA_DIR, "merged_water_data.csv")
    df_merged.to_csv(merged_data_path, index=False)
    print(f"Merged dataset saved to {merged_data_path} with shape {df_merged.shape}")
    
    # Features and target
    features = [
        "Population",
        "Per Capita Water Use (Liters per Day)",
        "Agricultural Water Use (%)",
        "Industrial Water Use (%)",
        "Household Water Use (%)",
        "Rainfall Impact (Annual Precipitation in mm)",
        "Groundwater Depletion Rate (%)"
    ]
    target = "Total Water Consumption (Billion Cubic Meters)"
    
    X = df_merged[features].copy()
    y = df_merged[target].copy()
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred_test))
    
    # Fit-on-full metrics for reporting
    full_preds = model.predict(X)
    full_mae = float(mean_absolute_error(y, full_preds))
    full_rmse = float(np.sqrt(mean_squared_error(y, full_preds)))
    full_r2 = float(r2_score(y, full_preds))
    
    feature_importances = {
        feat: round(float(imp), 4) for feat, imp in zip(features, model.feature_importances_)
    }
    
    print(f"Model Metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, Test R2: {r2:.4f}, Full R2: {full_r2:.4f}")
    
    # Save trained model
    model_file = os.path.join(ML_DIR, "water_demand_rf_model.joblib")
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    # Save metrics JSON
    metrics = {
        "model_name": "Random Forest Regressor",
        "n_estimators": 150,
        "max_depth": 12,
        "features": features,
        "target": target,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "full_dataset_r2": round(full_r2, 4),
        "full_dataset_mae": round(full_mae, 2),
        "full_dataset_rmse": round(full_rmse, 2),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "total_samples": int(len(df_merged)),
        "feature_importances": feature_importances
    }
    
    metrics_file = os.path.join(ML_DIR, "model_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    # Compute country baseline defaults for the frontend autofill
    country_defaults = {}
    for c in water_countries:
        c_rows = df_merged[df_merged["Country"] == c].sort_values("Year")
        latest = c_rows.iloc[-1]
        means = c_rows.mean(numeric_only=True)
        country_defaults[c] = {
            "Country": c,
            "LatestYear": int(latest["Year"]),
            "Population": round(float(latest["Population"])),
            "Per Capita Water Use (Liters per Day)": round(float(means["Per Capita Water Use (Liters per Day)"]), 1),
            "Agricultural Water Use (%)": round(float(means["Agricultural Water Use (%)"]), 1),
            "Industrial Water Use (%)": round(float(means["Industrial Water Use (%)"]), 1),
            "Household Water Use (%)": round(float(means["Household Water Use (%)"]), 1),
            "Rainfall Impact (Annual Precipitation in mm)": round(float(means["Rainfall Impact (Annual Precipitation in mm)"]), 1),
            "Groundwater Depletion Rate (%)": round(float(means["Groundwater Depletion Rate (%)"]), 2),
            "Avg Total Water Consumption": round(float(means[target]), 2)
        }
        
    defaults_file = os.path.join(ML_DIR, "country_defaults.json")
    with open(defaults_file, "w") as f:
        json.dump(country_defaults, f, indent=2)
    print(f"Country defaults saved to {defaults_file}")

if __name__ == "__main__":
    train_and_export()
