import os
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ML_DIR = os.path.join(BASE_DIR, "ml")

def train_and_evaluate_xgboost():
    merged_data_path = os.path.join(DATA_DIR, "merged_water_data.csv")
    
    # 1. Load dataset
    if os.path.exists(merged_data_path):
        df = pd.read_csv(merged_data_path)
    else:
        # Fallback to loading raw datasets and merging
        water_path = os.path.join(DATA_DIR, "cleaned_global_water_consumption.csv")
        pop_path = os.path.join(DATA_DIR, "world-population.csv")
        
        df_water = pd.read_csv(water_path)
        df_pop = pd.read_csv(pop_path)
        
        df_water.columns = df_water.columns.str.strip()
        df_pop.columns = df_pop.columns.str.strip()
        if "country" in df_pop.columns:
            df_pop = df_pop.rename(columns={"country": "Country"})
            
        df_pop["Country"] = df_pop["Country"].replace({"United Kingdom": "UK", "United States": "USA"})
        water_countries = sorted(df_water["Country"].unique())
        all_years = pd.DataFrame([(c, y) for c in water_countries for y in range(2000, 2025)], columns=["Country", "Year"])
        pop_filtered = df_pop[df_pop["Country"].isin(water_countries)][["Country", "Year", "Population"]].copy()
        merged_pop = pd.merge(all_years, pop_filtered, on=["Country", "Year"], how="left")
        merged_pop["Population"] = merged_pop.groupby("Country")["Population"].transform(lambda s: s.interpolate().bfill().ffill())
        df = pd.merge(df_water, merged_pop, on=["Country", "Year"], how="inner")

    # 2. Features and target (identical to Random Forest)
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
    
    X = df[features].copy()
    y = df[target].copy()
    
    # 3. Same 80/20 train/test split with identical random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("=" * 70)
    print("TRAINING XGBOOST REGRESSION MODEL")
    print("=" * 70)
    
    # 4. Train XGBRegressor
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # 5. Evaluate on Test set
    y_pred_test = xgb_model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_mse = float(mean_squared_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = float(r2_score(y_test, y_pred_test))
    
    # Evaluate on Full dataset
    y_pred_full = xgb_model.predict(X)
    full_mae = float(mean_absolute_error(y, y_pred_full))
    full_rmse = float(np.sqrt(mean_squared_error(y, y_pred_full)))
    full_r2 = float(r2_score(y, y_pred_full))
    
    feature_importances = {
        feat: round(float(imp), 4) for feat, imp in zip(features, xgb_model.feature_importances_)
    }
    
    # 6. Save trained XGBoost model as a separate model file
    xgb_model_file = os.path.join(ML_DIR, "water_demand_xgb_model.joblib")
    joblib.dump(xgb_model, xgb_model_file)
    print(f"[OK] Saved XGBoost model to: {xgb_model_file}")
    
    # Save XGBoost metrics as a separate metrics file
    xgb_metrics = {
        "model_name": "XGBoost Regressor",
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 3,
        "features": features,
        "target": target,
        "test_mae": round(test_mae, 2),
        "test_rmse": round(test_rmse, 2),
        "test_r2": round(test_r2, 4),
        "full_dataset_r2": round(full_r2, 4),
        "full_dataset_mae": round(full_mae, 2),
        "full_dataset_rmse": round(full_rmse, 2),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "total_samples": int(len(df)),
        "feature_importances": feature_importances
    }
    
    xgb_metrics_file = os.path.join(ML_DIR, "xgb_model_metrics.json")
    with open(xgb_metrics_file, "w") as f:
        json.dump(xgb_metrics, f, indent=2)
    print(f"[OK] Saved XGBoost metrics to: {xgb_metrics_file}\n")
    
    # 7. Load existing Random Forest metrics for direct comparison
    rf_metrics_file = os.path.join(ML_DIR, "model_metrics.json")
    rf_metrics = {}
    if os.path.exists(rf_metrics_file):
        with open(rf_metrics_file, "r") as f:
            rf_metrics = json.load(f)
            
    # 8. Print side-by-side comparison table
    print("=" * 70)
    print(f"{'METRIC / PARAMETER':<30} | {'RANDOM FOREST (CURRENT)':<20} | {'XGBOOST (NEW)':<15}")
    print("-" * 70)
    print(f"{'Algorithm':<30} | {rf_metrics.get('model_name', 'Random Forest'):<20} | {'XGBRegressor':<15}")
    print(f"{'Train Samples':<30} | {rf_metrics.get('train_samples', len(X_train)):<20} | {len(X_train):<15}")
    print(f"{'Test Samples':<30} | {rf_metrics.get('test_samples', len(X_test)):<20} | {len(X_test):<15}")
    print(f"{'Test MAE (lower is better)':<30} | {rf_metrics.get('mae', 'N/A'):<20} | {test_mae:.2f}")
    print(f"{'Test RMSE (lower is better)':<30} | {rf_metrics.get('rmse', 'N/A'):<20} | {test_rmse:.2f}")
    print(f"{'Test R² (higher is better)':<30} | {rf_metrics.get('r2_score', 'N/A'):<20} | {test_r2:.4f}")
    print(f"{'Full Dataset R²':<30} | {rf_metrics.get('full_dataset_r2', 'N/A'):<20} | {full_r2:.4f}")
    print(f"{'Full Dataset MAE':<30} | {rf_metrics.get('full_dataset_mae', 'N/A'):<20} | {full_mae:.2f}")
    print("=" * 70)
    
    print("\nXGBoost Feature Importances:")
    for feat, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:<45}: {imp * 100:.2f}%")
        
    # Analysis summary
    print("\nComparison Analysis:")
    rf_mae = rf_metrics.get('mae', 77.49)
    rf_rmse = rf_metrics.get('rmse', 101.10)
    rf_r2 = rf_metrics.get('r2_score', -0.0702)
    
    mae_diff = test_mae - rf_mae
    r2_diff = test_r2 - rf_r2
    
    print(f"  * Test MAE: XGBoost ({test_mae:.2f}) vs Random Forest ({rf_mae:.2f}) -> {'Improved' if mae_diff < 0 else 'Higher'} by {abs(mae_diff):.2f} BCM")
    print(f"  * Test R² : XGBoost ({test_r2:.4f}) vs Random Forest ({rf_r2:.4f}) -> {'Improved' if r2_diff > 0 else 'Lower'} by {abs(r2_diff):.4f}")
    print(f"  * Existing Random Forest model and backend endpoints remain untouched.")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate_xgboost()
