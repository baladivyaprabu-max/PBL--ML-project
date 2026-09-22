import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "water_demand.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            country TEXT,
            year INTEGER,
            population REAL NOT NULL,
            per_capita_water_use REAL NOT NULL,
            agricultural_water_use REAL NOT NULL,
            industrial_water_use REAL NOT NULL,
            household_water_use REAL NOT NULL,
            rainfall_impact REAL NOT NULL,
            groundwater_depletion_rate REAL NOT NULL,
            predicted_demand REAL NOT NULL,
            demand_status TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_prediction(data: dict) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO predictions (
            created_at, country, year, population,
            per_capita_water_use, agricultural_water_use, industrial_water_use,
            household_water_use, rainfall_impact, groundwater_depletion_rate,
            predicted_demand, demand_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        created_at,
        data.get("country", "Custom Scenario"),
        data.get("year", datetime.now().year),
        float(data["population"]),
        float(data["per_capita_water_use"]),
        float(data["agricultural_water_use"]),
        float(data["industrial_water_use"]),
        float(data["household_water_use"]),
        float(data["rainfall_impact"]),
        float(data["groundwater_depletion_rate"]),
        round(float(data["predicted_demand"]), 2),
        data["demand_status"]
    ))
    new_id = cursor.lastrowid
    conn.commit()
    
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (new_id,))
    row = cursor.fetchone()
    result = dict(row) if row else None
    conn.close()
    return result

def get_predictions(limit: int = 100) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    return results

def get_latest_prediction() -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    result = dict(row) if row else None
    conn.close()
    return result

def get_prediction_count() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
