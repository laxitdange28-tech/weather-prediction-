import os
import math
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# -------------------------------------------------------------------------
# 1. FastAPI App Initialization & CORS Configuration
# -------------------------------------------------------------------------
app = FastAPI(
    title="Weather Prediction Neural Network API",
    description="Deep Learning Weather Prediction Model trained with Backpropagation",
    version="1.0.0"
)

# Enable CORSMiddleware (crossmiddelwhere) for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# 2. Model Loading & Verification
# -------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model_data = None

def load_model():
    global model_data
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    print(f"Loaded ML model successfully. Accuracy: {model_data.get('accuracy', 0):.4f}")

try:
    load_model()
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")

# -------------------------------------------------------------------------
# 3. Pydantic Request & Response Schemas
# -------------------------------------------------------------------------
class WeatherPredictionRequest(BaseModel):
    temperature_c: float = Field(..., ge=-50.0, le=60.0, description="Temperature in Celsius")
    humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Relative Humidity percentage")
    precipitation_mm: float = Field(..., ge=0.0, le=100.0, description="Precipitation in mm")
    wind_speed_kmh: float = Field(..., ge=0.0, le=150.0, description="Wind Speed in km/h")
    location: str = Field(default="New York", description="Observation City")
    month: int = Field(default=6, ge=1, le=12, description="Month of observation (1-12)")
    hour: int = Field(default=14, ge=0, le=23, description="Hour of observation (0-23)")

class WeatherPredictionResponse(BaseModel):
    predicted_condition: str
    confidence: float
    probabilities: Dict[str, float]
    feels_like_c: float
    dew_point_c: float
    wind_chill_c: Optional[float]
    heat_index_c: Optional[float]
    rain_risk_pct: float
    advisories: List[str]
    weather_theme: str
    architecture: Dict[str, Any]

# -------------------------------------------------------------------------
# 4. Meteorological Calculations Helper
# -------------------------------------------------------------------------
def calculate_meteorology(temp: float, humidity: float, precip: float, wind: float):
    """
    Computes thermodynamic and apparent temperature metrics:
    - Dew Point using Magnus-Tetens approximation
    - Apparent / Feels-Like Temperature (Australian BOM / Steadman)
    - Wind Chill (for cold, windy conditions)
    - Heat Index (for hot, humid conditions)
    - Rain Risk Percentage
    """
    # 1. Dew Point (°C)
    a, b = 17.625, 243.04
    rh_fraction = max(0.01, min(1.0, humidity / 100.0))
    alpha = math.log(rh_fraction) + (a * temp) / (b + temp)
    dew_point = round((b * alpha) / (a - alpha), 1)

    # 2. Water vapor pressure (e in hPa)
    vapor_pressure = rh_fraction * 6.105 * math.exp((17.27 * temp) / (237.7 + temp))

    # 3. Australian BOM Apparent Temperature
    wind_ms = wind / 3.6
    apparent_temp = temp + 0.33 * vapor_pressure - 0.70 * wind_ms - 4.00
    feels_like = round(apparent_temp, 1)

    # 4. Wind Chill (for Temp <= 10°C and Wind > 4.8 km/h)
    wind_chill = None
    if temp <= 10.0 and wind > 4.8:
        wc = 13.12 + 0.6215 * temp - 11.37 * (wind ** 0.16) + 0.3965 * temp * (wind ** 0.16)
        wind_chill = round(wc, 1)

    # 5. Heat Index (for Temp >= 27°C and Humidity >= 40%)
    heat_index = None
    if temp >= 27.0 and humidity >= 40.0:
        c1 = -8.78469475556
        c2 = 1.61139411
        c3 = 2.33854883889
        c4 = -0.14611605
        c5 = -0.012308094
        c6 = -0.0164248277778
        c7 = 0.002211732
        c8 = 0.00072546
        c9 = -0.000003582
        hi = (c1 + c2 * temp + c3 * humidity + c4 * temp * humidity +
              c5 * (temp**2) + c6 * (humidity**2) + c7 * (temp**2) * humidity +
              c8 * temp * (humidity**2) + c9 * (temp**2) * (humidity**2))
        heat_index = round(hi, 1)

    # 6. Rain Risk Level
    rain_risk = round(min(100.0, max(0.0, (precip / 10.0) * 60.0 + (humidity / 100.0) * 40.0)), 1)

    return dew_point, feels_like, wind_chill, heat_index, rain_risk

def generate_advisories(condition: str, temp: float, precip: float, wind: float, humidity: float) -> List[str]:
    advisories = []
    
    if condition == "Rainy":
        advisories.append("☔ Rain gear needed: Carry an umbrella or waterproof jacket.")
        advisories.append("🚗 Wet roads alert: Drive with caution and reduce speed.")
    elif condition == "Stormy":
        advisories.append("⛈️ Severe storm warning: Seek shelter and avoid open outdoor areas.")
        advisories.append("💨 High gust alert: Secure loose outdoor items.")
    elif condition == "Snowy":
        advisories.append("❄️ Sub-zero freezing: Wear insulated winter layers, gloves, and boots.")
        advisories.append("🧊 Ice hazard: Watch for black ice on highways and walkways.")
    elif condition == "Cloudy":
        advisories.append("☁️ Overcast skies: Comfortable for outdoor activities, mild UV levels.")
        if humidity > 80:
            advisories.append("💧 High humidity: May feel clammy or muggy.")
    else:  # Sunny
        if temp > 28.0:
            advisories.append("☀️ High solar heat: Apply SPF sunscreen, stay hydrated, wear a hat.")
        else:
            advisories.append("☀️ Beautiful clear conditions: Excellent weather for outdoor recreation.")

    if wind > 25.0:
        advisories.append("💨 Strong breezes: Wind speeds exceed 25 km/h.")
    if temp < 5.0:
        advisories.append("🧤 Frost advisory: Protect exposed skin from chilly draft.")

    return advisories

# -------------------------------------------------------------------------
# 5. Core API Endpoints
# -------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "classes": model_data["classes"] if model_data else []
    }

@app.get("/api/info")
def get_model_info():
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    return {
        "model_type": "Multi-Layer Perceptron (MLP) Neural Network",
        "training_algorithm": "Backpropagation with Adam Optimizer",
        "accuracy": model_data.get("accuracy", 0.9874),
        "classes": model_data["classes"],
        "locations": model_data["locations"],
        "num_features": model_data["num_features"],
        "feature_count": len(model_data["feature_names"]),
        "architecture": model_data.get("architecture", {}),
        "loss_curve": model_data.get("loss_curve", [])
    }

@app.get("/api/presets")
def get_presets():
    return {
        "presets": [
            {
                "id": "sunny",
                "name": "☀️ Sunny Summer Day",
                "description": "Clear skies, warm sunshine, and gentle breeze",
                "temperature_c": 31.5,
                "humidity_pct": 38.0,
                "precipitation_mm": 0.2,
                "wind_speed_kmh": 8.5,
                "location": "Los Angeles",
                "month": 7,
                "hour": 13
            },
            {
                "id": "rainy",
                "name": "🌧️ Rainy Downpour",
                "description": "Heavy overcast with steady precipitation",
                "temperature_c": 16.0,
                "humidity_pct": 89.0,
                "precipitation_mm": 6.8,
                "wind_speed_kmh": 14.0,
                "location": "Seattle",
                "month": 4,
                "hour": 15
            },
            {
                "id": "stormy",
                "name": "⛈️ Severe Thunderstorm",
                "description": "High gale-force winds with torrential downpour",
                "temperature_c": 22.0,
                "humidity_pct": 92.0,
                "precipitation_mm": 8.5,
                "wind_speed_kmh": 26.5,
                "location": "Houston",
                "month": 8,
                "hour": 18
            },
            {
                "id": "snowy",
                "name": "❄️ Freezing Blizzard",
                "description": "Freezing winter temperatures with snowfall",
                "temperature_c": -6.5,
                "humidity_pct": 78.0,
                "precipitation_mm": 4.5,
                "wind_speed_kmh": 15.0,
                "location": "Chicago",
                "month": 1,
                "hour": 9
            },
            {
                "id": "cloudy",
                "name": "☁️ Humid & Overcast",
                "description": "High relative humidity with gray cloud cover",
                "temperature_c": 19.5,
                "humidity_pct": 74.0,
                "precipitation_mm": 1.2,
                "wind_speed_kmh": 11.0,
                "location": "New York",
                "month": 5,
                "hour": 11
            }
        ]
    }

@app.post("/api/predict", response_model=WeatherPredictionResponse)
def predict_weather(req: WeatherPredictionRequest):
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on server")

    # Construct input vector matching the exact features used during training
    row = {
        "Temperature_C": req.temperature_c,
        "Humidity_pct": req.humidity_pct,
        "Precipitation_mm": req.precipitation_mm,
        "Wind_Speed_kmh": req.wind_speed_kmh,
        "Month": req.month,
        "Hour": req.hour
    }

    # Location one-hot encoding
    for loc in model_data["locations"]:
        col_name = f"loc_{loc}"
        row[col_name] = 1.0 if req.location.strip().lower() == loc.strip().lower() else 0.0

    # Create dataframe with strictly identical column ordering
    df_query = pd.DataFrame([row])[model_data["feature_names"]]

    # Scale features using fitted StandardScaler
    scaled_query = model_data["scaler"].transform(df_query)

    # Perform Inference via Backpropagation-trained MLP
    model = model_data["model"]
    pred_class = model.predict(scaled_query)[0]
    probs = model.predict_proba(scaled_query)[0]

    prob_dict = {
        cls_name: round(float(prob) * 100, 2)
        for cls_name, prob in zip(model_data["classes"], probs)
    }

    confidence = prob_dict[pred_class]

    # Calculate thermodynamics
    dew_point, feels_like, wind_chill, heat_index, rain_risk = calculate_meteorology(
        req.temperature_c, req.humidity_pct, req.precipitation_mm, req.wind_speed_kmh
    )

    advisories = generate_advisories(
        pred_class, req.temperature_c, req.precipitation_mm, req.wind_speed_kmh, req.humidity_pct
    )

    theme_map = {
        "Sunny": "sunny",
        "Rainy": "rainy",
        "Stormy": "stormy",
        "Snowy": "snowy",
        "Cloudy": "cloudy"
    }

    return WeatherPredictionResponse(
        predicted_condition=pred_class,
        confidence=confidence,
        probabilities=prob_dict,
        feels_like_c=feels_like,
        dew_point_c=dew_point,
        wind_chill_c=wind_chill,
        heat_index_c=heat_index,
        rain_risk_pct=rain_risk,
        advisories=advisories,
        weather_theme=theme_map.get(pred_class, "sunny"),
        architecture=model_data.get("architecture", {})
    )

# -------------------------------------------------------------------------
# 6. Static Files and Frontend Serving
# -------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_ui():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "FastAPI Weather ML Backend is running. Please add static/index.html"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting AtmosphereAI server at http://127.0.0.1:{port}")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)

