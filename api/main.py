"""
FastAPI Backend for Bangalore Real Estate Price Prediction
===========================================================
REST API for real-time price predictions with map integration.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
import joblib
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.gnn_model import BangaloreGAT
from geocoder import geocode_location, get_nearby_properties, BANGALORE_LOCATIONS
from landmarks import get_nearby_landmarks, get_all_landmarks, LANDMARK_ICONS


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    location: str = Field(..., description="Location name in Bangalore")
    total_sqft: float = Field(..., gt=0, description="Total area in square feet")
    bhk: int = Field(..., ge=1, le=16, description="Number of bedrooms")
    bath: int = Field(..., ge=1, le=16, description="Number of bathrooms")
    balcony: int = Field(default=1, ge=0, le=5, description="Number of balconies")
    area_type: str = Field(default="Super built-up Area", description="Type of area")


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    success: bool
    location: str
    coordinates: dict
    predicted_price_per_sqft: float
    total_estimated_price: float
    total_estimated_price_formatted: str
    confidence_interval: dict
    nearby_comparables: List[dict]
    nearby_landmarks: List[dict]


class LocationInfo(BaseModel):
    """Location information."""
    name: str
    latitude: float
    longitude: float


# ============================================================================
# GLOBAL STATE
# ============================================================================

model = None
scaler_X = None
scaler_y = None
le_location = None
feature_names = None
processed_data = None


def load_artifacts():
    """Load trained model and preprocessing artifacts."""
    global model, scaler_X, scaler_y, le_location, feature_names, processed_data
    
    base_dir = Path(__file__).parent.parent
    checkpoints_dir = base_dir / 'checkpoints'
    
    # Load scalers
    scaler_X = joblib.load(checkpoints_dir / 'scaler_X.joblib')
    scaler_y = joblib.load(checkpoints_dir / 'scaler_y.joblib')
    le_location = joblib.load(checkpoints_dir / 'le_location.joblib')
    
    # Load feature names
    with open(checkpoints_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Load processed data for comparables
    processed_data = pd.read_csv(checkpoints_dir / 'processed_data.csv')
    
    # Load model
    num_features = len(feature_names)
    model = BangaloreGAT(in_channels=num_features)
    model.load_state_dict(torch.load(checkpoints_dir / 'best_gat_model.pt', map_location='cpu'))
    model.eval()
    
    print("✓ All artifacts loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    try:
        load_artifacts()
    except Exception as e:
        print(f"⚠ Could not load artifacts: {e}")
        print("  API will run in demo mode")
    yield


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Bangalore Real Estate Price Predictor",
    description="ML + GNN based real estate price prediction for Bangalore",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_features(request: PredictionRequest) -> np.ndarray:
    """Prepare features for prediction."""
    # Encode location
    location_clean = request.location.strip().title()
    
    # Check if location is known
    try:
        location_label = le_location.transform([location_clean])[0]
    except ValueError:
        # Unknown location - use most common
        location_label = 0
    
    # Area type encoding
    area_type_map = {
        'Super built-up Area': 1,
        'Built-up Area': 2,
        'Plot Area': 3,
        'Carpet Area': 4
    }
    area_type_encoded = area_type_map.get(request.area_type, 1)
    
    # Create feature vector
    features = np.array([[
        request.total_sqft,
        request.bhk,
        request.bath,
        request.balcony,
        area_type_encoded,
        location_label
    ]])
    
    # Scale features
    features_scaled = scaler_X.transform(features)
    
    return features_scaled


def predict_price(features_scaled: np.ndarray) -> float:
    """Make prediction using the model."""
    if model is None:
        # Demo mode: return average price for Bangalore
        return 6500.0
    
    # Convert to tensor
    x = torch.tensor(features_scaled, dtype=torch.float32)
    
    # Create self-loop edge for single prediction
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    with torch.no_grad():
        pred_scaled = model(x, edge_index)
    
    # Inverse transform
    pred = scaler_y.inverse_transform(pred_scaled.numpy())[0][0]
    
    # Clamp to reasonable range
    pred = max(2000, min(30000, pred))
    
    return float(pred)


def get_comparables(lat: float, lng: float, bhk: int, limit: int = 5) -> List[dict]:
    """Get nearby comparable properties."""
    if processed_data is None:
        return []
    
    try:
        # Filter by BHK
        df = processed_data[processed_data['bhk'] == bhk].copy()
        
        if len(df) == 0:
            df = processed_data.copy()
        
        # Get nearby
        nearby = get_nearby_properties(df, lat, lng, radius_km=3.0, limit=limit)
        
        comparables = []
        for _, row in nearby.iterrows():
            comparables.append({
                'location': row.get('location_clean', 'Unknown'),
                'bhk': int(row['bhk']),
                'total_sqft': float(row['total_sqft_clean']),
                'price_per_sqft': float(row['price_per_sqft']),
                'distance_km': round(float(row['distance_km']), 2),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude'])
            })
        
        return comparables
    except Exception as e:
        print(f"Error getting comparables: {e}")
        return []


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Bangalore Real Estate Price Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/predict": "Predict price for a property",
            "GET /api/locations": "Get list of known locations",
            "GET /api/health": "Health check"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": processed_data is not None
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict price for a property.
    
    Takes location, size, and property details to predict price per sqft
    and total estimated price.
    """
    try:
        # Geocode location
        coords = geocode_location(request.location)
        
        # Prepare features
        features = prepare_features(request)
        
        # Make prediction
        price_per_sqft = predict_price(features)
        
        # Calculate total price
        total_price = price_per_sqft * request.total_sqft
        
        # Get comparable properties
        comparables = get_comparables(coords[0], coords[1], request.bhk)
        
        # Confidence interval (simplified)
        ci_lower = price_per_sqft * 0.85
        ci_upper = price_per_sqft * 1.15
        
        # Get nearby landmarks
        landmarks = get_nearby_landmarks(coords[0], coords[1], radius_km=5.0, limit_per_type=2)
        
        return PredictionResponse(
            success=True,
            location=request.location.strip().title(),
            coordinates={"latitude": coords[0], "longitude": coords[1]},
            predicted_price_per_sqft=round(price_per_sqft, 2),
            total_estimated_price=round(total_price, 2),
            total_estimated_price_formatted=f"₹{total_price/100000:.2f} Lakhs",
            confidence_interval={
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2)
            },
            nearby_comparables=comparables,
            nearby_landmarks=landmarks
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/locations", response_model=List[LocationInfo])
async def get_locations():
    """Get list of known Bangalore locations with coordinates."""
    locations = []
    for name, coords in BANGALORE_LOCATIONS.items():
        locations.append(LocationInfo(
            name=name,
            latitude=coords[0],
            longitude=coords[1]
        ))
    return locations


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics."""
    if processed_data is None:
        return {"error": "Data not loaded"}
    
    return {
        "total_properties": len(processed_data),
        "unique_locations": processed_data['location_clean'].nunique(),
        "price_per_sqft": {
            "min": float(processed_data['price_per_sqft'].min()),
            "max": float(processed_data['price_per_sqft'].max()),
            "mean": float(processed_data['price_per_sqft'].mean()),
            "median": float(processed_data['price_per_sqft'].median())
        },
        "bhk_distribution": processed_data['bhk'].value_counts().to_dict()
    }


@app.get("/api/landmarks")
async def get_landmarks():
    """Get all Bangalore landmarks for map display."""
    landmarks = get_all_landmarks()
    return {
        "total": len(landmarks),
        "icons": LANDMARK_ICONS,
        "landmarks": landmarks
    }


# Serve frontend static files
frontend_path = Path(__file__).parent.parent / 'frontend'
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    
    @app.get("/app")
    async def serve_frontend():
        return FileResponse(str(frontend_path / 'index.html'))


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("  BANGALORE REAL ESTATE API")
    print("="*50)
    print("  Starting server...")
    print("  API docs: http://localhost:8000/docs")
    print("  Frontend: http://localhost:8000/app")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
