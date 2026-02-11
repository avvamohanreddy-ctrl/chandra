# 🏠 Bangalore Real Estate Price Predictor

A complete **ML + Graph Neural Network** based real estate price prediction system for Bangalore (Bengaluru), India. Uses Graph Attention Networks (GAT) to leverage spatial relationships between properties for more accurate predictions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)

## ✨ Features

- **Data Pipeline**: Automated cleaning, feature engineering, and geocoding for ~13,000 Bangalore properties
- **Baseline ML Models**: Linear Regression, Random Forest, XGBoost for comparison
- **Graph Neural Network**: Graph Attention Network (GAT) leveraging spatial proximity
- **REST API**: FastAPI backend for real-time predictions
- **Interactive Map UI**: Leaflet-based frontend with dark theme and glassmorphism design
- **Comparable Properties**: Find nearby properties with similar characteristics

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Bengaluru      │     │  Feature        │     │  Spatial        │
│  House Data     │────▶│  Engineering    │────▶│  Graph (k-NN)   │
│  (CSV)          │     │  + Geocoding    │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Interactive    │     │  FastAPI        │     │  GAT Model      │
│  Map UI         │◀────│  Backend        │◀────│  (PyTorch       │
│  (Leaflet)      │     │                 │     │   Geometric)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📂 Project Structure

```
bangalore-real-estate-predictor/
├── api/
│   └── main.py              # FastAPI backend
├── data/
│   └── Bengaluru_House_Data.csv
├── frontend/
│   ├── index.html           # Interactive UI
│   ├── styles.css           # Premium dark theme
│   └── app.js               # Map & API logic
├── src/
│   ├── data_loader.py       # Load raw data
│   ├── data_cleaner.py      # Clean & preprocess
│   ├── feature_engineering.py
│   ├── geocoder.py          # Location → coordinates
│   ├── graph_builder.py     # Spatial graph construction
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Metrics & visualizations
│   └── models/
│       ├── baseline_models.py  # LR, RF, XGBoost
│       └── gnn_model.py        # GAT, GCN models
├── checkpoints/             # Saved models
├── results/                 # Evaluation outputs
├── main.py                  # Complete pipeline
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd bangalore-real-estate-predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install PyTorch Geometric (may need separate installation)
pip install torch-geometric
```

### 2. Train the Models

```bash
# Full training (takes 5-10 minutes)
python main.py

# Quick test (subset of data)
python main.py --quick-test
```

This will:
- Clean and preprocess the Bengaluru House Data
- Train baseline ML models (Linear Regression, Random Forest, XGBoost)
- Build a spatial proximity graph
- Train the Graph Attention Network
- Generate evaluation plots in `results/`
- Save model checkpoints to `checkpoints/`

### 3. Start the API Server

```bash
cd api
uvicorn main:app --reload --port 8000
```

API will be available at:
- Docs: http://localhost:8000/docs
- Predict: POST http://localhost:8000/api/predict

### 4. Open the Frontend

Open `frontend/index.html` in your browser, or serve it:

```bash
cd frontend
python -m http.server 3000
# Then open http://localhost:3000
```

## 📊 API Endpoints

### POST /api/predict

Predict price for a property.

**Request:**
```json
{
  "location": "Whitefield",
  "total_sqft": 1500,
  "bhk": 3,
  "bath": 2,
  "balcony": 1,
  "area_type": "Super built-up Area"
}
```

**Response:**
```json
{
  "success": true,
  "location": "Whitefield",
  "coordinates": {"latitude": 12.9698, "longitude": 77.7500},
  "predicted_price_per_sqft": 6543.21,
  "total_estimated_price": 9814815.0,
  "total_estimated_price_formatted": "₹98.15 Lakhs",
  "confidence_interval": {"lower": 5562.73, "upper": 7523.69},
  "nearby_comparables": [...]
}
```

### GET /api/locations

Get list of known Bangalore locations with coordinates.

### GET /api/stats

Get dataset statistics.

## 🧠 Model Performance

After training, you can expect results similar to:

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | ~0.65 | ~800 | ~1200 |
| Random Forest | ~0.75 | ~600 | ~950 |
| XGBoost | ~0.78 | ~550 | ~900 |
| **GAT (GNN)** | **~0.82** | **~480** | **~780** |

*Actual results may vary based on data split and hyperparameters.*

## 🔧 Configuration

Key parameters in `main.py`:

```python
# Graph construction
k = 10  # Number of nearest neighbors

# GAT Model
hidden_channels = 64
heads = 4
dropout = 0.3

# Training
epochs = 500
lr = 0.01
patience = 50  # Early stopping
```

## 📈 Visualizations

After training, check the `results/` directory for:
- `training_history.png` - Loss and R² curves
- `actual_vs_predicted.png` - Scatter plot with R²
- `error_distribution.png` - Prediction error analysis
- `model_comparison.png` - Baseline vs GNN comparison

## 🌐 Technologies Used

- **Data**: Pandas, NumPy, Scikit-learn
- **ML**: XGBoost, Random Forest
- **GNN**: PyTorch, PyTorch Geometric (GAT, GCN)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript, Leaflet.js
- **Visualization**: Matplotlib

## 📝 Dataset

Using the [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) from Kaggle containing ~13,320 properties with:
- Area type, Location, Size (BHK)
- Total sqft, Bath, Balcony
- Price (in Lakhs)

## 🤝 Contributing

Contributions welcome! Some ideas:
- Add more GNN architectures (GraphSAGE, GIN)
- Implement real geocoding with Google Maps API
- Add time-series prediction for price trends
- Enhance frontend with property images

## 📄 License

MIT License - feel free to use for academic or commercial purposes.

---

Built with ❤️ using Graph Neural Networks
