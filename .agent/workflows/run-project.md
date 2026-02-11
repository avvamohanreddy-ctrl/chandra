---
description: Complete workflow to run the Bangalore Real Estate Price Predictor
---

# Bangalore Real Estate Predictor - Workflow

## Prerequisites

1. Python 3.9+ installed
2. Dataset: `Bengaluru_House_Data.csv` in the `data/` folder

---

## Step 1: Install Dependencies

```bash
cd /Users/aremkevin/Desktop/redemption\ arc/bangalore-real-estate-predictor
pip install -r requirements.txt
```

// turbo

## Step 2: Verify Dataset

```bash
python -c "from src.data_loader import load_raw_data; df = load_raw_data(); print(f'Dataset: {len(df)} records')"
```

---

## Step 3: Run Full Training Pipeline

### Option A: Quick Test (2-3 minutes, 1000 samples)

```bash
python main.py --quick-test
```

### Option B: Full Training (5-10 minutes, all data)

```bash
python main.py
```

**Outputs:**
- Model checkpoint: `checkpoints/best_gat_model.pt`
- Scalers: `checkpoints/scaler_X.joblib`, `scaler_y.joblib`
- Plots: `results/training_history.png`, `model_comparison.png`

---

## Step 4: Start API Server

```bash
uvicorn api.main:app --reload --port 8000
```

**Endpoints:**
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health
- Predict: POST http://localhost:8000/api/predict
- Locations: GET http://localhost:8000/api/locations

---

## Step 5: Open Frontend

### Option A: Direct file open

```bash
open frontend/index.html
```

### Option B: Serve via Python

```bash
cd frontend && python -m http.server 3000
```

Then open http://localhost:3000

---

## Step 6: Make a Prediction (API Test)

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Whitefield",
    "total_sqft": 1500,
    "bhk": 3,
    "bath": 2,
    "balcony": 1,
    "area_type": "Super built-up Area"
  }'
```

---

## Troubleshooting

### PyTorch Geometric not found
```bash
pip install torch-geometric
```

### CUDA/GPU issues
The model runs on CPU by default. No GPU required.

### API can't find model
Run `python main.py` first to train and save the model.

---

## Project Structure

```
bangalore-real-estate-predictor/
├── main.py              # Run this first (training)
├── api/main.py          # Then start API
├── frontend/            # Open in browser
├── src/                 # Core modules
├── checkpoints/         # Saved models
└── results/             # Evaluation plots
```
