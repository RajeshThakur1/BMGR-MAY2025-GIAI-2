"""
Fraud Detection API

This module provides a REST API for fraud detection model serving
with real-time predictions, model management, and monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path for imports  
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Custom imports
from features.feature_engineering import FeatureEngineer
from models.model_evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection service with machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model management
loaded_models = {}
feature_engineer = None
model_metadata = {}

# Pydantic models for API requests/responses
class TransactionRequest(BaseModel):
    """Transaction data for fraud prediction."""
    customer_id: str = Field(..., description="Unique customer identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_name: str = Field(..., description="Merchant name")
    merchant_category: str = Field(..., description="Merchant category")
    merchant_state: str = Field(..., description="Merchant state")
    customer_age: Optional[int] = Field(None, ge=18, le=100, description="Customer age")
    customer_income: Optional[float] = Field(None, gt=0, description="Customer income")
    account_age_days: Optional[int] = Field(None, ge=0, description="Account age in days")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001234",
                "amount": 150.75,
                "merchant_id": "MER_RETAIL_5678",
                "merchant_name": "TechStore",
                "merchant_category": "retail",
                "merchant_state": "NY",
                "customer_age": 35,
                "customer_income": 65000.0,
                "account_age_days": 730,
                "timestamp": "2023-12-01T14:30:00"
            }
        }

class BatchTransactionRequest(BaseModel):
    """Batch of transactions for fraud prediction."""
    transactions: List[TransactionRequest] = Field(..., description="List of transactions")
    
class FraudPredictionResponse(BaseModel):
    """Response for fraud prediction."""
    transaction_id: str = Field(..., description="Generated transaction ID")
    is_fraud: int = Field(..., description="Fraud prediction (0=normal, 1=fraud)")
    fraud_probability: float = Field(..., description="Fraud probability score")
    risk_score: str = Field(..., description="Risk level (low/medium/high)")
    model_used: str = Field(..., description="Model used for prediction")
    prediction_timestamp: str = Field(..., description="Prediction timestamp")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval")

class BatchFraudPredictionResponse(BaseModel):
    """Response for batch fraud prediction."""
    predictions: List[FraudPredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")

class ModelInfo(BaseModel):
    """Model information."""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    trained_at: str = Field(..., description="Training timestamp")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance")
    feature_count: int = Field(..., description="Number of features")

class HealthResponse(BaseModel):
    """API health response."""
    status: str = Field(..., description="API status")
    timestamp: str = Field(..., description="Current timestamp")
    models_loaded: int = Field(..., description="Number of loaded models")
    uptime_seconds: Optional[float] = Field(None, description="API uptime in seconds")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and feature engineering on startup."""
    global loaded_models, feature_engineer, model_metadata
    
    logger.info("Starting Fraud Detection API...")
    
    try:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer({
            'create_interactions': True,
            'create_ratios': True,
            'create_aggregations': False,
            'categorical_encoding': 'target',
            'scaling_method': 'robust'
        })
        
        # Load trained models
        models_dir = Path("models/trained")
        if models_dir.exists():
            for model_file in models_dir.glob("*_model.pkl"):
                model_name = model_file.stem.replace('_model', '')
                try:
                    model = joblib.load(model_file)
                    loaded_models[model_name] = model
                    
                    # Load model metadata if available
                    metadata_file = models_dir / "model_results.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            all_metadata = json.load(f)
                            if model_name in all_metadata:
                                model_metadata[model_name] = all_metadata[model_name]
                    
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {str(e)}")
        
        # Set default model if no models loaded
        if not loaded_models:
            logger.warning("No trained models found. API will use mock predictions.")
        
        logger.info(f"API startup complete. Loaded {len(loaded_models)} models.")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

# Dependency to get model
def get_model(model_name: Optional[str] = None):
    """Get model instance."""
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models available")
    
    if model_name and model_name in loaded_models:
        return loaded_models[model_name], model_name
    else:
        # Return default model (first available)
        default_model_name = list(loaded_models.keys())[0]
        return loaded_models[default_model_name], default_model_name

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(loaded_models),
        uptime_seconds=None  # Could implement uptime tracking
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    model_list = []
    
    for model_name, model in loaded_models.items():
        metadata = model_metadata.get(model_name, {})
        
        model_info = ModelInfo(
            model_name=model_name,
            model_type=type(model).__name__,
            version="1.0.0",
            trained_at=metadata.get('trained_at', datetime.now().isoformat()),
            performance_metrics=metadata.get('val_metrics', {}),
            feature_count=getattr(model, 'n_features_in_', 0)
        )
        model_list.append(model_info)
    
    return model_list

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    model_name: Optional[str] = None,
    model_info: tuple = Depends(get_model)
):
    """Predict fraud for a single transaction."""
    model, actual_model_name = model_info
    
    try:
        # Convert request to DataFrame
        transaction_data = transaction.dict()
        
        # Generate transaction ID if not provided
        transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(transaction_data)) % 10000:04d}"
        
        # Handle timestamp
        if transaction_data.get('timestamp'):
            transaction_data['timestamp'] = pd.to_datetime(transaction_data['timestamp'])
        else:
            transaction_data['timestamp'] = pd.Timestamp.now()
        
        # Create DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Apply feature engineering
        if feature_engineer:
            try:
                # For prediction, we need to transform without fitting
                df_transformed = feature_engineer.transform(df)
            except:
                # If transform fails, create minimal features
                df_transformed = _create_minimal_features(df)
        else:
            df_transformed = _create_minimal_features(df)
        
        # Prepare features for prediction
        feature_columns = [col for col in df_transformed.columns if col not in ['is_fraud', 'transaction_id', 'customer_id', 'merchant_id']]
        X = df_transformed[feature_columns].fillna(0)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            fraud_probability = float(model.predict_proba(X)[0, 1])
            prediction = int(model.predict(X)[0])
        else:
            # Fallback for models without predict_proba
            prediction = int(model.predict(X)[0])
            fraud_probability = float(prediction)  # Binary prediction as probability
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_score = "low"
        elif fraud_probability < 0.7:
            risk_score = "medium"
        else:
            risk_score = "high"
        
        return FraudPredictionResponse(
            transaction_id=transaction_id,
            is_fraud=prediction,
            fraud_probability=fraud_probability,
            risk_score=risk_score,
            model_used=actual_model_name,
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in fraud prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchFraudPredictionResponse)
async def predict_fraud_batch(
    batch_request: BatchTransactionRequest,
    model_name: Optional[str] = None,
    model_info: tuple = Depends(get_model)
):
    """Predict fraud for multiple transactions."""
    model, actual_model_name = model_info
    
    try:
        predictions = []
        fraud_count = 0
        high_risk_count = 0
        
        for transaction in batch_request.transactions:
            # Use single prediction endpoint for each transaction
            pred_response = await predict_fraud(transaction, model_name, model_info)
            predictions.append(pred_response)
            
            if pred_response.is_fraud:
                fraud_count += 1
            if pred_response.risk_score == "high":
                high_risk_count += 1
        
        # Create summary
        total_transactions = len(predictions)
        summary = {
            "total_transactions": total_transactions,
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / total_transactions if total_transactions > 0 else 0,
            "high_risk_transactions": high_risk_count,
            "processing_timestamp": datetime.now().isoformat(),
            "model_used": actual_model_name
        }
        
        return BatchFraudPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in batch fraud prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/models/{model_name}/info", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = loaded_models[model_name]
    metadata = model_metadata.get(model_name, {})
    
    return ModelInfo(
        model_name=model_name,
        model_type=type(model).__name__,
        version="1.0.0",
        trained_at=metadata.get('trained_at', datetime.now().isoformat()),
        performance_metrics=metadata.get('val_metrics', {}),
        feature_count=getattr(model, 'n_features_in_', 0)
    )

@app.post("/models/reload")
async def reload_models(background_tasks: BackgroundTasks):
    """Reload all models from disk."""
    background_tasks.add_task(reload_models_task)
    return {"message": "Model reload initiated", "timestamp": datetime.now().isoformat()}

async def reload_models_task():
    """Background task to reload models."""
    global loaded_models, model_metadata
    
    logger.info("Reloading models...")
    
    # Clear existing models
    loaded_models.clear()
    model_metadata.clear()
    
    # Reload models (same logic as startup)
    models_dir = Path("models/trained")
    if models_dir.exists():
        for model_file in models_dir.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            try:
                model = joblib.load(model_file)
                loaded_models[model_name] = model
                logger.info(f"Reloaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error reloading model {model_name}: {str(e)}")
    
    logger.info(f"Model reload complete. Loaded {len(loaded_models)} models.")

def _create_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create minimal features matching the training data feature set."""
    df_minimal = df.copy()
    
    # Handle timestamp features
    if 'timestamp' in df_minimal.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_minimal['timestamp']):
            df_minimal['timestamp'] = pd.to_datetime(df_minimal['timestamp'])
        
        df_minimal['hour'] = df_minimal['timestamp'].dt.hour
        df_minimal['day_of_week'] = df_minimal['timestamp'].dt.dayofweek
        df_minimal['week'] = df_minimal['timestamp'].dt.isocalendar().week
        df_minimal['is_weekend'] = (df_minimal['timestamp'].dt.dayofweek >= 5).astype(int)
        df_minimal['is_business_hours'] = ((df_minimal['timestamp'].dt.hour >= 9) & 
                                          (df_minimal['timestamp'].dt.hour <= 17)).astype(int)
        df_minimal['is_late_night'] = ((df_minimal['timestamp'].dt.hour >= 22) | 
                                      (df_minimal['timestamp'].dt.hour <= 6)).astype(int)
        
        # Cyclical encoding
        df_minimal['day_sin'] = np.sin(2 * np.pi * df_minimal['day_of_week'] / 7)
        df_minimal['day_cos'] = np.cos(2 * np.pi * df_minimal['day_of_week'] / 7)
        df_minimal['month_cos'] = np.cos(2 * np.pi * df_minimal['timestamp'].dt.month / 12)
    else:
        df_minimal['hour'] = 12
        df_minimal['day_of_week'] = 1
        df_minimal['week'] = 26
        df_minimal['is_weekend'] = 0
        df_minimal['is_business_hours'] = 1
        df_minimal['is_late_night'] = 0
        df_minimal['day_sin'] = 0.0
        df_minimal['day_cos'] = 1.0
        df_minimal['month_cos'] = 1.0
    
    # Basic amount features
    df_minimal['amount_log'] = np.log1p(df_minimal['amount'])
    df_minimal['amount_squared'] = df_minimal['amount'] ** 2
    df_minimal['amount_sqrt'] = np.sqrt(df_minimal['amount'])
    df_minimal['amount_cubed'] = df_minimal['amount'] ** 3
    
    # Amount percentile and binning (using approximate values for new data)
    df_minimal['is_high_amount'] = (df_minimal['amount'] > 1000).astype(int)  # Threshold for high amount
    df_minimal['amount_percentile_rank'] = np.minimum(df_minimal['amount'] / 10000, 1.0)  # Normalize to 0-1
    df_minimal['amount_bin'] = pd.cut(df_minimal['amount'], bins=10, labels=False).fillna(5)
    
    # Encode categorical features
    if 'merchant_category' in df_minimal.columns:
        category_mapping = {
            'grocery': 0, 'gas': 1, 'restaurant': 2, 'retail': 3, 
            'online': 4, 'entertainment': 5, 'travel': 6, 'healthcare': 7
        }
        df_minimal['merchant_category_encoded'] = df_minimal['merchant_category'].map(category_mapping).fillna(3)
    else:
        df_minimal['merchant_category_encoded'] = 3
    
    # Merchant name encoding (simple hash-based encoding for unknown merchants)
    if 'merchant_name' in df_minimal.columns:
        df_minimal['merchant_name_encoded'] = df_minimal['merchant_name'].apply(
            lambda x: abs(hash(str(x))) % 1000 if pd.notna(x) else 0
        )
    else:
        df_minimal['merchant_name_encoded'] = 0
    
    # Customer-merchant pair encoding
    if 'customer_id' in df_minimal.columns and 'merchant_id' in df_minimal.columns:
        df_minimal['customer_merchant_pair_encoded'] = (
            df_minimal['customer_id'].astype(str) + '_' + df_minimal['merchant_id'].astype(str)
        ).apply(lambda x: abs(hash(x)) % 10000)
    else:
        df_minimal['customer_merchant_pair_encoded'] = 0
    
    # Income-based features
    if 'customer_income' in df_minimal.columns and 'amount' in df_minimal.columns:
        df_minimal['amount_income_ratio'] = df_minimal['amount'] / (df_minimal['customer_income'] + 1e-8)
    else:
        df_minimal['amount_income_ratio'] = 0.1
    
    # Interaction features
    df_minimal['amount_hour_interaction'] = df_minimal['amount'] * df_minimal['hour']
    df_minimal['amount_weekend_interaction'] = df_minimal['amount'] * df_minimal['is_weekend']
    
    if 'customer_age' in df_minimal.columns:
        df_minimal['amount_age_interaction'] = df_minimal['amount'] * df_minimal['customer_age']
    else:
        df_minimal['amount_age_interaction'] = df_minimal['amount'] * 30  # Default age
    
    # Daily amount patterns (approximate for new predictions)
    df_minimal['amount_daily_deviation'] = 0.0  # No historical data for new predictions
    df_minimal['amount_daily_ratio'] = 1.0      # Default ratio
    
    # Fraud-related features (set to 0 for new predictions - no historical fraud data)
    df_minimal['customer_fraud_count'] = 0
    df_minimal['merchant_fraud_count'] = 0
    df_minimal['merchant_fraud_rate'] = 0.05  # Average fraud rate
    df_minimal['merchant_avg_amount'] = df_minimal['amount']  # Use current amount as proxy
    
    # Remove non-numeric columns
    columns_to_drop = ['timestamp', 'merchant_name', 'merchant_category', 'merchant_state', 
                      'customer_id', 'merchant_id', 'customer_age', 'customer_income', 'account_age_days']
    df_minimal = df_minimal.drop(columns=[col for col in columns_to_drop if col in df_minimal.columns])
    
    # Ensure all columns are numeric and handle any remaining issues
    for col in df_minimal.columns:
        if df_minimal[col].dtype == 'object':
            df_minimal[col] = pd.to_numeric(df_minimal[col], errors='coerce')
    
    # Fill any missing values
    df_minimal = df_minimal.fillna(0)
    
    # Ensure we have all the expected features in the correct order
    expected_features = [
        'amount', 'is_weekend', 'day_of_week', 'amount_log', 'amount_squared', 
        'customer_fraud_count', 'merchant_avg_amount', 'merchant_fraud_count', 
        'merchant_fraud_rate', 'week', 'hour', 'is_business_hours', 'is_late_night', 
        'day_sin', 'day_cos', 'month_cos', 'amount_hour_interaction', 
        'amount_weekend_interaction', 'amount_age_interaction', 'amount_daily_deviation', 
        'amount_daily_ratio', 'amount_income_ratio', 'amount_sqrt', 'amount_cubed', 
        'is_high_amount', 'amount_percentile_rank', 'merchant_name_encoded', 
        'merchant_category_encoded', 'customer_merchant_pair_encoded', 'amount_bin'
    ]
    
    # Add any missing expected features with default values
    for feature in expected_features:
        if feature not in df_minimal.columns:
            df_minimal[feature] = 0.0
    
    # Select only the expected features in the correct order
    df_minimal = df_minimal[expected_features]
    
    return df_minimal

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 