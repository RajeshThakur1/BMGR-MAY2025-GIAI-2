#!/usr/bin/env python3
"""
Gradio Web Interface for Fraud Detection API

This module provides a user-friendly web interface using Gradio to interact
with the fraud detection FastAPI backend. Users can input transaction details
and get real-time fraud predictions with detailed explanations.
"""

import gradio as gr
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import random
import logging
from typing import Dict, Any, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
MODELS = ["xgboost", "random_forest", "logistic_regression"]

# Sample data for quick testing
SAMPLE_CUSTOMERS = [
    {"customer_id": "CUST_000601", "age": 22, "income": 35000.0},
    {"customer_id": "CUST_001234", "age": 45, "income": 75000.0},
    {"customer_id": "CUST_005678", "age": 28, "income": 45000.0},
    {"customer_id": "CUST_009876", "age": 55, "income": 90000.0},
]

SAMPLE_MERCHANTS = [
    {"merchant_id": "MER_LUXURY_8934", "name": "Premium Electronics Outlet", "category": "retail", "state": "FL"},
    {"merchant_id": "MER_GROCERY_1234", "name": "Fresh Market Store", "category": "grocery", "state": "CA"},
    {"merchant_id": "MER_GAS_5678", "name": "QuickFill Gas Station", "category": "gas", "state": "TX"},
    {"merchant_id": "MER_REST_9012", "name": "Downtown Bistro", "category": "restaurant", "state": "NY"},
]

def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_sample_data() -> Tuple[str, float, str, str, str, str, int, float, int, str]:
    """Generate sample transaction data for quick testing."""
    customer = random.choice(SAMPLE_CUSTOMERS)
    merchant = random.choice(SAMPLE_MERCHANTS)
    
    # Generate realistic transaction amounts based on merchant category
    amount_ranges = {
        "grocery": (20, 200),
        "gas": (30, 100),
        "restaurant": (15, 150),
        "retail": (25, 2000),
        "online": (10, 500),
        "entertainment": (20, 300),
        "travel": (100, 5000),
        "healthcare": (50, 1000)
    }
    
    min_amt, max_amt = amount_ranges.get(merchant["category"], (10, 1000))
    amount = round(random.uniform(min_amt, max_amt), 2)
    
    # Generate timestamp (recent random time)
    base_time = datetime.now() - timedelta(days=random.randint(0, 30))
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    timestamp = base_time.replace(hour=hour, minute=minute, second=0).strftime("%Y-%m-%dT%H:%M:%S")
    
    # Account age (days)
    account_age = random.randint(1, 1000)
    
    return (
        customer["customer_id"],
        amount,
        merchant["merchant_id"],
        merchant["name"],
        merchant["category"],
        merchant["state"],
        customer["age"],
        customer["income"],
        account_age,
        timestamp
    )

def predict_fraud(customer_id: str, amount: float, merchant_id: str, merchant_name: str, 
                 merchant_category: str, merchant_state: str, customer_age: int, 
                 customer_income: float, account_age_days: int, timestamp: str, 
                 model_name: str) -> Tuple[str, str]:
    """
    Make fraud prediction using the API.
    
    Returns:
        Tuple of (prediction_result, api_status)
    """
    
    # Check API health first
    if not check_api_health():
        error_msg = "❌ **API Server Error**\n\nThe fraud detection API is not responding. Please ensure the API server is running on http://localhost:8000"
        return error_msg, "🔴 **API Status:** Offline"
    
    # Prepare request data
    request_data = {
        "customer_id": customer_id or f"CUST_{random.randint(100000, 999999)}",
        "amount": float(amount) if amount else 100.0,
        "merchant_id": merchant_id or f"MER_{random.randint(10000, 99999)}",
        "merchant_name": merchant_name or "Sample Merchant",
        "merchant_category": merchant_category or "retail",
        "merchant_state": merchant_state or "CA",
        "customer_age": int(customer_age) if customer_age else 30,
        "customer_income": float(customer_income) if customer_income else 50000.0,
        "account_age_days": int(account_age_days) if account_age_days else 30,
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }
    
    try:
        # Make API request
        response = requests.post(
            f"{API_BASE_URL}/predict",
            params={"model_name": model_name},
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Format the prediction result
            fraud_status = "🚨 **FRAUD DETECTED**" if result['is_fraud'] else "✅ **LEGITIMATE TRANSACTION**"
            probability = result['fraud_probability']
            risk_score = result['risk_score'].upper()
            
            # Risk score emoji
            risk_emoji = {
                "LOW": "🟢",
                "MEDIUM": "🟡", 
                "HIGH": "🔴"
            }.get(risk_score, "⚪")
            
            prediction_result = f"""
## {fraud_status}

### 📊 **Prediction Details**
- **Fraud Probability:** {probability:.1%}
- **Risk Score:** {risk_emoji} {risk_score}
- **Model Used:** {result['model_used'].title().replace('_', ' ')}
- **Transaction ID:** `{result['transaction_id']}`

### 💳 **Transaction Summary**
- **Amount:** ${request_data['amount']:,.2f}
- **Customer:** {request_data['customer_id']} (Age: {request_data['customer_age']})
- **Merchant:** {request_data['merchant_name']} ({request_data['merchant_category'].title()})
- **Location:** {request_data['merchant_state']}
- **Time:** {request_data['timestamp']}
- **Account Age:** {request_data['account_age_days']} days

### 🔍 **Risk Factors**
"""
            
            # Add risk factor analysis
            risk_factors = []
            if probability > 0.7:
                risk_factors.append("⚠️ Very high fraud probability")
            if request_data['amount'] > 5000:
                risk_factors.append("💰 High transaction amount")
            if request_data['customer_age'] < 25:
                risk_factors.append("👤 Young customer profile")
            if request_data['account_age_days'] < 30:
                risk_factors.append("🆕 New account (< 30 days)")
            
            # Check time-based risk factors
            try:
                trans_time = datetime.fromisoformat(request_data['timestamp'])
                if trans_time.hour < 6 or trans_time.hour > 22:
                    risk_factors.append("🌙 Late night/early morning transaction")
                if trans_time.weekday() >= 5:  # Weekend
                    risk_factors.append("📅 Weekend transaction")
            except:
                pass
            
            if risk_factors:
                prediction_result += "\n".join(f"- {factor}" for factor in risk_factors)
            else:
                prediction_result += "- ✅ No significant risk factors detected"
            
            api_status = f"🟢 **API Status:** Online | Response Time: {response.elapsed.total_seconds():.2f}s"
            
            return prediction_result, api_status
            
        else:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response content'
            error_msg = f"""
❌ **Prediction Failed**

**Status Code:** {response.status_code}
**Error:** {error_detail}

Please check your input data and try again.
"""
            return error_msg, f"🟡 **API Status:** Error ({response.status_code})"
            
    except requests.exceptions.Timeout:
        error_msg = "⏱️ **Request Timeout**\n\nThe API request took too long to respond. Please try again."
        return error_msg, "🟡 **API Status:** Timeout"
    except requests.exceptions.RequestException as e:
        error_msg = f"🔗 **Connection Error**\n\nFailed to connect to the API: {str(e)}"
        return error_msg, "🔴 **API Status:** Connection Failed"
    except Exception as e:
        error_msg = f"❌ **Unexpected Error**\n\n{str(e)}"
        return error_msg, "🔴 **API Status:** Error"

def batch_predict(file) -> str:
    """Process batch predictions from uploaded CSV file."""
    if file is None:
        return "❌ Please upload a CSV file for batch processing."
    
    try:
        df = pd.read_csv(file.name)
        required_columns = ['customer_id', 'amount', 'merchant_id', 'merchant_name', 
                           'merchant_category', 'merchant_state', 'customer_age', 
                           'customer_income', 'account_age_days', 'timestamp']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"❌ Missing required columns: {', '.join(missing_columns)}"
        
        results = []
        for _, row in df.iterrows():
            prediction, _ = predict_fraud(
                row['customer_id'], row['amount'], row['merchant_id'], 
                row['merchant_name'], row['merchant_category'], row['merchant_state'],
                row['customer_age'], row['customer_income'], row['account_age_days'],
                row['timestamp'], "xgboost"  # Default model for batch
            )
            
            # Extract key info from prediction
            is_fraud = "FRAUD DETECTED" in prediction
            results.append({
                'customer_id': row['customer_id'],
                'amount': row['amount'],
                'is_fraud': is_fraud,
                'prediction': prediction[:100] + "..." if len(prediction) > 100 else prediction
            })
            
        results_df = pd.DataFrame(results)
        return f"✅ Processed {len(results)} transactions. Results preview:\n\n{results_df.to_string(index=False)}"
        
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"

# Create Gradio Interface
def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="🛡️ Fraud Detection System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            🛡️ Fraud Detection System
        </div>
        <p style="text-align: center; font-size: 1.2em; color: #666; margin-bottom: 30px;">
            Advanced Machine Learning-powered fraud detection with real-time predictions
        </p>
        """)
        
        with gr.Tabs():
            # Single Prediction Tab
            with gr.Tab("🔍 Single Prediction", id="single"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 💳 Transaction Details")
                        
                        with gr.Row():
                            customer_id = gr.Textbox(
                                label="Customer ID",
                                placeholder="CUST_000001",
                                info="Unique customer identifier"
                            )
                            amount = gr.Number(
                                label="Transaction Amount ($)",
                                minimum=0.01,
                                value=100.0,
                                info="Transaction amount in USD"
                            )
                        
                        with gr.Row():
                            merchant_id = gr.Textbox(
                                label="Merchant ID",
                                placeholder="MER_12345",
                                info="Unique merchant identifier"
                            )
                            merchant_name = gr.Textbox(
                                label="Merchant Name",
                                placeholder="ABC Store",
                                info="Business name"
                            )
                        
                        with gr.Row():
                            merchant_category = gr.Dropdown(
                                label="Merchant Category",
                                choices=["grocery", "gas", "restaurant", "retail", "online", "entertainment", "travel", "healthcare"],
                                value="retail",
                                info="Type of business"
                            )
                            merchant_state = gr.Textbox(
                                label="Merchant State",
                                placeholder="CA",
                                max_lines=1,
                                info="State abbreviation"
                            )
                        
                        gr.Markdown("### 👤 Customer Information")
                        
                        with gr.Row():
                            customer_age = gr.Number(
                                label="Customer Age",
                                minimum=18,
                                maximum=100,
                                value=30,
                                info="Customer's age in years"
                            )
                            customer_income = gr.Number(
                                label="Annual Income ($)",
                                minimum=0,
                                value=50000.0,
                                info="Customer's annual income"
                            )
                        
                        with gr.Row():
                            account_age_days = gr.Number(
                                label="Account Age (days)",
                                minimum=0,
                                value=365,
                                info="Days since account creation"
                            )
                            timestamp = gr.Textbox(
                                label="Transaction Timestamp",
                                placeholder="2023-12-01T14:30:00",
                                info="Format: YYYY-MM-DDTHH:MM:SS"
                            )
                        
                        gr.Markdown("### 🤖 Model Selection")
                        model_name = gr.Dropdown(
                            label="ML Model",
                            choices=MODELS,
                            value="xgboost",
                            info="Choose the machine learning model for prediction"
                        )
                        
                        with gr.Row():
                            predict_btn = gr.Button("🔍 Predict Fraud", variant="primary", size="lg")
                            sample_btn = gr.Button("🎲 Load Sample Data", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Prediction Results")
                        prediction_output = gr.Markdown(
                            value="👋 **Welcome!**\n\nEnter transaction details and click 'Predict Fraud' to get started.",
                            container=True
                        )
                        
                        api_status = gr.Markdown(
                            value="🔄 **API Status:** Checking...",
                            container=True
                        )
            
            # Batch Processing Tab
            with gr.Tab("📋 Batch Processing", id="batch"):
                gr.Markdown("""
                ### 📊 Batch Fraud Detection
                Upload a CSV file with multiple transactions for batch processing.
                
                **Required Columns:**
                `customer_id, amount, merchant_id, merchant_name, merchant_category, merchant_state, customer_age, customer_income, account_age_days, timestamp`
                """)
                
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload CSV File",
                            file_types=[".csv"]
                        )
                        batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                    
                    with gr.Column():
                        batch_output = gr.Textbox(
                            label="Batch Results",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
            
            # API Information Tab
            with gr.Tab("ℹ️ API Info", id="info"):
                gr.Markdown(f"""
                ### 🔧 API Configuration
                
                **Base URL:** `{API_BASE_URL}`
                
                **Available Models:**
                - **XGBoost** - High-performance gradient boosting (Recommended)
                - **Random Forest** - Ensemble method with good interpretability
                - **Logistic Regression** - Fast linear model with probability outputs
                
                ### 📡 API Endpoints
                
                - **Health Check:** `GET /health`
                - **Single Prediction:** `POST /predict?model_name={{model}}`
                - **Model Info:** `GET /models`
                
                ### 📝 Sample Request
                ```json
                {{
                    "customer_id": "CUST_000601",
                    "amount": 1500.75,
                    "merchant_id": "MER_RETAIL_1234",
                    "merchant_name": "Electronics Store",
                    "merchant_category": "retail",
                    "merchant_state": "CA",
                    "customer_age": 28,
                    "customer_income": 65000.0,
                    "account_age_days": 45,
                    "timestamp": "2023-12-01T14:30:00"
                }}
                ```
                
                ### 🎯 Interpretation Guide
                
                **Fraud Probability:**
                - **0-30%:** Low risk (✅ Legitimate)
                - **30-70%:** Medium risk (⚠️ Review)
                - **70-100%:** High risk (🚨 Likely fraud)
                
                **Risk Factors:**
                - High transaction amounts
                - New customer accounts
                - Unusual transaction times
                - Age and income anomalies
                """)
        
        # Event handlers
        predict_btn.click(
            fn=predict_fraud,
            inputs=[customer_id, amount, merchant_id, merchant_name, merchant_category, 
                   merchant_state, customer_age, customer_income, account_age_days, timestamp, model_name],
            outputs=[prediction_output, api_status]
        )
        
        sample_btn.click(
            fn=get_sample_data,
            outputs=[customer_id, amount, merchant_id, merchant_name, merchant_category, 
                    merchant_state, customer_age, customer_income, account_age_days, timestamp]
        )
        
        batch_btn.click(
            fn=batch_predict,
            inputs=[file_upload],
            outputs=[batch_output]
        )
        
        # Auto-check API status on load
        interface.load(
            fn=lambda: "🟢 **API Status:** Online" if check_api_health() else "🔴 **API Status:** Offline",
            outputs=[api_status]
        )
    
    return interface

def main():
    """Main function to launch the Gradio interface."""
    logger.info("Starting Fraud Detection Gradio Interface...")
    
    # Check if API is running
    if not check_api_health():
        logger.warning("⚠️  API server is not responding. Please start the FastAPI server first.")
        print("⚠️  Warning: API server is not responding at http://localhost:8000")
        print("   Please ensure the FastAPI server is running before using the interface.")
        print("   You can start it with: python src/deployment/api.py")
        print()
    
    # Create and launch interface
    interface = create_interface()
    
    logger.info("🚀 Launching Gradio interface...")
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Gradio default port
        share=False,            # Set to True for public sharing
        debug=False,
        show_error=True,
        inbrowser=True          # Auto-open browser
    )

if __name__ == "__main__":
    main() 