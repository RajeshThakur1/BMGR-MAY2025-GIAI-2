"""
Comprehensive API Testing Script

This script tests all endpoints of the Fraud Detection API
with sample data and validates responses.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def print_response(endpoint: str, response: requests.Response):
    """Pretty print API response."""
    print(f"\n{'='*50}")
    print(f"🔗 Endpoint: {endpoint}")
    print(f"📊 Status Code: {response.status_code}")
    print(f"⏱️  Response Time: {response.elapsed.total_seconds():.3f}s")
    
    if response.status_code == 200:
        print("✅ Success!")
        try:
            response_json = response.json()
            print(f"📄 Response: {json.dumps(response_json, indent=2)}")
        except:
            print(f"📄 Response: {response.text}")
    else:
        print("❌ Error!")
        print(f"📄 Response: {response.text}")

def test_root_endpoint():
    """Test the root endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print_response("GET /", response)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: API server is not running!")
        return False

def test_health_endpoint():
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print_response("GET /health", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_models_list_endpoint():
    """Test the models list endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        print_response("GET /models", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Models list failed: {str(e)}")
        return False

def test_single_prediction():
    """Test single fraud prediction endpoint."""
    # Sample transaction data
    sample_transaction = {
        "customer_id": "CUST_001234",
        "amount": 1250.75,
        "merchant_id": "MER_RETAIL_5678",
        "merchant_name": "TechStore Plus",
        "merchant_category": "retail",
        "merchant_state": "NY",
        "customer_age": 35,
        "customer_income": 65000.0,
        "account_age_days": 730,
        "timestamp": "2023-12-01T14:30:00"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers=HEADERS,
            json=sample_transaction
        )
        print_response("POST /predict", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Single prediction failed: {str(e)}")
        return False

def test_batch_prediction():
    """Test batch fraud prediction endpoint."""
    # Sample batch of transactions
    batch_transactions = {
        "transactions": [
            {
                "customer_id": "CUST_001234",
                "amount": 1250.75,
                "merchant_id": "MER_RETAIL_5678",
                "merchant_name": "TechStore Plus",
                "merchant_category": "retail",
                "merchant_state": "NY",
                "customer_age": 35,
                "customer_income": 65000.0,
                "account_age_days": 730,
                "timestamp": "2023-12-01T14:30:00"
            },
            {
                "customer_id": "CUST_005678",
                "amount": 45.99,
                "merchant_id": "MER_GROCERY_1234",
                "merchant_name": "Fresh Market",
                "merchant_category": "grocery",
                "merchant_state": "CA",
                "customer_age": 28,
                "customer_income": 52000.0,
                "account_age_days": 365,
                "timestamp": "2023-12-01T16:45:00"
            },
            {
                "customer_id": "CUST_009999",
                "amount": 8750.00,
                "merchant_id": "MER_LUXURY_9999",
                "merchant_name": "Premium Jewelry",
                "merchant_category": "retail",
                "merchant_state": "FL",
                "customer_age": 55,
                "customer_income": 120000.0,
                "account_age_days": 90,
                "timestamp": "2023-12-01T23:15:00"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            headers=HEADERS,
            json=batch_transactions
        )
        print_response("POST /predict/batch", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Batch prediction failed: {str(e)}")
        return False

def test_model_info():
    """Test specific model info endpoint."""
    # First get available models
    try:
        models_response = requests.get(f"{API_BASE_URL}/models")
        if models_response.status_code == 200:
            models = models_response.json()
            if models and len(models) > 0:
                # Test with the first available model
                model_name = models[0]["model_name"]
                response = requests.get(f"{API_BASE_URL}/models/{model_name}/info")
                print_response(f"GET /models/{model_name}/info", response)
                return response.status_code == 200
            else:
                print("⚠️  No models available for testing model info endpoint")
                return True
        else:
            print("❌ Could not retrieve models list for model info test")
            return False
    except Exception as e:
        print(f"❌ Model info test failed: {str(e)}")
        return False

def test_model_reload():
    """Test model reload endpoint."""
    try:
        response = requests.post(f"{API_BASE_URL}/models/reload")
        print_response("POST /models/reload", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Model reload failed: {str(e)}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print(f"\n{'='*50}")
    print("🧪 Testing Edge Cases and Error Handling")
    
    # Test invalid transaction data
    invalid_transaction = {
        "customer_id": "INVALID",
        "amount": -100,  # Negative amount should fail validation
        "merchant_id": "",
        "merchant_name": "",
        "merchant_category": "invalid_category"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers=HEADERS,
            json=invalid_transaction
        )
        print_response("POST /predict (Invalid Data)", response)
        print("✅ Validation working correctly!" if response.status_code == 422 else "⚠️  Unexpected response")
    except Exception as e:
        print(f"❌ Edge case test failed: {str(e)}")
    
    # Test non-existent model info
    try:
        response = requests.get(f"{API_BASE_URL}/models/nonexistent_model/info")
        print_response("GET /models/nonexistent_model/info", response)
        print("✅ Error handling working!" if response.status_code == 404 else "⚠️  Unexpected response")
    except Exception as e:
        print(f"❌ Non-existent model test failed: {str(e)}")

def run_performance_test():
    """Run basic performance tests."""
    print(f"\n{'='*50}")
    print("⚡ Performance Testing")
    
    sample_transaction = {
        "customer_id": "PERF_TEST_001",
        "amount": 150.75,
        "merchant_id": "MER_PERF_TEST",
        "merchant_name": "Performance Test Store",
        "merchant_category": "retail",
        "merchant_state": "TX",
        "customer_age": 30,
        "customer_income": 50000.0,
        "account_age_days": 365
    }
    
    # Test multiple requests
    response_times = []
    success_count = 0
    num_requests = 10
    
    print(f"🚀 Sending {num_requests} requests...")
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                headers=HEADERS,
                json=sample_transaction,
                timeout=5
            )
            end_time = time.time()
            
            if response.status_code == 200:
                success_count += 1
                response_times.append(end_time - start_time)
            
        except Exception as e:
            print(f"❌ Request {i+1} failed: {str(e)}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"📊 Performance Results:")
        print(f"   • Success Rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
        print(f"   • Average Response Time: {avg_time*1000:.2f}ms")
        print(f"   • Min Response Time: {min_time*1000:.2f}ms")
        print(f"   • Max Response Time: {max_time*1000:.2f}ms")
        
        if avg_time < 0.1:  # Under 100ms
            print("✅ Performance: Excellent!")
        elif avg_time < 0.5:  # Under 500ms
            print("✅ Performance: Good!")
        else:
            print("⚠️  Performance: Could be improved")

def main():
    """Run all API tests."""
    print("🚀 Starting Fraud Detection API Testing")
    print("=" * 60)
    
    # Wait for server to start
    print("⏳ Waiting for API server to start...")
    time.sleep(3)
    
    test_results = []
    
    # Test all endpoints
    test_results.append(("Root Endpoint", test_root_endpoint()))
    test_results.append(("Health Check", test_health_endpoint()))
    test_results.append(("Models List", test_models_list_endpoint()))
    test_results.append(("Single Prediction", test_single_prediction()))
    test_results.append(("Batch Prediction", test_batch_prediction()))
    test_results.append(("Model Info", test_model_info()))
    test_results.append(("Model Reload", test_model_reload()))
    
    # Test edge cases
    test_edge_cases()
    
    # Performance testing
    run_performance_test()
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working perfectly!")
    else:
        print("⚠️  Some tests failed. Please check the API implementation.")
    
    print(f"\n📖 API Documentation: {API_BASE_URL}/docs")
    print(f"🔄 Alternative Docs: {API_BASE_URL}/redoc")

if __name__ == "__main__":
    main() 