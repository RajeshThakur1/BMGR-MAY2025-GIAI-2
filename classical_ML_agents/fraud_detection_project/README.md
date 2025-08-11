# Fraud Detection ML Pipeline

A comprehensive end-to-end machine learning system for detecting fraudulent transactions using classical ML algorithms.

## Project Structure

```
fraud_detection_project/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── src/
│   ├── data/
│   │   ├── data_generator.py
│   │   ├── data_loader.py
│   │   └── data_validator.py
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── classical_models.py
│   │   └── model_evaluation.py
│   ├── utils/
│   │   ├── preprocessing.py
│   │   └── visualization.py
│   └── deployment/
│       ├── api.py
│       └── monitoring.py
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── tests/
└── models/
    └── trained/
```

## ML Pipeline Stages

1. **Requirements & HLD**: System architecture and high-level design
2. **Data Generation**: Synthetic fraud detection dataset creation
3. **Data Collection**: Data loading and validation utilities
4. **EDA**: Comprehensive exploratory data analysis
5. **Feature Engineering**: Advanced feature creation and selection
6. **Data Splitting**: Stratified train/validation/test splits
7. **Model Building**: Multiple classical ML algorithms
8. **Evaluation**: Comprehensive model performance assessment
9. **Hyperparameter Tuning**: Grid search and cross-validation optimization
10. **Deployment**: API deployment and serving infrastructure
11. **Monitoring**: Performance monitoring and drift detection

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Generate synthetic data: `python src/data/data_generator.py`
3. Run EDA: Open `notebooks/02_eda_analysis.ipynb`
4. Train models: `python src/models/classical_models.py`

## Features

- **Realistic Synthetic Data**: Generates transaction data with fraud patterns
- **Multiple ML Models**: Random Forest, XGBoost, SVM, Logistic Regression
- **Advanced Feature Engineering**: Time-based, aggregation, and statistical features
- **Comprehensive Evaluation**: Precision, Recall, F1, AUC-ROC, Confusion Matrix
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Model Deployment**: REST API with real-time prediction
- **Monitoring**: Performance tracking and data drift detection 