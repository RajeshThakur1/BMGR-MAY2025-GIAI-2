"""
Classical Machine Learning Models for Fraud Detection

This module provides comprehensive training and evaluation of multiple
classical ML algorithms for fraud detection including ensemble methods,
tree-based models, and linear models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, f1_score, precision_score, 
                           recall_score, accuracy_score, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FraudModelTrainer:
    """
    Comprehensive trainer for classical ML models for fraud detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary for model training
        """
        self.config = config or {}
        self.models = {}
        self.model_results = {}
        self.feature_importance = {}
        self.is_trained = False
        
        # Default configuration
        self.default_config = {
            'random_state': 42,
            'cv_folds': 5,
            'test_size': 0.2,
            'handle_imbalance': True,
            'imbalance_method': 'smote',  # 'smote', 'undersample', 'smoteenn'
            'calibrate_probabilities': True,
            'feature_selection': True,
            'n_features': 50
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Initialize models with default parameters
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all classical ML models with default parameters."""
        logger.info("Initializing classical ML models")
        
        random_state = self.config['random_state']
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric='logloss',
                verbosity=0
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=-1
            ),
            
            'logistic_regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=random_state,
                max_iter=1000
            ),
            
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=random_state
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=random_state
            ),
            
            'naive_bayes': GaussianNB(),
            
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            
            'decision_tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            ),
            
            'ada_boost': AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=random_state
            )
        }
    
    def train_all_models(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train all classical ML models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training results for all models
        """
        logger.info("Starting training for all classical ML models")
        
        # Handle class imbalance if requested
        if self.config['handle_imbalance']:
            X_train, y_train = self._handle_class_imbalance(X_train, y_train)
        
        # Train each model
        all_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}")
            
            try:
                # Train the model
                model_results = self._train_single_model(
                    model_name, model, X_train, y_train, X_val, y_val
                )
                all_results[model_name] = model_results
                
                # Store the trained model
                self.models[model_name] = model_results['model']
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        self.model_results = all_results
        self.is_trained = True
        
        logger.info("Completed training all models")
        return all_results
    
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using specified method."""
        logger.info(f"Handling class imbalance using {self.config['imbalance_method']}")
        
        fraud_rate = y.mean()
        logger.info(f"Original fraud rate: {fraud_rate:.2%}")
        
        if self.config['imbalance_method'] == 'smote':
            sampler = SMOTE(random_state=self.config['random_state'])
        elif self.config['imbalance_method'] == 'undersample':
            sampler = RandomUnderSampler(random_state=self.config['random_state'])
        elif self.config['imbalance_method'] == 'smoteenn':
            sampler = SMOTEENN(random_state=self.config['random_state'])
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        new_fraud_rate = y_resampled.mean()
        logger.info(f"Resampled fraud rate: {new_fraud_rate:.2%}")
        logger.info(f"Dataset size: {len(X)} -> {len(X_resampled)}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def _train_single_model(self, 
                           model_name: str,
                           model: Any,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_val: Optional[pd.DataFrame] = None,
                           y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train a single model and return results."""
        
        # Record training start time
        start_time = datetime.now()
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Calibrate probabilities if requested
        if self.config['calibrate_probabilities'] and hasattr(model, 'predict_proba'):
            try:
                model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Could not calibrate {model_name}: {str(e)}")
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Cross-validation scores
        cv_scores = self._calculate_cv_scores(model, X_train, y_train)
        
        # Training predictions and metrics
        train_predictions = model.predict(X_train)
        train_probabilities = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        train_metrics = self._calculate_metrics(y_train, train_predictions, train_probabilities)
        
        # Validation predictions and metrics (if validation set provided)
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_predictions = model.predict(X_val)
            val_probabilities = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            val_metrics = self._calculate_metrics(y_val, val_predictions, val_probabilities)
        
        # Feature importance (if available)
        feature_importance = self._get_feature_importance(model, X_train.columns)
        
        return {
            'model': model,
            'model_name': model_name,
            'training_time': training_time,
            'cv_scores': cv_scores,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': feature_importance,
            'trained_at': datetime.now().isoformat()
        }
    
    def _calculate_cv_scores(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate cross-validation scores."""
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, 
                           random_state=self.config['random_state'])
        
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                cv_results[f'{metric}_mean'] = scores.mean()
                cv_results[f'{metric}_std'] = scores.std()
            except Exception as e:
                logger.warning(f"Could not calculate {metric} for cross-validation: {str(e)}")
                cv_results[f'{metric}_mean'] = 0.0
                cv_results[f'{metric}_std'] = 0.0
        
        return cv_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                metrics['average_precision'] = average_precision_score(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Could not calculate probability-based metrics: {str(e)}")
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return None
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        logger.info("Evaluating all models on test set")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            if isinstance(model, dict) and 'error' in model:
                evaluation_results[model_name] = model
                continue
                
            try:
                # Get predictions
                test_predictions = model.predict(X_test)
                test_probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                test_metrics = self._calculate_metrics(y_test, test_predictions, test_probabilities)
                
                # Get confusion matrix
                cm = confusion_matrix(y_test, test_predictions)
                
                evaluation_results[model_name] = {
                    'test_metrics': test_metrics,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': classification_report(y_test, test_predictions, output_dict=True)
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get a comparison of all trained models."""
        if not self.is_trained:
            raise ValueError("Models must be trained before comparison")
        
        comparison_data = []
        
        for model_name, results in self.model_results.items():
            if isinstance(results, dict) and 'error' in results:
                continue
                
            row = {'model': model_name}
            
            # Add training metrics
            if 'train_metrics' in results:
                for metric, value in results['train_metrics'].items():
                    row[f'train_{metric}'] = value
            
            # Add validation metrics
            if 'val_metrics' in results and results['val_metrics']:
                for metric, value in results['val_metrics'].items():
                    row[f'val_{metric}'] = value
            
            # Add CV scores
            if 'cv_scores' in results:
                for metric, value in results['cv_scores'].items():
                    row[f'cv_{metric}'] = value
            
            # Add training time
            if 'training_time' in results:
                row['training_time_seconds'] = results['training_time']
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric: str = 'f1_score', dataset: str = 'val') -> Tuple[str, Any]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
            dataset: Dataset to use for comparison ('train', 'val', 'cv')
            
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before getting best model")
        
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.model_results.items():
            if isinstance(results, dict) and 'error' in results:
                continue
            
            # Get score based on dataset
            score = None
            if dataset == 'train' and 'train_metrics' in results:
                score = results['train_metrics'].get(metric)
            elif dataset == 'val' and 'val_metrics' in results and results['val_metrics']:
                score = results['val_metrics'].get(metric)
            elif dataset == 'cv' and 'cv_scores' in results:
                score = results['cv_scores'].get(f'{metric}_mean')
            
            if score is not None and score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"Could not find best model using metric {metric} on {dataset}")
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, output_dir: str = "models/trained/") -> None:
        """Save all trained models to disk."""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            if not isinstance(model, dict):  # Skip error entries
                model_file = output_path / f"{model_name}_model.pkl"
                joblib.dump(model, model_file)
                logger.info(f"Saved {model_name} to {model_file}")
        
        # Save model results
        results_file = output_path / "model_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.model_results.items():
            if isinstance(results, dict) and 'error' not in results:
                serializable_results[model_name] = {
                    k: v for k, v in results.items() 
                    if k not in ['model']  # Exclude the actual model object
                }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Saved model results to {results_file}")
    
    def load_models(self, input_dir: str = "models/trained/") -> None:
        """Load trained models from disk."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
        
        # Load individual models
        for model_file in input_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            try:
                model = joblib.load(model_file)
                self.models[model_name] = model
                logger.info(f"Loaded {model_name} from {model_file}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
        
        # Load model results
        results_file = input_path / "model_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.model_results = json.load(f)
            logger.info(f"Loaded model results from {results_file}")
        
        self.is_trained = True

def main():
    """Main function for testing the model trainer."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    # Load processed data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Convert timestamp columns
    for df in [train_df, val_df, test_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Apply feature engineering
    from features.feature_engineering import FeatureEngineer
    
    config = {
        'create_interactions': True,
        'create_ratios': True,
        'create_aggregations': False,  # Skip to reduce processing time
        'categorical_encoding': 'target',
        'scaling_method': 'robust'
    }
    
    fe = FeatureEngineer(config)
    
    # Transform datasets
    train_transformed = fe.fit_transform(train_df, 'is_fraud')
    val_transformed = fe.transform(val_df)
    test_transformed = fe.transform(test_df)
    
    # Feature selection
    train_selected, selected_features = fe.select_features(train_transformed, 'is_fraud', k=30)
    val_selected = val_transformed[['is_fraud'] + selected_features]
    test_selected = test_transformed[['is_fraud'] + selected_features]
    
    # Prepare training data
    X_train = train_selected.drop('is_fraud', axis=1)
    y_train = train_selected['is_fraud']
    X_val = val_selected.drop('is_fraud', axis=1)
    y_val = val_selected['is_fraud']
    X_test = test_selected.drop('is_fraud', axis=1)
    y_test = test_selected['is_fraud']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize and train models
    model_config = {
        'handle_imbalance': True,
        'imbalance_method': 'smote',
        'calibrate_probabilities': True
    }
    
    trainer = FraudModelTrainer(model_config)
    
    # Train a subset of models for testing
    selected_models = ['random_forest', 'xgboost', 'logistic_regression']
    trainer.models = {k: v for k, v in trainer.models.items() if k in selected_models}
    
    # Train models
    training_results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    evaluation_results = trainer.evaluate_models(X_test, y_test)
    
    # Print results
    comparison_df = trainer.get_model_comparison()
    print("\nModel Comparison:")
    print(comparison_df[['model', 'val_f1_score', 'val_precision', 'val_recall', 'val_roc_auc']].round(4))
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model('f1_score', 'val')
    print(f"\nBest model: {best_model_name}")
    
    # Save models
    trainer.save_models()
    
    print("\nModel training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 