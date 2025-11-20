"""
Hyperparameter Tuning Module

This module provides advanced hyperparameter optimization for fraud detection models
using Optuna framework with Bayesian optimization, early stopping, and pruning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import joblib
import json
import time
from pathlib import Path
from datetime import datetime
import warnings

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter Optimization
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FraudHyperparameterTuner:
    """
    Advanced hyperparameter tuning for fraud detection models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            config: Configuration dictionary for tuning
        """
        self.config = config or {}
        self.studies = {}
        self.best_params = {}
        self.tuning_results = {}
        
        # Default configuration
        self.default_config = {
            'n_trials': 100,
            'timeout': 3600,  # 1 hour
            'cv_folds': 5,
            'random_state': 42,
            'optimization_metric': 'f1',  # 'f1', 'precision', 'recall', 'roc_auc'
            'direction': 'maximize',
            'pruner': 'median',  # 'median', 'none'
            'sampler': 'tpe',    # 'tpe', 'random'
            'early_stopping_rounds': 10,
            'verbose': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Initialize parameter spaces
        self._initialize_parameter_spaces()
        
    def _initialize_parameter_spaces(self):
        """Initialize hyperparameter search spaces for different models."""
        self.parameter_spaces = {
            'random_forest': {
                'n_estimators': (50, 300),
                'max_depth': (5, 30),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            
            'xgboost': {
                'n_estimators': (50, 300),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'colsample_bylevel': (0.6, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
                'min_child_weight': (1, 10)
            },
            
            'lightgbm': {
                'n_estimators': (50, 300),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'num_leaves': (10, 300),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
                'min_child_weight': (1, 50),
                'min_child_samples': (5, 100)
            },
            
            'logistic_regression': {
                'C': (0.001, 100.0),
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': (0.0, 1.0),  # Only for elasticnet
                'class_weight': ['balanced', None],
                'max_iter': (100, 2000)
            },
            
            'svm': {
                'C': (0.001, 100.0),
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'degree': (2, 5),  # Only for poly kernel
                'coef0': (0.0, 10.0),  # For poly and sigmoid
                'class_weight': ['balanced', None]
            },
            
            'gradient_boosting': {
                'n_estimators': (50, 300),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'subsample': (0.6, 1.0),
                'max_features': ['sqrt', 'log2', None]
            }
        }
    
    def tune_model(self,
                   model_name: str,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing tuning results
        """
        if model_name not in self.parameter_spaces:
            raise ValueError(f"Model {model_name} not supported. Available: {list(self.parameter_spaces.keys())}")
        
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        start_time = time.time()
        
        # Create Optuna study
        study_name = f"{model_name}_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure pruner
        if self.config['pruner'] == 'median':
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        else:
            pruner = None
        
        # Configure sampler
        if self.config['sampler'] == 'tpe':
            sampler = TPESampler(seed=self.config['random_state'])
        else:
            sampler = None
        
        study = optuna.create_study(
            direction=self.config['direction'],
            study_name=study_name,
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective function
        def objective(trial):
            return self._objective_function(trial, model_name, X_train, y_train, X_val, y_val)
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config['n_trials'],
            timeout=self.config['timeout'],
            callbacks=[self._logging_callback] if self.config['verbose'] else None
        )
        
        # Calculate tuning time
        tuning_time = time.time() - start_time
        
        # Store results
        self.studies[model_name] = study
        self.best_params[model_name] = study.best_params
        
        # Evaluate best model
        best_model = self._create_model_with_params(model_name, study.best_params)
        
        # Cross-validation with best parameters
        cv_scores = self._evaluate_model_cv(best_model, X_train, y_train)
        
        # Validation score if validation set provided
        val_score = None
        if X_val is not None and y_val is not None:
            best_model.fit(X_train, y_train)
            y_val_pred = best_model.predict(X_val)
            val_score = self._calculate_metric(y_val, y_val_pred)
        
        tuning_results = {
            'model_name': model_name,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'tuning_time_seconds': tuning_time,
            'cv_scores': cv_scores,
            'val_score': val_score,
            'study': study,
            'completed_at': datetime.now().isoformat()
        }
        
        self.tuning_results[model_name] = tuning_results
        
        logger.info(f"Completed tuning for {model_name}")
        logger.info(f"Best {self.config['optimization_metric']}: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return tuning_results
    
    def _objective_function(self,
                          trial: optuna.Trial,
                          model_name: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None) -> float:
        """Objective function for Optuna optimization."""
        
        # Sample hyperparameters
        params = self._sample_parameters(trial, model_name)
        
        # Create model with sampled parameters
        model = self._create_model_with_params(model_name, params)
        
        # Use validation set if provided, otherwise use cross-validation
        if X_val is not None and y_val is not None:
            # Train on training set, evaluate on validation set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = self._calculate_metric(y_val, y_pred)
        else:
            # Use cross-validation
            cv_scores = self._evaluate_model_cv(model, X_train, y_train)
            score = cv_scores['mean']
        
        return score
    
    def _sample_parameters(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Sample hyperparameters for a given model."""
        param_space = self.parameter_spaces[model_name]
        params = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, tuple) and len(param_config) == 2:
                # Continuous or integer range
                low, high = param_config
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            elif isinstance(param_config, list):
                # Categorical
                params[param_name] = trial.suggest_categorical(param_name, param_config)
        
        # Handle conditional parameters
        if model_name == 'logistic_regression':
            if params.get('penalty') == 'elasticnet':
                if 'l1_ratio' not in params:
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            elif 'l1_ratio' in params:
                del params['l1_ratio']
            
            # Solver compatibility
            if params.get('penalty') == 'elasticnet' and params.get('solver') != 'saga':
                params['solver'] = 'saga'
            elif params.get('penalty') == 'l1' and params.get('solver') not in ['liblinear', 'saga']:
                params['solver'] = trial.suggest_categorical('solver_l1', ['liblinear', 'saga'])
        
        if model_name == 'svm':
            if params.get('kernel') != 'poly' and 'degree' in params:
                del params['degree']
            if params.get('kernel') not in ['poly', 'sigmoid'] and 'coef0' in params:
                del params['coef0']
        
        return params
    
    def _create_model_with_params(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Create a model instance with given parameters."""
        base_params = {'random_state': self.config['random_state']}
        
        if model_name == 'random_forest':
            base_params.update({'n_jobs': -1})
            return RandomForestClassifier(**{**base_params, **params})
        
        elif model_name == 'xgboost':
            base_params.update({'eval_metric': 'logloss', 'verbosity': 0})
            return xgb.XGBClassifier(**{**base_params, **params})
        
        elif model_name == 'lightgbm':
            base_params.update({'verbosity': -1})
            return lgb.LGBMClassifier(**{**base_params, **params})
        
        elif model_name == 'logistic_regression':
            base_params.update({'max_iter': params.get('max_iter', 1000)})
            return LogisticRegression(**{**base_params, **params})
        
        elif model_name == 'svm':
            base_params.update({'probability': True})
            return SVC(**{**base_params, **params})
        
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(**{**base_params, **params})
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _evaluate_model_cv(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model using cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )
        
        metric = self.config['optimization_metric']
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the optimization metric."""
        metric = self.config['optimization_metric']
        
        if metric == 'f1':
            return f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            return precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            return recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'roc_auc':
            return roc_auc_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _logging_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback function for logging trial results."""
        if trial.number % 10 == 0:
            logger.info(f"Trial {trial.number}: {trial.value:.4f}")
    
    def tune_multiple_models(self,
                           model_names: List[str],
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_val: Optional[pd.DataFrame] = None,
                           y_val: Optional[pd.Series] = None) -> Dict[str, Dict[str, Any]]:
        """
        Tune hyperparameters for multiple models.
        
        Args:
            model_names: List of model names to tune
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing tuning results for all models
        """
        logger.info(f"Starting hyperparameter tuning for {len(model_names)} models")
        
        all_results = {}
        
        for model_name in model_names:
            try:
                results = self.tune_model(model_name, X_train, y_train, X_val, y_val)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error tuning {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        logger.info("Completed hyperparameter tuning for all models")
        return all_results
    
    def get_best_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Get the best model with tuned hyperparameters.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (best_model, best_params)
        """
        if model_name not in self.best_params:
            raise ValueError(f"Model {model_name} has not been tuned yet")
        
        best_params = self.best_params[model_name]
        best_model = self._create_model_with_params(model_name, best_params)
        
        return best_model, best_params
    
    def compare_tuned_models(self) -> pd.DataFrame:
        """Compare tuning results across all models."""
        if not self.tuning_results:
            raise ValueError("No tuning results available")
        
        comparison_data = []
        
        for model_name, results in self.tuning_results.items():
            if 'error' in results:
                continue
            
            row = {
                'model': model_name,
                'best_score': results['best_value'],
                'n_trials': results['n_trials'],
                'tuning_time_minutes': results['tuning_time_seconds'] / 60,
                'cv_mean': results['cv_scores']['mean'],
                'cv_std': results['cv_scores']['std']
            }
            
            if results['val_score']:
                row['val_score'] = results['val_score']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('best_score', ascending=False)
        
        return comparison_df
    
    def plot_optimization_history(self, model_name: str) -> None:
        """Plot optimization history for a specific model."""
        if model_name not in self.studies:
            raise ValueError(f"No tuning results for model {model_name}")
        
        study = self.studies[model_name]
        
        # Use Optuna's built-in plotting
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Optimization history
            trials = study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            ax1.plot(values)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel(f'{self.config["optimization_metric"].upper()} Score')
            ax1.set_title(f'Optimization History - {model_name}')
            ax1.grid(True, alpha=0.3)
            
            # Best value history
            best_values = []
            best_so_far = float('-inf')
            for value in values:
                if value > best_so_far:
                    best_so_far = value
                best_values.append(best_so_far)
            
            ax2.plot(best_values, color='red')
            ax2.set_xlabel('Trial')
            ax2.set_ylabel(f'Best {self.config["optimization_metric"].upper()} Score')
            ax2.set_title(f'Best Score Evolution - {model_name}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def save_tuning_results(self, output_dir: str = "models/tuning/") -> None:
        """Save tuning results to disk."""
        if not self.tuning_results:
            raise ValueError("No tuning results to save")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for model_name, results in self.tuning_results.items():
            if 'error' in results:
                continue
            
            # Save best parameters
            params_file = output_path / f"{model_name}_best_params.json"
            with open(params_file, 'w') as f:
                json.dump(results['best_params'], f, indent=2)
            
            # Save best model
            best_model, _ = self.get_best_model(model_name)
            model_file = output_path / f"{model_name}_tuned_model.pkl"
            joblib.dump(best_model, model_file)
            
            logger.info(f"Saved tuned {model_name} results to {output_dir}")
        
        # Save comparison results
        comparison_df = self.compare_tuned_models()
        comparison_file = output_path / "tuning_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        logger.info(f"Saved tuning comparison to {comparison_file}")
    
    def load_tuning_results(self, input_dir: str = "models/tuning/") -> None:
        """Load tuning results from disk."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
        
        # Load best parameters
        for params_file in input_path.glob("*_best_params.json"):
            model_name = params_file.stem.replace('_best_params', '')
            
            with open(params_file, 'r') as f:
                best_params = json.load(f)
            
            self.best_params[model_name] = best_params
            logger.info(f"Loaded best parameters for {model_name}")

def main():
    """Main function for testing hyperparameter tuning."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    # Load processed data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    
    # Convert timestamp columns
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
    
    # Apply feature engineering
    from features.feature_engineering import FeatureEngineer
    
    fe = FeatureEngineer({
        'create_interactions': True,
        'create_ratios': True,
        'create_aggregations': False,
        'categorical_encoding': 'target',
        'scaling_method': 'robust'
    })
    
    # Transform datasets
    train_transformed = fe.fit_transform(train_df, 'is_fraud')
    val_transformed = fe.transform(val_df)
    
    # Feature selection
    train_selected, selected_features = fe.select_features(train_transformed, 'is_fraud', k=20)
    val_selected = val_transformed[['is_fraud'] + selected_features]
    
    # Prepare data
    X_train = train_selected.drop('is_fraud', axis=1)
    y_train = train_selected['is_fraud']
    X_val = val_selected.drop('is_fraud', axis=1)
    y_val = val_selected['is_fraud']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Initialize tuner
    tuner_config = {
        'n_trials': 20,  # Reduced for testing
        'timeout': 300,  # 5 minutes
        'optimization_metric': 'f1',
        'verbose': True
    }
    
    tuner = FraudHyperparameterTuner(tuner_config)
    
    # Tune a few models
    models_to_tune = ['random_forest', 'xgboost']
    tuning_results = tuner.tune_multiple_models(models_to_tune, X_train, y_train, X_val, y_val)
    
    # Compare results
    comparison_df = tuner.compare_tuned_models()
    print("\nTuning Results Comparison:")
    print(comparison_df.round(4))
    
    # Save results
    tuner.save_tuning_results()
    
    print("\nHyperparameter tuning completed successfully!")

if __name__ == "__main__":
    main() 