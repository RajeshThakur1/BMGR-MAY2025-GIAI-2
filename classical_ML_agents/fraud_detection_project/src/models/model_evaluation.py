"""
Model Evaluation Module

This module provides comprehensive evaluation utilities for fraud detection models
including metrics calculation, visualizations, and detailed reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation for fraud detection.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        
    def evaluate_single_model(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None,
                             model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Basic classification metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                metrics['average_precision'] = average_precision_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Derived metrics from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = class_report
        
        # Business metrics for fraud detection
        total_transactions = len(y_true)
        total_fraud = np.sum(y_true)
        total_predicted_fraud = np.sum(y_pred)
        
        metrics['total_transactions'] = total_transactions
        metrics['total_actual_fraud'] = int(total_fraud)
        metrics['total_predicted_fraud'] = int(total_predicted_fraud)
        metrics['fraud_rate'] = total_fraud / total_transactions
        metrics['predicted_fraud_rate'] = total_predicted_fraud / total_transactions
        
        # Cost-based metrics (assuming costs)
        cost_per_fp = 10  # Cost of false positive (manual review)
        cost_per_fn = 100  # Cost of false negative (missed fraud)
        
        if cm.shape == (2, 2):
            total_cost = (fp * cost_per_fp) + (fn * cost_per_fn)
            metrics['total_cost'] = total_cost
            metrics['cost_per_transaction'] = total_cost / total_transactions
        
        return metrics
    
    def compare_models(self, 
                      models_results: Dict[str, Dict[str, Any]],
                      sort_by: str = 'f1_score') -> pd.DataFrame:
        """
        Compare multiple models and return comparison dataframe.
        
        Args:
            models_results: Dictionary of model evaluation results
            sort_by: Metric to sort models by
            
        Returns:
            DataFrame with model comparison
        """
        logger.info(f"Comparing {len(models_results)} models")
        
        comparison_data = []
        
        for model_name, results in models_results.items():
            if 'error' in results:
                continue
                
            row = {'model': model_name}
            
            # Key metrics for comparison
            key_metrics = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'roc_auc', 'average_precision', 'matthews_corrcoef',
                'specificity', 'false_positive_rate', 'total_cost'
            ]
            
            for metric in key_metrics:
                if metric in results:
                    row[metric] = results[metric]
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric (descending for most metrics, ascending for cost)
        ascending = sort_by in ['total_cost', 'false_positive_rate', 'false_negative_rate']
        if sort_by in comparison_df.columns:
            comparison_df = comparison_df.sort_values(sort_by, ascending=ascending)
        
        return comparison_df
    
    def plot_confusion_matrices(self, 
                               models_results: Dict[str, Dict[str, Any]],
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot confusion matrices for all models."""
        
        valid_models = {k: v for k, v in models_results.items() if 'confusion_matrix' in v}
        n_models = len(valid_models)
        
        if n_models == 0:
            logger.warning("No valid confusion matrices found")
            return
        
        # Calculate subplot layout
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(valid_models.items()):
            cm = np.array(results['confusion_matrix'])
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Fraud'],
                       yticklabels=['Normal', 'Fraud'],
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nF1: {results.get("f1_score", 0):.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, 
                       y_true: np.ndarray,
                       models_proba: Dict[str, np.ndarray],
                       figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot ROC curves for multiple models."""
        
        plt.figure(figsize=figsize)
        
        for model_name, y_proba in models_proba.items():
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = roc_auc_score(y_true, y_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
            except ValueError:
                logger.warning(f"Could not plot ROC curve for {model_name}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self,
                                    y_true: np.ndarray,
                                    models_proba: Dict[str, np.ndarray],
                                    figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot Precision-Recall curves for multiple models."""
        
        plt.figure(figsize=figsize)
        
        baseline_precision = np.mean(y_true)
        
        for model_name, y_proba in models_proba.items():
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                avg_precision = average_precision_score(y_true, y_proba)
                plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
            except ValueError:
                logger.warning(f"Could not plot PR curve for {model_name}")
        
        plt.axhline(y=baseline_precision, color='k', linestyle='--', 
                   label=f'Baseline ({baseline_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_calibration_curves(self,
                               y_true: np.ndarray,
                               models_proba: Dict[str, np.ndarray],
                               n_bins: int = 10,
                               figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot calibration curves for multiple models."""
        
        plt.figure(figsize=figsize)
        
        for model_name, y_proba in models_proba.items():
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_proba, n_bins=n_bins
                )
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        marker='o', label=model_name)
            except ValueError:
                logger.warning(f"Could not plot calibration curve for {model_name}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance_comparison(self,
                                         models_importance: Dict[str, Dict[str, float]],
                                         top_n: int = 15,
                                         figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot feature importance comparison across models."""
        
        if not models_importance:
            logger.warning("No feature importance data provided")
            return
        
        # Get all unique features
        all_features = set()
        for importance_dict in models_importance.values():
            if importance_dict:
                all_features.update(importance_dict.keys())
        
        if not all_features:
            logger.warning("No feature importance found")
            return
        
        # Create comparison dataframe
        importance_data = []
        for model_name, importance_dict in models_importance.items():
            if importance_dict:
                for feature, importance in importance_dict.items():
                    importance_data.append({
                        'model': model_name,
                        'feature': feature,
                        'importance': importance
                    })
        
        if not importance_data:
            return
        
        importance_df = pd.DataFrame(importance_data)
        
        # Get top features across all models
        avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        top_features = avg_importance.head(top_n).index.tolist()
        
        # Filter to top features
        filtered_df = importance_df[importance_df['feature'].isin(top_features)]
        
        # Create pivot for heatmap
        pivot_df = filtered_df.pivot(index='feature', columns='model', values='importance').fillna(0)
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Models')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    
    def generate_model_report(self,
                             model_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> str:
        """Generate a comprehensive model evaluation report."""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("FRAUD DETECTION MODEL EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model Overview
        report_lines.append("MODEL OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"Model Name: {model_results.get('model_name', 'Unknown')}")
        report_lines.append(f"Total Transactions: {model_results.get('total_transactions', 0):,}")
        report_lines.append(f"Actual Fraud Cases: {model_results.get('total_actual_fraud', 0):,}")
        report_lines.append(f"Predicted Fraud Cases: {model_results.get('total_predicted_fraud', 0):,}")
        report_lines.append(f"Actual Fraud Rate: {model_results.get('fraud_rate', 0):.2%}")
        report_lines.append(f"Predicted Fraud Rate: {model_results.get('predicted_fraud_rate', 0):.2%}")
        report_lines.append("")
        
        # Performance Metrics
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append("-" * 40)
        metrics_to_report = [
            ('Accuracy', 'accuracy'),
            ('Precision', 'precision'),
            ('Recall (Sensitivity)', 'recall'),
            ('F1-Score', 'f1_score'),
            ('Specificity', 'specificity'),
            ('ROC AUC', 'roc_auc'),
            ('Average Precision', 'average_precision'),
            ('Matthews Correlation Coefficient', 'matthews_corrcoef'),
            ('Cohen\'s Kappa', 'cohen_kappa')
        ]
        
        for metric_name, metric_key in metrics_to_report:
            value = model_results.get(metric_key, 0)
            report_lines.append(f"{metric_name:<30}: {value:.4f}")
        
        report_lines.append("")
        
        # Confusion Matrix
        if 'confusion_matrix' in model_results:
            cm = np.array(model_results['confusion_matrix'])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                report_lines.append("CONFUSION MATRIX")
                report_lines.append("-" * 40)
                report_lines.append("                 Predicted")
                report_lines.append("                Normal  Fraud")
                report_lines.append(f"Actual  Normal   {tn:6d}  {fp:5d}")
                report_lines.append(f"        Fraud    {fn:6d}  {tp:5d}")
                report_lines.append("")
                
                # Detailed breakdown
                report_lines.append("DETAILED BREAKDOWN")
                report_lines.append("-" * 40)
                report_lines.append(f"True Positives (Correctly identified fraud): {tp:,}")
                report_lines.append(f"True Negatives (Correctly identified normal): {tn:,}")
                report_lines.append(f"False Positives (Normal flagged as fraud): {fp:,}")
                report_lines.append(f"False Negatives (Fraud missed): {fn:,}")
                report_lines.append("")
        
        # Business Impact
        if 'total_cost' in model_results:
            report_lines.append("BUSINESS IMPACT")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Cost: ${model_results.get('total_cost', 0):,.2f}")
            report_lines.append(f"Cost per Transaction: ${model_results.get('cost_per_transaction', 0):.4f}")
            report_lines.append("")
        
        # Classification Report
        if 'classification_report' in model_results:
            class_report = model_results['classification_report']
            report_lines.append("DETAILED CLASSIFICATION REPORT")
            report_lines.append("-" * 40)
            report_lines.append(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            report_lines.append("-" * 50)
            
            for class_name, metrics in class_report.items():
                if class_name in ['0', '1']:  # Only show class metrics
                    class_label = 'Normal' if class_name == '0' else 'Fraud'
                    precision = metrics['precision']
                    recall = metrics['recall']
                    f1 = metrics['f1-score']
                    support = metrics['support']
                    report_lines.append(f"{class_label:<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10.0f}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def interactive_dashboard(self,
                            models_results: Dict[str, Dict[str, Any]],
                            y_true: np.ndarray,
                            models_proba: Dict[str, np.ndarray]) -> None:
        """Create an interactive dashboard using Plotly."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Comparison', 'ROC Curves', 'Precision-Recall Curves', 'Calibration'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Model comparison bar chart
        comparison_df = self.compare_models(models_results)
        if not comparison_df.empty:
            fig.add_trace(
                go.Bar(x=comparison_df['model'], y=comparison_df['f1_score'], name='F1-Score'),
                row=1, col=1
            )
        
        # ROC curves
        for model_name, y_proba in models_proba.items():
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = roc_auc_score(y_true, y_proba)
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f'{model_name} (AUC={auc_score:.3f})', mode='lines'),
                    row=1, col=2
                )
            except ValueError:
                continue
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', 
                      line=dict(dash='dash', color='black')),
            row=1, col=2
        )
        
        # Precision-Recall curves
        for model_name, y_proba in models_proba.items():
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                avg_precision = average_precision_score(y_true, y_proba)
                fig.add_trace(
                    go.Scatter(x=recall, y=precision, name=f'{model_name} (AP={avg_precision:.3f})', mode='lines'),
                    row=2, col=1
                )
            except ValueError:
                continue
        
        # Calibration curves
        for model_name, y_proba in models_proba.items():
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
                fig.add_trace(
                    go.Scatter(x=mean_predicted_value, y=fraction_of_positives, 
                             name=model_name, mode='lines+markers'),
                    row=2, col=2
                )
            except ValueError:
                continue
        
        # Add perfect calibration line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name='Perfect Calibration', mode='lines',
                      line=dict(dash='dash', color='black')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True, title_text="Fraud Detection Model Dashboard")
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="F1-Score", row=1, col=1)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        fig.update_xaxes(title_text="Recall", row=2, col=1)
        fig.update_yaxes(title_text="Precision", row=2, col=1)
        fig.update_xaxes(title_text="Mean Predicted Probability", row=2, col=2)
        fig.update_yaxes(title_text="Fraction of Positives", row=2, col=2)
        
        fig.show()

def main():
    """Main function for testing model evaluation."""
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic test data
    y_true = np.random.binomial(1, 0.1, n_samples)  # 10% fraud rate
    
    # Simulate model predictions
    models_proba = {
        'Model_A': np.random.beta(2, 8, n_samples),  # Conservative model
        'Model_B': np.random.beta(1, 3, n_samples),  # Aggressive model
        'Model_C': np.random.uniform(0, 1, n_samples)  # Random model
    }
    
    models_pred = {name: (proba > 0.5).astype(int) for name, proba in models_proba.items()}
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    models_results = {}
    for model_name in models_proba.keys():
        results = evaluator.evaluate_single_model(
            y_true, models_pred[model_name], models_proba[model_name], model_name
        )
        models_results[model_name] = results
    
    # Compare models
    comparison_df = evaluator.compare_models(models_results)
    print("Model Comparison:")
    print(comparison_df[['model', 'f1_score', 'precision', 'recall', 'roc_auc']].round(4))
    
    # Generate plots
    evaluator.plot_confusion_matrices(models_results)
    evaluator.plot_roc_curves(y_true, models_proba)
    evaluator.plot_precision_recall_curves(y_true, models_proba)
    
    # Generate report for best model
    best_model = comparison_df.iloc[0]['model']
    report = evaluator.generate_model_report(models_results[best_model])
    print("\nModel Report:")
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    print("\nModel evaluation testing completed!")

if __name__ == "__main__":
    main() 