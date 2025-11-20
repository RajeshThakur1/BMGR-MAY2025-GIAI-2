"""
Model Monitoring System

This module provides comprehensive monitoring for fraud detection models
including data drift detection, performance tracking, alerting, and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from dataclasses import dataclass, asdict
import pickle
import threading
import time

# Statistical libraries
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Data drift detection
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not available. Data drift detection will be limited.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Data classes for monitoring
@dataclass
class PredictionLog:
    """Log entry for model predictions."""
    timestamp: str
    transaction_id: str
    model_name: str
    prediction: int
    probability: float
    features: Dict[str, Any]
    processing_time_ms: float
    actual_label: Optional[int] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    timestamp: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    sample_size: int
    period: str

@dataclass
class DriftAlert:
    """Data drift alert."""
    timestamp: str
    model_name: str
    drift_type: str  # 'feature_drift', 'target_drift', 'prediction_drift'
    severity: str    # 'low', 'medium', 'high'
    description: str
    metrics: Dict[str, float]

@dataclass
class ModelHealth:
    """Overall model health status."""
    model_name: str
    status: str  # 'healthy', 'warning', 'critical'
    last_updated: str
    performance_score: float
    drift_score: float
    prediction_volume: int
    error_rate: float

class FraudModelMonitor:
    """
    Comprehensive monitoring system for fraud detection models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the model monitor.
        
        Args:
            config: Configuration dictionary for monitoring
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'db_path': 'monitoring/fraud_monitoring.db',
            'drift_threshold': 0.1,
            'performance_threshold': 0.8,
            'alert_frequency_hours': 24,
            'monitoring_window_days': 7,
            'reference_window_days': 30,
            'min_samples_for_drift': 1000,
            'enable_real_time_monitoring': True,
            'enable_drift_detection': True,
            'enable_performance_tracking': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Initialize database
        self.db_path = Path(self.config['db_path'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Monitoring state
        self.reference_data = {}
        self.alerts = []
        self.model_health = {}
        
        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info("Fraud model monitor initialized")
    
    def _init_database(self):
        """Initialize SQLite database for monitoring data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    transaction_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    features TEXT NOT NULL,
                    processing_time_ms REAL NOT NULL,
                    actual_label INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    roc_auc REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    period TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Drift alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    performance_score REAL NOT NULL,
                    drift_score REAL NOT NULL,
                    prediction_volume INTEGER NOT NULL,
                    error_rate REAL NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def log_prediction(self, prediction_log: PredictionLog):
        """Log a model prediction."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, transaction_id, model_name, prediction, probability, 
                 features, processing_time_ms, actual_label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_log.timestamp,
                prediction_log.transaction_id,
                prediction_log.model_name,
                prediction_log.prediction,
                prediction_log.probability,
                json.dumps(prediction_log.features),
                prediction_log.processing_time_ms,
                prediction_log.actual_label
            ))
            conn.commit()
    
    def update_actual_label(self, transaction_id: str, actual_label: int):
        """Update the actual label for a prediction."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE predictions 
                SET actual_label = ? 
                WHERE transaction_id = ?
            ''', (actual_label, transaction_id))
            conn.commit()
    
    def calculate_performance_metrics(self, 
                                    model_name: str, 
                                    period_hours: int = 24) -> Optional[PerformanceMetrics]:
        """Calculate performance metrics for a given period."""
        
        # Get predictions with actual labels from the specified period
        start_time = (datetime.now() - timedelta(hours=period_hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT prediction, actual_label, probability
                FROM predictions 
                WHERE model_name = ? 
                AND timestamp >= ? 
                AND actual_label IS NOT NULL
            ''', conn, params=(model_name, start_time))
        
        if len(df) < 10:  # Minimum samples required
            logger.warning(f"Insufficient data for performance calculation: {len(df)} samples")
            return None
        
        y_true = df['actual_label'].values
        y_pred = df['prediction'].values
        y_prob = df['probability'].values
        
        # Calculate metrics
        try:
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                model_name=model_name,
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1_score=f1_score(y_true, y_pred, zero_division=0),
                roc_auc=roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
                sample_size=len(df),
                period=f"{period_hours}h"
            )
            
            # Store metrics in database
            self._store_performance_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return None
    
    def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics 
                (timestamp, model_name, accuracy, precision, recall, f1_score, 
                 roc_auc, sample_size, period)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp,
                metrics.model_name,
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.roc_auc,
                metrics.sample_size,
                metrics.period
            ))
            conn.commit()
    
    def detect_data_drift(self, 
                         model_name: str,
                         current_data: pd.DataFrame,
                         reference_data: Optional[pd.DataFrame] = None) -> List[DriftAlert]:
        """Detect data drift using statistical tests."""
        alerts = []
        
        if reference_data is None:
            reference_data = self._get_reference_data(model_name)
        
        if reference_data is None or len(reference_data) < self.config['min_samples_for_drift']:
            logger.warning(f"Insufficient reference data for drift detection: {model_name}")
            return alerts
        
        if len(current_data) < 100:  # Minimum current data samples
            logger.warning(f"Insufficient current data for drift detection: {len(current_data)}")
            return alerts
        
        # Feature drift detection
        numeric_features = current_data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if feature in reference_data.columns:
                # Kolmogorov-Smirnov test for distribution drift
                ks_statistic, p_value = stats.ks_2samp(
                    reference_data[feature].dropna(),
                    current_data[feature].dropna()
                )
                
                if p_value < 0.05:  # Significant drift detected
                    severity = "high" if p_value < 0.01 else "medium"
                    
                    alert = DriftAlert(
                        timestamp=datetime.now().isoformat(),
                        model_name=model_name,
                        drift_type="feature_drift",
                        severity=severity,
                        description=f"Significant drift detected in feature '{feature}'",
                        metrics={
                            "feature": feature,
                            "ks_statistic": ks_statistic,
                            "p_value": p_value,
                            "reference_mean": float(reference_data[feature].mean()),
                            "current_mean": float(current_data[feature].mean()),
                            "reference_std": float(reference_data[feature].std()),
                            "current_std": float(current_data[feature].std())
                        }
                    )
                    alerts.append(alert)
        
        # Store alerts in database
        for alert in alerts:
            self._store_drift_alert(alert)
        
        return alerts
    
    def _get_reference_data(self, model_name: str) -> Optional[pd.DataFrame]:
        """Get reference data for drift detection."""
        # Get reference data from the past N days
        reference_start = (datetime.now() - timedelta(days=self.config['reference_window_days'])).isoformat()
        reference_end = (datetime.now() - timedelta(days=self.config['monitoring_window_days'])).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT features
                FROM predictions 
                WHERE model_name = ? 
                AND timestamp BETWEEN ? AND ?
            ''', conn, params=(model_name, reference_start, reference_end))
        
        if len(df) == 0:
            return None
        
        # Parse features JSON
        features_list = []
        for features_json in df['features']:
            try:
                features = json.loads(features_json)
                features_list.append(features)
            except json.JSONDecodeError:
                continue
        
        if not features_list:
            return None
        
        return pd.DataFrame(features_list)
    
    def _store_drift_alert(self, alert: DriftAlert):
        """Store drift alert in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO drift_alerts 
                (timestamp, model_name, drift_type, severity, description, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp,
                alert.model_name,
                alert.drift_type,
                alert.severity,
                alert.description,
                json.dumps(alert.metrics)
            ))
            conn.commit()
    
    def get_model_health(self, model_name: str) -> ModelHealth:
        """Get overall model health status."""
        
        # Get recent performance metrics
        performance_metrics = self.calculate_performance_metrics(model_name, period_hours=24)
        performance_score = performance_metrics.f1_score if performance_metrics else 0.0
        
        # Get recent drift alerts
        with sqlite3.connect(self.db_path) as conn:
            drift_alerts = pd.read_sql_query('''
                SELECT severity, COUNT(*) as count
                FROM drift_alerts 
                WHERE model_name = ? 
                AND timestamp >= ?
                GROUP BY severity
            ''', conn, params=(model_name, (datetime.now() - timedelta(hours=24)).isoformat()))
        
        # Calculate drift score (lower is better)
        drift_score = 0.0
        if len(drift_alerts) > 0:
            severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.6}
            for _, row in drift_alerts.iterrows():
                drift_score += severity_weights.get(row['severity'], 0.1) * row['count']
        
        # Get prediction volume
        with sqlite3.connect(self.db_path) as conn:
            volume_df = pd.read_sql_query('''
                SELECT COUNT(*) as volume
                FROM predictions 
                WHERE model_name = ? 
                AND timestamp >= ?
            ''', conn, params=(model_name, (datetime.now() - timedelta(hours=24)).isoformat()))
        
        prediction_volume = volume_df['volume'].iloc[0] if len(volume_df) > 0 else 0
        
        # Calculate error rate (predictions without actual labels after 24h)
        with sqlite3.connect(self.db_path) as conn:
            error_df = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN actual_label IS NULL THEN 1 ELSE 0 END) as without_labels
                FROM predictions 
                WHERE model_name = ? 
                AND timestamp BETWEEN ? AND ?
            ''', conn, params=(
                model_name, 
                (datetime.now() - timedelta(hours=48)).isoformat(),
                (datetime.now() - timedelta(hours=24)).isoformat()
            ))
        
        error_rate = 0.0
        if len(error_df) > 0 and error_df['total'].iloc[0] > 0:
            error_rate = error_df['without_labels'].iloc[0] / error_df['total'].iloc[0]
        
        # Determine overall status
        status = "healthy"
        if performance_score < self.config['performance_threshold'] or drift_score > 1.0 or error_rate > 0.1:
            status = "warning"
        if performance_score < 0.6 or drift_score > 2.0 or error_rate > 0.3:
            status = "critical"
        
        health = ModelHealth(
            model_name=model_name,
            status=status,
            last_updated=datetime.now().isoformat(),
            performance_score=performance_score,
            drift_score=drift_score,
            prediction_volume=prediction_volume,
            error_rate=error_rate
        )
        
        # Update health in database
        self._update_model_health(health)
        
        return health
    
    def _update_model_health(self, health: ModelHealth):
        """Update model health in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO model_health 
                (model_name, status, last_updated, performance_score, 
                 drift_score, prediction_volume, error_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                health.model_name,
                health.status,
                health.last_updated,
                health.performance_score,
                health.drift_score,
                health.prediction_volume,
                health.error_rate
            ))
            conn.commit()
    
    def generate_monitoring_report(self, 
                                 model_name: str, 
                                 days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get performance trends
        with sqlite3.connect(self.db_path) as conn:
            performance_df = pd.read_sql_query('''
                SELECT * FROM performance_metrics 
                WHERE model_name = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', conn, params=(model_name, start_date))
        
        # Get drift alerts
        with sqlite3.connect(self.db_path) as conn:
            alerts_df = pd.read_sql_query('''
                SELECT * FROM drift_alerts 
                WHERE model_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', conn, params=(model_name, start_date))
        
        # Get prediction volume
        with sqlite3.connect(self.db_path) as conn:
            volume_df = pd.read_sql_query('''
                SELECT DATE(timestamp) as date, COUNT(*) as volume
                FROM predictions 
                WHERE model_name = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', conn, params=(model_name, start_date))
        
        # Calculate summary statistics
        summary = {
            "model_name": model_name,
            "report_period_days": days,
            "generated_at": datetime.now().isoformat(),
            "total_predictions": len(performance_df) * performance_df['sample_size'].mean() if len(performance_df) > 0 else 0,
            "avg_daily_volume": volume_df['volume'].mean() if len(volume_df) > 0 else 0,
            "performance_metrics": {
                "latest_f1_score": performance_df['f1_score'].iloc[-1] if len(performance_df) > 0 else None,
                "avg_f1_score": performance_df['f1_score'].mean() if len(performance_df) > 0 else None,
                "f1_score_trend": "stable"  # Could implement trend calculation
            },
            "drift_summary": {
                "total_alerts": len(alerts_df),
                "high_severity_alerts": len(alerts_df[alerts_df['severity'] == 'high']) if len(alerts_df) > 0 else 0,
                "recent_alerts": alerts_df.head(5).to_dict('records') if len(alerts_df) > 0 else []
            },
            "health_status": self.get_model_health(model_name)
        }
        
        return summary
    
    def start_background_monitoring(self, models: List[str], interval_minutes: int = 60):
        """Start background monitoring for specified models."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Background monitoring already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            args=(models, interval_minutes),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started background monitoring for {len(models)} models")
    
    def stop_background_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join()
            logger.info("Background monitoring stopped")
    
    def _background_monitoring_loop(self, models: List[str], interval_minutes: int):
        """Background monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                for model_name in models:
                    # Calculate performance metrics
                    self.calculate_performance_metrics(model_name)
                    
                    # Check model health
                    health = self.get_model_health(model_name)
                    
                    if health.status in ['warning', 'critical']:
                        logger.warning(f"Model {model_name} health status: {health.status}")
                    
                    # Generate alerts if needed
                    if health.drift_score > 2.0:
                        logger.error(f"High drift detected for model {model_name}")
                
                # Sleep for specified interval
                self.stop_monitoring.wait(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")
                self.stop_monitoring.wait(60)  # Wait 1 minute before retrying

def main():
    """Main function for testing monitoring system."""
    # Initialize monitor
    monitor = FraudModelMonitor({
        'db_path': 'test_monitoring.db',
        'drift_threshold': 0.1
    })
    
    # Simulate some prediction logs
    for i in range(100):
        prediction_log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            transaction_id=f"TXN_{i:04d}",
            model_name="test_model",
            prediction=np.random.binomial(1, 0.1),
            probability=np.random.beta(2, 8),
            features={
                "amount": np.random.lognormal(5, 1),
                "hour": np.random.randint(0, 24),
                "merchant_category": np.random.choice(['retail', 'grocery', 'gas'])
            },
            processing_time_ms=np.random.uniform(10, 100),
            actual_label=np.random.binomial(1, 0.1) if np.random.random() > 0.3 else None
        )
        monitor.log_prediction(prediction_log)
    
    # Calculate performance metrics
    metrics = monitor.calculate_performance_metrics("test_model")
    if metrics:
        print(f"Performance Metrics: F1={metrics.f1_score:.3f}, Precision={metrics.precision:.3f}")
    
    # Get model health
    health = monitor.get_model_health("test_model")
    print(f"Model Health: {health.status} (Performance: {health.performance_score:.3f})")
    
    # Generate report
    report = monitor.generate_monitoring_report("test_model")
    print(f"Monitoring Report: {report['total_predictions']} predictions")
    
    print("Monitoring system test completed!")

if __name__ == "__main__":
    main() 