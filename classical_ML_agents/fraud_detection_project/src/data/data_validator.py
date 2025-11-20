"""
Data Validation Module

This module provides comprehensive data validation utilities
for fraud detection datasets including schema validation,
data quality checks, and anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validator for fraud detection datasets.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_dataset(self, df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive validation on the dataset.
        
        Args:
            df: Dataset to validate
            verbose: Whether to print detailed results
            
        Returns:
            Dict: Comprehensive validation results
        """
        logger.info("Starting comprehensive data validation")
        
        # Reset validation state
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._validate_schema(df)
        self._validate_data_types(df)
        self._validate_data_quality(df)
        self._validate_business_rules(df)
        self._validate_statistical_properties(df)
        self._detect_anomalies(df)
        
        # Compile results
        results = {
            'overall_status': 'PASS' if len(self.errors) == 0 else 'FAIL',
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'detailed_results': self.validation_results,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            self._print_validation_summary(results)
        
        logger.info(f"Validation complete: {results['overall_status']}")
        return results
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate dataset schema and structure."""
        logger.info("Validating dataset schema")
        
        # Required columns for fraud detection
        required_columns = [
            'transaction_id', 'customer_id', 'amount', 'timestamp',
            'merchant_id', 'merchant_category', 'is_fraud'
        ]
        
        # Check for missing required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate column names
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            self.errors.append(f"Duplicate column names: {duplicate_columns}")
        
        # Check for empty column names
        empty_columns = [col for col in df.columns if col == '' or pd.isna(col)]
        if empty_columns:
            self.errors.append(f"Empty or null column names found: {len(empty_columns)}")
        
        self.validation_results['schema'] = {
            'total_columns': len(df.columns),
            'missing_required': missing_columns,
            'duplicate_columns': duplicate_columns,
            'empty_columns': len(empty_columns)
        }
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types for key columns."""
        logger.info("Validating data types")
        
        type_issues = []
        
        # Check transaction_id is string/object
        if 'transaction_id' in df.columns:
            if not pd.api.types.is_object_dtype(df['transaction_id']):
                type_issues.append("transaction_id should be string/object type")
        
        # Check amount is numeric
        if 'amount' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['amount']):
                type_issues.append("amount should be numeric type")
        
        # Check timestamp is datetime
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                type_issues.append("timestamp should be datetime type")
        
        # Check is_fraud is integer or boolean
        if 'is_fraud' in df.columns:
            if df['is_fraud'].dtype not in ['int64', 'int32', 'bool']:
                type_issues.append("is_fraud should be integer or boolean type")
        
        # Check customer_age is numeric if present
        if 'customer_age' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['customer_age']):
                type_issues.append("customer_age should be numeric type")
        
        if type_issues:
            self.errors.extend(type_issues)
        
        self.validation_results['data_types'] = {
            'type_issues': type_issues,
            'dtypes': df.dtypes.to_dict()
        }
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """Validate data quality issues."""
        logger.info("Validating data quality")
        
        quality_issues = []
        
        # Check for missing values
        missing_stats = df.isnull().sum()
        high_missing_cols = missing_stats[missing_stats > len(df) * 0.05].index.tolist()
        if high_missing_cols:
            self.warnings.append(f"Columns with >5% missing values: {high_missing_cols}")
        
        # Check for duplicate transaction IDs
        if 'transaction_id' in df.columns:
            duplicate_ids = df['transaction_id'].duplicated().sum()
            if duplicate_ids > 0:
                quality_issues.append(f"Found {duplicate_ids} duplicate transaction IDs")
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            quality_issues.append(f"Found {empty_rows} completely empty rows")
        
        # Check for constant columns (no variation)
        constant_cols = []
        for col in df.select_dtypes(include=[np.number, 'object']).columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        if constant_cols:
            self.warnings.append(f"Constant columns (no variation): {constant_cols}")
        
        if quality_issues:
            self.errors.extend(quality_issues)
        
        self.validation_results['data_quality'] = {
            'total_rows': len(df),
            'missing_value_stats': missing_stats.to_dict(),
            'high_missing_columns': high_missing_cols,
            'duplicate_transaction_ids': duplicate_ids if 'transaction_id' in df.columns else 0,
            'empty_rows': empty_rows,
            'constant_columns': constant_cols
        }
    
    def _validate_business_rules(self, df: pd.DataFrame) -> None:
        """Validate business logic rules."""
        logger.info("Validating business rules")
        
        business_issues = []
        
        # Check transaction amounts
        if 'amount' in df.columns:
            negative_amounts = (df['amount'] < 0).sum()
            zero_amounts = (df['amount'] == 0).sum()
            extreme_amounts = (df['amount'] > 1000000).sum()  # > $1M
            
            if negative_amounts > 0:
                business_issues.append(f"Found {negative_amounts} negative transaction amounts")
            if zero_amounts > 0:
                self.warnings.append(f"Found {zero_amounts} zero transaction amounts")
            if extreme_amounts > 0:
                self.warnings.append(f"Found {extreme_amounts} extremely high amounts (>$1M)")
        
        # Check fraud labels
        if 'is_fraud' in df.columns:
            invalid_fraud_labels = (~df['is_fraud'].isin([0, 1])).sum()
            if invalid_fraud_labels > 0:
                business_issues.append(f"Found {invalid_fraud_labels} invalid fraud labels (not 0 or 1)")
            
            fraud_rate = df['is_fraud'].mean()
            if fraud_rate > 0.5:
                self.warnings.append(f"Unusually high fraud rate: {fraud_rate:.1%}")
            elif fraud_rate < 0.001:
                self.warnings.append(f"Unusually low fraud rate: {fraud_rate:.1%}")
        
        # Check customer age
        if 'customer_age' in df.columns:
            invalid_ages = ((df['customer_age'] < 0) | (df['customer_age'] > 150)).sum()
            if invalid_ages > 0:
                business_issues.append(f"Found {invalid_ages} invalid customer ages (<0 or >150)")
        
        # Check timestamps
        if 'timestamp' in df.columns:
            future_dates = (df['timestamp'] > datetime.now()).sum()
            old_dates = (df['timestamp'] < datetime(2000, 1, 1)).sum()
            
            if future_dates > 0:
                self.warnings.append(f"Found {future_dates} future transaction dates")
            if old_dates > 0:
                self.warnings.append(f"Found {old_dates} very old transaction dates (before 2000)")
        
        if business_issues:
            self.errors.extend(business_issues)
        
        self.validation_results['business_rules'] = {
            'amount_issues': {
                'negative': negative_amounts if 'amount' in df.columns else 0,
                'zero': zero_amounts if 'amount' in df.columns else 0,
                'extreme': extreme_amounts if 'amount' in df.columns else 0
            },
            'fraud_label_issues': invalid_fraud_labels if 'is_fraud' in df.columns else 0,
            'fraud_rate': fraud_rate if 'is_fraud' in df.columns else None,
            'age_issues': invalid_ages if 'customer_age' in df.columns else 0,
            'timestamp_issues': {
                'future_dates': future_dates if 'timestamp' in df.columns else 0,
                'old_dates': old_dates if 'timestamp' in df.columns else 0
            }
        }
    
    def _validate_statistical_properties(self, df: pd.DataFrame) -> None:
        """Validate statistical properties of the data."""
        logger.info("Validating statistical properties")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stats_results = {}
        
        for col in numeric_columns:
            if col in df.columns:
                col_stats = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'skewness': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis()),
                    'outlier_count': self._count_outliers(df[col])
                }
                
                # Check for extreme skewness
                if abs(col_stats['skewness']) > 3:
                    self.warnings.append(f"Column {col} has extreme skewness: {col_stats['skewness']:.2f}")
                
                # Check for extreme kurtosis
                if abs(col_stats['kurtosis']) > 10:
                    self.warnings.append(f"Column {col} has extreme kurtosis: {col_stats['kurtosis']:.2f}")
                
                stats_results[col] = col_stats
        
        self.validation_results['statistical_properties'] = stats_results
    
    def _detect_anomalies(self, df: pd.DataFrame) -> None:
        """Detect potential anomalies in the data."""
        logger.info("Detecting anomalies")
        
        anomalies = {}
        
        # Check for potential data entry errors
        if 'amount' in df.columns:
            # Detect rounded amounts (potential synthetic data)
            rounded_amounts = (df['amount'] % 100 == 0).sum()
            rounded_pct = rounded_amounts / len(df) * 100
            if rounded_pct > 50:
                self.warnings.append(f"High percentage of rounded amounts ({rounded_pct:.1f}%) - possible synthetic data")
            
            anomalies['rounded_amounts_pct'] = rounded_pct
        
        # Check for suspicious patterns in categorical data
        if 'merchant_category' in df.columns:
            category_dist = df['merchant_category'].value_counts(normalize=True)
            if category_dist.iloc[0] > 0.8:  # One category dominates
                self.warnings.append(f"One merchant category dominates: {category_dist.index[0]} ({category_dist.iloc[0]:.1%})")
        
        # Check for suspicious time patterns
        if 'timestamp' in df.columns:
            # Check if all transactions are on the same day
            unique_dates = df['timestamp'].dt.date.nunique()
            if unique_dates == 1:
                self.warnings.append("All transactions are on the same date - unusual pattern")
            
            # Check for regular intervals (potential synthetic data)
            time_diffs = df['timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                common_interval = time_diffs.mode()
                if len(common_interval) > 0 and (time_diffs == common_interval.iloc[0]).sum() > len(time_diffs) * 0.5:
                    self.warnings.append("Suspicious regular time intervals detected - possible synthetic data")
        
        self.validation_results['anomalies'] = anomalies
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def _print_validation_summary(self, results: Dict[str, Any]) -> None:
        """Print validation summary to console."""
        print("\n" + "="*60)
        print("DATA VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall Status: {results['overall_status']}")
        print(f"Errors: {results['error_count']}")
        print(f"Warnings: {results['warning_count']}")
        
        if results['errors']:
            print(f"\n❌ ERRORS ({len(results['errors'])}):")
            for i, error in enumerate(results['errors'], 1):
                print(f"   {i}. {error}")
        
        if results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
            for i, warning in enumerate(results['warnings'], 1):
                print(f"   {i}. {warning}")
        
        print(f"\nValidation completed at: {results['validation_timestamp']}")
        print("="*60)

class DataQualityMonitor:
    """
    Monitor data quality over time and detect drift.
    """
    
    def __init__(self):
        """Initialize the data quality monitor."""
        self.baseline_stats = None
        
    def set_baseline(self, df: pd.DataFrame) -> None:
        """Set baseline statistics for comparison."""
        logger.info("Setting baseline statistics for data quality monitoring")
        
        self.baseline_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.baseline_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        
        # Add categorical column stats
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.baseline_stats[col] = {
                'value_counts': df[col].value_counts().to_dict(),
                'nunique': df[col].nunique()
            }
    
    def detect_drift(self, df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect data drift compared to baseline.
        
        Args:
            df: New dataset to compare
            threshold: Threshold for detecting significant drift
            
        Returns:
            Dict: Drift detection results
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline statistics not set. Call set_baseline() first.")
        
        logger.info("Detecting data drift")
        drift_results = {'drifted_columns': [], 'drift_details': {}}
        
        # Check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.baseline_stats:
                current_mean = df[col].mean()
                baseline_mean = self.baseline_stats[col]['mean']
                baseline_std = self.baseline_stats[col]['std']
                
                if baseline_std > 0:
                    drift_score = abs(current_mean - baseline_mean) / baseline_std
                    if drift_score > threshold:
                        drift_results['drifted_columns'].append(col)
                        drift_results['drift_details'][col] = {
                            'type': 'numeric',
                            'drift_score': drift_score,
                            'baseline_mean': baseline_mean,
                            'current_mean': current_mean
                        }
        
        # Check categorical columns (simplified)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.baseline_stats:
                current_top_values = set(df[col].value_counts().head(5).index)
                baseline_top_values = set(list(self.baseline_stats[col]['value_counts'].keys())[:5])
                
                overlap = len(current_top_values.intersection(baseline_top_values)) / 5
                if overlap < (1 - threshold):
                    drift_results['drifted_columns'].append(col)
                    drift_results['drift_details'][col] = {
                        'type': 'categorical',
                        'overlap_score': overlap,
                        'baseline_top_values': list(baseline_top_values),
                        'current_top_values': list(current_top_values)
                    }
        
        return drift_results

def main():
    """Main function for testing the data validator."""
    # Create validator
    validator = DataValidator()
    
    # Load test data
    df = pd.read_csv("data/synthetic/fraud_transactions.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Run validation
    results = validator.validate_dataset(df, verbose=True)
    
    # Create quality monitor
    monitor = DataQualityMonitor()
    monitor.set_baseline(df)
    
    print(f"\nValidation complete with {results['error_count']} errors and {results['warning_count']} warnings")

if __name__ == "__main__":
    main() 