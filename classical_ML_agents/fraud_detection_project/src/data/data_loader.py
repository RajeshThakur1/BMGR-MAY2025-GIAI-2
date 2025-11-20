"""
Data Loading and Preprocessing Module

This module provides utilities for loading, validating, and preprocessing
fraud detection data from various sources.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for fraud detection datasets with validation and preprocessing.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data loader with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Expected schema for fraud detection data
        self.expected_schema = {
            'transaction_id': 'object',
            'customer_id': 'object', 
            'amount': 'float64',
            'timestamp': 'datetime64[ns]',
            'merchant_id': 'object',
            'merchant_name': 'object',
            'merchant_category': 'object',
            'merchant_state': 'object',
            'is_fraud': 'int64',
            'customer_age': 'int64',
            'customer_income': 'float64'
        }
        
        # Required columns that must be present
        self.required_columns = [
            'transaction_id', 'customer_id', 'amount', 'timestamp',
            'merchant_id', 'merchant_category', 'is_fraud'
        ]
        
    def load_data(self, 
                  file_path: Union[str, Path], 
                  validate: bool = True,
                  sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from various sources with optional validation.
        
        Args:
            file_path: Path to the data file
            validate: Whether to validate the data schema
            sample_size: If provided, return a random sample of this size
            
        Returns:
            pd.DataFrame: Loaded and optionally validated dataset
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Determine file type and load accordingly
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = self._load_csv(file_path)
            elif file_path.suffix.lower() == '.parquet':
                df = self._load_parquet(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = self._load_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Successfully loaded {len(df):,} records")
            
            # Sample data if requested
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled {sample_size:,} records")
            
            # Validate data if requested
            if validate:
                df = self.validate_data(df)
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(file_path, parse_dates=['timestamp'])
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load data from Parquet file."""
        return pd.read_parquet(file_path)
    
    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        """Load data from Excel file."""
        return pd.read_excel(file_path, parse_dates=['timestamp'])
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the dataset schema and data quality.
        
        Args:
            df: Input dataframe to validate
            
        Returns:
            pd.DataFrame: Validated dataframe
        """
        logger.info("Validating data schema and quality")
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data types for key columns
        df = self._validate_data_types(df)
        
        # Check for critical data quality issues
        self._check_data_quality(df)
        
        # Clean and preprocess
        df = self._clean_data(df)
        
        logger.info("Data validation completed successfully")
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data types."""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure amount is numeric
        if not pd.api.types.is_numeric_dtype(df['amount']):
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Ensure fraud flag is binary
        if df['is_fraud'].dtype not in ['int64', 'bool']:
            df['is_fraud'] = df['is_fraud'].astype(int)
        
        # Ensure customer_age is integer
        if 'customer_age' in df.columns:
            df['customer_age'] = pd.to_numeric(df['customer_age'], errors='coerce').astype('Int64')
        
        return df
    
    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """Check for data quality issues."""
        issues = []
        
        # Check for missing values in critical columns
        for col in self.required_columns:
            if col in df.columns:
                missing_pct = df[col].isna().mean() * 100
                if missing_pct > 5:  # More than 5% missing
                    issues.append(f"High missing values in {col}: {missing_pct:.1f}%")
        
        # Check for duplicate transaction IDs
        if df['transaction_id'].duplicated().any():
            duplicate_count = df['transaction_id'].duplicated().sum()
            issues.append(f"Found {duplicate_count} duplicate transaction IDs")
        
        # Check for unrealistic amounts
        if (df['amount'] <= 0).any():
            negative_count = (df['amount'] <= 0).sum()
            issues.append(f"Found {negative_count} non-positive transaction amounts")
        
        # Check fraud rate
        fraud_rate = df['is_fraud'].mean()
        if fraud_rate > 0.1:  # More than 10% fraud is suspicious
            issues.append(f"Unusually high fraud rate: {fraud_rate:.1%}")
        elif fraud_rate < 0.001:  # Less than 0.1% might indicate labeling issues
            issues.append(f"Unusually low fraud rate: {fraud_rate:.1%}")
        
        # Check date range
        date_range_days = (df['timestamp'].max() - df['timestamp'].min()).days
        if date_range_days > 2000:  # More than ~5 years
            issues.append(f"Very large date range: {date_range_days} days")
        
        # Log issues
        if issues:
            logger.warning("Data quality issues detected:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No major data quality issues detected")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        df = df.copy()
        
        # Remove duplicate transaction IDs (keep first occurrence)
        if df['transaction_id'].duplicated().any():
            logger.info("Removing duplicate transaction IDs")
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
        
        # Handle negative amounts (set to absolute value)
        if (df['amount'] < 0).any():
            logger.info("Converting negative amounts to positive")
            df['amount'] = df['amount'].abs()
        
        # Remove records with zero amounts
        if (df['amount'] == 0).any():
            zero_count = (df['amount'] == 0).sum()
            logger.info(f"Removing {zero_count} records with zero amounts")
            df = df[df['amount'] > 0]
        
        # Cap extremely high amounts (potential data entry errors)
        amount_99_percentile = df['amount'].quantile(0.99)
        if (df['amount'] > amount_99_percentile * 10).any():
            extreme_count = (df['amount'] > amount_99_percentile * 10).sum()
            logger.info(f"Capping {extreme_count} extremely high amounts")
            df.loc[df['amount'] > amount_99_percentile * 10, 'amount'] = amount_99_percentile * 10
        
        # Sort by timestamp for time-series analysis
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive data summary.
        
        Args:
            df: Dataset to summarize
            
        Returns:
            Dict: Data summary statistics
        """
        summary = {
            'basic_stats': {
                'total_records': len(df),
                'total_customers': df['customer_id'].nunique(),
                'total_merchants': df['merchant_id'].nunique() if 'merchant_id' in df.columns else None,
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat(),
                    'days': (df['timestamp'].max() - df['timestamp'].min()).days
                }
            },
            'fraud_stats': {
                'fraud_count': int(df['is_fraud'].sum()),
                'fraud_rate': float(df['is_fraud'].mean()),
                'fraud_amount_total': float(df[df['is_fraud'] == 1]['amount'].sum()),
                'normal_amount_total': float(df[df['is_fraud'] == 0]['amount'].sum())
            },
            'amount_stats': {
                'mean': float(df['amount'].mean()),
                'median': float(df['amount'].median()),
                'std': float(df['amount'].std()),
                'min': float(df['amount'].min()),
                'max': float(df['amount'].max()),
                'q25': float(df['amount'].quantile(0.25)),
                'q75': float(df['amount'].quantile(0.75))
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_transactions': int(df['transaction_id'].duplicated().sum()),
                'zero_amounts': int((df['amount'] == 0).sum()),
                'negative_amounts': int((df['amount'] < 0).sum())
            }
        }
        
        # Add merchant category stats if available
        if 'merchant_category' in df.columns:
            summary['merchant_stats'] = {
                'categories': df['merchant_category'].value_counts().to_dict(),
                'fraud_by_category': df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean']).to_dict()
            }
        
        return summary
    
    def train_test_split(self, 
                        df: pd.DataFrame, 
                        test_size: float = 0.2,
                        validation_size: float = 0.15,
                        stratify: bool = True,
                        time_based: bool = False,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input dataframe
            test_size: Proportion for test set
            validation_size: Proportion for validation set (from remaining data)
            stratify: Whether to stratify by fraud label
            time_based: Whether to use time-based split
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data: test={test_size:.1%}, val={validation_size:.1%}")
        
        if time_based:
            # Time-based split - most recent data for test
            df_sorted = df.sort_values('timestamp')
            n_total = len(df_sorted)
            n_test = int(n_total * test_size)
            n_val = int((n_total - n_test) * validation_size)
            
            test_df = df_sorted.tail(n_test).copy()
            val_df = df_sorted.iloc[-(n_test + n_val):-n_test].copy()
            train_df = df_sorted.iloc[:-(n_test + n_val)].copy()
            
        else:
            # Random split with optional stratification
            from sklearn.model_selection import train_test_split
            
            # First split: separate test set
            if stratify:
                train_val_df, test_df = train_test_split(
                    df, test_size=test_size, 
                    stratify=df['is_fraud'], 
                    random_state=random_state
                )
                
                # Second split: separate validation from training
                train_df, val_df = train_test_split(
                    train_val_df, 
                    test_size=validation_size/(1-test_size),
                    stratify=train_val_df['is_fraud'],
                    random_state=random_state
                )
            else:
                train_val_df, test_df = train_test_split(
                    df, test_size=test_size, random_state=random_state
                )
                train_df, val_df = train_test_split(
                    train_val_df, 
                    test_size=validation_size/(1-test_size),
                    random_state=random_state
                )
        
        logger.info(f"Split complete - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        
        # Log fraud distribution in each set
        for name, data in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            fraud_rate = data['is_fraud'].mean()
            logger.info(f"{name} fraud rate: {fraud_rate:.2%}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, 
                           train_df: pd.DataFrame,
                           val_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           output_dir: str = "data/processed/") -> None:
        """Save processed datasets to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "validation.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        # Save data summary
        combined_df = pd.concat([train_df, val_df, test_df])
        summary = self.get_data_summary(combined_df)
        
        with open(output_path / "data_summary.yaml", 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Processed data saved to {output_dir}")

def main():
    """Main function for testing the data loader."""
    loader = DataLoader()
    
    # Load the synthetic data
    df = loader.load_data("data/synthetic/fraud_transactions.csv")
    
    # Generate summary
    summary = loader.get_data_summary(df)
    print("Data Summary:")
    print(f"Total Records: {summary['basic_stats']['total_records']:,}")
    print(f"Fraud Rate: {summary['fraud_stats']['fraud_rate']:.2%}")
    print(f"Date Range: {summary['basic_stats']['date_range']['days']} days")
    
    # Split data
    train_df, val_df, test_df = loader.train_test_split(df)
    
    # Save processed data
    loader.save_processed_data(train_df, val_df, test_df)

if __name__ == "__main__":
    main() 