"""
Feature Engineering Module

This module provides comprehensive feature engineering capabilities for fraud detection
including time-based features, aggregations, interactions, and transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder, BinaryEncoder, OneHotEncoder
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for fraud detection datasets.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary for feature engineering
        """
        self.config = config or {}
        self.feature_transformers = {}
        self.feature_columns = {}
        self.is_fitted = False
        
        # Default configuration
        self.default_config = {
            'time_windows': [1, 3, 7, 14, 30],  # days
            'amount_percentiles': [10, 25, 50, 75, 90, 95, 99],
            'categorical_encoding': 'target',  # 'target', 'onehot', 'binary'
            'scaling_method': 'robust',  # 'standard', 'robust', 'minmax'
            'create_interactions': True,
            'create_ratios': True,
            'create_aggregations': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> pd.DataFrame:
        """
        Fit feature engineering pipeline and transform data.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            
        Returns:
            Transformed dataframe with engineered features
        """
        logger.info("Starting feature engineering fit_transform")
        
        df_transformed = df.copy()
        
        # 1. Create time-based features
        df_transformed = self._create_time_features(df_transformed)
        
        # 2. Create aggregation features
        if self.config['create_aggregations']:
            df_transformed = self._create_aggregation_features(df_transformed)
        
        # 3. Create interaction features
        if self.config['create_interactions']:
            df_transformed = self._create_interaction_features(df_transformed)
        
        # 4. Create ratio features
        if self.config['create_ratios']:
            df_transformed = self._create_ratio_features(df_transformed)
        
        # 5. Create statistical features
        df_transformed = self._create_statistical_features(df_transformed)
        
        # 6. Encode categorical features
        df_transformed = self._encode_categorical_features(df_transformed, target_col, fit=True)
        
        # 7. Create binning features
        df_transformed = self._create_binning_features(df_transformed, fit=True)
        
        # 8. Handle missing values
        df_transformed = self._handle_missing_values(df_transformed, fit=True)
        
        # 9. Scale numerical features
        df_transformed = self._scale_features(df_transformed, target_col, fit=True)
        
        self.is_fitted = True
        logger.info(f"Feature engineering completed. Shape: {df_transformed.shape}")
        
        return df_transformed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted feature engineering pipeline.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        logger.info("Applying feature engineering transform")
        
        df_transformed = df.copy()
        
        # Apply same transformations in same order
        df_transformed = self._create_time_features(df_transformed)
        
        if self.config['create_aggregations']:
            df_transformed = self._create_aggregation_features(df_transformed)
        
        if self.config['create_interactions']:
            df_transformed = self._create_interaction_features(df_transformed)
        
        if self.config['create_ratios']:
            df_transformed = self._create_ratio_features(df_transformed)
        
        df_transformed = self._create_statistical_features(df_transformed)
        df_transformed = self._encode_categorical_features(df_transformed, fit=False)
        df_transformed = self._create_binning_features(df_transformed, fit=False)
        df_transformed = self._handle_missing_values(df_transformed, fit=False)
        df_transformed = self._scale_features(df_transformed, fit=False)
        
        return df_transformed
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp."""
        if 'timestamp' not in df.columns:
            return df
        
        logger.info("Creating time-based features")
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['year'] = df['timestamp'].dt.year
        df['quarter'] = df['timestamp'].dt.quarter
        df['month'] = df['timestamp'].dt.month
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['day'] = df['timestamp'].dt.day
        df['dayofyear'] = df['timestamp'].dt.dayofyear
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Derived time features
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = (df['timestamp'].dt.day <= 5).astype(int)
        df['is_month_end'] = (df['timestamp'].dt.day >= 25).astype(int)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features for customers and merchants."""
        logger.info("Creating aggregation features")
        df = df.copy()
        
        # Customer aggregations
        if 'customer_id' in df.columns:
            customer_agg = df.groupby('customer_id').agg({
                'amount': ['count', 'sum', 'mean', 'std', 'min', 'max', 'median'],
                'hour': ['mean', 'std'],
                'is_weekend': 'mean'
            }).round(4)
            
            # Flatten column names
            customer_agg.columns = ['cust_' + '_'.join(col).strip() for col in customer_agg.columns]
            customer_agg.reset_index(inplace=True)
            
            # Merge back
            df = df.merge(customer_agg, on='customer_id', how='left')
        
        # Merchant aggregations
        if 'merchant_id' in df.columns:
            merchant_agg = df.groupby('merchant_id').agg({
                'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
                'customer_id': 'nunique'
            }).round(4)
            
            # Flatten column names
            merchant_agg.columns = ['merch_' + '_'.join(col).strip() for col in merchant_agg.columns]
            merchant_agg.reset_index(inplace=True)
            
            # Merge back
            df = df.merge(merchant_agg, on='merchant_id', how='left')
        
        # Category aggregations
        if 'merchant_category' in df.columns:
            category_agg = df.groupby('merchant_category').agg({
                'amount': ['mean', 'std', 'count'],
                'hour': 'mean'
            }).round(4)
            
            # Flatten column names
            category_agg.columns = ['cat_' + '_'.join(col).strip() for col in category_agg.columns]
            category_agg.reset_index(inplace=True)
            
            # Merge back
            df = df.merge(category_agg, on='merchant_category', how='left')
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        logger.info("Creating interaction features")
        df = df.copy()
        
        # Amount-based interactions
        if 'amount' in df.columns:
            if 'hour' in df.columns:
                df['amount_hour_interaction'] = df['amount'] * df['hour']
            
            if 'is_weekend' in df.columns:
                df['amount_weekend_interaction'] = df['amount'] * df['is_weekend']
            
            if 'customer_age' in df.columns:
                df['amount_age_interaction'] = df['amount'] * df['customer_age']
        
        # Customer-merchant interactions
        if 'customer_id' in df.columns and 'merchant_id' in df.columns:
            # Create customer-merchant pair identifier
            df['customer_merchant_pair'] = df['customer_id'].astype(str) + '_' + df['merchant_id'].astype(str)
            
            # Count unique pairs
            pair_counts = df['customer_merchant_pair'].value_counts()
            df['customer_merchant_frequency'] = df['customer_merchant_pair'].map(pair_counts)
            
            # First time interaction flag
            df['is_first_interaction'] = (df['customer_merchant_frequency'] == 1).astype(int)
        
        # Time-amount interactions
        if 'timestamp' in df.columns and 'amount' in df.columns:
            # Amount deviation from daily average
            daily_avg = df.groupby(df['timestamp'].dt.date)['amount'].mean()
            df['daily_avg_amount'] = df['timestamp'].dt.date.map(daily_avg)
            df['amount_daily_deviation'] = df['amount'] - df['daily_avg_amount']
            df['amount_daily_ratio'] = df['amount'] / df['daily_avg_amount']
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and comparison features."""
        logger.info("Creating ratio features")
        df = df.copy()
        
        # Customer ratios
        if 'amount' in df.columns:
            # Amount vs customer averages
            if 'cust_amount_mean' in df.columns:
                df['amount_vs_customer_avg'] = df['amount'] / (df['cust_amount_mean'] + 1e-8)
                df['amount_customer_zscore'] = (df['amount'] - df['cust_amount_mean']) / (df['cust_amount_std'] + 1e-8)
            
            # Amount vs merchant averages
            if 'merch_amount_mean' in df.columns:
                df['amount_vs_merchant_avg'] = df['amount'] / (df['merch_amount_mean'] + 1e-8)
            
            # Amount vs category averages
            if 'cat_amount_mean' in df.columns:
                df['amount_vs_category_avg'] = df['amount'] / (df['cat_amount_mean'] + 1e-8)
        
        # Transaction frequency ratios
        if 'customer_transaction_count' in df.columns and 'account_age_days' in df.columns:
            df['transactions_per_day'] = df['customer_transaction_count'] / (df['account_age_days'] + 1)
        
        # Income-based ratios
        if 'amount' in df.columns and 'customer_income' in df.columns:
            df['amount_income_ratio'] = df['amount'] / (df['customer_income'] + 1e-8)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        logger.info("Creating statistical features")
        df = df.copy()
        
        # Amount-based statistical features
        if 'amount' in df.columns:
            # Log transformation
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_sqrt'] = np.sqrt(df['amount'])
            
            # Power transformations
            df['amount_squared'] = df['amount'] ** 2
            df['amount_cubed'] = df['amount'] ** 3
            
            # Binomial features (high/low amounts)
            amount_median = df['amount'].median()
            df['is_high_amount'] = (df['amount'] > amount_median).astype(int)
            
            # Percentile rankings
            df['amount_percentile_rank'] = df['amount'].rank(pct=True)
        
        # Age-based features
        if 'customer_age' in df.columns:
            df['age_group'] = pd.cut(df['customer_age'], 
                                   bins=[0, 25, 35, 45, 55, 65, 100], 
                                   labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        # Velocity features (transaction patterns)
        if 'timestamp' in df.columns and 'customer_id' in df.columns:
            # Sort by customer and timestamp
            df_sorted = df.sort_values(['customer_id', 'timestamp'])
            
            # Time between transactions for same customer
            df_sorted['time_since_last_transaction'] = df_sorted.groupby('customer_id')['timestamp'].diff()
            df_sorted['time_since_last_transaction_hours'] = df_sorted['time_since_last_transaction'].dt.total_seconds() / 3600
            
            # Velocity indicators
            df_sorted['is_rapid_transaction'] = (df_sorted['time_since_last_transaction_hours'] < 1).astype(int)
            
            # Merge back
            df = df.merge(df_sorted[['transaction_id', 'time_since_last_transaction_hours', 'is_rapid_transaction']], 
                         on='transaction_id', how='left')
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, target_col: str = 'is_fraud', fit: bool = False) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info(f"Encoding categorical features (method: {self.config['categorical_encoding']})")
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Remove ID columns and target
        categorical_cols = [col for col in categorical_cols if col not in ['transaction_id', 'customer_id', 'merchant_id', target_col]]
        
        if fit:
            self.feature_transformers['categorical_encoders'] = {}
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    if self.config['categorical_encoding'] == 'target':
                        encoder = TargetEncoder()
                        df[f'{col}_encoded'] = encoder.fit_transform(df[col], df[target_col])
                        self.feature_transformers['categorical_encoders'][col] = encoder
                    elif self.config['categorical_encoding'] == 'onehot':
                        encoder = OneHotEncoder(cols=[col], use_cat_names=True)
                        encoded_df = encoder.fit_transform(df[[col]])
                        df = pd.concat([df, encoded_df], axis=1)
                        self.feature_transformers['categorical_encoders'][col] = encoder
                    elif self.config['categorical_encoding'] == 'binary':
                        encoder = BinaryEncoder(cols=[col])
                        encoded_df = encoder.fit_transform(df[[col]])
                        df = pd.concat([df, encoded_df], axis=1)
                        self.feature_transformers['categorical_encoders'][col] = encoder
                else:
                    # Transform using fitted encoders
                    encoder = self.feature_transformers['categorical_encoders'].get(col)
                    if encoder:
                        if self.config['categorical_encoding'] == 'target':
                            df[f'{col}_encoded'] = encoder.transform(df[col])
                        else:
                            encoded_df = encoder.transform(df[[col]])
                            df = pd.concat([df, encoded_df], axis=1)
        
        return df
    
    def _create_binning_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create binned versions of numerical features."""
        logger.info("Creating binning features")
        df = df.copy()
        
        if fit:
            self.feature_transformers['binning'] = {}
        
        # Amount binning
        if 'amount' in df.columns:
            if fit:
                # Create bins based on percentiles
                bins = np.percentile(df['amount'], self.config['amount_percentiles'])
                bins = np.unique(np.concatenate([[0], bins, [np.inf]]))
                self.feature_transformers['binning']['amount_bins'] = bins
            
            bins = self.feature_transformers['binning']['amount_bins']
            df['amount_bin'] = pd.cut(df['amount'], bins=bins, labels=False, include_lowest=True)
        
        # Age binning
        if 'customer_age' in df.columns:
            if fit:
                age_bins = [0, 25, 35, 45, 55, 65, 100]
                self.feature_transformers['binning']['age_bins'] = age_bins
            
            age_bins = self.feature_transformers['binning']['age_bins']
            df['age_bin'] = pd.cut(df['customer_age'], bins=age_bins, labels=False, include_lowest=True)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values")
        df = df.copy()
        
        if fit:
            self.feature_transformers['missing_handlers'] = {}
            
            # Calculate fill values for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    fill_value = df[col].median()
                    self.feature_transformers['missing_handlers'][col] = fill_value
            
            # Calculate fill values for categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    fill_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                    self.feature_transformers['missing_handlers'][col] = fill_value
        
        # Fill missing values
        for col, fill_value in self.feature_transformers['missing_handlers'].items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, target_col: str = 'is_fraud', fit: bool = False) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info(f"Scaling features (method: {self.config['scaling_method']})")
        df = df.copy()
        
        # Get numerical columns (exclude target and IDs)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_exclude = [target_col, 'transaction_id'] + [col for col in numerical_cols if 'id' in col.lower()]
        numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]
        
        if fit:
            # Initialize scaler
            if self.config['scaling_method'] == 'standard':
                scaler = StandardScaler()
            elif self.config['scaling_method'] == 'robust':
                scaler = RobustScaler()
            elif self.config['scaling_method'] == 'minmax':
                scaler = MinMaxScaler()
            else:
                return df
            
            # Fit and transform
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.feature_transformers['scaler'] = scaler
        else:
            # Transform using fitted scaler
            scaler = self.feature_transformers.get('scaler')
            if scaler:
                df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'is_fraud', 
                       method: str = 'mutual_info', k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using specified method.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            method: Feature selection method ('mutual_info', 'f_classif', 'rfe')
            k: Number of features to select
            
        Returns:
            Tuple of (selected_df, selected_features_list)
        """
        logger.info(f"Selecting top {k} features using {method}")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col and not col.endswith('_id')]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Handle missing values for feature selection
        X = X.fillna(X.median())
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Fit selector and get selected features
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create selected dataframe
        selected_df = df[[target_col] + selected_features].copy()
        
        logger.info(f"Selected {len(selected_features)} features")
        return selected_df, selected_features
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'is_fraud', 
                              method: str = 'mutual_info') -> pd.DataFrame:
        """
        Calculate feature importance scores.
        
        Args:
            df: Input dataframe
            target_col: Target column name  
            method: Importance calculation method
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Calculating feature importance using {method}")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col and not col.endswith('_id')]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        if method == 'mutual_info':
            scores = mutual_info_classif(X, y, random_state=42)
        elif method == 'f_classif':
            scores, _ = f_classif(X, y)
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': scores
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    """Main function for testing feature engineering."""
    # Load training data
    train_df = pd.read_csv("data/processed/train.csv")
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    
    print(f"Original shape: {train_df.shape}")
    
    # Initialize feature engineer
    config = {
        'create_interactions': True,
        'create_ratios': True,
        'create_aggregations': True,
        'categorical_encoding': 'target',
        'scaling_method': 'robust'
    }
    
    fe = FeatureEngineer(config)
    
    # Apply feature engineering
    train_transformed = fe.fit_transform(train_df, 'is_fraud')
    print(f"Transformed shape: {train_transformed.shape}")
    
    # Feature selection
    selected_df, selected_features = fe.select_features(train_transformed, 'is_fraud', k=30)
    print(f"Selected features shape: {selected_df.shape}")
    print(f"Top 10 selected features: {selected_features[:10]}")
    
    # Feature importance
    importance_df = fe.get_feature_importance(train_transformed, 'is_fraud')
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    print("\nFeature engineering completed successfully!")

if __name__ == "__main__":
    main() 