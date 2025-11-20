"""
Exploratory Data Analysis (EDA) Utilities

This module provides comprehensive utilities for analyzing fraud detection datasets
including visualization functions, statistical analysis, and insight generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import logging

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FraudEDA:
    """
    Comprehensive EDA class for fraud detection datasets.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with fraud detection dataset.
        
        Args:
            df: Fraud detection dataset
        """
        self.df = df.copy()
        self.fraud_df = self.df[self.df['is_fraud'] == 1]
        self.normal_df = self.df[self.df['is_fraud'] == 0]
        self.numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from features
        if 'is_fraud' in self.numeric_features:
            self.numeric_features.remove('is_fraud')
            
        logger.info(f"EDA initialized with {len(self.df):,} transactions ({self.df['is_fraud'].mean():.2%} fraud)")
    
    def dataset_overview(self) -> Dict[str, Any]:
        """Generate comprehensive dataset overview."""
        logger.info("Generating dataset overview")
        
        overview = {
            'basic_stats': {
                'total_transactions': len(self.df),
                'fraud_transactions': len(self.fraud_df),
                'normal_transactions': len(self.normal_df),
                'fraud_rate': self.df['is_fraud'].mean(),
                'unique_customers': self.df['customer_id'].nunique(),
                'unique_merchants': self.df['merchant_id'].nunique() if 'merchant_id' in self.df.columns else 0,
                'date_range_days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days,
                'total_features': len(self.df.columns) - 1  # Exclude target
            },
            'amount_stats': {
                'total_amount': self.df['amount'].sum(),
                'fraud_amount': self.fraud_df['amount'].sum(),
                'avg_transaction': self.df['amount'].mean(),
                'avg_fraud_transaction': self.fraud_df['amount'].mean(),
                'avg_normal_transaction': self.normal_df['amount'].mean(),
                'median_transaction': self.df['amount'].median()
            },
            'data_quality': {
                'missing_values': self.df.isnull().sum().sum(),
                'duplicate_rows': self.df.duplicated().sum(),
                'zero_amounts': (self.df['amount'] == 0).sum(),
                'negative_amounts': (self.df['amount'] < 0).sum()
            }
        }
        
        return overview
    
    def plot_fraud_distribution(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot comprehensive fraud distribution analysis."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Fraud Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Basic fraud distribution
        fraud_counts = self.df['is_fraud'].value_counts()
        axes[0, 0].pie(fraud_counts.values, labels=['Normal', 'Fraud'], autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Overall Transaction Distribution')
        
        # 2. Amount distribution by fraud status
        axes[0, 1].hist(self.normal_df['amount'], bins=50, alpha=0.7, label='Normal', density=True, color='blue')
        axes[0, 1].hist(self.fraud_df['amount'], bins=50, alpha=0.7, label='Fraud', density=True, color='red')
        axes[0, 1].set_xlabel('Transaction Amount ($)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Amount Distribution by Fraud Status')
        axes[0, 1].legend()
        axes[0, 1].set_xlim(0, np.percentile(self.df['amount'], 95))
        
        # 3. Fraud rate by amount bins
        amount_bins = pd.cut(self.df['amount'], bins=20)
        fraud_rate_by_amount = self.df.groupby(amount_bins)['is_fraud'].mean()
        fraud_rate_by_amount.plot(kind='bar', ax=axes[0, 2], rot=45)
        axes[0, 2].set_title('Fraud Rate by Amount Range')
        axes[0, 2].set_ylabel('Fraud Rate')
        
        # 4. Hourly fraud distribution
        hourly_fraud = self.df.groupby('hour_of_day')['is_fraud'].agg(['count', 'sum', 'mean'])
        axes[1, 0].bar(hourly_fraud.index, hourly_fraud['mean'], alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Fraud Rate')
        axes[1, 0].set_title('Fraud Rate by Hour')
        
        # 5. Day of week fraud distribution
        daily_fraud = self.df.groupby('day_of_week')['is_fraud'].agg(['count', 'sum', 'mean'])
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(range(7), daily_fraud['mean'], alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Fraud Rate')
        axes[1, 1].set_title('Fraud Rate by Day of Week')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(days)
        
        # 6. Merchant category fraud rates
        if 'merchant_category' in self.df.columns:
            category_fraud = self.df.groupby('merchant_category')['is_fraud'].agg(['count', 'mean'])
            category_fraud = category_fraud.sort_values('mean', ascending=True)
            category_fraud['mean'].plot(kind='barh', ax=axes[1, 2], color='purple')
            axes[1, 2].set_xlabel('Fraud Rate')
            axes[1, 2].set_title('Fraud Rate by Merchant Category')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_distributions(self, max_features: int = 12) -> None:
        """Analyze distributions of numerical features."""
        numeric_features = self.numeric_features[:max_features]
        n_features = len(numeric_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        fig.suptitle('Feature Distributions (Normal vs Fraud)', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_rows > 1 else [axes] if n_features == 1 else axes
        
        for i, feature in enumerate(numeric_features):
            if i < len(axes):
                # Plot distributions
                if feature in self.normal_df.columns and feature in self.fraud_df.columns:
                    axes[i].hist(self.normal_df[feature].dropna(), bins=30, alpha=0.6, 
                               label='Normal', density=True, color='blue')
                    axes[i].hist(self.fraud_df[feature].dropna(), bins=30, alpha=0.6, 
                               label='Fraud', density=True, color='red')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Density')
                    axes[i].set_title(f'{feature} Distribution')
                    axes[i].legend()
        
        # Hide empty subplots
        for i in range(len(numeric_features), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """Analyze feature correlations."""
        logger.info("Analyzing feature correlations")
        
        # Calculate correlation matrix
        numeric_df = self.df[self.numeric_features + ['is_fraud']]
        correlation_matrix = numeric_df.corr(method=method)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Feature Correlation Matrix ({method.title()})')
        plt.tight_layout()
        plt.show()
        
        # Show correlations with target
        target_correlations = correlation_matrix['is_fraud'].drop('is_fraud').sort_values(key=abs, ascending=False)
        print("Top correlations with fraud target:")
        print(target_correlations.head(10))
        
        return correlation_matrix
    
    def fraud_vs_normal_comparison(self) -> Dict[str, pd.DataFrame]:
        """Compare fraud vs normal transactions across all features."""
        logger.info("Comparing fraud vs normal transactions")
        
        results = {}
        
        # Numerical features comparison
        numeric_comparison = []
        for feature in self.numeric_features:
            if feature in self.df.columns:
                normal_stats = self.normal_df[feature].describe()
                fraud_stats = self.fraud_df[feature].describe()
                
                comparison = {
                    'feature': feature,
                    'normal_mean': normal_stats['mean'],
                    'fraud_mean': fraud_stats['mean'],
                    'normal_median': normal_stats['50%'],
                    'fraud_median': fraud_stats['50%'],
                    'normal_std': normal_stats['std'],
                    'fraud_std': fraud_stats['std'],
                    'mean_ratio': fraud_stats['mean'] / normal_stats['mean'] if normal_stats['mean'] != 0 else np.inf,
                    'ks_statistic': stats.ks_2samp(self.normal_df[feature].dropna(), 
                                                 self.fraud_df[feature].dropna())[0]
                }
                numeric_comparison.append(comparison)
        
        results['numeric_comparison'] = pd.DataFrame(numeric_comparison)
        
        # Categorical features comparison
        categorical_comparison = []
        for feature in self.categorical_features:
            if feature in self.df.columns and feature != 'transaction_id':
                # Top categories for normal vs fraud
                normal_top = self.normal_df[feature].value_counts().head(5)
                fraud_top = self.fraud_df[feature].value_counts().head(5)
                
                # Calculate fraud rate by category
                fraud_by_category = self.df.groupby(feature)['is_fraud'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
                
                categorical_comparison.append({
                    'feature': feature,
                    'normal_top_category': normal_top.index[0] if len(normal_top) > 0 else None,
                    'fraud_top_category': fraud_top.index[0] if len(fraud_top) > 0 else None,
                    'unique_values': self.df[feature].nunique(),
                    'highest_fraud_rate_category': fraud_by_category.index[0] if len(fraud_by_category) > 0 else None,
                    'highest_fraud_rate': fraud_by_category['mean'].iloc[0] if len(fraud_by_category) > 0 else None
                })
        
        results['categorical_comparison'] = pd.DataFrame(categorical_comparison)
        
        return results
    
    def temporal_analysis(self) -> None:
        """Analyze temporal patterns in fraud."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Fraud Analysis', fontsize=16, fontweight='bold')
        
        # 1. Fraud rate over time (daily)
        daily_fraud = self.df.groupby(self.df['timestamp'].dt.date)['is_fraud'].agg(['count', 'sum', 'mean'])
        axes[0, 0].plot(daily_fraud.index, daily_fraud['mean'], marker='o', alpha=0.7)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Fraud Rate')
        axes[0, 0].set_title('Daily Fraud Rate Trend')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Fraud volume over time
        axes[0, 1].plot(daily_fraud.index, daily_fraud['sum'], color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Fraud Count')
        axes[0, 1].set_title('Daily Fraud Volume')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Monthly patterns
        monthly_fraud = self.df.groupby('month')['is_fraud'].mean()
        axes[1, 0].bar(monthly_fraud.index, monthly_fraud.values, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Fraud Rate')
        axes[1, 0].set_title('Fraud Rate by Month')
        
        # 4. Weekend vs Weekday
        weekend_fraud = self.df.groupby('is_weekend')['is_fraud'].mean()
        axes[1, 1].bar(['Weekday', 'Weekend'], weekend_fraud.values, alpha=0.7, color='green')
        axes[1, 1].set_ylabel('Fraud Rate')
        axes[1, 1].set_title('Fraud Rate: Weekend vs Weekday')
        
        plt.tight_layout()
        plt.show()
    
    def outlier_analysis(self) -> Dict[str, pd.DataFrame]:
        """Detect and analyze outliers in numerical features."""
        logger.info("Analyzing outliers")
        
        outlier_results = {}
        
        # IQR method for outlier detection
        for feature in self.numeric_features[:8]:  # Limit to first 8 features
            if feature in self.df.columns:
                Q1 = self.df[feature].quantile(0.25)
                Q3 = self.df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
                outlier_fraud_rate = outliers['is_fraud'].mean() if len(outliers) > 0 else 0
                
                outlier_results[feature] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(self.df) * 100,
                    'outlier_fraud_rate': outlier_fraud_rate,
                    'normal_fraud_rate': self.df['is_fraud'].mean(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        # Create summary DataFrame
        outlier_summary = pd.DataFrame(outlier_results).T
        
        # Plot outlier analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')
        
        # Outlier counts
        outlier_summary['outlier_count'].plot(kind='bar', ax=axes[0, 0], color='red', alpha=0.7)
        axes[0, 0].set_title('Outlier Counts by Feature')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Outlier percentages
        outlier_summary['outlier_percentage'].plot(kind='bar', ax=axes[0, 1], color='orange', alpha=0.7)
        axes[0, 1].set_title('Outlier Percentages by Feature')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Outlier fraud rates vs normal
        x_pos = np.arange(len(outlier_summary))
        axes[1, 0].bar(x_pos - 0.2, outlier_summary['outlier_fraud_rate'], 0.4, label='Outlier Fraud Rate', alpha=0.7)
        axes[1, 0].bar(x_pos + 0.2, outlier_summary['normal_fraud_rate'], 0.4, label='Overall Fraud Rate', alpha=0.7)
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Fraud Rate')
        axes[1, 0].set_title('Fraud Rate: Outliers vs Overall')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(outlier_summary.index, rotation=45)
        axes[1, 0].legend()
        
        # Box plot of amount feature (most important)
        if 'amount' in self.df.columns:
            self.df.boxplot(column='amount', by='is_fraud', ax=axes[1, 1])
            axes[1, 1].set_title('Amount Distribution by Fraud Status')
            axes[1, 1].set_ylabel('Amount ($)')
        
        plt.tight_layout()
        plt.show()
        
        return outlier_summary
    
    def feature_importance_analysis(self) -> pd.DataFrame:
        """Calculate preliminary feature importance using mutual information."""
        logger.info("Calculating feature importance")
        
        # Prepare features
        feature_df = self.df[self.numeric_features].fillna(0)
        target = self.df['is_fraud']
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(feature_df, target, random_state=42)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.numeric_features,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
        plt.title('Feature Importance (Mutual Information)')
        plt.xlabel('Mutual Information Score')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def generate_insights_summary(self) -> Dict[str, Any]:
        """Generate comprehensive insights summary."""
        logger.info("Generating insights summary")
        
        insights = {
            'key_findings': [],
            'fraud_characteristics': {},
            'risk_factors': {},
            'recommendations': []
        }
        
        # Basic insights
        fraud_rate = self.df['is_fraud'].mean()
        avg_fraud_amount = self.fraud_df['amount'].mean()
        avg_normal_amount = self.normal_df['amount'].mean()
        
        insights['key_findings'].extend([
            f"Overall fraud rate: {fraud_rate:.2%}",
            f"Average fraud transaction: ${avg_fraud_amount:,.2f}",
            f"Average normal transaction: ${avg_normal_amount:,.2f}",
            f"Fraud transactions are {avg_fraud_amount/avg_normal_amount:.1f}x higher on average"
        ])
        
        # Temporal insights
        hourly_fraud = self.df.groupby('hour_of_day')['is_fraud'].mean()
        peak_fraud_hour = hourly_fraud.idxmax()
        insights['key_findings'].append(f"Peak fraud hour: {peak_fraud_hour}:00 ({hourly_fraud.max():.1%} fraud rate)")
        
        # Category insights
        if 'merchant_category' in self.df.columns:
            category_fraud = self.df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False)
            highest_risk_category = category_fraud.index[0]
            insights['key_findings'].append(f"Highest risk category: {highest_risk_category} ({category_fraud.iloc[0]:.1%} fraud rate)")
        
        return insights

def main():
    """Main function for testing EDA utilities."""
    # Load data
    df = pd.read_csv("data/processed/train.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize EDA
    eda = FraudEDA(df)
    
    # Run analyses
    overview = eda.dataset_overview()
    print("Dataset Overview:", overview['basic_stats'])
    
    # Generate plots
    eda.plot_fraud_distribution()
    eda.analyze_feature_distributions()
    correlation_matrix = eda.correlation_analysis()
    
    print("\nEDA analysis completed!")

if __name__ == "__main__":
    main() 