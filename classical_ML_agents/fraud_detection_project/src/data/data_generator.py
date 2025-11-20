"""
Synthetic Fraud Detection Data Generator

This module generates realistic synthetic transaction data for fraud detection
with proper fraud patterns, customer behaviors, and transaction characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, List, Dict
import yaml
import os
from pathlib import Path

class FraudDataGenerator:
    """
    Generates synthetic fraud detection dataset with realistic patterns.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data generator with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_config = self.config['data_generation']
        self.num_samples = self.data_config['num_samples']
        self.fraud_rate = self.data_config['fraud_rate']
        self.random_seed = self.data_config['random_seed']
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Transaction patterns
        self.transaction_patterns = self.data_config['transaction_patterns']
        self.merchant_categories = self.transaction_patterns['merchant_categories']
        
        # Initialize customer profiles
        self.num_customers = min(10000, self.num_samples // 3)  # Average 3 transactions per customer
        self.customer_profiles = self._create_customer_profiles()
        
    def _create_customer_profiles(self) -> pd.DataFrame:
        """Create customer profiles with varying risk levels."""
        customers = []
        
        for customer_id in range(self.num_customers):
            # Customer demographics
            age = np.random.normal(40, 15)
            age = max(18, min(80, age))  # Clip age between 18-80
            
            # Income affects spending patterns
            income = np.random.lognormal(10.5, 0.5)  # Log-normal distribution for income
            
            # Risk profile (affects fraud probability)
            risk_score = np.random.beta(2, 8)  # Most customers are low risk
            
            # Spending behavior
            avg_transaction_amount = income * 0.001 * (1 + np.random.normal(0, 0.3))
            transaction_frequency = np.random.poisson(5) + 1  # Transactions per month
            
            # Preferred merchant categories
            preferred_categories = np.random.choice(
                self.merchant_categories, 
                size=np.random.randint(2, 5), 
                replace=False
            ).tolist()
            
            customers.append({
                'customer_id': f'CUST_{customer_id:06d}',
                'age': int(age),
                'income': income,
                'risk_score': risk_score,
                'avg_transaction_amount': avg_transaction_amount,
                'transaction_frequency': transaction_frequency,
                'preferred_categories': preferred_categories,
                'account_age_days': np.random.randint(30, 3650)  # 1 month to 10 years
            })
        
        return pd.DataFrame(customers)
    
    def _generate_transaction_time(self, base_date: datetime) -> datetime:
        """Generate realistic transaction timestamp."""
        # Add random days (0-365)
        days_offset = int(np.random.randint(0, 365))
        
        # Business hours are more common (9 AM - 9 PM)
        if np.random.random() < 0.7:  # 70% during business hours
            hour = int(np.random.randint(9, 21))
        else:  # 30% outside business hours (potentially more suspicious)
            hour = int(np.random.choice(list(range(0, 9)) + list(range(21, 24))))
        
        minute = int(np.random.randint(0, 60))
        second = int(np.random.randint(0, 60))
        
        return base_date + timedelta(days=days_offset, hours=hour, minutes=minute, seconds=second)
    
    def _generate_merchant_info(self, category: str) -> Dict:
        """Generate merchant information based on category."""
        merchant_names = {
            'grocery': ['FreshMart', 'SuperSave', 'GreenGrocer', 'QuickShop'],
            'gas': ['SpeedyGas', 'FuelPlus', 'GasStation24', 'AutoFuel'],
            'restaurant': ['TastyBites', 'QuickEats', 'Finedining', 'CafeCorner'],
            'retail': ['MegaMall', 'ShopRite', 'FashionHub', 'TechStore'],
            'online': ['WebShop', 'DigitalMart', 'OnlineDeals', 'CyberStore'],
            'entertainment': ['MovieTheater', 'GameZone', 'SportsPlex', 'MusicHall'],
            'travel': ['AirlineBooking', 'HotelStay', 'CarRental', 'TravelAgency'],
            'healthcare': ['MedCenter', 'PharmaCare', 'HealthClinic', 'WellnessHub']
        }
        
        merchant_name = random.choice(merchant_names[category])
        
        # Generate merchant location
        states = ['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        state = random.choice(states)
        
        # Generate merchant ID
        merchant_id = f'MER_{category.upper()}_{random.randint(1000, 9999)}'
        
        return {
            'merchant_name': merchant_name,
            'merchant_category': category,
            'merchant_state': state,
            'merchant_id': merchant_id
        }
    
    def _calculate_fraud_probability(self, customer: Dict, transaction_data: Dict) -> float:
        """Calculate fraud probability based on customer and transaction features."""
        base_fraud_prob = self.fraud_rate
        
        # Customer risk factors
        risk_multiplier = 1 + customer['risk_score'] * 3
        
        # Transaction amount (higher amounts more likely to be fraud)
        amount_factor = 1 + (transaction_data['amount'] / 10000) * 0.5
        
        # Time-based factors (late night transactions more suspicious)
        hour = transaction_data['timestamp'].hour
        if hour < 6 or hour > 22:
            time_factor = 1.5
        else:
            time_factor = 1.0
        
        # Category-based factors
        high_risk_categories = ['online', 'travel', 'entertainment']
        if transaction_data['merchant_category'] in high_risk_categories:
            category_factor = 1.3
        else:
            category_factor = 1.0
        
        # Location factor (out-of-state transactions more suspicious)
        # For simplicity, assume 20% are out-of-state and more suspicious
        if np.random.random() < 0.2:
            location_factor = 1.4
        else:
            location_factor = 1.0
        
        fraud_prob = base_fraud_prob * risk_multiplier * amount_factor * time_factor * category_factor * location_factor
        
        return min(fraud_prob, 0.8)  # Cap at 80%
    
    def _generate_normal_transaction(self, customer: Dict) -> Dict:
        """Generate a normal (non-fraudulent) transaction."""
        # Choose category from customer's preferences with some randomness
        if np.random.random() < 0.8:
            category = random.choice(customer['preferred_categories'])
        else:
            category = random.choice(self.merchant_categories)
        
        # Generate amount based on customer's spending pattern
        avg_amount = max(1, customer['avg_transaction_amount'])  # Ensure positive
        amount = np.random.lognormal(
            np.log(avg_amount), 
            0.5
        )
        amount = max(1, min(amount, 5000))  # Clip between $1 and $5000
        
        # Generate merchant info
        merchant_info = self._generate_merchant_info(category)
        
        # Generate timestamp
        base_date = datetime(2023, 1, 1)
        timestamp = self._generate_transaction_time(base_date)
        
        return {
            'customer_id': customer['customer_id'],
            'amount': round(amount, 2),
            'timestamp': timestamp,
            **merchant_info,
            'is_weekend': timestamp.weekday() >= 5,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday()
        }
    
    def _generate_fraud_transaction(self, customer: Dict) -> Dict:
        """Generate a fraudulent transaction with suspicious patterns."""
        # Fraudulent transactions often have different patterns
        
        # Higher amounts are more common in fraud
        if np.random.random() < 0.6:
            amount = np.random.uniform(500, 20000)
        else:
            amount = np.random.uniform(1, 200)  # Some fraud is small amounts
        
        # Fraudsters often target high-risk categories
        high_risk_categories = ['online', 'travel', 'entertainment', 'retail']
        if np.random.random() < 0.7:
            category = random.choice(high_risk_categories)
        else:
            category = random.choice(self.merchant_categories)
        
        # Generate merchant info
        merchant_info = self._generate_merchant_info(category)
        
        # Fraudulent transactions often happen at odd hours
        base_date = datetime(2023, 1, 1)
        if np.random.random() < 0.4:  # 40% chance of odd hours
            timestamp = self._generate_transaction_time(base_date)
            # Adjust to make it more likely to be at odd hours
            if np.random.random() < 0.6:
                hour = int(np.random.choice(list(range(0, 6)) + list(range(22, 24))))
                timestamp = timestamp.replace(hour=hour)
        else:
            timestamp = self._generate_transaction_time(base_date)
        
        return {
            'customer_id': customer['customer_id'],
            'amount': round(amount, 2),
            'timestamp': timestamp,
            **merchant_info,
            'is_weekend': timestamp.weekday() >= 5,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday()
        }
    
    def generate_transactions(self) -> pd.DataFrame:
        """Generate the complete transaction dataset."""
        transactions = []
        transaction_id = 1
        
        print(f"Generating {self.num_samples} transactions...")
        
        for i in range(self.num_samples):
            if i % 10000 == 0:
                print(f"Progress: {i}/{self.num_samples} transactions generated")
            
            # Select random customer
            customer = self.customer_profiles.iloc[np.random.randint(0, len(self.customer_profiles))]
            
            # Generate base transaction
            if np.random.random() < 0.5:  # Start with normal transaction
                transaction = self._generate_normal_transaction(customer.to_dict())
            else:
                transaction = self._generate_fraud_transaction(customer.to_dict())
            
            # Calculate fraud probability and determine if fraud
            fraud_prob = self._calculate_fraud_probability(customer.to_dict(), transaction)
            is_fraud = np.random.random() < fraud_prob
            
            # If determined to be fraud, ensure it has fraud characteristics
            if is_fraud:
                transaction = self._generate_fraud_transaction(customer.to_dict())
            
            # Add transaction metadata
            transaction.update({
                'transaction_id': f'TXN_{transaction_id:08d}',
                'is_fraud': int(is_fraud),
                'customer_age': customer['age'],
                'customer_income': customer['income'],
                'account_age_days': customer['account_age_days']
            })
            
            transactions.append(transaction)
            transaction_id += 1
        
        print("Generating additional features...")
        df = pd.DataFrame(transactions)
        
        # Sort by timestamp for time-based features
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better fraud detection."""
        print("Adding derived features...")
        
        # Time-based features
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_month_end'] = (df['timestamp'].dt.day > 25).astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
        
        # Customer aggregations (simplified for synthetic data)
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['count', 'mean', 'std', 'max', 'min'],
            'is_fraud': 'sum'
        }).round(2)
        
        customer_stats.columns = [
            'customer_transaction_count',
            'customer_avg_amount', 
            'customer_std_amount',
            'customer_max_amount',
            'customer_min_amount',
            'customer_fraud_count'
        ]
        
        # Merge customer stats
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Fill NaN values for std (single transactions)
        df['customer_std_amount'] = df['customer_std_amount'].fillna(0)
        
        # Merchant aggregations
        merchant_stats = df.groupby('merchant_id').agg({
            'amount': ['count', 'mean'],
            'is_fraud': ['sum', 'mean']
        }).round(4)
        
        merchant_stats.columns = [
            'merchant_transaction_count',
            'merchant_avg_amount',
            'merchant_fraud_count', 
            'merchant_fraud_rate'
        ]
        
        df = df.merge(merchant_stats, on='merchant_id', how='left')
        
        # Add simple velocity features (for synthetic data, use approximations)
        # Count transactions per customer in the dataset as a proxy for velocity
        customer_transaction_counts = df.groupby('customer_id').size()
        df['hour_velocity'] = df['customer_id'].map(customer_transaction_counts) * 0.01  # Scale down as hourly proxy
        df['day_velocity'] = df['customer_id'].map(customer_transaction_counts) * 0.1   # Scale down as daily proxy
        
        return df
    
    def save_data(self, df: pd.DataFrame) -> None:
        """Save the generated dataset to CSV."""
        output_path = self.data_config['output_path']
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save dataset
        df.to_csv(output_path, index=False)
        
        # Print summary statistics
        print(f"\nDataset saved to: {output_path}")
        print(f"Total transactions: {len(df):,}")
        print(f"Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.2%})")
        print(f"Unique customers: {df['customer_id'].nunique():,}")
        print(f"Unique merchants: {df['merchant_id'].nunique():,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Amount range: ${df['amount'].min():.2f} to ${df['amount'].max():.2f}")
        
        # Category distribution
        print("\nMerchant category distribution:")
        print(df['merchant_category'].value_counts())
        
        # Feature summary
        print(f"\nDataset shape: {df.shape}")
        print("\nColumn data types:")
        print(df.dtypes)

def main():
    """Main function to generate fraud detection dataset."""
    print("Starting Fraud Detection Data Generation...")
    
    # Initialize generator
    generator = FraudDataGenerator()
    
    # Generate transactions
    df = generator.generate_transactions()
    
    # Save dataset
    generator.save_data(df)
    
    print("\nData generation completed successfully!")

if __name__ == "__main__":
    main() 