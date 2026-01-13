# src/feature_engineering.py
"""
Feature Engineering Pipeline
============================

Transforms raw client data into predictive features for the churn model.
Creates 25+ features across categories: Recency, Frequency, Monetary, 
Satisfaction, and Engagement.

Usage:
    from src.feature_engineering import FeatureEngineer
    
    fe = FeatureEngineer()
    features_df = fe.create_all_features(clients, bookings, feedback, communications)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CHURN_THRESHOLD_DAYS,
    TIME_WINDOWS, MIN_BOOKINGS_FOR_TREND, RAW_FILES, PROCESSED_FILES
)
from src.utils import (
    load_csv, save_csv, print_header, print_subheader,
    calculate_trend, safe_divide
)


class FeatureEngineer:
    """
    Feature engineering pipeline for churn prediction.
    
    This class transforms raw data into predictive features.
    Features are grouped into logical categories:
    
    - Recency: How recently the client engaged
    - Frequency: How often they book
    - Monetary: How much they spend
    - Satisfaction: Their ratings and feedback
    - Engagement: How deeply involved they are
    """
    
    def __init__(self, reference_date=None):
        """
        Initialize the feature engineer.
        
        Parameters:
            reference_date: Date to calculate features from (default: today)
        """
        self.reference_date = reference_date or pd.Timestamp.now()
        self.feature_names = []
    
    def create_all_features(self, clients_df, bookings_df, feedback_df, 
                           communications_df, churn_df=None):
        """
        Create all features for each client.
        
        Parameters:
            clients_df: Client master data
            bookings_df: Booking history
            feedback_df: Feedback records
            communications_df: Communication logs
            churn_df: Optional churn labels (for training data)
        
        Returns:
            DataFrame with all features and optional churn label
        """
        print_header("FEATURE ENGINEERING")
        
        # Ensure date columns are datetime
        bookings_df = bookings_df.copy()
        feedback_df = feedback_df.copy()
        communications_df = communications_df.copy()
        
        bookings_df['booking_date'] = pd.to_datetime(bookings_df['booking_date'])
        feedback_df['feedback_date'] = pd.to_datetime(feedback_df['feedback_date'])
        communications_df['contact_date'] = pd.to_datetime(communications_df['contact_date'])
        
        features_list = []
        
        print(f"Processing {len(clients_df)} clients...")
        
        for idx, client in clients_df.iterrows():
            client_id = client['client_id']
            
            # Get client-specific data
            client_bookings = bookings_df[bookings_df['client_id'] == client_id].copy()
            client_feedback = feedback_df[feedback_df['client_id'] == client_id].copy()
            client_comms = communications_df[communications_df['client_id'] == client_id].copy()
            
            # Skip clients with no bookings
            if len(client_bookings) == 0:
                continue
            
            # Sort by date
            client_bookings = client_bookings.sort_values('booking_date')
            
            # Create feature dictionary
            features = {'client_id': client_id}
            
            # Add all feature categories
            features.update(self._recency_features(client, client_bookings))
            features.update(self._frequency_features(client_bookings))
            features.update(self._monetary_features(client_bookings))
            features.update(self._satisfaction_features(client_feedback))
            features.update(self._engagement_features(client_bookings, client_comms))
            features.update(self._client_profile_features(client))
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Add churn labels if provided
        if churn_df is not None:
            churn_labels = churn_df[['client_id', 'churned']].copy()
            churn_labels['churned'] = (churn_labels['churned'] == 'Yes').astype(int)
            features_df = features_df.merge(churn_labels, on='client_id', how='left')
        
        # Store feature names (excluding client_id and churned)
        self.feature_names = [col for col in features_df.columns 
                             if col not in ['client_id', 'churned']]
        
        print(f"\n✓ Created {len(self.feature_names)} features for {len(features_df)} clients")
        
        return features_df
    
    def _recency_features(self, client, bookings):
        """
        Create recency-based features (how recently did they engage?).
        
        Features:
            - days_since_last_booking: Days since most recent booking
            - days_since_first_booking: Client tenure in days
            - months_as_client: Tenure in months
        """
        last_booking = bookings['booking_date'].max()
        first_booking = pd.to_datetime(client['first_booking_date'])
        
        return {
            'days_since_last_booking': (self.reference_date - last_booking).days,
            'days_since_first_booking': (self.reference_date - first_booking).days,
            'months_as_client': (self.reference_date - first_booking).days / 30
        }
    
    def _frequency_features(self, bookings):
        """
        Create frequency-based features (how often do they book?).
        
        Features:
            - total_bookings: Total number of bookings ever
            - bookings_last_30_days: Bookings in last 30 days
            - bookings_last_90_days: Bookings in last 90 days
            - bookings_last_180_days: Bookings in last 180 days
            - avg_days_between_bookings: Average gap between bookings
            - max_gap_between_bookings: Longest gap between bookings
            - booking_frequency_trend: Is frequency increasing or decreasing?
        """
        features = {
            'total_bookings': len(bookings)
        }
        
        # Bookings in time windows
        for window in TIME_WINDOWS:
            cutoff = self.reference_date - pd.Timedelta(days=window)
            count = len(bookings[bookings['booking_date'] >= cutoff])
            features[f'bookings_last_{window}_days'] = count
        
        # Days between bookings
        if len(bookings) > 1:
            bookings = bookings.sort_values('booking_date')
            gaps = bookings['booking_date'].diff().dt.days.dropna()
            features['avg_days_between_bookings'] = gaps.mean()
            features['max_gap_between_bookings'] = gaps.max()
        else:
            features['avg_days_between_bookings'] = 0
            features['max_gap_between_bookings'] = 0
        
        # Booking frequency trend (comparing first half to second half of tenure)
        if len(bookings) >= MIN_BOOKINGS_FOR_TREND:
            mid_point = bookings['booking_date'].min() + \
                       (bookings['booking_date'].max() - bookings['booking_date'].min()) / 2
            
            first_half_count = len(bookings[bookings['booking_date'] < mid_point])
            second_half_count = len(bookings[bookings['booking_date'] >= mid_point])
            
            features['booking_frequency_trend'] = safe_divide(
                second_half_count - first_half_count,
                first_half_count,
                0
            )
        else:
            features['booking_frequency_trend'] = 0
        
        return features
    
    def _monetary_features(self, bookings):
        """
        Create monetary features (how much do they spend?).
        
        Features:
            - total_revenue: Lifetime revenue from client
            - avg_booking_value: Average revenue per booking
            - max_booking_value: Largest single booking
            - revenue_last_90_days: Recent revenue
            - revenue_trend: Is spending increasing or decreasing?
            - total_participants: Total people trained
            - avg_participants_per_booking: Average group size
        """
        features = {
            'total_revenue': bookings['revenue'].sum(),
            'avg_booking_value': bookings['revenue'].mean(),
            'max_booking_value': bookings['revenue'].max(),
            'total_participants': bookings['number_of_participants'].sum(),
            'avg_participants_per_booking': bookings['number_of_participants'].mean()
        }
        
        # Revenue in last 90 days
        cutoff_90 = self.reference_date - pd.Timedelta(days=90)
        recent_bookings = bookings[bookings['booking_date'] >= cutoff_90]
        features['revenue_last_90_days'] = recent_bookings['revenue'].sum()
        
        # Revenue trend
        if len(bookings) >= MIN_BOOKINGS_FOR_TREND:
            revenues = bookings.sort_values('booking_date')['revenue'].tolist()
            features['revenue_trend'] = calculate_trend(revenues)
        else:
            features['revenue_trend'] = 0
        
        return features
    
    def _satisfaction_features(self, feedback):
        """
        Create satisfaction features (are they happy?).
        
        Features:
            - avg_overall_rating: Average rating given
            - avg_nps_score: Average NPS score
            - avg_trainer_rating: Average trainer rating
            - latest_rating: Most recent rating
            - min_rating_ever: Lowest rating given
            - rating_trend: Is satisfaction improving or declining?
            - feedback_response_rate: Percentage of bookings with feedback
        """
        if len(feedback) == 0:
            return {
                'avg_overall_rating': np.nan,
                'avg_nps_score': np.nan,
                'avg_trainer_rating': np.nan,
                'latest_rating': np.nan,
                'min_rating_ever': np.nan,
                'rating_trend': 0,
                'has_given_feedback': 0
            }
        
        feedback = feedback.sort_values('feedback_date')
        
        features = {
            'avg_overall_rating': feedback['overall_rating'].mean(),
            'avg_nps_score': feedback['nps_score'].mean(),
            'avg_trainer_rating': feedback['trainer_rating'].mean(),
            'latest_rating': feedback.iloc[-1]['overall_rating'],
            'min_rating_ever': feedback['overall_rating'].min(),
            'has_given_feedback': 1
        }
        
        # Rating trend
        if len(feedback) >= MIN_BOOKINGS_FOR_TREND:
            ratings = feedback['overall_rating'].tolist()
            features['rating_trend'] = calculate_trend(ratings)
        else:
            features['rating_trend'] = 0
        
        return features
    
    def _engagement_features(self, bookings, communications):
        """
        Create engagement features (how engaged are they?).
        
        Features:
            - num_course_categories: Unique course types taken
            - category_diversity_score: How diverse is their course selection?
            - total_communications: Total touchpoints
            - communications_last_90_days: Recent communication frequency
            - avg_communications_per_booking: Engagement ratio
        """
        features = {
            'num_course_categories': bookings['course_category'].nunique(),
            'num_unique_courses': bookings['course_name'].nunique()
        }
        
        # Category diversity score (entropy-like measure)
        category_counts = bookings['course_category'].value_counts()
        if len(category_counts) > 1:
            proportions = category_counts / len(bookings)
            features['category_diversity_score'] = -sum(proportions * np.log2(proportions))
        else:
            features['category_diversity_score'] = 0
        
        # Communication features
        features['total_communications'] = len(communications)
        
        cutoff_90 = self.reference_date - pd.Timedelta(days=90)
        recent_comms = communications[communications['contact_date'] >= cutoff_90]
        features['communications_last_90_days'] = len(recent_comms)
        
        features['avg_communications_per_booking'] = safe_divide(
            len(communications), len(bookings), 0
        )
        
        return features
    
    def _client_profile_features(self, client):
        """
        Create client profile features (who are they?).
        
        Features:
            - industry_encoded: Industry sector as category
            - company_size_encoded: Company size as category
            - region_encoded: Geographic region as category
        """
        return {
            'industry': client['industry_sector'],
            'company_size': client['company_size'],
            'region': client['region'],
            'account_manager': client['account_manager']
        }
    
    def get_feature_names(self):
        """Return list of feature names (excluding identifiers and target)."""
        return self.feature_names


def preprocess_features(features_df, label_encoders=None, fit=True):
    """
    Preprocess features for model training.
    
    - Handles missing values
    - Encodes categorical variables
    - Returns numeric feature matrix
    
    Parameters:
        features_df: DataFrame with features
        label_encoders: Dict of LabelEncoders (for transform mode)
        fit: Whether to fit encoders (True) or just transform (False)
    
    Returns:
        Processed DataFrame and label encoders
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    
    print_subheader("Preprocessing Features")
    
    df = features_df.copy()
    
    # Identify column types
    categorical_cols = ['industry', 'company_size', 'region', 'account_manager']
    numeric_cols = [col for col in df.columns 
                   if col not in categorical_cols + ['client_id', 'churned']]
    
    # Handle missing values in numeric columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Encode categorical columns
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        if fit:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le:
                # Handle unseen categories
                df[col + '_encoded'] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] 
                    if str(x) in le.classes_ else -1
                )
    
    # Drop original categorical columns
    df = df.drop(columns=categorical_cols, errors='ignore')
    
    print(f"✓ Processed {len(numeric_cols)} numeric features")
    print(f"✓ Encoded {len(categorical_cols)} categorical features")
    
    return df, label_encoders


def load_and_engineer_features():
    """
    Load raw data and create features.
    
    This is a convenience function that runs the full pipeline.
    
    Returns:
        features_df: DataFrame with all features and churn labels
        label_encoders: Dict of fitted label encoders
    """
    print_header("LOADING AND ENGINEERING FEATURES")
    
    # Load raw data
    clients_df = load_csv(RAW_DATA_DIR / RAW_FILES['clients'])
    bookings_df = load_csv(RAW_DATA_DIR / RAW_FILES['bookings'])
    feedback_df = load_csv(RAW_DATA_DIR / RAW_FILES['feedback'])
    comms_df = load_csv(RAW_DATA_DIR / RAW_FILES['communications'])
    churn_df = load_csv(RAW_DATA_DIR / RAW_FILES['churn_labels'])
    
    # Create features
    fe = FeatureEngineer()
    features_df = fe.create_all_features(
        clients_df, bookings_df, feedback_df, comms_df, churn_df
    )
    
    # Preprocess features
    processed_df, label_encoders = preprocess_features(features_df)
    
    # Save processed features
    save_csv(processed_df, PROCESSED_DATA_DIR / PROCESSED_FILES['features'])
    
    return processed_df, label_encoders


if __name__ == "__main__":
    # Run feature engineering pipeline
    features_df, encoders = load_and_engineer_features()
    
    print_header("FEATURE ENGINEERING COMPLETE")
    print(f"\nFeature matrix shape: {features_df.shape}")
    print(f"\nChurn distribution:")
    if 'churned' in features_df.columns:
        print(features_df['churned'].value_counts())
    
    print(f"\nSample features (first 5 rows):")
    print(features_df.head())
