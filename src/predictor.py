# src/predictor.py
"""
Production Prediction Pipeline
==============================

Provides an easy-to-use interface for making churn predictions
on individual clients or batches.

Usage:
    from src.predictor import ChurnPredictor
    
    predictor = ChurnPredictor()
    result = predictor.predict_single(client_features)
    
    # Or for batch predictions
    results = predictor.predict_batch(features_df)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import joblib
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MODELS_DIR, MODEL_FILES, RISK_THRESHOLDS,
    CHURN_PROBABILITY_THRESHOLD, AVG_CLIENT_VALUE
)
from src.utils import (
    get_risk_level, get_risk_emoji, format_currency,
    format_percentage, days_to_readable
)


class ChurnPredictor:
    """
    Production-ready churn prediction system.
    
    This class provides:
    - Single client predictions with explanations
    - Batch predictions for multiple clients
    - Risk categorization (High/Medium/Low)
    - Action recommendations
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor.
        
        Parameters:
            model_path: Path to saved model (uses default if None)
        """
        self.model_path = model_path or MODELS_DIR / MODEL_FILES['xgboost']
        self.threshold = CHURN_PROBABILITY_THRESHOLD
        
        # Load model and artifacts
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and related artifacts."""
        try:
            self.model = joblib.load(self.model_path)
            self.label_encoders = joblib.load(
                MODELS_DIR / MODEL_FILES['label_encoders']
            )
            self.feature_columns = joblib.load(
                MODELS_DIR / MODEL_FILES['feature_columns']
            )
            print(f"✓ Loaded model from {self.model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model not found. Run 'python src/model.py' first to train.\n{e}"
            )
    
    def predict_single(self, client_features):
        """
        Make a prediction for a single client.
        
        Parameters:
            client_features: DataFrame row or dict with client features
        
        Returns:
            Dictionary with prediction results and recommendations
        """
        # Convert to DataFrame if needed
        if isinstance(client_features, dict):
            client_features = pd.DataFrame([client_features])
        elif isinstance(client_features, pd.Series):
            client_features = pd.DataFrame([client_features])
        
        # Ensure we have the right columns
        features = client_features[self.feature_columns].copy()
        
        # Get prediction
        churn_prob = self.model.predict_proba(features)[0][1]
        will_churn = churn_prob >= self.threshold
        risk_level = get_risk_level(churn_prob, RISK_THRESHOLDS)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            client_features.iloc[0], churn_prob, risk_level
        )
        
        # Build result
        result = {
            'churn_probability': float(churn_prob),
            'churn_probability_pct': format_percentage(churn_prob),
            'will_churn': bool(will_churn),
            'risk_level': risk_level,
            'risk_emoji': get_risk_emoji(risk_level),
            'days_to_churn': int(90 * churn_prob) if will_churn else None,
            'estimated_revenue_at_risk': AVG_CLIENT_VALUE if will_churn else 0,
            'recommendations': recommendations,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, features_df):
        """
        Make predictions for multiple clients.
        
        Parameters:
            features_df: DataFrame with client features
        
        Returns:
            DataFrame with original data plus predictions
        """
        # Ensure we have the right columns
        feature_cols = [c for c in self.feature_columns if c in features_df.columns]
        features = features_df[feature_cols].copy()
        
        # Get predictions
        probabilities = self.model.predict_proba(features)[:, 1]
        
        # Create results
        results = features_df.copy()
        results['churn_probability'] = probabilities
        results['will_churn'] = probabilities >= self.threshold
        results['risk_level'] = [
            get_risk_level(p, RISK_THRESHOLDS) for p in probabilities
        ]
        results['revenue_at_risk'] = [
            AVG_CLIENT_VALUE if p >= self.threshold else 0 
            for p in probabilities
        ]
        
        return results.sort_values('churn_probability', ascending=False)
    
    def _generate_recommendations(self, features, churn_prob, risk_level):
        """
        Generate specific action recommendations based on client data.
        
        Parameters:
            features: Client feature values
            churn_prob: Predicted churn probability
            risk_level: Risk categorization
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Check days since last booking
        days_since = features.get('days_since_last_booking', 0)
        if days_since > 90:
            recommendations.append({
                'priority': 'URGENT',
                'action': 'Immediate Contact Required',
                'detail': f'No booking in {int(days_since)} days. Call within 24 hours.',
                'expected_impact': 'High'
            })
        elif days_since > 60:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Schedule Check-in Call',
                'detail': f'Gap of {int(days_since)} days detected. Proactive outreach recommended.',
                'expected_impact': 'Medium-High'
            })
        
        # Check satisfaction
        rating = features.get('avg_overall_rating', 5)
        if pd.notna(rating) and rating < 4.0:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Address Satisfaction Issues',
                'detail': f'Average rating is {rating:.1f}/5.0. Schedule feedback discussion.',
                'expected_impact': 'High'
            })
        elif pd.notna(rating) and rating < 4.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Satisfaction Follow-up',
                'detail': f'Rating of {rating:.1f}/5.0 could be improved. Send satisfaction survey.',
                'expected_impact': 'Medium'
            })
        
        # Check revenue trend
        rev_trend = features.get('revenue_trend', 0)
        if pd.notna(rev_trend) and rev_trend < -0.2:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Explore Upsell Opportunities',
                'detail': 'Spending is declining. Offer training needs assessment.',
                'expected_impact': 'Medium'
            })
        
        # Check booking frequency trend
        freq_trend = features.get('booking_frequency_trend', 0)
        if pd.notna(freq_trend) and freq_trend < -0.3:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Re-engagement Campaign',
                'detail': 'Booking frequency declining significantly. Propose quarterly training plan.',
                'expected_impact': 'Medium-High'
            })
        
        # Check course diversity
        num_categories = features.get('num_course_categories', 1)
        if num_categories <= 1:
            recommendations.append({
                'priority': 'LOW',
                'action': 'Cross-sell Other Courses',
                'detail': 'Client only books one course type. Suggest related programs.',
                'expected_impact': 'Low-Medium'
            })
        
        # Executive escalation for very high risk
        if churn_prob >= 0.85:
            recommendations.insert(0, {
                'priority': 'URGENT',
                'action': 'Executive Escalation',
                'detail': 'Critical churn risk (>85%). Escalate to senior management immediately.',
                'expected_impact': 'Critical'
            })
        
        # Default recommendation if none generated
        if len(recommendations) == 0:
            recommendations.append({
                'priority': 'LOW',
                'action': 'Routine Check-in',
                'detail': 'Send quarterly satisfaction survey or newsletter.',
                'expected_impact': 'Low'
            })
        
        return recommendations
    
    def get_top_risk_clients(self, features_df, top_n=10):
        """
        Get the highest risk clients from a batch.
        
        Parameters:
            features_df: DataFrame with client features
            top_n: Number of top risk clients to return
        
        Returns:
            DataFrame with top risk clients and their predictions
        """
        results = self.predict_batch(features_df)
        return results.head(top_n)
    
    def get_risk_summary(self, features_df):
        """
        Get a summary of risk distribution.
        
        Parameters:
            features_df: DataFrame with client features
        
        Returns:
            Dictionary with risk summary statistics
        """
        results = self.predict_batch(features_df)
        
        risk_counts = results['risk_level'].value_counts()
        total = len(results)
        
        summary = {
            'total_clients': total,
            'high_risk': risk_counts.get('HIGH', 0),
            'medium_risk': risk_counts.get('MEDIUM', 0),
            'low_risk': risk_counts.get('LOW', 0),
            'high_risk_pct': risk_counts.get('HIGH', 0) / total,
            'total_revenue_at_risk': results['revenue_at_risk'].sum(),
            'avg_churn_probability': results['churn_probability'].mean()
        }
        
        return summary


def demo_prediction():
    """
    Demonstrate the prediction pipeline with sample data.
    """
    from src.feature_engineering import load_and_engineer_features
    from src.utils import print_header
    
    print_header("CHURN PREDICTION DEMO")
    
    # Load and process data
    print("\nLoading data and features...")
    features_df, _ = load_and_engineer_features()
    
    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = ChurnPredictor()
    
    # Single prediction example
    print("\n" + "=" * 50)
    print("SINGLE CLIENT PREDICTION")
    print("=" * 50)
    
    sample_client = features_df.iloc[0]
    result = predictor.predict_single(sample_client)
    
    print(f"\nClient ID: {sample_client.get('client_id', 'Unknown')}")
    print(f"Churn Probability: {result['churn_probability_pct']}")
    print(f"Risk Level: {result['risk_emoji']} {result['risk_level']}")
    print(f"Will Churn: {'Yes' if result['will_churn'] else 'No'}")
    print(f"Revenue at Risk: {format_currency(result['estimated_revenue_at_risk'])}")
    
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • [{rec['priority']}] {rec['action']}")
        print(f"    {rec['detail']}")
    
    # Risk summary
    print("\n" + "=" * 50)
    print("PORTFOLIO RISK SUMMARY")
    print("=" * 50)
    
    summary = predictor.get_risk_summary(features_df)
    
    print(f"\nTotal Clients: {summary['total_clients']}")
    print(f"  🔴 High Risk:   {summary['high_risk']} ({format_percentage(summary['high_risk_pct'])})")
    print(f"  🟡 Medium Risk: {summary['medium_risk']}")
    print(f"  🟢 Low Risk:    {summary['low_risk']}")
    print(f"\nTotal Revenue at Risk: {format_currency(summary['total_revenue_at_risk'])}")
    
    # Top 5 highest risk
    print("\n" + "=" * 50)
    print("TOP 5 HIGHEST RISK CLIENTS")
    print("=" * 50)
    
    top_risk = predictor.get_top_risk_clients(features_df, top_n=5)
    for _, client in top_risk.iterrows():
        emoji = get_risk_emoji(client['risk_level'])
        print(f"  {emoji} {client.get('client_id', 'N/A'):8s} "
              f"- {format_percentage(client['churn_probability'])} probability")


if __name__ == "__main__":
    demo_prediction()
