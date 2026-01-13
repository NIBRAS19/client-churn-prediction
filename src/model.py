# src/model.py
"""
Model Training and Evaluation
=============================

Trains an XGBoost classifier to predict client churn.
Includes model evaluation, feature importance analysis,
and threshold optimization.

Usage:
    python src/model.py

This will:
    1. Load and preprocess data
    2. Train the model
    3. Evaluate performance
    4. Save the trained model
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import warnings
import joblib

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
    RAW_FILES, PROCESSED_FILES, MODEL_FILES,
    TEST_SIZE, RANDOM_SEED, XGBOOST_PARAMS, 
    CHURN_PROBABILITY_THRESHOLD, AVG_CLIENT_VALUE, INTERVENTION_COST
)
from src.utils import (
    load_csv, save_csv, save_model, print_header, print_subheader,
    format_currency, format_percentage
)
from src.feature_engineering import FeatureEngineer, preprocess_features


class ChurnModel:
    """
    XGBoost-based churn prediction model.
    
    This class handles:
    - Data loading and preprocessing
    - Model training with hyperparameters
    - Evaluation and metrics
    - Feature importance analysis
    - Business impact calculation
    """
    
    def __init__(self, params=None):
        """
        Initialize the model.
        
        Parameters:
            params: XGBoost hyperparameters (uses defaults from config if None)
        """
        self.params = params or XGBOOST_PARAMS
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.threshold = CHURN_PROBABILITY_THRESHOLD
    
    def load_data(self):
        """
        Load and prepare data for training.
        
        Returns:
            X_train, X_test, y_train, y_test: Split data
        """
        print_header("LOADING DATA")
        
        # Load raw data
        clients = load_csv(RAW_DATA_DIR / RAW_FILES['clients'])
        bookings = load_csv(RAW_DATA_DIR / RAW_FILES['bookings'])
        feedback = load_csv(RAW_DATA_DIR / RAW_FILES['feedback'])
        comms = load_csv(RAW_DATA_DIR / RAW_FILES['communications'])
        churn = load_csv(RAW_DATA_DIR / RAW_FILES['churn_labels'])
        
        # Create features
        print_subheader("Creating Features")
        fe = FeatureEngineer()
        features_df = fe.create_all_features(clients, bookings, feedback, comms, churn)
        
        # Preprocess
        print_subheader("Preprocessing")
        processed_df, self.label_encoders = preprocess_features(features_df)
        
        # Prepare X and y
        self.feature_columns = [col for col in processed_df.columns 
                               if col not in ['client_id', 'churned']]
        
        X = processed_df[self.feature_columns]
        y = processed_df['churned']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_SEED, 
            stratify=y
        )
        
        print(f"\n✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        print(f"✓ Churn rate (train): {y_train.mean():.1%}")
        print(f"✓ Churn rate (test): {y_test.mean():.1%}")
        
        # Save processed data
        save_csv(X_train, PROCESSED_DATA_DIR / PROCESSED_FILES['X_train'])
        save_csv(X_test, PROCESSED_DATA_DIR / PROCESSED_FILES['X_test'])
        save_csv(y_train.to_frame(), PROCESSED_DATA_DIR / PROCESSED_FILES['y_train'])
        save_csv(y_test.to_frame(), PROCESSED_DATA_DIR / PROCESSED_FILES['y_test'])
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train the XGBoost model.
        
        Parameters:
            X_train: Training features
            y_train: Training labels
        """
        print_header("TRAINING MODEL")
        
        # Handle class imbalance with scale_pos_weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        params = self.params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        
        print(f"Class distribution: {neg_count} active, {pos_count} churned")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Create and train model
        self.model = xgb.XGBClassifier(**params)
        
        print("\nTraining XGBoost classifier...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='roc_auc'
        )
        print(f"CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        print("\n✓ Model training complete!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Parameters:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of evaluation metrics
        """
        print_header("MODEL EVALUATION")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Apply custom threshold
        y_pred_threshold = (y_pred_proba >= self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_threshold),
            'precision': precision_score(y_test, y_pred_threshold),
            'recall': recall_score(y_test, y_pred_threshold),
            'f1': f1_score(y_test, y_pred_threshold),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\n📊 PERFORMANCE METRICS")
        print("=" * 50)
        print(f"  Accuracy:  {metrics['accuracy']:.1%} (correct predictions)")
        print(f"  Precision: {metrics['precision']:.1%} (when we predict churn, we're right)")
        print(f"  Recall:    {metrics['recall']:.1%} (we catch this % of actual churners)")
        print(f"  F1 Score:  {metrics['f1']:.3f} (balance of precision/recall)")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f} (discrimination ability)")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_threshold)
        print("\n📋 CONFUSION MATRIX")
        print("=" * 50)
        print(f"  True Negatives:  {cm[0,0]:4d} (correctly predicted active)")
        print(f"  False Positives: {cm[0,1]:4d} (false alarms)")
        print(f"  False Negatives: {cm[1,0]:4d} (missed churners)")
        print(f"  True Positives:  {cm[1,1]:4d} (correctly predicted churn)")
        
        # Classification report
        print("\n📝 CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(y_test, y_pred_threshold, 
                                   target_names=['Active', 'Churned']))
        
        # Save evaluation plots
        if PLOTTING_AVAILABLE:
            self._plot_evaluation(y_test, y_pred_proba, cm)
        
        return metrics
    
    def _plot_evaluation(self, y_test, y_pred_proba, cm):
        """Generate and save evaluation plots."""
        print("\nGenerating evaluation plots...")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: ROC Curve
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1].plot(recall, precision, color='blue', lw=2)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                   xticklabels=['Active', 'Churned'],
                   yticklabels=['Active', 'Churned'])
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'model_evaluation.png', dpi=150)
        plt.close()
        
        print(f"✓ Saved evaluation plots to {OUTPUTS_DIR / 'model_evaluation.png'}")
    
    def analyze_feature_importance(self, top_n=15):
        """
        Analyze and display feature importance.
        
        Parameters:
            top_n: Number of top features to display
        """
        print_header("FEATURE IMPORTANCE")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print("=" * 50)
        
        for idx, row in importance.head(top_n).iterrows():
            bar = "▓" * int(row['importance'] * 50)
            print(f"  {row['feature'][:30]:30s} {row['importance']:.3f} {bar}")
        
        # Save to file
        importance.to_csv(OUTPUTS_DIR / 'feature_importance.csv', index=False)
        
        # Plot if available
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(10, 8))
            top_features = importance.head(top_n)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Features for Churn Prediction')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR / 'feature_importance.png', dpi=150)
            plt.close()
            print(f"\n✓ Saved feature importance plot to {OUTPUTS_DIR / 'feature_importance.png'}")
        
        return importance
    
    def calculate_business_impact(self, y_test, y_pred_proba):
        """
        Calculate business impact of the model.
        
        Parameters:
            y_test: Actual labels
            y_pred_proba: Predicted probabilities
        """
        print_header("BUSINESS IMPACT ANALYSIS")
        
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate financial metrics
        saved_revenue = tp * AVG_CLIENT_VALUE
        intervention_costs = tp * INTERVENTION_COST
        lost_revenue = fn * AVG_CLIENT_VALUE
        wasted_costs = fp * INTERVENTION_COST
        
        net_benefit = saved_revenue - intervention_costs - wasted_costs
        total_intervention_cost = intervention_costs + wasted_costs
        roi = net_benefit / total_intervention_cost if total_intervention_cost > 0 else 0
        
        print("\n💰 FINANCIAL IMPACT (Test Set)")
        print("=" * 50)
        print(f"\n  Clients correctly identified as at-risk: {tp}")
        print(f"  Clients missed (false negatives):        {fn}")
        print(f"  False alarms (false positives):          {fp}")
        
        print(f"\n  Potential Revenue Saved:    {format_currency(saved_revenue)}")
        print(f"  Lost Revenue (missed):      {format_currency(lost_revenue)}")
        print(f"  Intervention Costs:         {format_currency(intervention_costs)}")
        print(f"  Wasted Costs (false alarms):{format_currency(wasted_costs)}")
        
        print(f"\n  NET BENEFIT:                {format_currency(net_benefit)}")
        print(f"  ROI:                        {roi:.1f}:1")
        
        # Annualized projection (assuming test set is representative sample)
        sample_ratio = len(y_test) / 500  # Assuming 500 total clients
        annual_benefit = net_benefit / sample_ratio
        
        print(f"\n📈 ANNUALIZED PROJECTION")
        print("=" * 50)
        print(f"  Estimated Annual Benefit:   {format_currency(annual_benefit)}")
        print(f"  Estimated Clients Saved:    {int(tp / sample_ratio)} per year")
    
    def save(self):
        """Save the trained model and associated artifacts."""
        print_header("SAVING MODEL")
        
        # Save main model
        save_model(self.model, MODELS_DIR / MODEL_FILES['xgboost'])
        
        # Save label encoders
        save_model(self.label_encoders, MODELS_DIR / MODEL_FILES['label_encoders'])
        
        # Save feature columns
        save_model(self.feature_columns, MODELS_DIR / MODEL_FILES['feature_columns'])
        
        print("\n✓ All model artifacts saved successfully!")


def train_and_evaluate():
    """
    Main function to train and evaluate the churn model.
    
    Returns:
        Trained ChurnModel instance
    """
    print_header("CLIENT CHURN PREDICTION MODEL")
    print("Select Training & Management Consultancy LLC")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize model
    model = ChurnModel()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = model.load_data()
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    # Feature importance
    importance = model.analyze_feature_importance()
    
    # Business impact
    y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    model.calculate_business_impact(y_test.values, y_pred_proba)
    
    # Save model
    model.save()
    
    print_header("TRAINING COMPLETE")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext Steps:")
    print("1. Run: streamlit run app.py  (to launch dashboard)")
    print("2. Or use the predictor module for batch predictions")
    
    return model


if __name__ == "__main__":
    model = train_and_evaluate()
