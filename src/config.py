# src/config.py
"""
Configuration Settings
======================

Central configuration for paths, parameters, and constants.
Modify these values to customize the system behavior.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory (automatically detected)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Output directory (for visualizations and reports)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA GENERATION SETTINGS
# =============================================================================

# Number of synthetic clients to generate
N_CLIENTS = 500

# Random seed for reproducibility
RANDOM_SEED = 42

# Client behavior distribution (must sum to 1.0)
CLIENT_BEHAVIOR_PROBS = {
    'loyal': 0.40,      # Regular bookings, low churn risk
    'declining': 0.25,  # Was active, now declining (HIGH churn risk)
    'sporadic': 0.20,   # Irregular patterns
    'new': 0.10,        # Recently joined
    'dormant': 0.05     # Already churned
}


# =============================================================================
# CHURN DEFINITION
# =============================================================================

# A client is considered "churned" if no booking in this many days
CHURN_THRESHOLD_DAYS = 90


# =============================================================================
# FEATURE ENGINEERING SETTINGS
# =============================================================================

# Time windows for aggregating features (in days)
TIME_WINDOWS = [30, 90, 180]

# Minimum bookings required for reliable trend calculation
MIN_BOOKINGS_FOR_TREND = 3


# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Train/test split ratio
TEST_SIZE = 0.2

# XGBoost hyperparameters (tuned for this dataset)
XGBOOST_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# Probability threshold for churn classification
# (optimized for business impact - balances precision and recall)
CHURN_PROBABILITY_THRESHOLD = 0.45


# =============================================================================
# RISK LEVEL THRESHOLDS
# =============================================================================

# Probability thresholds for risk categorization
RISK_THRESHOLDS = {
    'high': 0.70,    # >= 70% probability = HIGH RISK
    'medium': 0.40,  # >= 40% probability = MEDIUM RISK
    # < 40% = LOW RISK
}


# =============================================================================
# BUSINESS PARAMETERS
# =============================================================================

# Average annual value of a client (in AED)
AVG_CLIENT_VALUE = 156000

# Cost of intervention per client (in AED)
INTERVENTION_COST = 5000


# =============================================================================
# UAE-SPECIFIC DATA
# =============================================================================

# Industry sectors for synthetic data
INDUSTRIES = [
    'Banking & Finance', 'Oil & Gas', 'Government', 'Healthcare',
    'Education', 'Retail', 'Technology', 'Construction',
    'Real Estate', 'Tourism & Hospitality', 'Manufacturing',
    'Telecommunications', 'Aviation', 'Logistics'
]

# Company size categories
COMPANY_SIZES = [
    'Small (1-50)',
    'Medium (51-200)',
    'Large (201-500)',
    'Enterprise (500+)'
]

# UAE regions
REGIONS = ['Abu Dhabi', 'Dubai', 'Sharjah', 'Ajman', 'Ras Al Khaimah', 'Fujairah']

# Account managers
ACCOUNT_MANAGERS = ['Fatima Ahmed', 'Mohammed Ali', 'Sara Hassan', 'Ahmed Abdullah', 'Noura Salem']

# Course categories and their base prices (AED per participant)
COURSE_CATALOG = {
    'Leadership & Management': {
        'courses': [
            'Leadership Excellence Program', 'Strategic Leadership',
            'Team Management Essentials', 'Executive Leadership',
            'Change Management', 'Performance Management'
        ],
        'price_range': (7000, 12000)
    },
    'Project Management': {
        'courses': [
            'PMP Certification Prep', 'Agile & Scrum Fundamentals',
            'Project Planning & Execution', 'Risk Management',
            'Prince2 Foundation'
        ],
        'price_range': (8000, 15000)
    },
    'Human Resources': {
        'courses': [
            'HR Management Essentials', 'Talent Acquisition',
            'Performance Appraisal', 'Employee Relations',
            'Compensation & Benefits', 'HR Analytics'
        ],
        'price_range': (6000, 10000)
    },
    'Finance & Accounting': {
        'courses': [
            'Finance for Non-Finance', 'Financial Analysis',
            'Budgeting & Forecasting', 'Internal Audit'
        ],
        'price_range': (6500, 11000)
    },
    'Communication Skills': {
        'courses': [
            'Business Communication', 'Presentation Skills',
            'Negotiation Skills', 'Conflict Resolution',
            'Email Writing Mastery'
        ],
        'price_range': (3500, 7000)
    },
    'Customer Service': {
        'courses': [
            'Customer Service Excellence', 'Complaint Handling',
            'Service Quality Management'
        ],
        'price_range': (3000, 6000)
    },
    'Sales & Marketing': {
        'courses': [
            'Sales Fundamentals', 'Digital Marketing',
            'Strategic Marketing', 'Key Account Management'
        ],
        'price_range': (5000, 9000)
    },
    'AI & Technology': {
        'courses': [
            'AI for Business Leaders', 'Generative AI Essentials',
            'Data Literacy for Leaders', 'Digital Transformation',
            'Prompt Engineering for Professionals'
        ],
        'price_range': (6500, 13000)
    },
    'Personal Development': {
        'courses': [
            'Time Management', 'Emotional Intelligence',
            'Critical Thinking', 'Problem Solving'
        ],
        'price_range': (3000, 6000)
    },
    'Administration': {
        'courses': [
            'Executive Assistant Excellence', 'Office Management',
            'Business Writing', 'Advanced Excel'
        ],
        'price_range': (3500, 6500)
    },
    'Team Building': {
        'courses': [
            'Team Building Workshop', 'Virtual Team Building',
            'Leadership Team Offsite', 'Innovation Workshop'
        ],
        'price_range': (4000, 8000)
    },
    'Compliance & Quality': {
        'courses': [
            'ISO 9001 Quality Management', 'Risk & Compliance',
            'Internal Control Systems'
        ],
        'price_range': (7000, 12000)
    }
}


# =============================================================================
# FILE NAMES
# =============================================================================

# Raw data files
RAW_FILES = {
    'clients': 'clients.csv',
    'bookings': 'bookings.csv',
    'feedback': 'feedback.csv',
    'communications': 'communications.csv',
    'churn_labels': 'churn_labels.csv'
}

# Processed data files
PROCESSED_FILES = {
    'features': 'features.csv',
    'X_train': 'X_train.csv',
    'X_test': 'X_test.csv',
    'y_train': 'y_train.csv',
    'y_test': 'y_test.csv'
}

# Model files
MODEL_FILES = {
    'xgboost': 'churn_model_xgboost.pkl',
    'pipeline': 'churn_predictor_pipeline.pkl',
    'label_encoders': 'label_encoders.pkl',
    'feature_columns': 'feature_columns.pkl'
}
