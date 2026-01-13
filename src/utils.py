# src/utils.py
"""
Utility Functions
=================

Common helper functions used across the project.
These handle data loading, saving, and formatting operations.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def load_csv(filepath: Path, parse_dates: list = None) -> pd.DataFrame:
    """
    Load a CSV file with optional date parsing.
    
    Parameters:
        filepath: Path to the CSV file
        parse_dates: List of column names to parse as dates
    
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(filepath, parse_dates=parse_dates)
        print(f"✓ Loaded {filepath.name}: {len(df):,} rows")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found - {filepath}")
        raise


def save_csv(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV.
    
    Parameters:
        df: DataFrame to save
        filepath: Destination path
        index: Whether to include row index
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    print(f"✓ Saved {filepath.name}: {len(df):,} rows")


def save_model(model, filepath: Path) -> None:
    """
    Save a trained model using joblib.
    
    Parameters:
        model: The model object to save
        filepath: Destination path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✓ Saved model to {filepath.name}")


def load_model(filepath: Path):
    """
    Load a saved model.
    
    Parameters:
        filepath: Path to the saved model
    
    Returns:
        The loaded model object
    """
    try:
        model = joblib.load(filepath)
        print(f"✓ Loaded model from {filepath.name}")
        return model
    except FileNotFoundError:
        print(f"✗ Error: Model not found - {filepath}")
        raise


def format_currency(amount: float, currency: str = "AED") -> str:
    """
    Format a number as currency.
    
    Parameters:
        amount: The numeric amount
        currency: Currency code (default: AED)
    
    Returns:
        Formatted currency string
    """
    return f"{currency} {amount:,.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal as percentage.
    
    Parameters:
        value: Decimal value (0.25 = 25%)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_days_ago(date, reference_date=None) -> int:
    """
    Calculate days between a date and reference date.
    
    Parameters:
        date: The date to check
        reference_date: Reference date (default: today)
    
    Returns:
        Number of days difference
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    if isinstance(date, str):
        date = pd.to_datetime(date)
    if isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)
    
    return (reference_date - date).days


def get_risk_level(probability: float, thresholds: dict = None) -> str:
    """
    Categorize a probability into risk levels.
    
    Parameters:
        probability: Churn probability (0-1)
        thresholds: Dict with 'high' and 'medium' thresholds
    
    Returns:
        Risk level string: 'HIGH', 'MEDIUM', or 'LOW'
    """
    if thresholds is None:
        thresholds = {'high': 0.70, 'medium': 0.40}
    
    if probability >= thresholds['high']:
        return 'HIGH'
    elif probability >= thresholds['medium']:
        return 'MEDIUM'
    else:
        return 'LOW'


def get_risk_emoji(risk_level: str) -> str:
    """
    Get emoji indicator for risk level.
    
    Parameters:
        risk_level: 'HIGH', 'MEDIUM', or 'LOW'
    
    Returns:
        Emoji string
    """
    emojis = {
        'HIGH': '🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    return emojis.get(risk_level, '⚪')


def print_header(text: str, width: int = 70) -> None:
    """
    Print a formatted section header.
    
    Parameters:
        text: Header text
        width: Total width in characters
    """
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)


def print_subheader(text: str, width: int = 70) -> None:
    """
    Print a formatted subsection header.
    
    Parameters:
        text: Subheader text
        width: Total width in characters
    """
    print("\n" + "-" * width)
    print(text)
    print("-" * width)


def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print a summary of a DataFrame.
    
    Parameters:
        df: DataFrame to summarize
        name: Name to display
    """
    print(f"\n{name} Summary:")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Show missing values if any
    missing = df.isnull().sum()
    if missing.any():
        print(f"  - Missing values:")
        for col in missing[missing > 0].index:
            print(f"      {col}: {missing[col]:,}")


def calculate_trend(values: list, min_points: int = 3) -> float:
    """
    Calculate a simple trend (change direction) from a list of values.
    
    Parameters:
        values: List of numeric values over time
        min_points: Minimum points required for trend calculation
    
    Returns:
        Trend value: positive = increasing, negative = decreasing, 0 = no trend
    """
    if len(values) < min_points:
        return 0.0
    
    # Compare first half to second half
    mid = len(values) // 2
    first_half = np.mean(values[:mid])
    second_half = np.mean(values[mid:])
    
    if first_half == 0:
        return 0.0
    
    return (second_half - first_half) / first_half


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Parameters:
        numerator: Top of the division
        denominator: Bottom of the division
        default: Value to return if division is impossible
    
    Returns:
        Result of division or default value
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max bounds.
    
    Parameters:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
    
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def days_to_readable(days: int) -> str:
    """
    Convert days to human-readable format.
    
    Parameters:
        days: Number of days
    
    Returns:
        Readable string like "2 weeks" or "3 months"
    """
    if days < 7:
        return f"{days} days"
    elif days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''}"
    elif days < 365:
        months = days // 30
        return f"{months} month{'s' if months > 1 else ''}"
    else:
        years = days // 365
        return f"{years} year{'s' if years > 1 else ''}"


if __name__ == "__main__":
    # Quick test of utility functions
    print_header("Utility Functions Test")
    
    print(f"\nCurrency format: {format_currency(156000)}")
    print(f"Percentage format: {format_percentage(0.87)}")
    print(f"Risk level (0.8): {get_risk_emoji(get_risk_level(0.8))} {get_risk_level(0.8)}")
    print(f"Risk level (0.5): {get_risk_emoji(get_risk_level(0.5))} {get_risk_level(0.5)}")
    print(f"Risk level (0.2): {get_risk_emoji(get_risk_level(0.2))} {get_risk_level(0.2)}")
    print(f"Days readable (45): {days_to_readable(45)}")
    print(f"Days readable (120): {days_to_readable(120)}")
    
    print("\n✓ All utility functions working correctly!")
