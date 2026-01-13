# src/generate_data.py
"""
Synthetic Data Generator
========================

Generates realistic demo data for the Client Churn Prediction System.
This creates 5 interconnected CSV files that mimic real corporate training data.

Usage:
    python src/generate_data.py

Output Files:
    - data/raw/clients.csv          (500 client records)
    - data/raw/bookings.csv         (~3,000 booking records)
    - data/raw/feedback.csv         (~2,500 feedback records)
    - data/raw/communications.csv   (~5,000 communication logs)
    - data/raw/churn_labels.csv     (churn status for all clients)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_DIR, RANDOM_SEED, N_CLIENTS, CHURN_THRESHOLD_DAYS,
    CLIENT_BEHAVIOR_PROBS, INDUSTRIES, COMPANY_SIZES, REGIONS,
    ACCOUNT_MANAGERS, COURSE_CATALOG, RAW_FILES
)
from src.utils import print_header, print_subheader, save_csv


# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_company_name() -> str:
    """
    Generate a realistic UAE company name.
    
    Returns:
        A randomly generated company name
    """
    prefixes = [
        'Al', 'Emirates', 'Abu Dhabi', 'Dubai', 'National', 'Gulf',
        'Arabian', 'United', 'International', 'Global', 'Middle East'
    ]
    suffixes = [
        'Group', 'Holdings', 'Corporation', 'Company', 'Industries',
        'Enterprises', 'Solutions', 'Services', 'International', 'LLC'
    ]
    types = [
        'Bank', 'Petroleum', 'Aviation', 'Trading', 'Construction',
        'Development', 'Investment', 'Insurance', 'Healthcare', 'Education',
        'Technology', 'Manufacturing', 'Real Estate', 'Logistics'
    ]
    
    structure = random.choice([
        f"{random.choice(prefixes)} {random.choice(types)} {random.choice(suffixes)}",
        f"{random.choice(prefixes)} {random.choice(suffixes)}",
        f"{random.choice(types)} {random.choice(suffixes)}"
    ])
    return structure


def generate_clients(n_clients: int = N_CLIENTS) -> pd.DataFrame:
    """
    Generate client master data.
    
    Parameters:
        n_clients: Number of clients to generate
    
    Returns:
        DataFrame with client information
    """
    print_subheader("1. Generating Client Master Data")
    
    clients_data = []
    
    for i in range(1, n_clients + 1):
        # First booking date (spread over last 5 years)
        days_ago = random.randint(30, 1825)  # 1 month to 5 years
        first_booking = datetime.now() - timedelta(days=days_ago)
        
        client = {
            'client_id': f'C{str(i).zfill(4)}',
            'client_name': generate_company_name(),
            'industry_sector': random.choice(INDUSTRIES),
            'company_size': random.choice(COMPANY_SIZES),
            'first_booking_date': first_booking.strftime('%Y-%m-%d'),
            'account_manager': random.choice(ACCOUNT_MANAGERS),
            'region': random.choice(REGIONS)
        }
        clients_data.append(client)
    
    clients_df = pd.DataFrame(clients_data)
    print(f"✓ Generated {len(clients_df)} clients")
    
    return clients_df


def generate_bookings(clients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate booking history based on client behavior patterns.
    
    Parameters:
        clients_df: DataFrame containing client information
    
    Returns:
        DataFrame with booking history
    """
    print_subheader("2. Generating Booking History")
    
    # Flatten course catalog
    all_courses = []
    for category, info in COURSE_CATALOG.items():
        for course in info['courses']:
            all_courses.append({
                'course_name': course,
                'course_category': category,
                'price_range': info['price_range']
            })
    
    bookings_data = []
    booking_id = 1
    
    behaviors = list(CLIENT_BEHAVIOR_PROBS.keys())
    behavior_probs = list(CLIENT_BEHAVIOR_PROBS.values())
    
    for _, client in clients_df.iterrows():
        client_id = client['client_id']
        first_booking_date = pd.to_datetime(client['first_booking_date'])
        
        # Assign behavior pattern to this client
        client_behavior = np.random.choice(behaviors, p=behavior_probs)
        
        # Set booking parameters based on behavior
        if client_behavior == 'loyal':
            n_bookings = random.randint(8, 20)
            avg_gap_days = random.randint(30, 60)
            last_booking_days_ago = random.randint(5, 45)
            
        elif client_behavior == 'declining':
            n_bookings = random.randint(6, 15)
            avg_gap_days = random.randint(40, 90)
            last_booking_days_ago = random.randint(100, 150)
            
        elif client_behavior == 'sporadic':
            n_bookings = random.randint(3, 10)
            avg_gap_days = random.randint(60, 120)
            last_booking_days_ago = random.randint(30, 90)
            
        elif client_behavior == 'new':
            n_bookings = random.randint(1, 4)
            avg_gap_days = random.randint(20, 60)
            last_booking_days_ago = random.randint(5, 30)
            
        else:  # dormant
            n_bookings = random.randint(2, 8)
            avg_gap_days = random.randint(60, 120)
            last_booking_days_ago = random.randint(180, 400)
        
        # Calculate the last booking date
        current_date = datetime.now() - timedelta(days=last_booking_days_ago)
        
        # Generate booking dates
        booking_dates = []
        for i in range(n_bookings):
            if i == 0:
                booking_date = first_booking_date
            else:
                gap = avg_gap_days + random.randint(-15, 15)
                gap = max(7, gap)  # Minimum 7 days between bookings
                booking_date = booking_dates[-1] + timedelta(days=gap)
                
                if booking_date > current_date:
                    break
            
            booking_dates.append(booking_date)
        
        # Create booking records
        for booking_date in booking_dates:
            course = random.choice(all_courses)
            
            # Participants based on company size
            if 'Enterprise' in client['company_size']:
                participants = random.randint(15, 35)
            elif 'Large' in client['company_size']:
                participants = random.randint(10, 25)
            elif 'Medium' in client['company_size']:
                participants = random.randint(5, 15)
            else:
                participants = random.randint(2, 10)
            
            # Calculate revenue
            price_per_person = random.randint(*course['price_range'])
            
            # Volume discount for 10+ participants
            if participants >= 10:
                price_per_person = int(price_per_person * 0.85)
            
            revenue = price_per_person * participants
            
            # Payment status
            payment_delay = random.randint(0, 30)
            payment_date = booking_date + timedelta(days=payment_delay)
            
            if payment_delay > 20:
                payment_status = random.choice(['Pending', 'Overdue'])
            else:
                payment_status = 'Paid'
            
            booking = {
                'booking_id': f'B{str(booking_id).zfill(5)}',
                'client_id': client_id,
                'booking_date': booking_date.strftime('%Y-%m-%d'),
                'course_name': course['course_name'],
                'course_category': course['course_category'],
                'number_of_participants': participants,
                'revenue': revenue,
                'payment_date': payment_date.strftime('%Y-%m-%d'),
                'payment_status': payment_status
            }
            
            bookings_data.append(booking)
            booking_id += 1
    
    bookings_df = pd.DataFrame(bookings_data)
    
    print(f"✓ Generated {len(bookings_df):,} bookings")
    print(f"  Average bookings per client: {len(bookings_df) / len(clients_df):.1f}")
    print(f"  Total revenue: AED {bookings_df['revenue'].sum():,.0f}")
    
    return bookings_df


def generate_feedback(bookings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate feedback data for bookings.
    
    Parameters:
        bookings_df: DataFrame containing booking history
    
    Returns:
        DataFrame with feedback records
    """
    print_subheader("3. Generating Feedback Data")
    
    positive_comments = [
        "Excellent trainer, very practical content",
        "Highly engaging and interactive session",
        "Great real-world examples and case studies",
        "Trainer was knowledgeable and approachable",
        "Very useful tools and frameworks provided",
        "Good pace, well-structured content",
        "Exceeded expectations, highly recommend"
    ]
    
    neutral_comments = [
        "Good course overall, met expectations",
        "Useful content, could use more examples",
        "Trainer was knowledgeable",
        "Decent workshop, learned some new things"
    ]
    
    negative_comments = [
        "Pace was too fast, hard to follow",
        "Too theoretical, needed more practical exercises",
        "Content was basic, expected more depth",
        "Trainer rushed through important topics",
        "Not enough time for questions"
    ]
    
    feedback_data = []
    
    for _, booking in bookings_df.iterrows():
        # 80% feedback response rate
        if random.random() > 0.20:
            booking_date = pd.to_datetime(booking['booking_date'])
            feedback_date = booking_date + timedelta(days=random.randint(1, 7))
            
            # Base rating (generally positive)
            base_rating = random.uniform(3.5, 5.0)
            
            # Overall rating
            overall_rating = round(base_rating + random.uniform(-0.3, 0.3), 1)
            overall_rating = max(1.0, min(5.0, overall_rating))
            
            # Trainer rating
            trainer_rating = round(base_rating + random.uniform(-0.2, 0.4), 1)
            trainer_rating = max(1.0, min(5.0, trainer_rating))
            
            # NPS score (correlated with ratings)
            if overall_rating >= 4.5:
                nps_score = random.randint(9, 10)
            elif overall_rating >= 3.5:
                nps_score = random.randint(7, 8)
            else:
                nps_score = random.randint(0, 6)
            
            # Comment selection
            if overall_rating >= 4.5:
                comment = random.choice(positive_comments) if random.random() > 0.3 else None
            elif overall_rating >= 3.5:
                comment = random.choice(neutral_comments) if random.random() > 0.5 else None
            else:
                comment = random.choice(negative_comments) if random.random() > 0.4 else None
            
            feedback = {
                'feedback_id': f'F{str(len(feedback_data) + 1).zfill(5)}',
                'client_id': booking['client_id'],
                'booking_id': booking['booking_id'],
                'feedback_date': feedback_date.strftime('%Y-%m-%d'),
                'overall_rating': overall_rating,
                'nps_score': nps_score,
                'trainer_rating': trainer_rating,
                'comments': comment
            }
            
            feedback_data.append(feedback)
    
    feedback_df = pd.DataFrame(feedback_data)
    
    print(f"✓ Generated {len(feedback_df):,} feedback records")
    print(f"  Response rate: {len(feedback_df) / len(bookings_df) * 100:.1f}%")
    print(f"  Average rating: {feedback_df['overall_rating'].mean():.2f}/5.0")
    print(f"  Average NPS: {feedback_df['nps_score'].mean():.1f}/10")
    
    return feedback_df


def generate_communications(clients_df: pd.DataFrame, bookings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate communication log data.
    
    Parameters:
        clients_df: DataFrame containing client information
        bookings_df: DataFrame containing booking history
    
    Returns:
        DataFrame with communication records
    """
    print_subheader("4. Generating Communication Log")
    
    contact_types = ['Email', 'Phone Call', 'Meeting', 'Video Call', 'WhatsApp']
    contact_reasons = [
        'Follow-up on booking', 'Training needs assessment', 'Quotation request',
        'Feedback discussion', 'Course recommendation', 'Invoice query',
        'Schedule confirmation', 'General inquiry', 'Renewal discussion',
        'Complaint resolution', 'Thank you message', 'New course announcement',
        'Satisfaction survey', 'Account review'
    ]
    
    communication_data = []
    
    for _, client in clients_df.iterrows():
        client_id = client['client_id']
        
        # Get client's booking history
        client_bookings = bookings_df[bookings_df['client_id'] == client_id]
        
        if len(client_bookings) == 0:
            continue
        
        # Number of communications (1-3 per booking + some proactive)
        n_communications = len(client_bookings) * random.randint(1, 3) + random.randint(0, 5)
        
        first_booking = pd.to_datetime(client_bookings['booking_date'].min())
        
        for i in range(n_communications):
            # Communication date
            date_range = (datetime.now() - first_booking).days
            if date_range > 0:
                contact_date = first_booking + timedelta(days=random.randint(0, date_range + 30))
            else:
                contact_date = first_booking
            
            # Don't create future communications
            if contact_date > datetime.now():
                contact_date = datetime.now() - timedelta(days=random.randint(1, 30))
            
            communication = {
                'communication_id': f'COM{str(len(communication_data) + 1).zfill(5)}',
                'client_id': client_id,
                'contact_date': contact_date.strftime('%Y-%m-%d'),
                'contact_type': random.choice(contact_types),
                'contact_reason': random.choice(contact_reasons),
                'contacted_by': client['account_manager']
            }
            
            communication_data.append(communication)
    
    communication_df = pd.DataFrame(communication_data)
    communication_df = communication_df.sort_values('contact_date')
    
    print(f"✓ Generated {len(communication_df):,} communication records")
    print(f"  Average contacts per client: {len(communication_df) / len(clients_df):.1f}")
    
    return communication_df


def generate_churn_labels(clients_df: pd.DataFrame, bookings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate churn labels based on booking activity.
    
    A client is considered churned if they haven't made a booking
    in the last CHURN_THRESHOLD_DAYS days.
    
    Parameters:
        clients_df: DataFrame containing client information
        bookings_df: DataFrame containing booking history
    
    Returns:
        DataFrame with churn labels
    """
    print_subheader("5. Generating Churn Labels")
    
    reference_date = datetime.now()
    cutoff_date = reference_date - timedelta(days=CHURN_THRESHOLD_DAYS)
    
    churn_labels = []
    
    for _, client in clients_df.iterrows():
        client_id = client['client_id']
        
        # Get client's bookings
        client_bookings = bookings_df[bookings_df['client_id'] == client_id]
        
        if len(client_bookings) == 0:
            # No bookings = churned
            churned = 'Yes'
            last_booking_date = None
            churn_date = None
        else:
            last_booking_date = pd.to_datetime(client_bookings['booking_date'].max())
            
            if last_booking_date < cutoff_date:
                churned = 'Yes'
                churn_date = last_booking_date + timedelta(days=CHURN_THRESHOLD_DAYS)
            else:
                churned = 'No'
                churn_date = None
        
        label = {
            'client_id': client_id,
            'last_booking_date': last_booking_date.strftime('%Y-%m-%d') if last_booking_date else None,
            'churned': churned,
            'churn_date': churn_date.strftime('%Y-%m-%d') if churn_date else None
        }
        
        churn_labels.append(label)
    
    churn_df = pd.DataFrame(churn_labels)
    churn_rate = (churn_df['churned'] == 'Yes').sum() / len(churn_df) * 100
    
    print(f"✓ Generated churn labels for {len(churn_df)} clients")
    print(f"  Churn rate: {churn_rate:.1f}%")
    print(f"  Active clients: {(churn_df['churned'] == 'No').sum()}")
    print(f"  Churned clients: {(churn_df['churned'] == 'Yes').sum()}")
    
    return churn_df


def print_summary(clients_df, bookings_df, feedback_df, communication_df, churn_df):
    """
    Print a comprehensive summary of generated data.
    """
    print_header("DATASET SUMMARY")
    
    print(f"\n📊 CLIENTS: {len(clients_df)}")
    print(f"   Industries: {clients_df['industry_sector'].nunique()}")
    print(f"   Regions: {clients_df['region'].nunique()}")
    print(f"   Account Managers: {clients_df['account_manager'].nunique()}")
    
    print(f"\n📅 BOOKINGS: {len(bookings_df):,}")
    print(f"   Date Range: {bookings_df['booking_date'].min()} to {bookings_df['booking_date'].max()}")
    print(f"   Course Categories: {bookings_df['course_category'].nunique()}")
    print(f"   Unique Courses: {bookings_df['course_name'].nunique()}")
    print(f"   Total Revenue: AED {bookings_df['revenue'].sum():,.0f}")
    print(f"   Average Deal Size: AED {bookings_df['revenue'].mean():,.0f}")
    
    print(f"\n⭐ FEEDBACK: {len(feedback_df):,}")
    print(f"   Response Rate: {len(feedback_df) / len(bookings_df) * 100:.1f}%")
    print(f"   Average Overall Rating: {feedback_df['overall_rating'].mean():.2f}/5.0")
    print(f"   Average NPS Score: {feedback_df['nps_score'].mean():.1f}/10")
    print(f"   With Comments: {feedback_df['comments'].notna().sum()}")
    
    print(f"\n📞 COMMUNICATIONS: {len(communication_df):,}")
    print(f"   Date Range: {communication_df['contact_date'].min()} to {communication_df['contact_date'].max()}")
    print(f"   Average per Client: {len(communication_df) / len(clients_df):.1f}")
    
    churn_rate = (churn_df['churned'] == 'Yes').sum() / len(churn_df) * 100
    print(f"\n🎯 CHURN STATUS:")
    print(f"   Active Clients: {(churn_df['churned'] == 'No').sum()} ({100 - churn_rate:.1f}%)")
    print(f"   Churned Clients: {(churn_df['churned'] == 'Yes').sum()} ({churn_rate:.1f}%)")


def main():
    """
    Main function to generate all datasets.
    """
    print_header("GENERATING DEMO DATA FOR CHURN PREDICTION SYSTEM")
    print("Select Training & Management Consultancy LLC")
    
    # Generate all datasets
    clients_df = generate_clients()
    bookings_df = generate_bookings(clients_df)
    feedback_df = generate_feedback(bookings_df)
    communication_df = generate_communications(clients_df, bookings_df)
    churn_df = generate_churn_labels(clients_df, bookings_df)
    
    # Save all files
    print_header("SAVING FILES")
    
    save_csv(clients_df, RAW_DATA_DIR / RAW_FILES['clients'])
    save_csv(bookings_df, RAW_DATA_DIR / RAW_FILES['bookings'])
    save_csv(feedback_df, RAW_DATA_DIR / RAW_FILES['feedback'])
    save_csv(communication_df, RAW_DATA_DIR / RAW_FILES['communications'])
    save_csv(churn_df, RAW_DATA_DIR / RAW_FILES['churn_labels'])
    
    # Print summary
    print_summary(clients_df, bookings_df, feedback_df, communication_df, churn_df)
    
    print_header("✅ DEMO DATA GENERATION COMPLETE!")
    print("\nNext Steps:")
    print("1. Review the CSV files in data/raw/")
    print("2. Run: python src/model.py  (to train the model)")
    print("3. Run: streamlit run app.py  (to launch dashboard)")
    
    return clients_df, bookings_df, feedback_df, communication_df, churn_df


if __name__ == "__main__":
    main()
