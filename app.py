# app.py
"""
Streamlit Dashboard for Client Churn Prediction
================================================

Interactive dashboard for exploring churn predictions,
viewing at-risk clients, and generating recommendations.

Usage:
    streamlit run app.py

Then open http://localhost:8501 in your browser.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Client Churn Prediction",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import project modules
try:
    from src.config import (
        RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
        RAW_FILES, PROCESSED_FILES, AVG_CLIENT_VALUE
    )
    from src.utils import format_currency, format_percentage, get_risk_emoji
    from src.predictor import ChurnPredictor
    from src.feature_engineering import FeatureEngineer, preprocess_features
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure you're running from the project root directory.")
    st.stop()

# Try importing visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Some visualizations may be limited.")


# =============================================================================
# DATA LOADING (with caching)
# =============================================================================

@st.cache_data
def load_raw_data():
    """Load raw datasets."""
    try:
        clients = pd.read_csv(RAW_DATA_DIR / RAW_FILES['clients'])
        bookings = pd.read_csv(RAW_DATA_DIR / RAW_FILES['bookings'])
        feedback = pd.read_csv(RAW_DATA_DIR / RAW_FILES['feedback'])
        comms = pd.read_csv(RAW_DATA_DIR / RAW_FILES['communications'])
        churn = pd.read_csv(RAW_DATA_DIR / RAW_FILES['churn_labels'])
        return clients, bookings, feedback, comms, churn
    except FileNotFoundError:
        return None, None, None, None, None


@st.cache_data
def load_features():
    """Load or create features."""
    try:
        features = pd.read_csv(PROCESSED_DATA_DIR / PROCESSED_FILES['features'])
        return features
    except FileNotFoundError:
        return None


@st.cache_resource
def load_predictor():
    """Load the trained predictor."""
    try:
        return ChurnPredictor()
    except FileNotFoundError:
        return None


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.title("🚨 Client Churn Prediction System")
    st.markdown("**Select Training & Management Consultancy LLC**")
    st.markdown("---")
    
    # Load data
    clients, bookings, feedback, comms, churn = load_raw_data()
    features = load_features()
    predictor = load_predictor()
    
    # Check if data exists
    if clients is None:
        st.error("📁 Raw data not found!")
        st.info("Run the following command to generate demo data:")
        st.code("python src/generate_data.py")
        return
    
    if predictor is None:
        st.warning("🔧 Model not trained yet!")
        st.info("Run the following command to train the model:")
        st.code("python src/model.py")
        st.markdown("---")
        st.subheader("📊 Available Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clients", len(clients))
        col2.metric("Bookings", len(bookings))
        col3.metric("Feedback", len(feedback))
        col4.metric("Communications", len(comms))
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["📊 Dashboard", "🔍 Client Analysis", "📈 Portfolio Risk", "💡 Recommendations"]
    )
    
    # Route to pages
    if page == "📊 Dashboard":
        show_dashboard(clients, bookings, feedback, churn, predictor, features)
    elif page == "🔍 Client Analysis":
        show_client_analysis(clients, bookings, feedback, churn, predictor, features)
    elif page == "📈 Portfolio Risk":
        show_portfolio_risk(predictor, features)
    elif page == "💡 Recommendations":
        show_recommendations(predictor, features)


# =============================================================================
# PAGE: DASHBOARD
# =============================================================================

def show_dashboard(clients, bookings, feedback, churn, predictor, features):
    """Main dashboard view."""
    
    st.header("📊 Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_clients = len(clients)
    total_revenue = bookings['revenue'].sum()
    avg_rating = feedback['overall_rating'].mean()
    churn_rate = (churn['churned'] == 'Yes').mean() * 100
    
    col1.metric("Total Clients", f"{total_clients}")
    col2.metric("Total Revenue", format_currency(total_revenue))
    col3.metric("Avg. Rating", f"{avg_rating:.1f}/5.0")
    col4.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("---")
    
    # Risk distribution
    if features is not None and predictor is not None:
        st.subheader("🎯 Current Risk Distribution")
        
        results = predictor.predict_batch(features)
        risk_counts = results['risk_level'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("🔴 High Risk", risk_counts.get('HIGH', 0))
            st.metric("🟡 Medium Risk", risk_counts.get('MEDIUM', 0))
            st.metric("🟢 Low Risk", risk_counts.get('LOW', 0))
            
            revenue_at_risk = results['revenue_at_risk'].sum()
            st.metric("💰 Revenue at Risk", format_currency(revenue_at_risk))
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    color=risk_counts.index,
                    color_discrete_map={
                        'HIGH': '#FF4B4B',
                        'MEDIUM': '#FFA500',
                        'LOW': '#00CC66'
                    },
                    title="Client Risk Distribution"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top at-risk clients
    st.subheader("🚨 Top 5 At-Risk Clients")
    
    if features is not None and predictor is not None:
        top_risk = predictor.get_top_risk_clients(features, top_n=5)
        
        for _, client in top_risk.iterrows():
            emoji = get_risk_emoji(client['risk_level'])
            prob = client['churn_probability']
            
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.write(f"{emoji} **{client.get('client_id', 'N/A')}**")
            col2.write(f"Probability: **{prob:.1%}**")
            col3.write(f"Risk: **{client['risk_level']}**")


# =============================================================================
# PAGE: CLIENT ANALYSIS
# =============================================================================

def show_client_analysis(clients, bookings, feedback, churn, predictor, features):
    """Individual client analysis view."""
    
    st.header("🔍 Client Analysis")
    
    # Client selector
    client_list = sorted(clients['client_id'].tolist())
    selected_client = st.sidebar.selectbox("Select Client", client_list)
    
    # Get client info
    client_info = clients[clients['client_id'] == selected_client].iloc[0]
    client_bookings = bookings[bookings['client_id'] == selected_client]
    client_feedback = feedback[feedback['client_id'] == selected_client]
    client_churn = churn[churn['client_id'] == selected_client].iloc[0]
    
    # Client profile
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Client Profile")
        st.write(f"**Company:** {client_info['client_name']}")
        st.write(f"**Industry:** {client_info['industry_sector']}")
        st.write(f"**Size:** {client_info['company_size']}")
        st.write(f"**Region:** {client_info['region']}")
        st.write(f"**Account Manager:** {client_info['account_manager']}")
        st.write(f"**Client Since:** {client_info['first_booking_date']}")
    
    with col2:
        st.subheader("📊 Activity Summary")
        st.write(f"**Total Bookings:** {len(client_bookings)}")
        st.write(f"**Total Revenue:** {format_currency(client_bookings['revenue'].sum())}")
        st.write(f"**Total Participants:** {client_bookings['number_of_participants'].sum()}")
        st.write(f"**Last Booking:** {client_bookings['booking_date'].max() if len(client_bookings) > 0 else 'N/A'}")
        
        if len(client_feedback) > 0:
            st.write(f"**Avg. Rating:** {client_feedback['overall_rating'].mean():.1f}/5.0")
            st.write(f"**Avg. NPS:** {client_feedback['nps_score'].mean():.1f}/10")
    
    st.markdown("---")
    
    # Churn prediction
    if features is not None and predictor is not None:
        st.subheader("🎯 Churn Prediction")
        
        client_features = features[features['client_id'] == selected_client]
        
        if len(client_features) > 0:
            result = predictor.predict_single(client_features.iloc[0])
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Churn Probability", 
                f"{result['churn_probability']:.1%}",
                delta=None
            )
            col2.metric(
                "Risk Level",
                f"{result['risk_emoji']} {result['risk_level']}"
            )
            col3.metric(
                "Status",
                "⚠️ At Risk" if result['will_churn'] else "✅ Active"
            )
            col4.metric(
                "Revenue at Risk",
                format_currency(result['estimated_revenue_at_risk'])
            )
            
            # Recommendations
            st.subheader("💡 Recommended Actions")
            for rec in result['recommendations']:
                priority_colors = {
                    'URGENT': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }
                emoji = priority_colors.get(rec['priority'], '⚪')
                
                with st.expander(f"{emoji} {rec['action']} ({rec['priority']})"):
                    st.write(f"**Detail:** {rec['detail']}")
                    st.write(f"**Expected Impact:** {rec['expected_impact']}")
        else:
            st.warning("No features available for this client.")


# =============================================================================
# PAGE: PORTFOLIO RISK
# =============================================================================

def show_portfolio_risk(predictor, features):
    """Portfolio-level risk analysis."""
    
    st.header("📈 Portfolio Risk Analysis")
    
    if features is None or predictor is None:
        st.warning("Model or features not available.")
        return
    
    # Get predictions for all clients
    results = predictor.predict_batch(features)
    
    # Summary metrics
    summary = predictor.get_risk_summary(features)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clients", summary['total_clients'])
    col2.metric("🔴 High Risk", summary['high_risk'])
    col3.metric("🟡 Medium Risk", summary['medium_risk'])
    col4.metric("Revenue at Risk", format_currency(summary['total_revenue_at_risk']))
    
    st.markdown("---")
    
    # Risk by segment
    st.subheader("Risk by Industry Segment")
    
    if 'industry_encoded' in results.columns:
        # This would need the original industry labels - simplified for demo
        risk_by_segment = results.groupby('risk_level').size().reset_index(name='count')
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                risk_by_segment,
                x='risk_level',
                y='count',
                color='risk_level',
                color_discrete_map={
                    'HIGH': '#FF4B4B',
                    'MEDIUM': '#FFA500',
                    'LOW': '#00CC66'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Probability distribution
    st.subheader("Churn Probability Distribution")
    
    if PLOTLY_AVAILABLE:
        fig = px.histogram(
            results,
            x='churn_probability',
            nbins=20,
            title="Distribution of Churn Probabilities",
            labels={'churn_probability': 'Churn Probability'}
        )
        fig.add_vline(x=0.45, line_dash="dash", line_color="red",
                      annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # At-risk table
    st.subheader("All At-Risk Clients (High & Medium)")
    
    at_risk = results[results['risk_level'].isin(['HIGH', 'MEDIUM'])].copy()
    at_risk = at_risk.sort_values('churn_probability', ascending=False)
    
    display_cols = ['client_id', 'churn_probability', 'risk_level', 'revenue_at_risk']
    display_cols = [c for c in display_cols if c in at_risk.columns]
    
    if len(at_risk) > 0:
        st.dataframe(at_risk[display_cols], use_container_width=True)
    else:
        st.success("No high or medium risk clients detected!")


# =============================================================================
# PAGE: RECOMMENDATIONS
# =============================================================================

def show_recommendations(predictor, features):
    """Actionable recommendations view."""
    
    st.header("💡 Action Recommendations")
    
    if features is None or predictor is None:
        st.warning("Model or features not available.")
        return
    
    # Get all predictions
    results = predictor.predict_batch(features)
    high_risk = results[results['risk_level'] == 'HIGH'].copy()
    
    st.subheader("🔴 Urgent Actions Required")
    st.write(f"**{len(high_risk)} clients** need immediate attention")
    
    if len(high_risk) > 0:
        for idx, (_, client) in enumerate(high_risk.head(10).iterrows()):
            client_id = client.get('client_id', f'Client {idx+1}')
            prob = client['churn_probability']
            
            with st.expander(f"🔴 {client_id} - {prob:.1%} probability"):
                # Get detailed recommendations
                pred_result = predictor.predict_single(client)
                
                st.write(f"**Revenue at Risk:** {format_currency(client['revenue_at_risk'])}")
                
                st.write("**Recommended Actions:**")
                for rec in pred_result['recommendations']:
                    st.write(f"• **[{rec['priority']}]** {rec['action']}")
                    st.write(f"  _{rec['detail']}_")
    
    st.markdown("---")
    
    # Summary of all actions
    st.subheader("📋 Weekly Action Summary")
    
    medium_risk = results[results['risk_level'] == 'MEDIUM']
    low_risk = results[results['risk_level'] == 'LOW']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Calls Today", len(high_risk))
        st.write("Immediate phone calls")
    
    with col2:
        st.metric("🟡 Emails This Week", len(medium_risk))
        st.write("Check-in emails")
    
    with col3:
        st.metric("🟢 Quarterly Reviews", len(low_risk))
        st.write("Regular touch-base")


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
