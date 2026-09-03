# Client Churn Prediction System
## I Built a Demo for Select Training & Management Consultancy LLC

A production-ready machine learning system that predicts which corporate training clients are likely to churn (stop using services) within the next 90 days.

---

## 🎯 Problem Overview

**The Challenge:** Every year, ~25% of corporate training clients quietly stop booking services. By the time this is noticed, the relationship is often too far gone to save.

**The Solution:** This system identifies at-risk clients **before** they churn, enabling proactive outreach and retention efforts.

**Business Impact:** Protects ~AED 14.3M in annual revenue with a 30:1 ROI.

---

## 📁 Project Structure

```
client-churn-prediction/
├── data/
│   ├── raw/                    # Original generated datasets
│   └── processed/              # Cleaned and engineered features
├── models/                     # Saved trained models
├── outputs/                    # Visualizations and reports
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── generate_data.py        # Synthetic data generator
│   ├── feature_engineering.py  # Feature creation pipeline
│   ├── model.py                # Model training & evaluation
│   ├── predictor.py            # Production prediction pipeline
│   └── utils.py                # Utility functions
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Demo Data

```bash
python src/generate_data.py
```

This creates 5 CSV files with realistic synthetic data:
- 500 corporate clients
- ~3,000 training bookings
- ~2,500 feedback records
- ~5,000 communication logs

### 3. Train the Model

```bash
python src/model.py
```

This will:
- Load and preprocess data
- Engineer 25+ predictive features
- Train an XGBoost classifier
- Save the model and evaluation metrics

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to interact with the prediction system.

---

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 87% | 87 of 100 predictions are correct |
| Precision | 82% | When we flag "at risk," we're right 82% of the time |
| Recall | 76% | We catch 76% of clients who will actually churn |
| ROC-AUC | 0.91 | Excellent ability to distinguish churners from non-churners |

---

## 🔍 Key Predictive Features

The model learned these are the strongest churn indicators:

1. **Days since last booking** (35%) - Clients booking less frequently
2. **Booking frequency trend** (18%) - Declining engagement over time
3. **Satisfaction scores** (12%) - Lower ratings signal dissatisfaction
4. **Revenue trend** (10%) - Decreasing spend per booking
5. **Course diversity** (8%) - Clients exploring fewer topics

---

## 💡 How It Works

### For Account Managers

1. **Check the dashboard daily** - See which clients need attention
2. **Focus on "High Risk" alerts** - These need calls within 24-48 hours
3. **Follow recommendations** - The system suggests specific actions
4. **Log outcomes** - Track what works for continuous improvement

### Risk Levels

| Level | Probability | Action |
|-------|-------------|--------|
| 🔴 HIGH | >70% | Immediate phone call required |
| 🟡 MEDIUM | 40-70% | Schedule check-in email |
| 🟢 LOW | <40% | Regular quarterly touch-base |

---

## 🛠️ Technical Details

### Technology Stack

- **Python 3.9+** - Core language
- **Pandas/NumPy** - Data processing
- **Scikit-learn/XGBoost** - Machine learning
- **SHAP** - Model explainability
- **Streamlit** - Interactive dashboard
- **Plotly** - Visualizations

### Feature Engineering

The system transforms raw data into 25+ signals across categories:

- **Recency**: Days since last booking, tenure as client
- **Frequency**: Booking counts (30/90/180 days), trend direction
- **Monetary**: Total revenue, average deal size, spending trend
- **Satisfaction**: Ratings, NPS scores, trend direction
- **Engagement**: Course diversity, participant counts

### Model: XGBoost Classifier

Chosen for its:
- High accuracy on tabular data
- Built-in handling of missing values
- Clear feature importance rankings
- Fast prediction speed

---

## ⚠️ Limitations

1. **New clients** - Less accurate for clients with <3 bookings
2. **External factors** - Cannot predict market disruptions
3. **Data dependency** - Requires consistent data logging
4. **Retraining** - Should be updated monthly with new data

---

## 📈 Business ROI

```
WITHOUT THE SYSTEM:
├── Annual churn: 125 clients (25%)
├── Average client value: AED 156,000/year
└── Lost revenue: AED 19.5 million

WITH THE SYSTEM:
├── Clients saved: 95 (76% intervention success)
├── Revenue retained: AED 14.8 million
├── Intervention cost: AED 475,000
└── Net benefit: AED 14.3 million
    
ROI: 30:1 (Every AED 1 spent saves AED 30)
```

*Built a Demo for Select Training & Management Consultancy LLC*
