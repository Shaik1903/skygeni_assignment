# 📊 SkyGeni Sales Intelligence Dashboard

A comprehensive **Sales Decision Intelligence Platform** built with Streamlit, featuring three ML-powered decision engines, interactive visualizations, and actionable recommendations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Shaik1903/skygeni_assignment.git
cd skygeni_assignment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
skygeni_opus/
├── app.py                          # Main Streamlit dashboard (Home Page)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_🎯_Win_Rate_Drivers.py    # Factor analysis page
│   ├── 2_📈_Revenue_Forecast.py    # Forecasting page
│   ├── 3_🚨_Anomaly_Detection.py   # Anomaly alerts page
│   └── 4_💡_Recommendations.py     # Action recommendations
│
├── src/                            # Core Python modules
│   ├── __init__.py
│   ├── data_loader.py              # Data loading & preprocessing
│   ├── metrics.py                  # Custom business metrics
│   └── eda.py                      # EDA & insight generation
│
└── data/
    └── skygeni_sales_data.csv      # Source data
```

---

## 📊 Dashboard Pages

### 🏠 Home (Overview)
- **5 KPI Cards**: Win Rate, Revenue, Deals, Cycle Time, Deal Velocity Index
- **Interactive Charts**: Win rate trend, revenue by quarter, segment heatmap
- **Lead Source Analysis**: Pie chart showing conversion by source
- **Rep Leaderboard**: Top performing sales reps
- **Auto-Generated Insights**: Key findings with severity levels
- **Top Recommendations**: Quick action items

### 🎯 Win Rate Drivers
- Feature importance analysis (simulated SHAP-like)
- Segment deep dives (region, industry, product, lead source)
- Win Pressure Score visualization
- Best vs. worst performing segments

### 📈 Revenue Forecast
- Time series forecasting with trend extrapolation
- Confidence intervals (80% and 95%)
- Scenario comparison (Conservative, Most Likely, Optimistic)
- Monthly breakdown table
- Methodology documentation

### 🚨 Anomaly Detection
- Z-score based anomaly detection
- Segment-level deviation alerts
- Severity classification (High/Medium/Low)
- Metric trend monitoring
- Rolling baseline visualization

### 💡 Recommendations
- Prioritized action cards
- Impact vs. Effort matrix
- Implementation timeline
- Export to CSV or Markdown

---

## 📈 Custom Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Deal Velocity Index (DVI)** | `(Amount / Cycle Days) / Median` | Measures revenue efficiency per day |
| **Win Pressure Score** | `(Segment Win Rate × Volume) / Overall Rate` | Shows segment impact on overall performance |
| **Rep Consistency Score** | `1 - (StdDev / Mean) of monthly rates` | Identifies stable vs. volatile performers |

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly, Altair
- **Data Processing**: Pandas, NumPy
- **Statistics**: SciPy
- **ML (planned)**: scikit-learn, SHAP, Prophet

---

## 📝 Development

### Running Tests
```bash
python test_data_modules.py
```

### Adding New Features
1. Add analysis functions to `src/` modules
2. Create new page in `pages/` directory
3. Update `requirements.txt` if new dependencies added
4. Update this README

---

## 📌 Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1.0 | 2026-02-09 | Initial release with data analysis and Streamlit dashboard |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is part of the SkyGeni Sales Intelligence Challenge.

---

**Built with ❤️ for data-driven sales decisions**
