# 📈 Stock Prediction App

A modern **Streamlit-based** stock prediction application that uses LSTM neural networks to forecast stock prices with real-time data visualization and public prediction history.

## ✨ Features

- **🤖 AI-Powered Predictions**: LSTM neural networks for accurate stock forecasting
- **📊 Interactive Charts**: Beautiful Plotly visualizations with technical indicators
- **🔮 30-Day Forecasts**: Detailed future price predictions
- **💾 Public History**: Community prediction tracking via Supabase
- **📱 Responsive Design**: Works perfectly on desktop and mobile
- **⚡ Fast Performance**: Optimized with Streamlit caching

## 🚀 Quick Start

### 1. **Streamlit Cloud (Recommended)**
The easiest way to deploy for free:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select this repo
4. Set main file: `streamlit_app.py`
5. Add your Supabase secrets (see configuration below)
6. Deploy!

### 2. **Local Development**
```bash
# Clone the repository
git clone <your-repo-url>
cd GetYourStock

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (copy from env.example)
cp env.example .env
# Edit .env with your Supabase credentials

# Run the app
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` to see the app!

## ⚙️ Configuration

### Supabase Setup
1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Go to Settings → API to get your credentials
4. Run the SQL schema from `supabase_schema.sql`

### Environment Variables

**For Streamlit Cloud:**
Add to your app's secrets:
```toml
[secrets]
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your_anon_key_here"
SUPABASE_REQUIRED = "false"
```

**For Local Development:**
Create a `.env` file:
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_REQUIRED=false
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML/AI**: TensorFlow/Keras LSTM
- **Database**: Supabase (PostgreSQL)
- **Data Source**: Yahoo Finance
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud

## 📱 How It Works

1. **Enter Stock Symbol**: Choose from popular stocks or enter any valid symbol
2. **AI Processing**: LSTM model analyzes 3 years of historical data
3. **Generate Predictions**: Creates 30-day forecast with confidence intervals
4. **Interactive Charts**: Explore predictions with multiple visualization tabs
5. **Save & Share**: All predictions saved to public database

## 🎯 Key Features

### 📈 **Prediction Tabs**
- **Price Prediction**: Historical vs predicted accuracy
- **30-Day Forecast**: Future price projections
- **Technical Analysis**: Moving averages and indicators

### 🏥 **System Status**
- Real-time model loading status
- Database connectivity monitoring
- Performance indicators

### 💾 **Public Database**
- Community prediction history
- No authentication required
- Transparent and accessible

## 📦 Dependencies

Core packages:
- `streamlit` - Web application framework
- `tensorflow` - LSTM neural network model
- `yfinance` - Real-time stock data
- `plotly` - Interactive visualizations
- `supabase` - Database integration
- `pandas` & `numpy` - Data processing

See `requirements.txt` for complete list with versions.

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error:**
- Ensure `stock_price_model.h5` exists in root directory
- For large models, consider using Git LFS

**Stock Data Error:**
- Verify symbol format (1-5 alphanumeric characters)
- Check internet connection for Yahoo Finance API

**Database Connection:**
- Verify Supabase credentials in environment variables
- Check Supabase project status

**Streamlit Cloud:**
- Review app logs in dashboard
- Ensure all dependencies in requirements.txt
- Check Python version compatibility (3.9-3.12)

## 📚 Documentation

- **[Streamlit Deployment Guide](STREAMLIT_DEPLOYMENT.md)** - Comprehensive deployment instructions
- **[Supabase Setup Guide](SUPABASE_SETUP.md)** - Database configuration
- **[Supabase Schema](supabase_schema.sql)** - Database structure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This application is for educational purposes only. Stock predictions are estimates and should not be used as financial advice. Always consult with financial professionals before making investment decisions.

---

**🎉 Ready to predict the future of stocks? Deploy your app on Streamlit Cloud today!**
