# Streamlit Stock Prediction App - Deployment Guide

## 🚀 Overview

This is a **modern Streamlit-based** stock prediction application that uses LSTM neural networks to forecast stock prices. The app is designed for easy deployment on Streamlit Cloud and other cloud platforms.

## ✨ Features

- **Real-time stock price prediction** using LSTM neural networks
- **Interactive Plotly charts** with historical and predicted data
- **30-day forecast** with technical indicators
- **Public prediction history** stored in Supabase
- **Responsive design** optimized for web and mobile
- **Modern UI/UX** with Streamlit's native components
- **Easy deployment** with minimal configuration

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML/AI**: TensorFlow/Keras LSTM model
- **Database**: Supabase (PostgreSQL)
- **Data Source**: Yahoo Finance (yfinance)
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud (recommended) or any cloud platform

## 📦 Dependencies

```txt
streamlit>=1.28.0
numpy>=1.21.0,<2.0.0
pandas>=1.3.0,<3.0.0
matplotlib>=3.5.0,<4.0.0
scikit-learn>=1.0.0,<2.0.0
tensorflow>=2.15.0,<2.20.0
yfinance>=0.2.0,<1.0.0
plotly>=5.0.0,<6.0.0
python-dotenv>=0.19.0
requests>=2.25.0,<3.0.0
psutil>=5.8.0
supabase>=1.0.0,<3.0.0
postgrest>=0.10.0
gotrue>=0.5.0
```

## 🏗️ Deployment Options

### 1. 🎯 Streamlit Cloud (Recommended)

**Streamlit Cloud** is the easiest way to deploy your app for free.

#### Steps:

1. **Prepare Your Repository**:
   ```
   GetYourStock/
   ├── streamlit_app.py          # Main application
   ├── requirements.txt          # Dependencies
   ├── supabase_client.py        # Database integration
   ├── stock_price_model.h5      # Pre-trained LSTM model
   ├── .env.example              # Environment variables template
   └── .streamlit/
       └── config.toml           # Streamlit configuration
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `streamlit_app.py`
   - Click "Deploy!"

3. **Configure Environment Variables**:
   In Streamlit Cloud dashboard, add these secrets:
   ```toml
   [secrets]
   SUPABASE_URL = "https://gxtlqnrevmdvlggwjloz.supabase.co"
   SUPABASE_ANON_KEY = "your_supabase_anon_key_here"
   SUPABASE_REQUIRED = "false"
   ```

4. **Access Your App**:
   Your app will be available at:
   `https://your-username-app-name-streamlit-app-random.streamlit.app/`

### 2. 🐳 Docker Deployment

For custom cloud platforms or self-hosting:

#### Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run:
```bash
docker build -t stock-prediction-app .
docker run -p 8501:8501 stock-prediction-app
```

### 3. 🌐 Other Cloud Platforms

#### Heroku:
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### Google Cloud Run:
```bash
gcloud run deploy stock-prediction \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🔧 Configuration Files

### .streamlit/config.toml
```toml
[global]
developmentMode = false

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### .env.example
```bash
# Supabase Configuration
SUPABASE_URL=https://gxtlqnrevmdvlggwjloz.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key_here
SUPABASE_REQUIRED=false
```

## 🏃‍♂️ Local Development

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd GetYourStock
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file:
```bash
SUPABASE_URL=https://gxtlqnrevmdvlggwjloz.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_REQUIRED=false
```

### 3. Run Locally
```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## ⚙️ Environment Variables Setup

### For Streamlit Cloud:
Add these to your `secrets.toml` in the Streamlit dashboard:

```toml
[secrets]
SUPABASE_URL = "https://gxtlqnrevmdvlggwjloz.supabase.co"
SUPABASE_ANON_KEY = "your_supabase_anon_key_here"
SUPABASE_REQUIRED = "false"
```

### For Other Platforms:
Set these environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `SUPABASE_URL` | `https://gxtlqnrevmdvlggwjloz.supabase.co` | Your Supabase project URL |
| `SUPABASE_ANON_KEY` | `your_anon_key` | Supabase anonymous key |
| `SUPABASE_REQUIRED` | `false` | Whether Supabase is required for app to work |

### Getting Supabase Keys:
1. Go to [supabase.com](https://supabase.com)
2. Navigate to your project dashboard
3. Go to Settings → API
4. Copy the **URL** and **anon/public** key

## 📊 Features Overview

### 🔮 **Stock Prediction**
- Enter any valid stock symbol (AAPL, TSLA, etc.)
- Get 30-day price forecasts using LSTM model
- View historical vs predicted price accuracy

### 📈 **Interactive Charts**
- **Price Prediction**: Historical vs predicted prices
- **30-Day Forecast**: Future price projections
- **Technical Analysis**: Moving averages and indicators

### 💾 **Public Database**
- All predictions stored in Supabase
- View recent community predictions
- No user authentication required
- Fully public and accessible

### 🏥 **System Monitoring**
- Real-time system status in sidebar
- Model loading status
- Database connectivity status

## 🎯 Key Features of Streamlit Deployment

### ✅ **Simplicity**
- **One-click deployment** on Streamlit Cloud
- **No complex configuration** files needed
- **Automatic SSL** and custom domains
- **Built-in secrets management**

### ✅ **Performance**
- **Automatic caching** with `@st.cache_data` and `@st.cache_resource`
- **Efficient state management** with Streamlit
- **Fast cold starts** compared to traditional web frameworks
- **Optimized for data science workloads**

### ✅ **Developer Experience**
- **Hot reloading** during development
- **Easy debugging** with built-in error handling
- **Interactive widgets** and components
- **No frontend/backend separation needed**

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**:
   - Ensure `stock_price_model.h5` exists in root directory
   - Check file size limits (GitHub: 100MB, Streamlit Cloud: 200MB)
   - Use Git LFS for large model files if needed

2. **Stock Data Error**:
   - Verify stock symbol is valid (1-5 alphanumeric characters)
   - Check internet connectivity for Yahoo Finance API
   - Yahoo Finance may have rate limits

3. **Database Connection**:
   - Verify Supabase environment variables are set correctly
   - Check Supabase project status in dashboard
   - Ensure database schema is properly set up

4. **Streamlit Cloud Issues**:
   - Check app logs in Streamlit Cloud dashboard
   - Verify requirements.txt has all dependencies
   - Ensure Python version compatibility (3.9-3.12)

### Git LFS for Large Models

If your model file is too large for Git:

```bash
# Install Git LFS
git lfs install

# Track the model file
git lfs track "*.h5"

# Add and commit
git add .gitattributes
git add stock_price_model.h5
git commit -m "Add model with Git LFS"
```

## 🚀 Performance Optimization

### Caching Strategy
The app uses Streamlit's caching decorators:

- `@st.cache_resource` - For loading the ML model (persists across sessions)
- `@st.cache_data(ttl=300)` - For stock data (cached for 5 minutes)

### Memory Management
- **Model loaded once** and cached globally
- **Data cached** to avoid repeated API calls
- **Efficient data structures** with pandas and numpy

## 🔐 Security Best Practices

- **Environment variables** for all sensitive data
- **Input validation** for stock symbols
- **Error handling** prevents crashes
- **Rate limiting** awareness for external APIs
- **No sensitive data** in logs or UI

## 📱 Mobile Experience

The Streamlit app is fully responsive:
- **Responsive layout** adapts to screen size
- **Touch-friendly** interface elements
- **Mobile-optimized** charts and inputs
- **Fast loading** on mobile networks

## 🎉 Success!

Once deployed, your app will provide:

- **Professional-looking interface** with modern design
- **Real-time stock predictions** with LSTM neural networks
- **Interactive visualizations** with Plotly
- **Public prediction history** via Supabase
- **Responsive design** for all devices

## 📞 Support

### Streamlit Resources:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Troubleshooting Steps:
1. Check Streamlit Cloud app logs
2. Test locally first
3. Verify environment variables
4. Check Supabase configuration
5. Review requirements.txt dependencies

---

**🎯 The Streamlit approach provides the best developer experience with minimal configuration and maximum functionality!** 