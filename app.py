import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import gradio as gr
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
PREDICTION_DAYS = 30
TIME_STEP = 60
DATA_YEARS = 3

# Load model
model = load_model('stock_price_model.h5')
model.make_predict_function()  # For faster inference


def preprocess_data(df):
    """Process yfinance data"""
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index().rename(columns={'index': 'Date'})
    df = df[['Date', 'High', 'Low', 'Open', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def get_stock_data(stock_symbol):
    """Fetch stock data with caching"""
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * DATA_YEARS)
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    return preprocess_data(df)


def prepare_data(df):
    """Prepare data for LSTM prediction"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Create dataset using sliding window
    X = np.array([scaled_data[i:i + TIME_STEP, 0]
                  for i in range(len(scaled_data) - TIME_STEP - 1)])
    y = scaled_data[TIME_STEP + 1:, 0]

    return X.reshape(X.shape[0], TIME_STEP, 1), y, scaler


def predict_future(model, data, scaler):
    """Generate future predictions"""
    last_data = data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    future_preds = np.zeros(PREDICTION_DAYS, dtype='float32')

    for i in range(PREDICTION_DAYS):
        next_pred = model.predict(last_data, verbose=0)[0, 0]
        future_preds[i] = next_pred
        last_data = np.roll(last_data, -1, axis=1)
        last_data[0, -1, 0] = next_pred

    return scaler.inverse_transform(future_preds.reshape(-1, 1))


def create_plot(df, pred_data=None, future_data=None, title=""):
    """Create interactive Plotly figure"""
    fig = go.Figure()

    # Main price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Prediction line
    if pred_data is not None:
        fig.add_trace(go.Scatter(
            x=df.index[TIME_STEP + 1:],
            y=pred_data[:, 0],
            name='Predicted',
            line=dict(color='orange')
        ))

    # Future prediction
    if future_data is not None:
        future_dates = pd.date_range(
            start=df.index[-1],
            periods=PREDICTION_DAYS + 1
        )[1:]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_data[:, 0],
            name='30-Day Forecast',
            line=dict(color='green')
        ))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def predict_stock(stock_symbol):
    """Main prediction function for Gradio"""
    try:
        df = get_stock_data(stock_symbol)
        X, y, scaler = prepare_data(df)

        # Make predictions
        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred)

        # Future prediction
        future_prices = predict_future(
            model,
            scaler.transform(df['Close'].values.reshape(-1, 1)),
            scaler
        )

        # Create plots
        main_plot = create_plot(
            df,
            pred_data=y_pred,
            title=f"{stock_symbol} Price Prediction"
        )

        future_plot = create_plot(
            df,
            future_data=future_prices,
            title=f"{stock_symbol} 30-Day Forecast"
        )

        # Technical indicators
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()

        tech_fig = go.Figure()
        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            name='Price', line=dict(color='blue')))
        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='50-Day SMA', line=dict(color='orange')))
        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_200'],
            name='200-Day SMA', line=dict(color='red')))
        tech_fig.update_layout(
            title=f"{stock_symbol} Technical Indicators",
            template='plotly_dark'
        )

        return (
            f"${df['Close'].iloc[-1]:.2f}",
            df.index[-1].strftime('%Y-%m-%d'),
            main_plot,
            future_plot,
            tech_fig
        )

    except Exception as e:
        raise gr.Error(f"Prediction failed: {str(e)}")


# Gradio Interface
with gr.Blocks(title="Stock Prediction", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 📈 Real-Time Stock Predictor")
    gr.Markdown("Predict stock prices using LSTM neural networks")

    with gr.Row():
        stock_input = gr.Textbox(
            label="Stock Symbol (Examples: TSLA, AAPL, MSFT, AMZN, GOOG, AEP)",
            value="TSLA",
            placeholder="Enter stock symbol (e.g. AAPL, MSFT)"
        )
        submit_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        with gr.Column():
            last_price = gr.Textbox(label="Last Price")
            last_date = gr.Textbox(label="Last Date")

    with gr.Tabs():
        with gr.Tab("Price Prediction"):
            main_plot = gr.Plot(label="Price Prediction")
        with gr.Tab("30-Day Forecast"):
            future_plot = gr.Plot(label="Future Prediction")
        with gr.Tab("Technical Indicators"):
            tech_plot = gr.Plot(label="Technical Analysis")

    submit_btn.click(
        fn=predict_stock,
        inputs=stock_input,
        outputs=[last_price, last_date, main_plot, future_plot, tech_plot]
    )

# For Hugging Face Spaces
demo.launch(debug=False)