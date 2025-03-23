import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objs as go
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_absolute_error

# Define the path to the models folder
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def fetch_stock_data(ticker):
    """
    Fetch historical stock data using yfinance (last 5 years).
    Cache the data to avoid redundant API calls.
    """
    # Path to the cached data file
    cache_file = os.path.join(MODELS_DIR, f"{ticker}_data.csv")
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        print("Loading cached data...")
        data = pd.read_csv(cache_file)
        data['ds'] = pd.to_datetime(data['ds'])
    else:
        print("Fetching data from yfinance...")
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=5)  # Fetch 5 years of data
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        data = data[['Close']].reset_index()
        data.columns = ['ds', 'y']  # Prophet requires columns named 'ds' (date) and 'y' (value)
        data.to_csv(cache_file, index=False)  # Cache the data
    
    return data

def train_prophet_model(data):
    """
    Train a Prophet model on the stock data.
    """
    model = Prophet(
        changepoint_prior_scale=0.05,  # Adjust sensitivity to trend changes
        seasonality_prior_scale=10.0,  # Adjust seasonality strength
        yearly_seasonality=False,      # Disable yearly seasonality
        weekly_seasonality=False,      # Disable weekly seasonality
        daily_seasonality=False,       # Disable daily seasonality
        interval_width=0.95            # Set confidence interval width
    )
    model.fit(data)
    return model

def train_sarimax_model(data, p, d, q, P):
    """
    Train a SARIMAX model on the stock data.
    """
    model = SARIMAX(
        data['y'],
        order=(p, d, q),
        seasonal_order=(P, 0, 0, 12),  # Seasonal AR order (P) with a seasonal period of 12
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    return model_fit

def train_random_forest_model(data):
    """
    Train a Random Forest model on the stock data.
    """
    # Feature engineering (e.g., lag features)
    data['lag_1'] = data['y'].shift(1)
    data['lag_2'] = data['y'].shift(2)
    data['lag_3'] = data['y'].shift(3)
    data.dropna(inplace=True)
    
    X = data[['lag_1', 'lag_2', 'lag_3']]
    y = data['y']
    
    model = RandomForestRegressor(
        n_estimators=200,  # Increase the number of trees
        random_state=42,
        max_depth=10       # Limit the depth of trees
    )
    model.fit(X, y)
    return model

def prepare_lstm_data(data, lookback=60):
    """
    Prepare data for LSTM model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['y']])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
    
    return X, y, scaler

def train_lstm_model(data):
    """
    Train an LSTM model on the stock data.
    """
    # Prepare data
    lookback = 60  # Adjust lookback period as needed
    X, y, scaler = prepare_lstm_data(data, lookback)
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))  # Reduced units
    model.add(Dropout(0.2))  # Reduced dropout to prevent overfitting
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))  # Reduced dropout to prevent overfitting
    model.add(Dense(units=25))  # Reduced dense layer units
    model.add(Dense(units=1))
    
    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(
        X, y,
        batch_size=32,
        epochs=50,  # Reduced epochs to prevent overfitting
        validation_split=0.2,
        verbose=1
    )
    
    # Print training and validation loss
    print("Training Loss:", history.history['loss'])
    print("Validation Loss:", history.history['val_loss'])
    
    return model, scaler

def predict_future_prices(model, model_name, periods, data=None, scaler=None, sarimax_params=None):
    """
    Predict future stock prices using the selected model.
    """
    if model_name == "prophet":
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    elif model_name == "sarimax":
        p, d, q, P = sarimax_params
        forecast = model.get_forecast(steps=periods)
        conf_int = forecast.conf_int()
        future_dates = pd.date_range(start=data['ds'].max(), periods=periods + 1, freq='D')[1:]
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast.predicted_mean,
            'yhat_lower': conf_int.iloc[:, 0],
            'yhat_upper': conf_int.iloc[:, 1]
        })
    elif model_name == "random_forest":
        # Predict using the last available data points
        last_lag_1 = data['y'].iloc[-1]
        last_lag_2 = data['y'].iloc[-2]
        last_lag_3 = data['y'].iloc[-3]
        predictions = []
        for _ in range(periods):
            prediction = model.predict([[last_lag_1, last_lag_2, last_lag_3]])[0]
            predictions.append(prediction)
            last_lag_3 = last_lag_2
            last_lag_2 = last_lag_1
            last_lag_1 = prediction
        future_dates = pd.date_range(start=data['ds'].max(), periods=periods + 1, freq='D')[1:]
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions,
            'yhat_lower': [p - 0.1 * p for p in predictions],  # Add variation to lower bound
            'yhat_upper': [p + 0.1 * p for p in predictions]   # Add variation to upper bound
        })
    elif model_name == "lstm":
        # Predict using the LSTM model
        lookback = 60
        inputs = data['y'].values[-lookback:]
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        
        predictions = []
        for _ in range(periods):
            X_test = np.array(inputs[-lookback:])
            X_test = np.reshape(X_test, (1, X_test.shape[0], 1))
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred[0][0])
            inputs = np.append(inputs, pred)  # Update inputs with the new prediction
        
        # Inverse transform the predictions to original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=data['ds'].max(), periods=periods + 1, freq='D')[1:]
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions.flatten(),
            'yhat_lower': predictions.flatten() - 0.1 * predictions.flatten(),  # Add variation to lower bound
            'yhat_upper': predictions.flatten() + 0.1 * predictions.flatten()   # Add variation to upper bound
        })
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def calculate_error(data, forecast):
    """
    Calculate the Mean Absolute Error (MAE) between actual and predicted prices.
    """
    actual = data['y'].values[-len(forecast):]
    predicted = forecast['yhat'].values
    mae = mean_absolute_error(actual, predicted)
    return mae

def create_interactive_plot(data, forecast, ticker, model_name):
    """
    Create an interactive Plotly graph showing historical and predicted prices.
    """
    # Path to the saved plot file
    plot_file = f"static/{ticker}_{model_name}_forecast.html"
    
    # Generate the plot
    historical_trace = go.Scatter(
        x=data['ds'],
        y=data['y'],
        mode='lines',
        name='Historical Close Price'
    )
    
    predicted_trace = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange')
    )
    
    upper_trace = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        fill=None,
        line=dict(width=0),
        showlegend=False
    )
    lower_trace = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        fill='tonexty',
        line=dict(width=0),
        name='Confidence Interval'
    )
    
    layout = go.Layout(
        title=f"{ticker} Stock Price Forecast ({model_name})",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price (USD)'),
        hovermode='x unified'
    )
    
    fig = go.Figure(data=[historical_trace, predicted_trace, upper_trace, lower_trace], layout=layout)
    
    # Save the plot as an HTML file
    os.makedirs("static", exist_ok=True)  # Create static folder if it doesn't exist
    fig.write_html(plot_file)
    
    return plot_file

def create_error_plot(data, forecast, ticker, model_name):
    """
    Create an interactive Plotly graph showing the error between actual and predicted prices.
    """
    # Path to the saved error plot file
    error_plot_file = f"static/{ticker}_{model_name}_error.html"
    
    # Generate the error plot
    actual_trace = go.Scatter(
        x=data['ds'][-len(forecast):],
        y=data['y'][-len(forecast):],
        mode='lines',
        name='Actual Price'
    )
    
    predicted_trace = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red')
    )
    
    error_trace = go.Scatter(
        x=forecast['ds'],
        y=data['y'][-len(forecast):] - forecast['yhat'],
        mode='lines',
        name='Error',
        line=dict(color='green')
    )
    
    layout = go.Layout(
        title=f"{ticker} Stock Price Error ({model_name})",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price (USD)'),
        hovermode='x unified'
    )
    
    fig = go.Figure(data=[actual_trace, predicted_trace, error_trace], layout=layout)
    
    # Save the error plot as an HTML file
    os.makedirs("static", exist_ok=True)  # Create static folder if it doesn't exist
    fig.write_html(error_plot_file)
    
    return error_plot_file

def main(ticker, periods, model_name, sarimax_params=None):
    """
    Main function to fetch data, train the model, and make predictions.
    """
    try:
        # Fetch data (last 5 years)
        data = fetch_stock_data(ticker)
        
        # Train or load the selected model
        if model_name == "prophet":
            model = train_prophet_model(data)
            forecast = predict_future_prices(model, model_name, periods)
        elif model_name == "sarimax":
            p, d, q, P = sarimax_params
            model = train_sarimax_model(data, p, d, q, P)
            forecast = predict_future_prices(model, model_name, periods, data=data, sarimax_params=sarimax_params)
        elif model_name == "random_forest":
            model = train_random_forest_model(data)
            forecast = predict_future_prices(model, model_name, periods, data=data)
        elif model_name == "lstm":
            model, scaler = train_lstm_model(data)
            forecast = predict_future_prices(model, model_name, periods, data=data, scaler=scaler)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Create interactive plot
        plot_url = create_interactive_plot(data, forecast, ticker, model_name)
        
        # Create error plot
        error_plot_url = create_error_plot(data, forecast, ticker, model_name)
        
        return forecast, plot_url, error_plot_url
    except Exception as e:
        raise RuntimeError(f"Error in main function: {e}")