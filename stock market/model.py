import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(df, target_col='Close', sequence_length=10):
    """
    Prepare data for training a stock price forecasting model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock price data.
    target_col : str, optional
        Column to use as the target variable. Default is 'Close'.
    sequence_length : int, optional
        Number of previous days to use for prediction. Default is 10.
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    # Extract target variable
    target = df[target_col].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    target_scaled = scaler.fit_transform(target)
    
    # Create sequences
    X, y = [], []
    for i in range(len(target_scaled) - sequence_length):
        X.append(target_scaled[i:i+sequence_length])
        y.append(target_scaled[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Train a Random Forest model for stock price forecasting.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training target.
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())
    return model

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the trained model on the test data.
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test target.
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used to transform the target variable.
        
    Returns:
    --------
    tuple
        mse, r2, y_pred
    """
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    y_pred = y_pred.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Inverse transform the predictions and actual values
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, y_pred

def plot_predictions(y_test, y_pred):
    """
    Plot the actual vs. predicted stock prices.
    
    Parameters:
    -----------
    y_test : numpy.ndarray
        Actual stock prices.
    y_pred : numpy.ndarray
        Predicted stock prices.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Actual vs. Predicted Stock Prices')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Load dummy data
    from data_generator import generate_dummy_stock_data
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    dummy_data = generate_dummy_stock_data(start_date, end_date)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(dummy_data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    mse, r2, y_pred = evaluate_model(model, X_test, y_test, scaler)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    
    # Plot predictions
    plot_predictions(y_test, y_pred) 