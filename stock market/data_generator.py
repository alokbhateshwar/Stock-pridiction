import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_dummy_stock_data(start_date, end_date, ticker='DUMMY'):
    """
    Generate dummy stock price data for a given date range.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    ticker : str, optional
        Ticker symbol for the dummy stock. Default is 'DUMMY'.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing dummy stock price data.
    """
    # Convert start and end dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Generate random stock prices
    np.random.seed(42)  # For reproducibility
    initial_price = 100.0
    prices = [initial_price]
    
    for i in range(1, len(date_range)):
        # Generate a random price change
        change = np.random.normal(0, 1)
        new_price = prices[-1] * (1 + change/100)
        prices.append(new_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'Volume': np.random.randint(1000, 10000, size=len(date_range))
    })
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    return df

if __name__ == '__main__':
    # Generate dummy data for the last 365 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    dummy_data = generate_dummy_stock_data(start_date, end_date)
    print(dummy_data.head()) 