# Stock Price Forecasting with Machine Learning

This project demonstrates how to forecast stock prices using machine learning techniques. It uses dummy data for demonstration purposes.

## Project Structure

- `data_generator.py`: Script to generate dummy stock price data.
- `model.py`: Script for training and evaluating a stock price forecasting model.
- `requirements.txt`: List of required Python packages.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Generate dummy stock price data:
   ```
   python data_generator.py
   ```

2. Train and evaluate the model:
   ```
   python model.py
   ```

## Model

The model used in this project is a Random Forest Regressor, which is trained on a sequence of previous stock prices to predict the next day's price.

## Evaluation

The model is evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics. The actual vs. predicted stock prices are also plotted for visual comparison.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 