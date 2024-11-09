import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to get stock data and make predictions
def stock_price_prediction():
    # User input prompts
    ticker = input("Enter the stock ticker (e.g., AAPL): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    future_days = int(input("Enter the number of future days to predict: "))
    
    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Close']]
    stock_data['Days'] = np.arange(len(stock_data))

    # Prepare data for model
    X = stock_data[['Days']].values
    y = stock_data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    predictions = model.predict(X_test)

    # Predict future prices
    future_X = np.arange(len(stock_data), len(stock_data) + future_days).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    # Calculate prediction margin for uncertainty (e.g., 5% margin)
    prediction_margin = 0.05 * future_predictions  # 5% margin for uncertainty
    
    # Ensure margin and predictions are 1D
    future_predictions = future_predictions.flatten()
    prediction_margin = prediction_margin.flatten()

    # Create a DataFrame for future predictions and historical data
    future_dates = pd.date_range(start=end_date, periods=future_days, freq='D')
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_predictions,
        'Lower Bound': future_predictions - prediction_margin,
        'Upper Bound': future_predictions + prediction_margin
    })
    
    # Display the table for future predictions
    print("\nFuture Price Predictions (with confidence intervals):")
    print(future_df.to_string(index=False))

    # Plot results with enhanced details
    plt.figure(figsize=(14, 10))

    # Plot historical prices
    plt.plot(stock_data['Days'], stock_data['Close'], label="Historical Prices", color="blue", linewidth=2)

    # Scatter plot for test predictions
    plt.scatter(X_test, predictions, color='red', label="Test Predictions", marker='x', s=100, zorder=5)

    # Plot regression line for the training data
    plt.plot(stock_data['Days'], model.predict(X), color='green', linestyle='-', linewidth=2, label="Regression Line (Training)")

    # Plot future predictions
    future_days_range = np.arange(len(stock_data), len(stock_data) + future_days)
    plt.plot(future_days_range, future_predictions, color='orange', linestyle='dashed', linewidth=3, label="Future Predictions")

    # Add prediction interval (simulating with 5% margin for demonstration)
    plt.fill_between(future_days_range, 
                     future_predictions - prediction_margin, 
                     future_predictions + prediction_margin, 
                     color='orange', alpha=0.2, label="Prediction Interval (±5%)")

    # Formatting the time axis with dates
    date_range = pd.date_range(start=end_date, periods=future_days, freq='D').strftime('%Y-%m-%d')
    plt.xticks(ticks=future_days_range, labels=date_range, rotation=45, ha="right", fontsize=12)

    # Add grid, titles, and axis labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Stock Price ($)", fontsize=14)
    plt.title(f"Stock Price Prediction for {ticker} (Past and Future)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show legend
    plt.legend(loc='upper left', fontsize=12)

    # Add a horizontal line for the average stock price in the training period
    avg_price = np.mean(y_train)
    plt.axhline(y=avg_price, color='green', linestyle='--', label=f"Avg Price ($ {avg_price:.2f})")

    # Show plot
    plt.tight_layout()
    plt.show()

    return future_predictions

# Run the prediction function
predicted_prices = stock_price_prediction()
print("Predicted Future Prices:", predicted_prices)
