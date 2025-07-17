import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load stock data
ticker = "AAPL"  # You can change this to any stock symbol
data = yf.download(ticker, start="2020-01-01", end="2024-12-31")

# Prepare data
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date'])
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
X = data[['Days']]
y = data['Close']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'], label="Actual Prices")
plt.plot(X_test.index, predictions, label="Predicted Prices", color="red")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Print error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

