#price prediction.ipynb file codes

#block 1
import pandas as pd
df = pd.read_csv('combined_stock_data.csv')
df.columns = df.columns.str.strip()
df.head()

#block 2
df.columns

#block 3
df.shape

#block 4
df.size

#block 5
df.isnull().sum()

#block 6
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#block 7
cols_to_convert = ['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'ltp', 'close', 'vwap',
                   '52W H', '52W L', 'VOLUME', 'VALUE']
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
df.dropna(inplace=True)

#block 8
df['close_lag_1'] = df['close'].shift(1)
df['close_lag_2'] = df['close'].shift(2)

df['SMA_5'] = df['close'].rolling(window=5).mean()
df['SMA_10'] = df['close'].rolling(window=10).mean()

df['volatility_5'] = df['close'].rolling(window=5).std()
df['daily_return'] = df['close'].pct_change()

df.dropna(inplace=True)

#block 9
features = ['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'ltp', 'close', 'vwap',
            '52W H', '52W L', 'VOLUME', 'VALUE', 'close_lag_1', 'close_lag_2',
            'SMA_5', 'SMA_10', 'volatility_5', 'daily_return']

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

#block 10
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
import math

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

data = df[['close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(actual, predictions)
mse = mean_squared_error(actual, predictions)
rmse = math.sqrt(mse)
r2 = r2_score(actual, predictions)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
rmse = math.sqrt(mean_squared_error(actual, predictions))
print("Root Mean Squared Error (RMSE):", rmse)

plt.figure(figsize=(14, 6))
plt.plot(actual, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

def predict_next_price(manual_input):
    if len(manual_input) != 60:
        raise ValueError("Input must contain exactly 60 closing prices.")

    manual_input = np.array(manual_input).reshape(-1, 1)
    scaled_input = scaler.transform(manual_input)
    scaled_input = np.reshape(scaled_input, (1, 60, 1))
    pred_scaled = model.predict(scaled_input)
    pred_price = scaler.inverse_transform(pred_scaled)

    return float(pred_price[0][0])

last_60 = df['close'].values[-60:]
predicted = predict_next_price(last_60)
print("Predicted next stock price from last 60 closing prices:", predicted)

#block 11
your_input = [
    101.5, 102.1, 100.7, 99.4, 98.3, 100.2, 101.0, 101.3, 102.4, 103.0,
    103.5, 104.2, 103.9, 102.7, 101.9, 100.5, 99.8, 99.0, 98.6, 98.3,
    98.7, 99.2, 99.9, 100.3, 100.8, 101.5, 102.0, 102.6, 103.1, 103.9,
    104.6, 105.2, 104.8, 104.4, 104.0, 103.5, 103.1, 102.8, 102.3, 101.9,
    101.6, 101.2, 100.9, 100.5, 100.2, 99.8, 99.5, 99.3, 99.0, 98.8,
    98.5, 98.3, 98.1, 98.0, 97.9, 97.8, 97.7, 97.6, 97.5, 97.4
]
predicted = predict_next_price(your_input)
print("Predicted next price:", predicted)
