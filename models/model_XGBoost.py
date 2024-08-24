import pandas as pd 
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import utils.data_preprocessing as dp 

class XGBoostModel():
    def __init__(self, data_src): 
        self.data_src = data_src
        self.data_df = self.data_prep()

    def data_prep(self): 
        data_fullpath = f"data/raw/{self.data_src}.csv"
        data = dp.data_load(data_fullpath)
        data = dp.adding_features(data)
        
        # Shifted close as the target for next day prediction
        data['shifted_close'] = data['close'].shift(-1)

        data.dropna(inplace=True)

        return data 

    def run_predictor(self): 
        # define features X and target Y 
        df = self.data_df
        X = df[['close', 'volume', 'ema_20', 'ma_50', 'rsi']].values
        y = df['close'].shift(-7).dropna().values # Target is the price 7 days in the future

        # Drop the last 7 rows from X since they have no corresponding y
        X = X[:-7]
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        # Initialize the XGBoost regressor
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

        # Train the model
        xg_reg.fit(X_train, y_train)

        # Predict on the test set
        y_pred = xg_reg.predict(X_test)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")


        # Predict the next 7 days using the last available feature set
        last_features = X_scaled[-1]

        predicted_prices = []
        for i in range(7):
            next_price = xg_reg.predict(last_features.reshape(1, -1))
            
            # Append the predicted price to the list
            predicted_prices.append(next_price[0])
            
            # Update the features (recursive forecasting)
            last_features = np.array([next_price[0], df['volume'].iloc[-1], df['ema_20'].iloc[-1], df['ma_50'].iloc[-1], df['rsi'].iloc[-1]])
            last_features = scaler.transform(last_features.reshape(1, -1))

        print("XGBoost forecast for the next 7 days:", predicted_prices)


        return 