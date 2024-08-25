import pandas as pd 

from statsmodels.tsa.arima.model import ARIMA

import utils.data_preprocessing as dp 

class ARIMAModel():
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

    def run_predictor(self, forecast_days=3): 
        
        data = self.data_df
        data = data.asfreq('D')

        # Select the features you want to include as exogenous variables
        exog = data[['volume', 'marketCap']]


        # Fit ARIMA model
        p = 1  # the number of lag observations included in the model (AR component).
        d = 1  # the number of times that the raw observations are differenced (I component). number of differencing
        q = 5  #  the size of the moving average window (MA component).
        model = ARIMA(data['close'], exog=exog, order=(p, d, q))
        model_fit = model.fit(method_kwargs={'maxiter': 500})
        # print(model_fit.summary())
        
        # Forecast
        forecast = model_fit.get_forecast(steps=forecast_days, exog=exog[-forecast_days:])
        forecasted_mean = forecast.predicted_mean
        forecasted_conf_int = forecast.conf_int()

        # Create DataFrame for forecasted prices
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        forecast_df = pd.DataFrame({
            'predicted_price': forecasted_mean,
            'lower_conf_int': forecasted_conf_int.iloc[:, 0],
            'upper_conf_int': forecasted_conf_int.iloc[:, 1]
        }, index=future_dates)


        print(forecast_df)

        return forecast_df