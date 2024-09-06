import pandas as pd 

from statsmodels.tsa.arima.model import ARIMA

import utils.data_preprocessing as dp 

class ARIMAModel():
    ''' ARIMA (Auto Regressive Integrated Moving Average)
        It is a class of models that explains a given time series based on its own past values, 
        its own past errors, and it integrates differencing to make the time series stationary.

        KEY COMPONENT of ARIMA: 
        1) AR (AutoRegressive) Component
        p: This represents the number of lag observations included in the model (i.e., the number of terms). 
            It shows the relationship between an observation and some number of lagged observations.
        
        2) I (Integrated) Component
        d: This represents the number of times the data have had past values subtracted to make 
            the series stationary (i.e., the number of differencing required to remove trends and seasonality). 
        
        3) MA (Moving Average) Component
        q: This represents the number of lagged forecast errors in the prediction equation.

        HOW ARIMA WORKS: 
        1. Identify if the time series is stationary. If not, apply differencing until it becomes stationary. 
        This determines the order d. Use autocorrelation function (ACF) and 
        partial autocorrelation function (PACF) plots to determine the appropriate values for p and q.
        2. Estimate the coefficients used in AR and MA Components using methods like maximum likelihood estimation.
        3. Check if the residuals (errors) of the model resemble white noise, 
        meaning they are independently and identically distributed with a mean of zero.
        If the residuals are not white noise, adjust the model accordingly.
        4. Use the fitted ARIMA model to forecast future values.
    '''
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