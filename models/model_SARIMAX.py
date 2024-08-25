import pandas as pd 
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

import utils.data_preprocessing as dp
from utils.plots import plotly_line_chart


class SARIMAXModel():
    def __init__(self, data_src):
        self.data_src = data_src

    def data_prep(self):

        data_fullpath = f"data/raw/{self.data_src}.csv"
        data = dp.data_load(data_fullpath)
        data = dp.adding_features(data)

        data.dropna(inplace=True)

        return data
    
    def create_model(self, target_series, exog, pdq=(1,1,1), PDQS=(1,1,1,1), summary=False): 
        p, d, q = pdq 
        P, D, Q, S = PDQS
        
        model = SARIMAX(
            target_series, 
            exog=exog, 
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model_fit = model.fit(
            method='powell',       # You can try different methods such as 'nm' or 'powell', 'lbfgs'
            maxiter=500,          # Increase the number of iterations
            disp=True             # Display convergence information
        )
        
        if summary: print(model_fit.summary())
        return model_fit 
    
    def run_predictor(self, forecast_days=3, summary=False):
        
        data = self.data_prep()
        
        # Ensure the index has a daily frequency
        data = data.asfreq('D')
        if summary: print(data.columns) 

        # Exogenous variables (e.g., RSI, Bollinger Bands, Volume, etc.)
        exog = data[['rsi', 'ema_20', 'ema_100', 'volume']] 

        # Target variable (close prices)
        target_series = data['close']

        # Creating the model 
        sarimax_model_fit = self.create_model(target_series, exog, pdq=(1,1,7), PDQS=(1,1,1,12))

        # Forecast
        forecast = sarimax_model_fit.get_forecast(steps=forecast_days, exog=exog.iloc[-forecast_days:])
        forecasted_mean = forecast.predicted_mean
        forecasted_conf_int = forecast.conf_int()

        # Create a date range for the forecasted prices
        future_dates = pd.date_range(start=target_series.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

        # Create DataFrame for forecasted prices
        forecast_df = pd.DataFrame({
            'predicted_price': forecasted_mean,
            'lower_conf_int': forecasted_conf_int.iloc[:, 0],
            'upper_conf_int': forecasted_conf_int.iloc[:, 1]
        }, index=future_dates)

        print(forecast_df)

        return forecast_df, target_series, sarimax_model_fit
    
    def evaluate_result(self):
        forecast_prices, target_series, model_fit = self.run_predictor(forecast_days=7)

        # Historical forecast to check model fit
        historical_forecast = model_fit.get_prediction(start=0, end=len(target_series)-1)
        historical_forecast_mean = historical_forecast.predicted_mean
        historical_conf_int = historical_forecast.conf_int()

        # Create DataFrame for historical forecasted prices
        historical_forecast_df = pd.DataFrame({
            'predicted_price': historical_forecast_mean,
            'lower_conf_int': historical_conf_int.iloc[:, 0],
            'upper_conf_int': historical_conf_int.iloc[:, 1]
        }, index=target_series.index)

        plot_data = [
            {"xvals": target_series.index, 'yvals': target_series, 'label': 'Actual Closing Price', 'marker': ',', 'plotly_mode': 'lines'},
            {"xvals": historical_forecast_df.index, 'yvals': historical_forecast_df['predicted_price'], 'label': 'Historical Forecasted Price', 'marker': 'x', 'plotly_mode': 'lines+markers'},
            {"xvals": forecast_prices.index, 'yvals': forecast_prices['predicted_price'], 'label': 'Forecasted Closing Price', 'marker': 'x', 'plotly_mode': 'lines+markers'}
        ]

        fig = plotly_line_chart(plot_data, 'SARIMAX + GARCH Model Forecast', 'Date', 'Price')
        fig.show()
        