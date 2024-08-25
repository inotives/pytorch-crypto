import utils.data_preprocessing as dp 

class MovingAverageModel():
    def __init__(self, data_src): 
        self.data_src = data_src
        self.data_df = self.data_prep()

    def data_prep(self): 
        data_fullpath = f"data/raw/{self.data_src}.csv"
        data = dp.data_load(data_fullpath)
        data = dp.adding_features(data)

        data.dropna(inplace=True)

        return data 

    def run_predictor(self, forecast_days=3):

        df = self.data_df

        # Forecast SMA for the next 7 days
        last_sma = df['ma_20'].iloc[-1]
        sma_forecast = [last_sma] * forecast_days
        print(f"SMA-20 forecast for the next {forecast_days} days:", sma_forecast)

        # Forecast EMA for the next 7 days
        last_ema = df['ema_20'].iloc[-1]
        ema_forecast = [last_ema] * forecast_days
        print(f"EMA-20 forecast for the next {forecast_days} days:", ema_forecast)

        # Forecast WMA for the next 7 days
        last_wma = df['wma_20'].iloc[-1]
        wma_forecast = [last_wma] * forecast_days
        print(f"WMA-20 forecast for the next {forecast_days} days:", wma_forecast)

        # Forecast WMA for the next 7 days
        last_wma200 = df['wma_200'].iloc[-1]
        wma200_forecast = [last_wma200] * forecast_days
        print(f"WMA-200 forecast for the next {forecast_days} days:", wma200_forecast)

