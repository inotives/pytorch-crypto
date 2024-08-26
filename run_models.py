from models.model_XGBoost import XGBoostModel
from models.model_RandomForest import RandomForestModel
from models.model_MA import MovingAverageModel
from models.model_ARIMA import ARIMAModel
from models.model_SARIMAX import SARIMAXModel
from models.model_VAR import VARModel
from models.model_LSTM import run_predictor as lstm_predictor




if __name__=='__main__':

    ohlcv_dataset = 'ohlcv_bitcoin_20240821'



    ''' Comparing Price Prediction using different type of models... '''

    print('\nPRICE PREDICTOR USING Moving Average \n--------------------------------------------')
    ma = MovingAverageModel(ohlcv_dataset)
    ma.run_predictor(forecast_days=7)

    print('\nPRICE PREDICTOR USING Vector Autoregression (VAR) \n--------------------------------------------')
    va = VARModel(ohlcv_dataset)
    va.run_predictor(forecast_days=7)

    # print('\nPRICE PREDICTOR USING Random Forest \n--------------------------------------------')
    # rf = RandomForestModel(ohlcv_dataset)
    # rf.run_predictor(forecast_days=7)

    # print('\nPRICE PREDICTOR USING XGBOOST \n--------------------------------------------')
    # xg = XGBoostModel(ohlcv_dataset)
    # xg.run_predictor(forecast_days=7)

    # print('\nPRICE PREDICTOR USING ARIMA \n--------------------------------------------\n<might take awhile to run>')
    # ar = ARIMAModel(ohlcv_dataset)
    # ar.run_predictor(forecast_days=7)

    # print('\nPRICE PREDICTOR USING SARIMAX \n--------------------------------------------\n<might take awhile to run>')
    # ar = SARIMAXModel(ohlcv_dataset)
    # ar.run_predictor(forecast_days=7)

    