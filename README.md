# pytorch-crypto

## Overview

This project demonstrates how use various data in crypto (e.q OHLCV, onchain_metrics, etc) collected from [data-collector repos](https://github.com/inotives/data-collectors) in machine learning models using PyTorch. Aims to experiments building models such as price predictors with various model like LSTM, etc.


## How to install
- Once clone the repos, cd into the project root folder and create a python virtual env.
```
python -m venv venv
```
- next, activate the virtual env using `source venv/bin/activate`
- once you activate your virtual env, make sure you are in the correct virtual evn `which python`
- then you can start to install the require packages. `pip install -r requirements.txt`

## How to run the predictor 
- The OHLCV data used in the predictor was downloaded from coinmarketcap public endpoint of the historical price, for example here are the link to download [bitcoin historical OHCLV data](https://coinmarketcap.com/currencies/bitcoin/historical-data/).
- Download the historical data as csv and store in the data/raw folder
- update the csv name of the ohclv data in run_models.py
- then go to terminal and ran `python run_models.py`


## Predictor Models Implemented or Work in Progress
Below are list of forecasting model implemented along with the use of OHLCV data.  
- Simple predictors with Moving Average (Simple Moving Average, Exponential Moving Average, Weighted Moving Average)
- ARIMA (Auto Regressive Integrated Moving Average)
- SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
- LSTM (Long Short-Term Memory) networks
- GRU (Gated Recurrent Unit)
- Random Forest 
- XGBoost
- SVR (wip)
- 
