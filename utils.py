import plotly.graph_objs as go
import pandas as pd

def plot_predictions(y_test, predictions, scaler):
    # Inverse transform the predictions and y_test for plotting
    y_test = scaler.inverse_transform(
        pd.DataFrame({
            'close': y_test,
            'TR': [0]*len(y_test),  # placeholder columns for inverse transform
            'ATR': [0]*len(y_test),
            'EMA': [0]*len(y_test),
            'WMA': [0]*len(y_test),
            'cumulative_return': [0]*len(y_test),
            'VWAP': [0]*len(y_test),
            'rsi': [0]*len(y_test),
            'Bollinger_Upper': [0]*len(y_test),
            'Bollinger_Lower': [0]*len(y_test),
        })
    )['close']

    predictions = scaler.inverse_transform(
        pd.DataFrame({
            'close': predictions,
            'TR': [0]*len(predictions),  # placeholder columns for inverse transform
            'ATR': [0]*len(predictions),
            'EMA': [0]*len(predictions),
            'WMA': [0]*len(predictions),
            'cumulative_return': [0]*len(predictions),
            'VWAP': [0]*len(predictions),
            'rsi': [0]*len(predictions),
            'Bollinger_Upper': [0]*len(predictions),
            'Bollinger_Lower': [0]*len(predictions),
        })
    )['close']

    # Create traces for actual and predicted values
    trace1 = go.Scatter(
        x=list(range(len(y_test))),
        y=y_test,
        mode='lines',
        name='Actual Prices'
    )

    trace2 = go.Scatter(
        x=list(range(len(predictions))),
        y=predictions,
        mode='lines',
        name='Predicted Prices'
    )

    layout = go.Layout(
        title='Crypto Price Prediction',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Price'),
        showlegend=True
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()
