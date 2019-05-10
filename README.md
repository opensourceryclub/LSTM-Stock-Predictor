
# LSTM-STOCK-PREDICTOR

  

Use the command line to train and visualize stock data

  

## Getting Started

Get an api key for stock data from [AlphaVantage](https://www.alphavantage.co/)

Run `python StockPredict.py` to generate an apikey.txt file for the project

Put key in apikey.txt file

## Features

### Single

`python StockPredict.py single [stock-ticker]`

To overwrite any existing model, train on, then visualize an existing stock


### All
`python StockPredict.py all`

To output predictions for the next day values for all stock tickers in stockList.txt


Notice this will create models for any models that cannot be found and recreate these models when it deems they are too old to be reliable.

### Show

`python StockPredict.py show [stock-ticker] [days]`

Takes a pre-made model (or making a new one if none is found) it visualizes prediction information for a span of days

The graph will show the stocks price over the most recent `days` days aswell as red lines indicating what the model predicted the change to be. Days where the stock correctly predicted increases are in green. Days were the model predicted increases, but the stock price decreased are in red.

## Future plans

Currently working on ways to visualize the training proccess over time and customize the training to the stock more. The predictions as is are  in nead of some work aswell.