'''
    Controller of the stock predictor

'''

from scripts.data import DataLoader
from scripts.model import StockModel
from scripts import charts

import os
import numpy as np

''' Initialization '''
# Initialises the files if they need to be
def init_program():

    # Files to be made
    files = ['apikey.txt']
    directories = ['./save_data/', './save_data/data/', './save_data/models/']

    fs = 0; ds = 0
    
    for d in directories:
        if not os.path.exists(d):
            os.mkdir(d)
            ds += 1
    for f in files: 
        if not os.path.exists(f):
            os.open(f,"w+").close()
            fs += 1
        
    print('Made', fs, 'files and', ds, 'directories')

''' Loading in valued '''
# Given a stock ticker, load in the data for that stock, either from the api or localy
def load_data(ticker):

    key = load_key()
    if key == "": return

    data_loader = DataLoader(ticker, key)
    data_loader.load()

    return data_loader.get_data()

# Load in the api key
def load_key():

    key = './apikey.txt'

    keyFile = open(key, "r")
    key = keyFile.read()
    keyFile.close()

    if key == "": print('Please place alphavantage key in ./apikey.txt')
    return key

# Creates a new model
def create_model(ticker):
    
    key = load_key()
    if key == "": return
    
    print("Creating a new model")
    model = StockModel(ticker)
    model.create_model()

# Deletes a model
def delete_model(ticker):

    path = './save_data/models/' + ticker + '.h5'
    
    if os.path.isfile(path):
        print("Removed model")
        os.remove(path)
    else:
        print("No model to remove")
    
# Loads in a pre-made model
def load_model(ticker):
    
    key = load_key()
    if key == "": return
    
    model = StockModel(ticker)
    return (model, model.load_model())


''' Training Models '''
def train_model(ticker):

    model, loaded = load_model(ticker)
    if not loaded: return

    data = load_data(ticker)
    windows, x_list, y_list = data.generate_all_windows(100)

    x = np.stack(x_list, axis = 0)
    y = np.stack(y_list, axis = 0)

    print('Doing Training')
    trainings = [ (30, 0), (20, 1000), (20, 500), (20, 100) ]
    for count, size in trainings:    

        if size != 0: print(' - Training', count, 'epochs', 'on last', size, 'days')
        else: print(' - Training', count, 'epochs', 'on all days')    
        
        x_test, y_test, windows_test = data.get_train_data(x, y, windows, size)
        model.train( x_test, y_test, windows_test, count)

    print('Saving Model to file                ')
    model.save()

def train_specific(ticker, count, size):

    model, loaded = load_model(ticker)
    if not loaded: return

    data = load_data(ticker)
    windows, x_list, y_list = data.generate_all_windows(100)

    x = np.stack(x_list, axis = 0)
    y = np.stack(y_list, axis = 0)

    print('Doing Training')
    if size != 0: print(' - Training', count, 'epochs', 'on last', size, 'days')
    else: print(' - Training', count, 'epochs', 'on all days')    
    
    x_test, y_test, windows_test = data.get_train_data(x, y, windows, size)
    model.train( x_test, y_test, windows_test, count)

    print('Saving Model to file                   ')
    model.save()

# Deletes, remakes, trains, and visualizes model
def retrain(ticker):

    delete_model(ticker)
    create_model(ticker)
    train_model(ticker)
    graph_predictions(ticker, 100)

''' Graphing values '''
# Graphs the last few days of a stock's value
def graph_data(ticker, days):
    charts.visualize(load_data(ticker), days)

# Generates a window for stock for a day and show it
def graph_window(ticker, predict_day):
    stockData = load_data(ticker)
    windowData = stockData.generate_window(predict_day - 100, 100)
    charts.visualize_window(windowData)

def graph_predictions(ticker, days):

    model, loaded = load_model(ticker)
    if not loaded: return

    stockData = load_data(ticker)

    print("Getting model predictions")
    actual_values = []
    predict_next = []
    
    for i in range(days, 1, -1):

        window, next_value = stockData.generate_window(- i - 100, 100)

        predict_next.append( model.predict(window) )
        actual_values.append( window.unscale(next_value) )
    
    print("Graphing model")    
    charts.visualize_predictions(actual_values, predict_next)

# Predicts the next value of a given stock
def predict(ticker, show = True):
    
    model, loaded = load_model(ticker)
    if not loaded: return

    stockData = load_data(ticker)

    window, last_value = stockData.generate_prediction_window(100)

    next_raw = model.predict(window) 
    change = round((next_raw - last_value) / last_value, 6) * 100

    if show:
        print("\nStock: ", ticker)
        print(" - Current Value:", last_value)
        print(" - Prediction of next:", round(next_raw, 2))
        print(" - Change:", round(change, 6), "%")

    return last_value, next_raw, change

# predict all models
def predict_all():
    models = os.listdir('./save_data/models/')

    outputs = []

    # go through all the models you have saved
    for model in models:
        name = model.replace('.h5', '')
        _last, _next, _change = predict(name, False)

        outputs.append(  (_change, _next, model ))

    # sort them by change
    outputs.sort(key=lambda tup: tup[0]) 

    print("\nFinal Predictions: ")
    for i in outputs:
        print(" - ",i[2]," : ", i[0])

''' EVALUATION '''
def evaluate(ticker):
    
    model, loaded = load_model(ticker)
    if not loaded: return

    stockData = load_data(ticker)

    print("Getting model predictions")

    for scope in [1000, 200, 100, 20]:

        delta = 0
        total = 0

        for i in range(scope, 1, -1):

            window, next_value = stockData.generate_window(- i - 100, 100)

            last = window.last_value()

            predict = model.predict(window)
            actual =  window.unscale(next_value)
        
            # Buy it if you though you were going to make a profit
            if (predict > last):
                delta += actual - last
            total += actual - last
        
        print("In the last", scope, "days :")
        print(" - delta of ", round(total, 2), " dollars if you had held")
        print(" - delta of ", round(delta, 2), " dollars if you had used model")

    print("* Notice the model was trained on everything but the last 20 days")
    print("* Ranges other than this one will be unreliable")
