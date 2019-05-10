'''

This PY file takes in a stock ticker value and gives an expected output for the stock for tomorrow

Made by Orcus50

'''

import dataLoader, model
from keras.backend import clear_session
import sys, os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   

# Generate files that do not exist yet
if not os.path.exists('apikey.txt'): open("apikey.txt","w+").close()
if not os.path.exists('stockList.txt'): open("stockList.txt","w+").close()
if not os.path.exists('./Saved_Models'): os.makedirs('./Saved_Models')
if not os.path.exists('./Saved_Data'): os.makedirs('./Saved_Data')

# Load api key
API_KEY = open('apikey.txt', 'r').read()
if API_KEY == '': exit ('Please put api key in file.')

# Takes a ticker and makes a model for prediction
def make_model(stock_ticker):

    # Class to load in needed data from api
    print("Making new model for stock", stock_ticker) 
    
    # load data in with api key (I really should move this key elsewhere, RIP)
    stocks = dataLoader.StockData(stock_ticker, API_KEY)

    # Makes X and Y training data from stockdata
    X, Y, _scaleinfo = stocks.generate_test_data(100)
    
    # Create model with hyper parameters 
    stock_model = model.SequentialModel(sequence_length = 99, classes = 1, hidden_size = 128)

    # Do training (will take a while)
    print("Training:")
    print("Training on full history of stock")
    stock_model.train(X, Y, 20)
    print("Done.       \nTraining on recent few years")
    stock_model.train(X[-1000:], Y[-1000:], 20)
    print("Done.       \nTraining on last year")
    stock_model.train(X[-200:], Y[-200:], 10)
    print("Done.       \nTraining on last week")
    stock_model.train(X[-20:], Y[-20:], 2)
    print("Training Finished!")

    # Model is now trained, lets save it
    stock_model.model.save('./Saved_Models/' + stock_ticker + '.h5')

    print("Model saved in Saved_Models directory", "\n")

# Takes a ticker and predicts the next value assuming a pre-existing model
def predict_model(stock_ticker):

    import dataLoader
    from keras.models import load_model
  
    # Load in the model
    model = load_model('./Saved_Models/' + stock_ticker + '.h5')

    # Load in most recent data for training
    stocks = dataLoader.StockData(stock_ticker, API_KEY)

    # Get most recent 99 days of close data
    close_data = stocks.get_smooth_data()[-99:]
    
    # Model wants specific format
    close_window, _max, _min = stocks.generate_quick_window(close_data)
    shaped_close_window = close_window.reshape((1,99,1))

    # Get output prediction and scale it back up to be useful
    raw_output = model.predict(shaped_close_window)[0,0]
    prediction = (raw_output * (_max - _min)) + _min

    # We have seen into the future
    print(stock_ticker,"Prediction:", close_data[-1], "=>", prediction)

    # Maybe?
    clear_session()

    return prediction, prediction - close_data[-1], (prediction - close_data[-1]) / (prediction)

# Loads in data and generates model if need be
def verify_model_in(stock_ticker):
    
    # First check and see if we have a model for this stock allready
    saved_model_path = './Saved_Models/' + stock_ticker + '.h5'
    exists = os.path.isfile(saved_model_path)

    # If file exists then check how long ago it was created
    if exists:

        print("Previous model found, checking viability...")

        creation_date = os.path.getctime(saved_model_path)
        current_date = time.time()

        # created more than a week ago? -> no longer reliable
        if current_date - creation_date > 604800:
            os.remove(saved_model_path)
            print("Previous model too old. It has been removed.")

        # use it then
        else:
            print("Previous model is good use it.")
            return
    
    make_model(stock_ticker)

# Visualizes what the model looks like
def visualize_model(stock_ticker, span):
    
    import dataLoader
    from keras.models import load_model
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 5
  
    # Load in the model
    print("Loading model for visualization.")
    model = load_model('./Saved_Models/' + stock_ticker + '.h5')

    # Load in most recent data for training
    print("Loading data for visualization.")
    stocks = dataLoader.StockData(stock_ticker, "C816MWPYBH85MAWB")
    data = stocks.get_smooth_data()

    # Get the last 40 days
    last_windows, _real_outputs, _scale_info = stocks.generate_specific_windows(data[-100 - span:], 100)
    
    # get the output predictions
    print("Doing all predictions.")
    output = model.predict(last_windows)[:,0]
    output_scaled = (output * (_scale_info[:,0]-_scale_info[:,1]) + _scale_info[:,1])

    # Graph it all
    build = []
    for i in range(span - 1):
        
        last_real = data[-span - 1 + i]
        next_real = data[-span + i]
        next_predict = output_scaled[-span + 1 + i]
        
        build.append((i,i+1))
        build.append((last_real, next_predict))
        build.append('r')

        if (next_predict > last_real):
            if (next_real > last_real):
                plt.axvspan(i+0.1, i+0.9, color='green', alpha=0.2)
            else:
                plt.axvspan(i+0.1, i+0.9, color='red', alpha=0.2)
                
    # Each red line point from actual value of day to predicted value of the next day
    print("Graphing.")
    plt.plot(*build)

    # Blue is actual
    plt.plot(data[-span - 1:])

    plt.show()

    # Maybe?
    clear_session()


# Store requested ticker value and print info
if len(sys.argv) == 1: exit("Please give run mode when calling.")
run_type = sys.argv[1]

# Train on single stock
if (run_type == 'single'):
    
    # Make sure they provide a stock too
    if len(sys.argv) == 2: exit("Please give run mode when calling.")
    stock_ticker = sys.argv[2]

    make_model(stock_ticker)
    visualize_model(stock_ticker, 40)

# Do all stocks in stockList.txt file
if (run_type == 'all'):

    # Read from file list
    stocks = open('stockList.txt', 'r').read().split("\n")
    outputCSV = open('predictionOutput.csv', 'w')

    for stock in stocks:
        if stock == '': continue

        # Call guerentees model exists
        verify_model_in(stock)

        # Do prediction
        prediction, delta, percent = predict_model(stock)    
        outputCSV.write(stock + ',' + str(prediction) + ',' + str(delta) + ',' + str(percent) + '\n')
        outputCSV.flush()

# Shows the models output
if (run_type == 'show'):

    # Make sure they provide a stock too
    if len(sys.argv) == 2: exit("Please give run mode when calling.")
    stock_ticker = sys.argv[2]
    if len(sys.argv) == 3: exit("Please give range of days when calling.")
    days = int(sys.argv[3])

    # Load model if it doesnt exist
    verify_model_in(stock_ticker)

    # Do visualization
    visualize_model(stock_ticker, days)


