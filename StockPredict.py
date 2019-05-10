'''

This PY file takes in a stock ticker value and gives an expected output for the stock for tomorrow

Made by Orcus50

'''

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

# If no args were given, thats it, else, go to the end of the specific commands
if len(sys.argv) == 1: exit("Init Done")
run_type = sys.argv[1]

# Only import this if you get this far, big import
import runModes

# Train on single stock
if (run_type == 'single'):
    
    # Make sure they provide a stock too
    if len(sys.argv) == 2: exit("Please give stock when calling.")
    stock_ticker = sys.argv[2]

    runModes.make_model(stock_ticker, API_KEY)
    runModes.visualize_model(stock_ticker, 40, API_KEY)

# Do all stocks in stockList.txt file
if (run_type == 'all'):

    # Read from file list
    stocks = open('stockList.txt', 'r').read().split("\n")
    outputCSV = open('predictionOutput.csv', 'w')

    for stock in stocks:
        if stock == '': continue

        # Call guerentees model exists
        runModes.verify_model_in(stock, API_KEY)

        # Do prediction
        prediction, delta, percent = runModes.predict_model(stock, API_KEY)    
        outputCSV.write(stock + ',' + str(prediction) + ',' + str(delta) + ',' + str(percent) + '\n')
        outputCSV.flush()

# Shows the models output
if (run_type == 'show'):

    # Make sure they provide a stock too
    if len(sys.argv) == 2: exit("Please give stock when calling.")
    stock_ticker = sys.argv[2]
    if len(sys.argv) == 3: exit("Please give range of days when calling.")
    days = int(sys.argv[3])

    # Load model if it doesnt exist
    runModes.verify_model_in(stock_ticker, API_KEY)

    # Do visualization
    runModes.visualize_model(stock_ticker, days, API_KEY)


