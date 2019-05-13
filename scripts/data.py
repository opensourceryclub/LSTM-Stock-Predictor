'''
    Handles data loading and pre-processing
'''

import requests
from datetime import timedelta
import datetime
import numpy as np
import statistics
import os, time

class DataLoader:
    
    def __init__(self, stock_ticker, api_key):

        self.ticker = stock_ticker
        self.key = api_key

    # Send api request, mode = 'compact' or 'full'
    def request_data(self, mode):
        
        # Send the request
        params = {'function':'TIME_SERIES_DAILY_ADJUSTED',
                'symbol':self.ticker,'apikey':self.key,
                'outputsize':mode} 
        responce = requests.get('https://www.alphavantage.co/query', params).json()

        if not "Time Series (Daily)" in responce: exit("Failed to load " + self.ticker)

        # return json responce
        return self.parse_json_data(responce["Time Series (Daily)"])

    # Takes json from alphavantage and gets readable array output
    def parse_json_data(self, json):

        # start 20 years ago if you can
        offset = timedelta(days=1)
        date = datetime.date.today() + offset
        end = datetime.date.today() - timedelta(days = 20 * 365) 

        # keep track of if stocks split
        split_mult = 1.0
        
        # output of each day
        processed_data = []
        
        while True:
        
            # shift day
            date -= offset
            day = str(date)
            if date < end:  break
            
            # account for some days not being in the list
            if day in json.keys():  
        
                # stock splits
                split_coef = float(json[day]["8. split coefficient"])
                split_mult /= split_coef     
                
                # append the days data
                processed_data.insert( 0 , float(json[day]['4. close']) * split_mult )

        return processed_data

    # Takes in a stock ticker and loads it either from api or files
    def load (self):
        
        print("Loading in data")

        # Check and see if file allready exists?
        file_path = './save_data/data/' + self.ticker + '.txt'
        
        if os.path.isfile(file_path):

            # Loads the data in from a file
            past_data = self.read_data(file_path)

            # Check when file was updated
            creation_date = os.path.getctime(file_path)
            # Newer than two hours, dont worry about syncing with the api again
            if time.time() - creation_date < 7200:
                print(" - Prior saved data found, using")
                self.data = np.array(past_data)
                return
                
            # We need to see what to update it with
            recent = self.request_data('compact')

            # Check last twenty traiding days
            for i in range(20):
                
                # Find the point of overlap
                if recent[-i] == past_data[-2] and recent[- i - 1] == past_data[-3]:
                    
                    print(" - Prior saved data found, fetching recent data and using")

                    # add the values back ontop
                    past_data.pop()
                    past_data.pop()
                    past_data.extend(recent[-i:])
                    
                    self.data = np.array(past_data)
                    self.save_data(file_path, past_data)
                    return

            # If you got here there was somehthing wrong with the data, remove it and just re-download
            print(" - Prior saved data found, but corrupted, deleting")
            os.remove(file_path)

        # We have no local copy, download in full
        print(" - Downloading and saving full stock data")
        data = self.request_data('full')
        self.data = np.array(data)
        self.save_data(file_path, data)
        
    # Saves the data to a file
    def save_data(self, file_path, data):

        save_file = open(file_path, 'w')
        for i in data: save_file.write(str(i) + "\n")
        save_file.close()

    def read_data(self, file_path):
        
        past_file = open(file_path, 'r')
        saved_data = past_file.read().split("\n")
        past_file.close()

        past_values = []
        for i in saved_data:
            if (i != ''):
                past_values.append(float(i))

        return past_values

    # Get data as StockData object    
    def get_data(self):

        # Do a roll on the data to smooth it out?
        smoothed = ( self.data + np.roll(self.data, 1) ) / 2

        return StockData(smoothed)


# Datatype that stores and maninpulates stock data once loaded
class StockData:

    def __init__(self, data):
        self.data = data

    # Generates a trainable window for the stock at an index
    def generate_window(self, start_index, size):

        X = self.data[start_index : start_index + size]
        Y = self.data[start_index + size]

        window = Window(X)
        next_value = window.scale(Y)

        return (window, next_value)

    def generate_prediction_window(self, size):
        
        X = self.data[-size:]
        window = Window(X)

        return (window, X[-1])

    def generate_all_windows(self, size):

        count = len(self.data) - size
        windows = []; X = []; Y = []

        for i in range(count):
            
            window, next_value = self.generate_window(i, size)
            windows.append( window )
            X.append( window.buffer_scaled )
            Y.append( next_value )

        return windows, X, Y

    # Gets train data and drops off test data (20 days)
    def get_train_data(self, x, y, windows, size):
        if size == 0:
            return x[:-20], y[:-20], windows[:-20]
        return x[-size - 20: -20], y[-size - 20 : -20], windows[-size -20: -20]

# Datatype that converts a range of data into a window
class Window:

    def __init__ (self, data):
        self.data = data

        self.max = data.max()
        self.min = data.min()
        self.span = self.max - self.min
        
        # Max set to 1, min set to 0
        self.full_scaled = ((data - self.min) / self.span) 

        # Max set to 0.8, min set to 0.2
        self.buffer_scaled = (self.full_scaled * 0.6) + 0.2

    # last value before prediction
    def last_value(self):
        return self.data[-1]

    # Takes another point and fits it onto the graph
    def scale(self, data):
        
        data_full_scaled = (data - self.min) / self.span
        data_buffer_scaled = (data_full_scaled * 0.6) + 0.2

        return data_buffer_scaled

    # Take a point scaled onto the graph and scaled it back to a normal value
    def unscale(self, data_buffer_scaled):

        data_full_scaled = (data_buffer_scaled - 0.2) / 0.6
        data_no_scaled = (data_full_scaled * self.span) + self.min
        
        return data_no_scaled
