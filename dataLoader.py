'''

Loads the data through api calls and processes it for the model

'''

#Data Class
import requests
from datetime import timedelta
import datetime
import numpy as np
import statistics
import pandas as pd
import os, time

class StockData:

    # init functions
    def __init__(self, stock_ticker, api_key, use_full = True):

        self.ticker = stock_ticker
        self.key = api_key
        self.use_full = use_full

        # Do loading
        self.load_data()           # Load in raw json

    # Send api request
    def request_data(self, mode):
        
        # Send the request
        params = {'function':'TIME_SERIES_DAILY_ADJUSTED',
                'symbol':self.ticker,'apikey':self.key,
                'outputsize':mode} 
        responce = requests.get('https://www.alphavantage.co/query', params).json()

        if not "Time Series (Daily)" in responce: exit("Failed to load " + self.ticker)
        return self.populate_data(responce["Time Series (Daily)"])

    # Saves the data to a file
    def save_data(self, data):

        file_path = './Saved_Data/' + self.ticker + '.txt'
        save_file = open(file_path, 'w')

        for i in data:
            save_file.write(str(i) + "\n")

    # Loads in the data if it can from file, else url
    def load_data(self):

        # Check and see if file allready exists?
        file_path = './Saved_Data/' + self.ticker + '.txt'
        
        if os.path.isfile(file_path):

            # Load in past values as float
            past_file = open(file_path, 'r')
            saved_data = past_file.read().split("\n")
            past_values = []
            for i in saved_data:
                if (i != ''):
                    past_values.append(float(i))

            # Check when file was updated
            creation_date = os.path.getctime(file_path)
            current_date = time.time()

            # Newer than an hour, dont worry bout it
            if current_date - creation_date < 3600:
                self.data = np.array(past_values)
                past_file.close()
                return
                
            # We need to see what to update it with
            recent = self.request_data('compact')

            # Check last ten traiding days
            for i in range(10):
                
                # Find the point of overlap
                if recent[-i] == past_values[-2] and recent[- i - 1] == past_values[-3]:

                    # add the values back ontop
                    past_values.pop()
                    past_values.pop()
                    past_values.extend(recent[-i:])
                    
                    self.data = np.array(past_values)
                    past_file.close()
                    self.save_data(past_values)
                    return

            # If you got here there was somehthing wrong with the data, remove it
            past_file.close()
            os.remove(file_path)

        # File does not exist, load that in
        data = self.request_data('full')
        self.data = np.array(data)
        self.save_data(data)
       
    # Read the close data out
    def populate_data(self, data):
    
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
            if day in data.keys():  
        
                # stock splits
                split_coef = float(data[day]["8. split coefficient"])
                split_mult /= split_coef     
                
                # append the days data
                processed_data.insert( 0 , float(data[day]['4. close']) * split_mult )

        return processed_data
    
    # window normalization, I is the first in span, i[-1] is prediction day
    def generate_window(self, model_data, i, window_size):
        
        data = model_data[i: i + window_size]

        ma = data.max()
        mi = data.min()
        
        data = (data - mi) / (ma-mi)
        
        return data, ma, mi

    # makes a window for a specific, predifined data range
    def generate_quick_window(self, data):
        ma = data.max()
        mi = data.min()
        data = (data - mi) / (ma-mi)
        
        return data, ma, mi

    # Generates all windows for specific test data
    def generate_specific_windows(self, data, window_size):
        
        window_count = data.shape[0] - window_size
        
        X = []; Y = []
        scale_info = [];
        
        for i in range(window_count):
            
            window_data, max_val, min_val = self.generate_window(data, i, window_size)
            
            scale_info.append([max_val, min_val])
            X.append(window_data[0 : -1].reshape(window_size-1, 1))
            Y.append([window_data[-1]])
            
        return np.stack(X, axis=0), np.stack(Y, axis=0), np.stack(scale_info, axis=0)
    
    # Get single day running average data
    def get_smooth_data (self):
        return ( self.data + np.roll(self.data, 1) ) / 2
        
    # Generates all test data for a given key
    def generate_test_data(self, window_size):
        
        window_count = self.data.shape[0] - window_size
        
        X = []; Y = []
        scale_info = [];
        
        for i in range(window_count):
            
            window_data, max_val, min_val = self.generate_window(self.data, i, window_size)
            
            scale_info.append([max_val, min_val])
            X.append(window_data[0 : -1].reshape(window_size-1, 1))
            Y.append([window_data[-1]])
            
        return np.stack(X, axis=0), np.stack(Y, axis=0), np.stack(scale_info, axis=0)
    