'''

    Implements sequential python machien leaning model
    LSTM using Keras

'''

from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model

import os, time
import numpy as np

class StockModel:

    def __init__(self, ticker):
        
        # Train info
        self.total_training = 0
        self.sequence_length = 100
        self.hidden_size = 128

        self.ticker = ticker

    # Tries to load new model if it exists, else just make a new one    
    def load_model(self):

        print("Loading in model")

        # Check and see if file allready exists?
        file_path = './save_data/models/' + self.ticker + '.h5'
        
        if os.path.isfile(file_path):

            # Check for new model
            creation_date = os.path.getctime(file_path)
            if time.time() - creation_date < 7 * 24 * 60 * 60:

                print(" - Prior model found, loading it")
                self.model = load_model(file_path)
                return True
    
            # Old, lets not use it
            print(" - Prior model found, but older than a week, not using")
    
        print(" - No model found to load, create a model first with 'create'")
        return False

    # Creates a new model for a stock
    def create_model(self):

        file_path = './save_data/models/' + self.ticker + '.h5'
        if os.path.isfile(file_path):
            print("This model allready exists! Aborting. Delete model first")
            return False

        # Initialize Sequential Model
        model = Sequential()

        # LSTM
        model.add(LSTM(self.hidden_size, input_shape=(self.sequence_length,1)))

        # Add the output layer 
        model.add(Dense(1))     
        model.add(Activation('sigmoid'))
        
        # Consider cross Entropy loss??
        model.compile(loss='mean_squared_error', optimizer=RMSprop()) 

        print("New model created and saved")
        self.model = model
        self.save()
        return True

    def save(self):
        self.model.save('./save_data/models/' + self.ticker + '.h5')
    
    # Visualizes process
    def visualize_train(self, epoch, logs):
        print("   > ", epoch, " / ", self.epochs, "  E:", round(logs['loss'], 3), end="\r")

    # train on data
    def train(self, x, y, windows, epochs):

        # Print info
        self.epochs = epochs

        # Add callbacks
        callbacks = []
        callbacks.append(LambdaCallback(on_epoch_end=self.visualize_train))

        # Shape inputs
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = y.reshape(y.shape[0], 1)

        self.model.fit(x, y,
            batch_size=128,
            epochs=epochs, 
            verbose = 0,
            callbacks=callbacks)

        self.total_training += epochs

    # make a prediction
    def predict(self, window):

        shaped = window.buffer_scaled.reshape(1, window.buffer_scaled.shape[0], 1)

        return window.unscale( self.model.predict(np.array( shaped ))[0,0] )
        