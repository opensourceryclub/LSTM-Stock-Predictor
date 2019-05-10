'''

Generates a model to predict stock prices given data

'''
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop

class SequentialModel:
    
     # init functions
    def __init__(self, sequence_length, classes, hidden_size):

        self.sequence_length = sequence_length
        self.classes = classes
        self.hidden_size = hidden_size

        self.create_model()
        
    def create_model(self):

        # Size of vector in the hidden layer.
        hidden_size = 64 
        # Initialize Sequential Model
        model = Sequential()
        model.add(LSTM(self.hidden_size, input_shape=(self.sequence_length,self.classes)))
        # Add the output layer 
        model.add(Dense(self.classes)) 
        # Optimization through RMSprop
        optimizer_new = RMSprop() 
        # Consider cross Entropy loss. Why? MLE of P(D | theta)
        model.compile(loss='mean_squared_error', optimizer=optimizer_new) 

        self.model = model
        return self.model
    
    def visualizeTrain(self, epoch, logs):
        print(epoch, " / ", self.epochs, end="\r")

    def train(self, x, y, epochs):

        self.epochs = epochs;
        viewOut = LambdaCallback(on_epoch_end=self.visualizeTrain)
        
        self.model.fit(x, y,
              batch_size=128,
              epochs=epochs, 
              verbose = 0,
              callbacks=[viewOut])
        
    def predict(self, data):
        return self.model.predict(data)