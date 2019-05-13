'''
    Handles the graphing of the project

'''

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

# Takes StockData element to view
def visualize(data, days):
    print("Showing last", days, "days worth of data")
    rcParams['figure.figsize'] = 10, 5
    plt.plot(data.data[-days:])
    plt.show()

def visualize_window(window_data):

    print("Showing input window and next output")
    window, next_point = window_data

    rcParams['figure.figsize'] = 10, 5
    plt.plot(window.buffer_scaled)
    plt.plot(100, next_point, marker = 'X')
    plt.show()

def visualize_predictions(actual, predicted):
    
    rcParams['figure.figsize'] = 10, 5

    # Graph it all
    build = []
    fail = []; correct = []
    all_values = []

    for i in range(len(actual)-1):
        
        build.append((i,i+1))
        build.append((actual[i], predicted[i+1]))
        build.append('r')

        if (predicted[i+1] > actual[i]):

            dif =  abs(predicted[i + 1] - actual[i + 1])
            all_values.append(dif)

            if (actual[i+1] > actual[i]):
                correct.append ( (dif, i) )
            else:
                fail.append ( (dif, i) )
                
    # Do error bars
    error_array = np.array(all_values)
    _max = error_array.max()
    _min = error_array.min()

    for yes, i in correct:
        alpha = 1 - (yes - _min) / (_max - _min)
        plt.axvspan(i+0.1, i+0.9, color='green', alpha=alpha * 0.5)
    for no, i in fail:
        alpha = (no - _min) / (_max - _min)
        plt.axvspan(i+0.1, i+0.9, color='red', alpha= alpha * 0.5)

    # Each red line point from actual value of day to predicted value of the next day
    plt.plot(*build)

    # Blue is actual
    plt.plot(actual[:])
    
    plt.show()