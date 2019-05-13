

# LSTM-STOCK-PREDICTOR


Use the command line to train and visualize stock data


## Getting Started

1. Install Dependencies w/ pip

```
pip install keras tenserflow numpy matplotlib pylab requests
```

2. Get an api key for stock data from [AlphaVantage](https://www.alphavantage.co/)

3. Navigate to project and start CLI for predictor

``` 
python Stock-Shell.py 
```

5. Type `init` to initialize files

6. Put key in apikey.txt file

## Example

#### Type `help` for a more detailed list of commands

1. Create model for stock ticker, lets use AAPL
```
> create AAPL

Creating a new model
New model created and saved
```

2. Chart downloaded stock data if you want
```
> chart data AAPL

Loading in data
 - Downloading and saving full stock data
Showing last 100 days worth of data
```
![](http://drive.google.com/uc?export=view&id=1woK0nVOoIt-BMYyGJStSuI9q_ImcEXKt)
3. Train model on data. More training options availible. Look at help file
```
> train AAPL normal

Doing Training
 - Training 30 epochs on all days
 - Training 20 epochs on last 1000 days
 - Training 20 epochs on last 500 days
 - Training 20 epochs on last 100 days
Saving Model to file
```
4. You may need to train more in the future, you can further train existing models if you want
```
> train AAPL 40 200

Doing Training
 - Training 40 epochs on last 200 days
Saving Model to file
```
5. Visualize model predictions for stock over last 100 days
```
> chart model AAPL
```
 Blue shows actual value of stock. Red shows predicted next value for that day.
 Days where the model predicts profit are highlighted by how much you would have that day.
![](http://drive.google.com/uc?export=view&id=1MHdXtx6p9lqfdCFZRWUTlujoBSoOGOPx)
6. Evaluates the models prediction potential profit
```
> evaluate AAPL

In the last 1000 days :
 - delta of  68.23  dollars if you had held
 - delta of  300.7  dollars if you had used model
In the last 200 days :
 - delta of  5.04  dollars if you had held
 - delta of  94.02  dollars if you had used model
In the last 100 days :
 - delta of  34.24  dollars if you had held
 - delta of  47.13  dollars if you had used model
In the last 20 days :
 - delta of  0.04  dollars if you had held
 - delta of  0.99  dollars if you had used model
* Notice the model was trained on everything but the last 20 days
* Ranges other than this may be massivley unreliable
```
7. Predict the next day
```
> predict AAPL

Stock:  AAPL
 - Current Value: 192.085
 - Prediction of next: 190.53
 - Change: -0.8105 %
```
8. Exit
```
> exit
```

## Special thanks to Zack:

HEY U STUOIPD WHY TEHYERE BO DEPENENENCIES IDK WHAT TO FINSALL VIRtuAK ENV MARBE THANKS U GOOD PROGRAMmer  