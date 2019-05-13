'''

    Starts a CLI that runs the stock application

    Made by Orcus50

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
from scripts import controller

# Little utility thing for command line
def check(args, length, error):

    if len(args) == length:
        print(error)
        return True
    return False

def get_value(args, index, default):
    if len(args) <= index: return default
    if type(default) == type('str'): return args[index]
    return int(args[index])

print('\n-- Stock Market Shell -- \n')
print(' type help for commands and exit to close \n')

while True:

    # Read in input
    args = input('\n> ').split(' ')

    if len(args) == 0: continue

    # EXIT
    elif args[0] == 'exit': exit('bye ;)')

    # INIT
    elif args[0] == 'init':
        controller.init_program()

    # HELP
    elif args[0] == 'help':
        helpFile = open("./helpFile.txt", "r")
        print(helpFile.read())
        helpFile.close()
        
    # DATA
    elif args[0] == 'data':
        if check(args, 1, 'Please provide ticker value.'): continue

        controller.load_data(args[1])

    # Create model
    elif args[0] == 'create':
        if check(args, 1, 'Please provide ticker value.'): continue

        ticker = args[1]
        controller.create_model(ticker)

    # Delete model
    elif args[0] == 'delete':
        if check(args, 1, 'Please provide ticker value.'): continue

        ticker = args[1]
        controller.delete_model(ticker)

    # CHART
    elif args[0] == 'chart':
        if check(args, 1, 'Please privide what to chart.'): continue
        
        # VIEW RAW STOCK DATA
        if args[1] == 'data':
            if check(args, 2, 'Please provide ticker value.'): continue
            
            ticker = args[2]
            days = get_value(args, 3, 100)

            controller.graph_data(ticker, days)
            
        # VIEW MODEL WINDOW DATA
        if args[1] == 'window':
            if check(args, 2, 'Please provide ticker value.'): continue
            
            ticker = args[2]
            predict_day = - get_value(args, 3, 1)

            controller.graph_window(ticker, predict_day)

        # VIEW MODEL PREDICTION DATA
        if args[1] == 'model':
            if check(args, 2, 'Please provide ticker value.'): continue

            ticker = args[2]
            days = get_value(args, 3, 100)

            controller.graph_predictions(ticker, days)

    # Train model
    elif args[0] == 'train':
        if check(args, 1, 'Please provide ticker value.'): continue
        ticker = args[1]

        if check(args, 2, 'Please provide train type'): continue

        if args[2] == 'normal':
            controller.train_model(ticker)

        else:
            epochs = get_value(args, 2, 5)
            days = get_value(args, 3, 500)

            controller.train_specific(ticker, epochs, days)

    # Retrain
    elif args[0] == 'retrain':
        if check(args, 1, 'Please provide ticker value.'): continue
        ticker = args[1]
        controller.retrain(ticker)

    elif args[0] == 'evaluate':
        if check(args, 1, 'Please provide ticker value.'): continue
        ticker = args[1]
        controller.evaluate(ticker)

    elif args[0] == 'predict':
        if check(args, 1, 'Please provide ticker value.'): continue

        if args[1] == 'all':
            controller.predict_all()
        else:
            ticker = args[1]
            controller.predict(ticker)

    else:
        print('Unknown command')

