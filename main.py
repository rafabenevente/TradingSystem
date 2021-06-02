import pandas as pd
import pickle
import logging
import os
import pandas_ta as ta
import shutil
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from datetime import datetime
from enum import Enum


class Operation(Enum):
    HOLD = 1
    LONG = 2
    SHORT = 3


class PredictResult(Enum):
    DOWN = 0
    UP = 1


folder_path = os.path.join("models", datetime.today().strftime("%Y_%m_%d_%H_%M_%S"))

# def get_train_data():
#     x = pd.read_csv("./DadosTreino.csv", sep=";")
#     x = x.iloc[::-1]
#     x.pop(x.columns[0])
#     y = x.pop(x.columns[-1])
#
#     x = add_features(x, "train")
#
#     return x, y
#
#
# def get_test_data():
#     x = pd.read_csv("./DadosTeste.csv", sep=";")
#     x = x.iloc[::-1]
#     x.pop(x.columns[0])
#     y = x.pop(x.columns[-1])
#
#
#
#     return x, y

def add_features(x):
    # Simple moving average
    x.ta.sma(length=6, fillna=0, append=True)
    x.ta.sma(length=10, fillna=0, append=True)
    x.ta.sma(length=12, fillna=0, append=True)
    x.ta.sma(length=24, fillna=0, append=True)
    x.ta.sma(length=30, fillna=0, append=True)

    # Relative Strength Index
    x.ta.rsi(length=6, fillna=0, append=True)
    x.ta.rsi(length=10, fillna=0, append=True)
    x.ta.rsi(length=12, fillna=0, append=True)
    x.ta.rsi(length=24, fillna=0, append=True)
    x.ta.rsi(length=30, fillna=0, append=True)

    # Rate of Change
    x.ta.roc(length=6, fillna=0, append=True)
    x.ta.roc(length=12, fillna=0, append=True)
    x.ta.roc(length=24, fillna=0, append=True)
    x.ta.roc(length=28, fillna=0, append=True)

    # Commodity Channel Index
    x.ta.cci(length=6, fillna=0, high="max", low="min", append=True)
    x.ta.cci(length=12, fillna=0, high="max", low="min", append=True)
    x.ta.cci(length=14, fillna=0, high="max", low="min", append=True)
    x.ta.cci(length=28, fillna=0, high="max", low="min", append=True)

    # Save
    x.to_csv(path_or_buf=os.path.join(folder_path, "data.csv"), sep=";")
    return x



def run():
    x = pd.read_csv("./DadosCompletos.csv", sep=";")
    x = x.iloc[::-1]  # Reverse data
    x = add_features(x)
    x = x.dropna()
    x.pop(x.columns[0])
    y = x.pop("ClassBinVarFut")

    x_train = x.iloc[:464,]  # >= 31/01/2018
    y_train = y.iloc[:464,]


    x_val = x.iloc[464:,]  # < 31/01/2018
    y_val = y.iloc[464:,]

    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    # y_pred = gnb.predict(x_test)
    # print(f"GaussianNB - total {x_test.shape[0]} mislabeled : {(y_test != y_pred).sum()}")

    bnb = BernoulliNB()
    bnb.fit(x_train, y_train)
    # y_pred = bnb.predict(x_test)
    # print(f"BernoulliNB - total {x_test.shape[0]} mislabeled : {(y_test != y_pred).sum()}")
    # y_proba = bnb.predict_proba(x_test)

    # Save model
    file = open(os.path.join(folder_path, "classifier.pickle"), "wb")
    pickle.dump(bnb, file)
    file.close()

    do_backtest(bnb, x_val, y_val)


def do_backtest(model, x_val, y_val):
    # y_test = model.predict_proba(x_test)
    y_pred = model.predict(x_val)
    logging.basicConfig(filename=os.path.join(folder_path, "execution.log"), level=logging.INFO)
    logging.info(f"TEST SET - total {x_val.shape[0]} mislabeled : {(y_pred != y_val).sum()}")
    apply_trading_strategy(x_val, y_pred)


def apply_trading_strategy(x_test, y_test):
    logging.basicConfig(filename=os.path.join(folder_path, "execution.log"), level=logging.INFO)
    operation = Operation.HOLD
    total_money = x_test.close.iloc[0] * 100
    stock_number = 0
    stop_loss = 0.03  # 2% stop loss
    operation_price = 0
    logging.info(f"Stop Loss percent:{stop_loss}")
    logging.info("#---------------------------------------------------------------------------------------------------------------#")
    logging.info(f"Start money:{total_money}")

    for index, row in x_test.iterrows():
        sugested_operation = PredictResult(y_test[index])

        current_price = row.close

        # Stop Loss
        if operation_price > (current_price * (1 + stop_loss)) and operation == Operation.SHORT:
            operation = Operation.HOLD
            total_money = total_money + (((current_price - operation_price) * stock_number) * -1)
            stock_number = 0
            operation_price = 0
            logging.info(
                f"SELL - STOP LOSS - Operation:{Operation.SHORT} Price:{current_price} - RemainCash = {total_money}")
        if operation_price < (current_price * (1 - stop_loss)) and operation == Operation.LONG:
            operation = Operation.HOLD
            total_money = total_money + (current_price * stock_number)
            stock_number = 0
            operation_price = 0
            logging.info(
                f"SELL - STOP LOSS - Operation:{Operation.LONG} Price:{current_price} - RemainCash = {total_money}")

        if operation == Operation.HOLD:
            if sugested_operation == PredictResult.UP:  # Long operation condition
                # Start long operation
                operation = Operation.LONG
                operation_price = current_price
                stock_number = min(int(total_money / operation_price), 100)
                total_money = total_money - (stock_number * operation_price)
                logging.info(
                    f"BUY - Operation:{operation} - Price:{operation_price} - Qt: {stock_number} - RemainCash = {total_money}")
            else:  # Short operation condition
                # Start short operation
                operation = Operation.SHORT
                operation_price = current_price
                stock_number = min(int(total_money / operation_price), 100)
                # total_money = total_money - (stock_number * operation_price)
                logging.info(
                    f"BUY - Operation:{operation} - Price:{operation_price} - Qt: {stock_number} - RemainCash = {total_money}")
        elif operation == Operation.LONG:
            if sugested_operation == PredictResult.DOWN:
                # In a long with down predict end long operation
                operation = Operation.HOLD
                total_money = total_money + (current_price * stock_number)
                stock_number = 0
                operation_price = 0
                logging.info(f"SELL - Operation:{Operation.LONG} Price:{current_price} - RemainCash = {total_money}")
            else:
                # In a long with up predict maintain long operation
                operation = Operation.LONG
        else:  # SHORT
            if sugested_operation == PredictResult.UP:
                # In a short with up predict end short operation
                operation = Operation.HOLD
                total_money = total_money + (((current_price - operation_price) * stock_number) * -1)
                stock_number = 0
                operation_price = 0
                logging.info(f"SELL - Operation:{Operation.SHORT} Price:{current_price} - RemainCash = {total_money}")
            else:
                # In a short with up predict maintain short operation
                operation = Operation.SHORT

    if operation == Operation.LONG:
        total_money = total_money + (current_price * stock_number)
        stock_number = 0
        operation_price = 0
        logging.info(f"SELL - Operation:{Operation.LONG} Price:{current_price} - RemainCash = {total_money}")
    elif operation == Operation.SHORT:
        total_money = total_money + (((current_price - operation_price) * stock_number) * -1)
        stock_number = 0
        operation_price = 0
        logging.info(f"SELL - Operation:{Operation.SHORT} Price:{current_price} - RemainCash = {total_money}")

    logging.info(f"End money:{total_money}")
    shutil.move(folder_path, folder_path+"__"+str(total_money).replace(".", "_"))





if __name__ == '__main__':
    os.mkdir(folder_path)
    run()

