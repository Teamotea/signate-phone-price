import os
import pandas as pd


def load_train_data():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    train = pd.read_csv(dir_name + '/train.csv')
    train_x = train.drop('price_range', axis=1)
    train_y = train['price_range']
    return train_x, train_y


def load_test_data():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    test_x = pd.read_csv(dir_name + '/test.csv')
    return test_x


def load_data():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    train = pd.read_csv(dir_name + '/train.csv')
    train_x = train.drop('price_range', axis=1)
    train_y = train['price_range']
    test_x = pd.read_csv(dir_name + '/test.csv')
    return train_x, train_y, test_x
